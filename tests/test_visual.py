from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from gnss_sim.visual import parse_result, render_case
from gnss_sim.visual_runner import _engineering_cases, _pilot_inputs, _request, load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "p6-visual.json"


def test_renderer_is_deterministic_and_fixed_size(tmp_path):
    case = _engineering_cases()[2]
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    render_case(case, first)
    render_case(case, second)
    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    with Image.open(first) as image:
        assert image.size == (1800, 1200)
        assert image.format == "PNG"


@pytest.mark.parametrize(
    ("task", "response", "expected"),
    [
        ("point", '{"N":[0,364],"E":[],"U":[120]}', [0, 364]),
        ("range", '```json\n{"N":[[0,364]],"E":[],"U":[]}\n```', [(0, 364)]),
    ],
)
def test_valid_response_parses_to_p4_schema(task, response, expected):
    result = parse_result(response, "case_0001", task)
    assert result.status == "success"
    assert result.method == "visual-v1"
    assert result.predictions.N == expected


@pytest.mark.parametrize(
    ("task", "response"),
    [
        ("point", "There are no anomalies: {\"N\":[],\"E\":[],\"U\":[]}"),
        ("point", '{"N":[365],"E":[],"U":[]}'),
        ("point", '{"N":[true],"E":[],"U":[]}'),
        ("point", '{"N":[],"N":[4],"E":[],"U":[]}'),
        ("point", '{"N":[],"E":[]}'),
        ("range", '{"N":[[4,3]],"E":[],"U":[]}'),
        ("range", '{"N":[[2,4,5]],"E":[],"U":[]}'),
        ("range", '{"N":[[0.5,4]],"E":[],"U":[]}'),
        ("range", ""),
    ],
)
def test_invalid_response_is_not_repaired(task, response):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_result(response, "case_0001", task)


def test_model_request_is_image_only_and_exactly_once(tmp_path):
    config = load_config(CONFIG)
    image = tmp_path / "image.png"
    render_case(_engineering_cases()[0], image)
    captured = []

    def respond(request):
        body = json.loads(request.content)
        captured.append(body)
        return httpx.Response(200, json={"model": "qwen3.8-flash", "id": "resp-test",
                                         "usage": {"prompt_tokens": 10},
                                         "choices": [{"finish_reason": "stop", "message": {
                                             "content": '{"N":[],"E":[],"U":[]}'}}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result, raw = _request(client, "https://example.test/v1", "test-secret",
                               image.read_bytes(), "point", "engineering_01", config)
    assert result.status == raw["status"] == "success"
    assert len(captured) == 1
    assert captured[0]["model"] == "qwen3.8-flash"
    assert captured[0]["enable_thinking"] is False
    content = captured[0]["messages"][0]["content"]
    assert [item["type"] for item in content] == ["text", "image_url"]
    assert "case_type" not in json.dumps(captured[0])
    assert "truth" not in json.dumps(captured[0])


def test_missing_model_response_becomes_failed_row(tmp_path):
    config = load_config(CONFIG)
    image = tmp_path / "image.png"
    render_case(_engineering_cases()[0], image)

    def respond(_request):
        return httpx.Response(200, json={"model": "qwen3.8-flash", "choices": []})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result, raw = _request(client, "https://example.test/v1", "test-secret",
                               image.read_bytes(), "range", "engineering_01", config)
    assert result.status == raw["status"] == "failed"
    assert result.predictions.N == result.predictions.E == result.predictions.U == []
    assert raw["error_type"] == "IndexError"


def test_pilot_input_scan_does_not_open_truth(tmp_path, monkeypatch):
    pilot = tmp_path / "pilot-v1"
    cases = []
    for index in range(1, 301):
        case_id = f"case_{index:04d}"
        path = pilot / "cases" / case_id / "input.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"input-only")
        cases.append({"case_id": case_id,
                      "input_sha256": hashlib.sha256(b"input-only").hexdigest()})
    manifest = pilot / "manifest.json"
    summary = pilot / "summary.json"
    manifest.write_text(json.dumps({"pilot_id": "pilot-v1", "cases": cases}), encoding="utf-8")
    summary.write_text("{}", encoding="utf-8")
    config = load_config(CONFIG)
    config["pilot_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config["pilot_summary_sha256"] = hashlib.sha256(summary.read_bytes()).hexdigest()
    original = Path.read_bytes

    def guarded_read(path):
        if path.name == "truth.json":
            raise AssertionError("inference opened truth")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    assert len(_pilot_inputs(pilot, config)) == 300
