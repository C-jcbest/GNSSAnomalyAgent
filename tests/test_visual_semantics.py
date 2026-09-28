import json
from pathlib import Path

import httpx
import pytest

from gnss_sim import visual_semantics as sem

CONFIG = sem.ROOT / "configs/p7a-visual-semantics.json"


def test_configuration_only_changes_identity_and_prompts(tmp_path):
    config = sem.load_config(CONFIG)
    assert config["method"] == "visual-semantics-v2"
    config["model"]["temperature"] = 0.5
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Only protocol"):
        sem.load_config(path)


@pytest.mark.parametrize("body, expected", [
    ('{"N":[100],"E":[],"U":[]}', "success"),
    ('[100]', "failed"),
    ('{"N":[365],"E":[],"U":[]}', "failed"),
])
def test_prompt_only_request_preserves_transport_and_failure_rule(body, expected):
    config = sem.load_config(CONFIG)
    captured = []

    def respond(request):
        payload = json.loads(request.content)
        captured.append(payload)
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["enable_thinking"] is False
        assert payload["model"] == "qwen3.8-flash"
        assert payload["messages"][0]["content"][0]["text"] == config["prompts"]["point"]
        assert "case_id" not in json.dumps(payload)
        return httpx.Response(200, json={"model": "qwen3.8-flash", "choices": [
            {"finish_reason": "stop", "message": {"content": body}}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result, raw = sem.request_once(client, "https://example.test", "secret", b"png",
                                       "point", "case_0001", config)
    assert len(captured) == 1
    assert result.method == raw["method"] == sem.METHOD
    assert result.status == raw["status"] == expected


def test_journal_never_resubmits_claimed_or_completed_calls(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    for name in ("manifest.json", "summary.json"):
        (pilot / name).write_text("{}")
    out = tmp_path / "run"
    sem.write_json(out / "preflight/report.json", {
        "config_sha256": sem._sha(CONFIG), "source_sha256": sem.source_hashes(),
        "calls": 12, "valid_responses": 12})
    # Simulate a request interrupted after dispatch, before its response was saved.
    claim = out / "calls/point/case_0001.started"
    claim.parent.mkdir(parents=True)
    claim.write_text("already submitted")
    monkeypatch.setattr(sem, "frozen_images", lambda *a: {"case_0001": image})
    monkeypatch.setattr(sem, "_pilot_inputs", lambda *a: [])
    monkeypatch.setattr(sem, "_credentials", lambda *a: ("https://example.test", "secret"))
    calls = []

    def fake_request(client, base, key, image, task, case_id, config):
        calls.append((case_id, task))
        result = sem.failed_result(case_id, task, sem.METHOD)
        return result, {"case_id": case_id, "status": "failed", "usage": None,
                        "image_sha256": sem._sha(tmp_path / "image.png")}

    original_read = Path.read_bytes

    def guard(path):
        assert path.name != "truth.json"
        return original_read(path)

    monkeypatch.setattr(sem, "request_once", fake_request)
    monkeypatch.setattr(Path, "read_bytes", guard)
    first = sem.run_visual(CONFIG, pilot, out, tmp_path, None)
    second = sem.run_visual(CONFIG, pilot, out, tmp_path, None)
    assert first == second
    assert calls == [("case_0001", "range")]
    record = json.loads((out / "calls/point/case_0001.json").read_bytes())
    assert record["raw"]["error_type"] == "InterruptedRequest"
    assert first["tasks"]["point"]["success"] == 0
