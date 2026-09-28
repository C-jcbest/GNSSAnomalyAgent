import json
from pathlib import Path

import httpx
import pytest

from gnss_sim import candidate_review as review
from gnss_sim.input_only import read_inputs, sha, write_json, write_rows
from gnss_sim.p7b import select_cases
from gnss_sim.schemas import PointResult, RangeResult
from gnss_sim.visual_runner import _engineering_cases

ROOT = Path(__file__).resolve().parents[1]


def config():
    return review.load_config(ROOT / "configs/p7b-candidate-review.json")


def row(values=(), task="point", status="success", axis="N"):
    model = PointResult if task == "point" else RangeResult
    return model(case_id="case_0001", method="test", status=status,
                 predictions={a: list(values) if a == axis else [] for a in review.AXES})


def test_point_dedup_provenance_and_order():
    actual = review.candidates_from(row([11, 3, 3]), row([3, 8]), "point")
    assert [c["start"] for c in actual] == [3, 8, 11]
    assert actual[0]["sources"] == ["N", "V"]
    assert actual == review.candidates_from(row([3, 11]), row([8, 3]), "point")
    assert len(review.candidates_from(row([3]), row([3], axis="E"), "point")) == 2


def test_ranges_remain_individually_reviewable():
    actual = review.candidates_from(row([(3, 9), (12, 18)], "range"),
                                    row([(3, 9), (7, 14)], "range"), "range")
    assert [(c["start"], c["end"]) for c in actual] == [(3, 9), (7, 14), (12, 18)]


@pytest.mark.parametrize("n,v", [("failed", "success"), ("success", "failed"), ("failed", "failed")])
def test_dependency_failures(n, v):
    assert review.candidates_from(row(status=n), row(status=v), "point") is None


def test_mismatched_task_identity():
    with pytest.raises(ValueError):
        review.candidates_from(row(), row(task="range"), "point")
    with pytest.raises(ValueError):
        review.candidates_from(row(), row().model_copy(update={"case_id": "case_0002"}), "point")


@pytest.mark.parametrize("text", [
    '{"keep_ids":["unknown"]}', '{"keep_ids":["candidate_0001","candidate_0001"]}',
    '{"keep_ids":[1]}', '{"keep_ids":null}', '{"keep_ids":[],"N":[]}',
    '{"keep_ids":[],"keep_ids":[]}', '```json\n{"keep_ids":[]}\n```', '[]', 'invalid',
])
def test_strict_parser_rejects(text):
    with pytest.raises((ValueError, TypeError)):
        review.parse_keep(text, [{"id": "candidate_0001"}])


def test_parser_empty_and_kept():
    candidates = [{"id": "candidate_0001"}]
    assert review.parse_keep('{"keep_ids":[]}', candidates) == []
    assert review.parse_keep('{"keep_ids":["candidate_0001"]}', candidates) == ["candidate_0001"]


def test_batch_limit_no_truncation():
    assert review.batches([], config()) == []
    assert [len(b) for b in review.batches(list(range(65)), config())] == [64, 1]
    assert len(review.batches(list(range(1152)), config())) == 18
    with pytest.raises(ValueError):
        review.batches(list(range(1153)), config())


def test_whole_case_failure_and_empty():
    candidates = review.candidates_from(row([7]), row([9]), "point")
    failed = review.reviewed_result("case_0001", "point", candidates,
        [{"status": "success", "keep_ids": [candidates[0]["id"]]}, {"status": "failed"}])
    assert failed.status == "failed" and failed.predictions.N == []
    assert review.reviewed_result("case_0001", "point", [], []).status == "success"


def test_request_allowlist_and_no_retry():
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={"model": "qwen3.8-flash", "id": "test",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"keep_ids":[]}'}}]})

    candidates = review.candidates_from(row([7]), row([7]), "point")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        record = review.request_once(client, "https://example.test", "secret", b"png", "point",
                                     "case_0001", candidates, config())
    assert record["status"] == "success" and len(calls) == 1
    assert "sources" not in record["prompt"] and "secret" not in json.dumps(record)
    assert calls[0]["enable_thinking"] is False and calls[0]["max_tokens"] == 2048
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
        record = review.request_once(client, "https://example.test", "secret", b"png", "point",
                                     "case_0001", candidates, config())
    assert record["status"] == "failed" and record["attempts"] == 1
    assert record["usage"] is None


def test_selection_is_metadata_only():
    from gnss_sim.schemas import CASE_TYPES, SCENARIO_TYPES
    entries = []
    for kind in ("normal", *CASE_TYPES[1:], *SCENARIO_TYPES):
        if kind in CASE_TYPES[1:]:
            for axis in review.AXES:
                for sign in ("positive", "negative"):
                    for _ in range(4):
                        entries.append({"case_type": kind, "axis": axis, "sign": sign})
        else:
            entries.extend({"case_type": kind} for _ in range(30 if kind == "normal" else 25))
    for i, entry in enumerate(entries, 1):
        entry["case_id"] = f"case_{i:04d}"
    selected = select_cases({"cases": entries[::-1]})
    assert len(selected) == 60
    assert sum(e["case_type"] == "normal" for e in selected) == 6
    assert selected == select_cases({"cases": entries})


def test_input_only_integration_and_failure_denominator(tmp_path, monkeypatch):
    package = tmp_path / "inputs"
    case = _engineering_cases()[0].model_copy(update={"case_id": "case_0001"})
    path = package / "cases/case_0001/input.json"
    write_json(path, case.model_dump(mode="json"))
    image = package / "images/case_0001.png"
    image.parent.mkdir()
    image.write_bytes(b"test png")
    write_json(package / "manifest.json", {"protocol": "input-only-v1", "cases": [
        {"case_id": case.case_id, "input_sha256": sha(path), "image_sha256": sha(image)}]})
    hashes = {}
    for task in review.TASKS:
        for condition in ("N", "V"):
            name = f"{condition}-{task}.jsonl"
            status = "failed" if condition == "V" and task == "point" else "success"
            write_rows(package / name, [row(task=task, status=status)])
            hashes[name] = sha(package / name)
    write_json(package / "upstream.json", {"sha256": hashes, "cost": {}})
    pre = tmp_path / "preflight"
    cfg = ROOT / "configs/p7b-candidate-review.json"
    write_json(pre / "report.json", {"config_sha256": sha(cfg), "source_sha256": review.source_hashes(),
                                      "calls": 4, "valid": 4, "cost": {}})
    monkeypatch.setattr(review, "_credentials", lambda _: ("https://example.test", "secret"))
    monkeypatch.setattr(review, "request_once", lambda *a: pytest.fail("empty/failed must not call API"))
    original = Path.open

    def guard(self, *args, **kwargs):
        if self.name == "truth.json":
            pytest.fail("inference must not open truth")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    report = review.run(cfg, package, pre, tmp_path / "run", None)
    assert report["cases"] == 1 and report["review_cost"]["requests"] == 0
    assert report["success"]["point"]["U"] == report["success"]["point"]["C"] == 0
    assert report["success"]["range"]["C"] == 1
    with pytest.raises(FileExistsError):
        review.run(cfg, package, pre, tmp_path / "run", None)
    manifest = json.loads((package / "manifest.json").read_bytes())
    manifest["cases"][0]["type"] = "normal"
    (package / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_inputs(package)
