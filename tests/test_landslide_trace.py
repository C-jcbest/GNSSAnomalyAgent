from __future__ import annotations

import base64
import hashlib
import json

import numpy as np
import pytest

from gnss_sim.landslide_trace import checked_asset, replay_response, request_evidence


def test_same_filename_from_different_rounds_cannot_replace_input_evidence(tmp_path):
    first = tmp_path / "v2"
    second = tmp_path / "v3"
    first.mkdir()
    second.mkdir()
    (first / "raw.png").write_bytes(b"old-image")
    (second / "raw.png").write_bytes(b"new-image")
    out = tmp_path / "report"
    older = checked_asset(first / "raw.png", out)
    newer = checked_asset(second / "raw.png", out)
    assert older["url"] != newer["url"]
    assert (out / older["url"]).read_bytes() == b"old-image"
    (out / older["url"]).write_bytes(b"damaged-export")
    checked_asset(first / "raw.png", out, older["sha256"])
    assert (out / older["url"]).read_bytes() == b"old-image"
    with pytest.raises(ValueError, match="Image SHA256 mismatch"):
        checked_asset(second / "raw.png", out, older["sha256"])


def test_request_hash_binds_exact_prompt_images_order_and_parameters(tmp_path):
    content = [{"type": "text", "text": "Exact archived prompt\n</script> is text"}]
    hashes = {}
    for name in ("raw.png", "auxiliary.png"):
        data = name.encode()
        (tmp_path / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}})
    payload = {"model": "test-model", "temperature": 0.1, "top_p": 0.3, "max_tokens": 2048,
               "messages": [{"role": "user", "content": content}], "enable_thinking": False,
               "response_format": {"type": "json_object"}}
    receipt = {"request_id": "example", "prompt": content[0]["text"], "image_sha256": hashes,
               "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
               **{key: payload[key] for key in ("model", "temperature", "top_p", "max_tokens")}}
    images = request_evidence(receipt, tmp_path, tmp_path / "report")
    assert [image["name"] for image in images] == ["raw.png", "auxiliary.png"]
    receipt["image_sha256"] = dict(reversed(list(hashes.items())))
    with pytest.raises(ValueError, match="Request reconstruction SHA256 mismatch"):
        request_evidence(receipt, tmp_path, tmp_path / "report")


def replay(stages, activity, observed, saved_stage, kind="visual_model"):
    gate = [[index, index] for index, value in enumerate(activity) if value == 1]
    started = {"prompt": "Confirmed inclusive activity intervals: " + json.dumps(gate)}
    output = json.dumps({"stages": stages})
    response = {"status": "returned_json", "output": output}
    dates = [f"2026-01-{index + 1:02}" for index in range(len(activity))]
    return replay_response(started, response, np.array(observed), dates, np.array(activity), kind,
                           {"activity": activity, "stage": saved_stage})


def test_missing_day_is_not_reported_as_activity_gate_violation():
    result = replay([[0, 2, "A"]], [1, -1, 1], [True, False, True], [1, -1, 1])
    assert result["status"] == "passed"
    assert result["saved_matches"]
    assert result["details"] == []


def test_observed_gate_hole_is_reported_with_exact_calendar_dates():
    result = replay([[0, 3, "A"]], [1, 0, 0, 1], [True] * 4, [-1] * 4)
    assert result["status"] == "failed"
    assert "no automatic clipping" in result["error"]
    assert result["details"][0]["interval"] == [1, 2]
    assert result["details"][0]["dates"] == ["2026-01-02", "2026-01-03"]


def test_shared_inclusive_endpoint_is_reported_without_repair():
    result = replay([[0, 1, "A"], [1, 2, "D"]], [1] * 3, [True] * 3, [-1] * 3)
    assert result["status"] == "failed"
    assert result["details"][0]["kind"] == "overlap"
    assert result["details"][0]["previous"] == [0, 1, "A"]
    assert result["details"][0]["interval"] == [1, 2, "D"]


def test_ungated_ablation_does_not_turn_outside_activity_into_contract_failure():
    result = replay([[0, 2, "A"]], [0, 1, 1], [True] * 3, [1] * 3, "visual_model_ungated")
    assert result["status"] == "passed"
    assert result["saved_matches"]
    assert result["details"][0]["kind"] == "ungated_outside"


def test_logged_gate_disagreement_is_not_silently_replaced_by_saved_gate():
    result = replay_response({"prompt": "Confirmed inclusive activity intervals: [[1, 2]]"},
                             {"status": "returned_json", "output": '{"stages": []}'},
                             np.ones(3, dtype=bool), ["a", "b", "c"], np.ones(3, dtype=int),
                             "visual_model", {"activity": [1] * 3, "stage": [-1] * 3})
    assert result["status"] == "failed"
    assert result["error"] == "Logged prompt gate differs from its saved source"


def test_local_activity_request_reports_activity_uncertainty_shared_endpoint():
    result = replay_response({"prompt": "frozen localization"},
                             {"status": "returned_json", "output": '{"activity": [[0, 1]], "uncertain": [[1, 2]]}'},
                             np.ones(3, dtype=bool), ["a", "b", "c"], np.full(3, -1),
                             "locate-global", {"activity": [-1] * 3, "stage": [-1] * 3})
    assert result["status"] == "failed"
    assert result["details"][0]["interval"] == [1, 1]
    assert result["details"][0]["dates"] == ["b", "b"]
