import json

import httpx
import pytest

from gnss_sim import p8a
from gnss_sim.input_only import write_json
from gnss_sim.visual import parse_result


def initial(body='{"N":[[100,189]],"E":[],"U":[]}'):
    return parse_result(body, "case_0001", "range")


def config():
    return p8a.load_config(p8a.ROOT / "configs/p8a-range-context.json")


def test_window_selection_covers_edges_and_empty_case_without_truncation():
    assert p8a.selected_windows(initial()) == [("N", 0, 119), ("N", 90, 209), ("N", 180, 299)]
    empty = initial('{"N":[],"E":[],"U":[]}')
    assert len(p8a.selected_windows(empty)) == 12
    ends = initial('{"N":[[0,2],[360,364]],"E":[],"U":[]}')
    assert p8a.selected_windows(ends) == [("N", 0, 119), ("N", 245, 364)]


def test_review_control_same_prompt_initial_and_budget_only_images_differ():
    c = config()
    p1 = p8a.payload_for(c, "V1", [b"overview"], initial())
    p2 = p8a.payload_for(c, "V2", [b"overview", b"detail"], initial())
    assert p1["messages"][0]["content"] == p2["messages"][0]["content"][:2]
    assert p1["max_tokens"] == p2["max_tokens"] == 8192
    assert p2["response_format"] == {"type": "json_object"}
    assert p2["enable_thinking"] is False
    with pytest.raises(ValueError, match="budget"):
        p8a.payload_for(c, "V2", [b"image"] * 6, initial())


@pytest.mark.parametrize("body,status", [
    ('{"N":[[90,200]],"E":[],"U":[]}', "success"),
    ('[[90,200]]', "failed"),
    ('{"N":[[90,365]],"E":[],"U":[]}', "failed"),
    ('{"N":[[200,90]],"E":[],"U":[]}', "failed"),
    ('{"N":[],"N":[],"E":[],"U":[]}', "failed"),
])
def test_one_attempt_schema_and_resume(body, status, tmp_path):
    calls = []
    def reply(request):
        calls.append(request)
        return httpx.Response(200, json={"model": "qwen3.8-flash", "choices": [{
            "finish_reason": "stop", "message": {"content": body}}]})
    payload = p8a.payload_for(config(), "V1", [b"png"], initial())
    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        for _ in range(2):
            row, record = p8a.journal_request(client, "https://example.test", "SECRET",
                                              "case_0001", "V1", payload, tmp_path)
            assert row.status == record["status"] == status
    assert len(calls) == 1
    assert "SECRET" not in (tmp_path / "attempt-started.json").read_text()


def test_pending_request_marked_failed_without_retry(tmp_path):
    payload = p8a.payload_for(config(), "V0", [b"png"])
    import hashlib
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    write_json(tmp_path / "attempt-started.json", {"payload_sha256": digest})
    def never_send(request):
        pytest.fail("uncertain interrupted attempt must not be sent again")
    with httpx.Client(transport=httpx.MockTransport(never_send)) as client:
        row, record = p8a.journal_request(client, "https://example.test", "SECRET",
                                          "case_0001", "V0", payload, tmp_path)
    assert row.status == "failed"
    assert record["error_type"] == "UncertainInterruptedAttempt"


def test_failed_upstream_never_enters_review():
    failed = p8a.failed_result("case_0001", "range")
    with pytest.raises(ValueError):
        p8a.payload_for(config(), "V2", [b"png"], failed)


def test_global_vertical_scale_matches_original_renderer():
    from matplotlib import pyplot as plt
    fig, panel = plt.subplots()
    try:
        values = [0, 1, 4, -2]
        panel.plot(values)
        panel.margins(y=0.08)
        assert p8a.y_limits(values) == pytest.approx(panel.get_ylim())
    finally:
        plt.close(fig)
