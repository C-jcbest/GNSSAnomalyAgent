import json

import httpx
import pytest

from gnss_sim import visual_8k


@pytest.mark.parametrize("task,body,status", [
    ("point", '{"N":[100],"E":[],"U":[]}', "success"),
    ("range", '{"N":[[100,189]],"E":[],"U":[]}', "success"),
    ("point", '[100]', "failed"),
    ("range", '{"N":[[100,365]],"E":[],"U":[]}', "failed"),
])
def test_actual_payload_uses_8192_and_object_mode_with_local_schema_check(task, body, status):
    config = visual_8k.load_config()
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert payload["max_tokens"] == 8192
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["model"] == "qwen3.8-flash"
        assert payload["enable_thinking"] is False
        assert payload["messages"][0]["content"][0]["text"] == config["prompts"][task]
        return httpx.Response(200, json={"model": "qwen3.8-flash", "choices": [
            {"finish_reason": "stop", "message": {"content": body}}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        prediction, record = visual_8k.request_once(
            client, "https://example.test", "secret", b"png", task, "case_0001", config)
    assert len(calls) == 1
    assert prediction.status == record["status"] == status
    assert prediction.method == record["method"] == visual_8k.METHOD
    assert record["max_tokens"] == 8192


def test_only_declared_budget_change_allowed_and_historical_configs_retained():
    config = visual_8k.load_config()
    config["model"]["max_tokens"] = 512
    with pytest.raises(ValueError, match="max_tokens=8192"):
        visual_8k.validate_config(config)
    for name, budget in (("p6-visual.json", 512), ("p7a-visual-semantics.json", 512),
                         ("p7b-candidate-review.json", 2048)):
        historical = json.loads((visual_8k.ROOT / "configs" / name).read_bytes())
        assert historical["model"]["max_tokens"] == budget
