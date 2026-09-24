import json

import httpx
import pytest

from gnss_anomaly.contracts import Prediction
from gnss_anomaly.models import ModelClient, resolve_model
from gnss_anomaly.policies import Choice, Executor


def make_client(monkeypatch, config, handler):
    monkeypatch.setenv("QWEN_BASE_URL", "https://model.test/v1")
    monkeypatch.setenv("QWEN_API_KEY", "private-key")
    return ModelClient(config["model"], httpx.Client(transport=httpx.MockTransport(handler)))


def test_model_json_image_usage_and_budget(monkeypatch, config):
    def handler(request):
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert body["messages"][1]["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"status":"ok","events":[]}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    config["model"]["max_requests"] = 1
    model = make_client(monkeypatch, config, handler)
    assert model.ask("test", Prediction, [b"png"]).status == "ok"
    assert model.calls[0]["usage"]["prompt_tokens"] == 10
    with pytest.raises(RuntimeError, match="budget"):
        model.ask("test", Prediction)
    assert "private-key" not in json.dumps(model.calls)
    model.close()


def test_bad_output_counts_against_budget(monkeypatch, config):
    config["model"]["max_requests"] = 1
    model = make_client(
        monkeypatch,
        config,
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": '{"tool":"shell","reason":"execute"}'}}]}
        ),
    )
    with pytest.raises(RuntimeError):
        model.ask("test", Choice)
    assert len(model.calls) == 1 and model.calls[0]["status"] == "error"
    with pytest.raises(RuntimeError, match="budget"):
        model.ask("test", Choice)
    model.close()


def test_empty_model_object_is_not_a_normal_prediction(monkeypatch, config):
    model = make_client(
        monkeypatch,
        config,
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]}),
    )
    with pytest.raises(RuntimeError):
        model.ask("test", Prediction)
    model.close()


def test_explicit_model_and_qwen_image_options(monkeypatch, config):
    monkeypatch.setenv("QWEN_PRICE_MODEL", "other-model")
    monkeypatch.setenv("QWEN_INPUT_PRICE_PER_MILLION", "1")
    monkeypatch.setenv("QWEN_OUTPUT_PRICE_PER_MILLION", "2")
    config["model"].update(
        name="qwen3.8-flash",
        enable_thinking=False,
        vl_high_resolution_images=False,
        max_pixels=2621440,
    )

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == resolve_model(config["model"]) == "qwen3.8-flash"
        assert body["enable_thinking"] is False
        assert body["vl_high_resolution_images"] is False
        assert "extra_body" not in body
        block = body["messages"][1]["content"][1]
        assert block["max_pixels"] == 2621440
        assert "max_pixels" not in block["image_url"]
        return httpx.Response(
            200,
            json={
                "model": "qwen3.8-flash",
                "id": "response-test",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"status":"ok","events":[]}',
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    model = make_client(monkeypatch, config, handler)
    model.ask("JSON test", Prediction, [b"png"])
    record = model.calls[0]
    assert record["finish_reason"] == "stop" and not record["reasoning_present"]
    assert record["request_settings"]["enable_thinking"] is False
    assert record["cost"] is None  # Never reuse the old model's price.
    model.close()


def test_model_name_resolution(monkeypatch):
    assert resolve_model({}) == "qwen3.8-flash"
    monkeypatch.setenv("QWEN_MODEL", "legacy-model")
    assert resolve_model({}) == "qwen3.8-flash"
    assert resolve_model({"name": "explicit-model"}) == "explicit-model"
    with pytest.raises(ValueError):
        resolve_model({"name": " "})


def test_visual_model_defaults_to_explicit_non_thinking(monkeypatch, config):
    config["model"]["name"] = "qwen3.8-flash"
    config["model"].pop("enable_thinking", None)

    def handler(request):
        assert json.loads(request.content)["enable_thinking"] is False
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"status":"ok","events":[]}'}}]},
        )

    model = make_client(monkeypatch, config, handler)
    assert model.ask("test", Prediction).status == "ok"
    assert model.calls[0]["request_settings"]["enable_thinking"] is False
    model.close()


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def ask(self, prompt, schema, images=None):
        self.prompts.append(prompt)
        return schema.model_validate(next(self.replies))


@pytest.mark.parametrize("method", ["generic", "lma", "lma_no_vision", "lma_no_review"])
def test_agents_use_bounded_tools_without_truth(method, config, window):
    model = ScriptedModel(
        [{"tool": "hampel", "reason": "inspect"}, {"tool": "stop", "reason": "sufficient"}]
    )
    executor = Executor(config, model, {"source_split": "development", "best_by_missing": {}})
    assert executor.run(window(), method).status == "ok"
    assert len(executor.tools) == 1
    assert len(model.prompts) == (1 if method == "lma_no_review" else 2)
    assert all('"labels"' not in p and '"injection"' not in p for p in model.prompts)


def test_no_vision_ablation_enforced(config, window):
    model = ScriptedModel([{"tool": "visual", "reason": "ignore restriction"}])
    with pytest.raises(ValueError, match="unavailable"):
        Executor(config, model, {"source_split": "development"}).run(window(), "lma_no_vision")


def test_visual_output_bounds_checked(config, window):
    model = ScriptedModel([{"events": [{"start": 0, "end": 999, "channels": ["N"]}]}])
    with pytest.raises(ValueError, match="outside"):
        Executor(config, model).run(window(), "visual")
