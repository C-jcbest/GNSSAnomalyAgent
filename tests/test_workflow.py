import json

import httpx
import pytest

from gnss_anomaly.cli import main
from gnss_anomaly.datasets import build_dataset, demo_backgrounds
from gnss_anomaly.experiments import calibrate, capabilities, run_experiment
from gnss_anomaly.models import ModelClient
from gnss_anomaly.policies import load_capabilities
from gnss_anomaly.storage import read_json, write_json


def small_dataset(root):
    demo_backgrounds(root / "bg")
    catalog = read_json(root / "bg/backgrounds.json")
    catalog["items"] = catalog["items"][:5]
    write_json(root / "bg/backgrounds.json", catalog)
    build_dataset(root / "bg", root / "dataset", 5, 12)
    return root / "dataset"


def test_offline_calibration_run_and_resume(tmp_path, config, monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    dataset = small_dataset(tmp_path)
    config["detectors"]["iforest_trees"] = 5
    with pytest.raises(ValueError, match="calibrated"):
        run_experiment(dataset, tmp_path / "denied", config, "test", ["hampel"])
    config = calibrate(dataset, config, tmp_path / "calibrated.json")
    result = run_experiment(
        dataset, tmp_path / "run", config, "test", ["hampel", "cusum", "iforest"]
    )
    assert len([r for r in result["summary"] if r["dimension"] == "all"]) == 3
    before = {p.name: p.read_bytes() for p in (tmp_path / "run/results").glob("*.json")}
    assert (
        run_experiment(
            dataset, tmp_path / "run", config, "test", ["hampel", "cusum", "iforest"], resume=True
        )
        == result
    )
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "run/results").glob("*.json")}
    assert result["all_records_present"]
    with pytest.raises(ValueError, match="identity"):
        run_experiment(dataset, tmp_path / "run", config, "test", ["hampel"], resume=True)
    with pytest.raises(ValueError, match="test"):
        capabilities(tmp_path / "run", tmp_path / "card.json")
    config["detectors"]["hampel_threshold"] += 1
    with pytest.raises(ValueError, match="changed after"):
        run_experiment(dataset, tmp_path / "changed", config, "test", ["hampel"])


def test_capability_card_forbids_test_and_partial_runs(tmp_path, config):
    write_json(tmp_path / "test-card.json", {"source_split": "test"})
    with pytest.raises(ValueError, match="never test"):
        load_capabilities(str(tmp_path / "test-card.json"))
    dataset = small_dataset(tmp_path)
    run_experiment(dataset, tmp_path / "partial", config, "development", ["hampel"], limit=1)
    with pytest.raises(ValueError, match="full split"):
        capabilities(tmp_path / "partial", tmp_path / "card.json")


def test_provisional_pilot_cannot_be_promoted_to_formal_evidence(tmp_path, config):
    dataset = small_dataset(tmp_path)
    m = read_json(dataset / "manifest.json")
    m["kind"] = "platform_pilot_provisional"
    write_json(dataset / "manifest.json", m)
    with pytest.raises(ValueError, match="development baseline"):
        run_experiment(dataset, tmp_path / "denied", config, "test", ["hampel"])
    with pytest.raises(ValueError, match="development baseline"):
        run_experiment(dataset, tmp_path / "denied", config, "development", ["lma"])
    with pytest.raises(ValueError, match="cannot calibrate"):
        calibrate(dataset, config, tmp_path / "config.json")
    summary = run_experiment(
        dataset, tmp_path / "pilot", config, "development", ["hampel"], limit=1
    )
    assert "not paper evidence" in summary["note"]
    with pytest.raises(ValueError, match="provisional pilot"):
        capabilities(tmp_path / "pilot", tmp_path / "card.json")


def test_cli_does_not_need_online_environment(tmp_path):
    assert (
        main(["--env", str(tmp_path / "missing.env"), "demo", "--out", str(tmp_path / "demo")]) == 0
    )
    assert (tmp_path / "demo/dataset/manifest.json").is_file()


def test_measured_card_and_all_agent_methods_end_to_end(tmp_path, config, monkeypatch):
    dataset = small_dataset(tmp_path)
    config["detectors"]["iforest_trees"] = 5
    config["repeats"] = 2
    monkeypatch.setenv("QWEN_BASE_URL", "https://test.invalid/v1")
    monkeypatch.setenv("QWEN_API_KEY", "fixture-key")

    def response(request):
        prompt = json.loads(request.content)["messages"][1]["content"][0]["text"]
        if "首轮工具" in prompt:
            result = {"tool": "hampel", "reason": "fixture selection"}
        elif "是否需要复核" in prompt:
            result = {"tool": "stop", "reason": "fixture stop"}
        else:
            result = {"status": "ok", "events": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}}]})

    monkeypatch.setattr(
        "gnss_anomaly.experiments.ModelClient",
        lambda cfg: ModelClient(cfg, httpx.Client(transport=httpx.MockTransport(response))),
    )
    monkeypatch.setattr("gnss_anomaly.policies.render", lambda *args: b"fixture-png")
    run_experiment(
        dataset,
        tmp_path / "dev",
        config,
        "development",
        ["hampel", "cusum", "iforest", "visual", "fixed"],
    )
    card = capabilities(tmp_path / "dev", tmp_path / "card.json")
    assert card["source_split"] == "development" and len(card["by_type_and_state"]) == 96
    config["capability_file"] = str(tmp_path / "card.json")
    result = run_experiment(
        dataset,
        tmp_path / "agents",
        config,
        "development",
        ["rule", "generic", "lma", "lma_no_vision", "lma_no_review"],
        limit=2,
    )
    assert all(r["completion_rate_mean"] == 1 for r in result["summary"])
    # Remove one of the visual repetitions: case sets remain equal but card must be refused.
    next((tmp_path / "dev/results").glob("*_visual_1.json")).unlink()
    with pytest.raises(ValueError, match="repeats"):
        capabilities(tmp_path / "dev", tmp_path / "partial-card.json")
