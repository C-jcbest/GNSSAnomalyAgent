from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnss_anomaly.contracts import Event, Prediction, Window
from gnss_anomaly.experiments import execution_schedule
from gnss_anomaly.fusion import merge_branches
from gnss_anomaly.policies import Executor
from gnss_anomaly.storage import read_json


def test_independent_branch_does_not_receive_local_candidates(monkeypatch):
    config = read_json(Path("configs/fusion-pilot-v1.json"))
    local = Prediction(events=[Event(start=20, end=20, channels=["N"], reason="PRIVATE_PRIOR")])
    seen = []

    def detect(window, name, settings, seed, diagnostics):
        seen.append(name)
        return local

    monkeypatch.setattr("gnss_anomaly.policies.detect", detect)
    renders = []

    def render(window, prior=None, bounds=None):
        renders.append((prior, bounds))
        return b"image"

    monkeypatch.setattr("gnss_anomaly.policies.render", render)

    class Capture:
        def ask(self, prompt, schema, images):
            assert "PRIVATE_PRIOR" not in prompt and "数值/先前候选" not in prompt
            assert '"legal_index": [0, 179]' in prompt
            assert '"U": [[60, 69]]' in prompt
            assert len(images) == 1
            return schema.model_validate(
                {
                    "status": "ok",
                    "events": [{"start": 45, "end": 179, "channels": ["U"]}],
                }
            )

    values = np.zeros((180, 3)).tolist()
    for i in range(60, 70):
        values[i][2] = None
    window = Window(
        case_id="opaque",
        sampling_hours=24,
        timestamps=list(pd.date_range("2026-01-01T07:00Z", periods=180, freq="D")),
        values=values,
    )
    executor = Executor(config, Capture())
    before = window.model_dump_json()
    p = executor.run(window, "split_fusion")
    assert p.mask(180)[20, 0] and p.mask(180)[179, 2]
    assert not p.mask(180)[:, 1].any()
    assert executor.tools[0]["result"] == local.model_dump()
    assert executor.tools[1]["tool"] == "visual_sustained"
    assert renders == [(None, None)]
    executor.run(window, "split_fusion")
    assert seen == ["hampel"] and executor.tools[0]["cache_hit"]
    assert window.model_dump_json() == before


@pytest.mark.parametrize("status", ["error", "insufficient"])
def test_branch_failure_is_not_silently_replaced_by_local_prediction(status):
    local = Prediction(events=[Event(start=5, end=5, channels=["N"])])
    failed = Prediction(status=status)
    for a, b in ((local, failed), (failed, local)):
        result = merge_branches(a, b, 180)
        assert result.status == status and not result.events


def test_union_preserves_exact_cells_and_deduplicates_overlaps():
    local = Prediction(events=[Event(start=5, end=5, channels=["N"])])
    visual = Prediction(events=[Event(start=4, end=7, channels=["N", "U"])])
    result = merge_branches(local, visual, 180)
    np.testing.assert_array_equal(result.mask(180), local.mask(180) | visual.mask(180))
    assert len(result.events) == 2


def test_schedule_is_reproducible_and_interleaves_paired_conditions():
    config = read_json(Path("configs/fusion-pilot-v1.json"))
    methods = ["visual", "fixed", "split_fusion"]
    first = execution_schedule("a", methods, config)
    assert first == execution_schedule("a", methods, config)
    assert len(first) == len(set(first)) == 9
    for repeat in range(3):
        block = first[repeat * 3 : (repeat + 1) * 3]
        assert {m for m, r in block} == set(methods) and {r for m, r in block} == {repeat}
    schedules = {tuple(execution_schedule(str(i), methods, config)) for i in range(20)}
    assert len(schedules) > 1


def test_evidence_prompt_changes_only_independent_visual_task(monkeypatch):
    config = read_json(Path("configs/fusion-evidence-v1.json"))
    window = Window(
        case_id="opaque",
        sampling_hours=24,
        timestamps=list(pd.date_range("2026-01-01T07:00Z", periods=180, freq="D")),
        values=np.zeros((180, 3)).tolist(),
    )
    prompts = []

    class Capture:
        def ask(self, prompt, schema, images):
            prompts.append(prompt)
            assert len(images) == 1
            return schema.model_validate({"status": "ok", "events": []})

    monkeypatch.setattr("gnss_anomaly.policies.render", lambda *a, **k: b"image")
    executor = Executor(config, Capture())
    executor.run(window, "visual")
    executor.run(window, "split_fusion")
    executor.run(window, "split_fusion_evidence")
    assert "实际有效观测依据" not in prompts[0] + prompts[1]
    assert "实际有效观测依据" in prompts[2]
    assert "数值/先前候选" not in prompts[2]
    assert '"legal_index": [0, 179]' in prompts[2]
    assert '"labels"' not in "".join(prompts)
    assert executor.tools[1]["tool"] == "visual_sustained_evidence"
    assert config["model"]["name"] == "qwen3.8-flash"
    assert config["model"]["enable_thinking"] is False
    methods = ["visual", "split_fusion", "split_fusion_evidence"]
    schedule = execution_schedule("opaque", methods, config)
    assert len(schedule) == 9
    for repeat in range(3):
        assert {method for method, _ in schedule[repeat * 3 : (repeat + 1) * 3]} == set(methods)


def test_interleaved_run_logs_failures_and_resume_without_resampling(tmp_path, monkeypatch):
    from gnss_anomaly.experiments import run_experiment
    from gnss_anomaly.storage import file_hash, write_json

    config = read_json(Path("configs/fusion-pilot-v1.json"))
    dataset = tmp_path / "data"
    window = Window(
        case_id="opaque",
        sampling_hours=24,
        timestamps=list(pd.date_range("2026-01-01T07:00Z", periods=180, freq="D")),
        values=np.zeros((180, 3)).tolist(),
    )
    write_json(dataset / "inputs/opaque.json", window.model_dump(mode="json"))
    write_json(dataset / "labels.json", {"opaque": {"events": []}})
    record = dict(
        case_id="opaque",
        file="inputs/opaque.json",
        split="development",
        group="group",
        source_group_id="source",
        base_id="base",
        kind="normal",
        state="native",
        semantic="undetermined",
    )
    write_json(
        dataset / "manifest.json",
        {
            "kind": "platform_pilot_provisional",
            "sampling_hours": 24,
            "records": [record],
            "files": {
                name: file_hash(dataset / name) for name in ["inputs/opaque.json", "labels.json"]
            },
        },
    )
    requests = []

    class FakeModel:
        def __init__(self, config):
            self.calls = []
            self.cost = 0

        def reset(self):
            self.calls = []

        def ask(self, prompt, schema, images=None):
            requests.append(prompt)
            self.calls.append({"status": "ok", "logical_call": 1, "attempt": 1})
            return Prediction(status="insufficient" if "你负责独立分析" in prompt else "ok")

        def close(self):
            pass

    monkeypatch.setattr("gnss_anomaly.experiments.ModelClient", FakeModel)
    monkeypatch.setattr("gnss_anomaly.policies.render", lambda *a, **k: b"image")
    methods = ["visual", "fixed", "split_fusion"]
    result = run_experiment(dataset, tmp_path / "run", config, "development", methods)
    assert result["records_present"] == 9 and result["all_records_present"]
    rows = [read_json(p) for p in (tmp_path / "run/results").glob("*.json")]
    rows.sort(key=lambda r: r["schedule_index"])
    assert [(r["method"], r["repeat"]) for r in rows] == execution_schedule(
        "opaque", methods, config
    )
    assert all(r["started_at"] <= r["finished_at"] for r in rows)
    assert all(not r["metrics"]["completed"] for r in rows if r["method"] == "split_fusion")
    before = {p.name: p.read_bytes() for p in (tmp_path / "run/results").glob("*.json")}
    run_experiment(dataset, tmp_path / "run", config, "development", methods, resume=True)
    assert len(requests) == 9
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "run/results").glob("*.json")}
