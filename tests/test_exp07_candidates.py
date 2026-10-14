import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "exp07_candidates", Path(__file__).resolve().parents[1] / "scripts/experiment_exp07.py")
exp07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp07)


def observation(values):
    return {"dates": list(range(len(values))), "background_end": 89,
            "displacement_mm": values.tolist()}


def event(kind, start, end, status="confirmed"):
    return {"kind": kind, "start": start, "end": end, "status": status,
            "evidence": "observed displacement", "phases": []}


def test_isolated_competition_distinguishes_spike_from_neighboring_step_and_scale():
    values = np.random.default_rng(33).normal(0, .2, (210, 3))
    values[133] += [7, -5, 3]
    values[169:] += [8, -6, 4]
    numerical = {"prediction": {"events": []}}
    obs = observation(values)
    result = exp07.isolated_competition(obs, numerical)
    assert [(e["kind"], e["start"]) for e in result["prediction"]["events"]] == [
        ("spike", 133)]
    scaled = exp07.isolated_competition(observation(values * 100), numerical)
    assert scaled["prediction"]["events"] == result["prediction"]["events"]
    obs["displacement_mm"][134] = None
    masked = exp07.isolated_competition(obs, numerical)
    assert masked["prediction"]["events"] == []
    assert not masked["generic_mask"][134]


def test_coexisting_step_requires_explicit_visual_confirmation_and_preserves_activity():
    obs = observation(np.zeros((210, 3)))
    obs["displacement_mm"][145] = None
    numerical = [event("step", 133, 133), event("step", 158, 158),
                 event("deformation", 120, 175)]
    candidates = {"numerical_prediction": {"prediction": {"events": numerical}},
                  "proposals": [{"id": "pair", **event("transient", 133, 157, "uncertain")}]}
    visual = {"status": "success", "prediction": {"events": [
        event("transient", 133, 157), event("step", 133, 133)]}}
    result = exp07.coexist_repair(obs, visual, candidates)
    assert result["coexisting_step_indices"] == [133]
    assert {(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]} == {
        ("step", 133, 133), ("transient", 133, 157), ("deformation", 120, 175)}
    assert not result["generic_mask"][145]
    visual["prediction"]["events"][-1]["status"] = "uncertain"
    assert exp07.coexist_repair(obs, visual, candidates)["coexisting_step_indices"] == []
    failed = exp07.coexist_repair(obs, {"status": "failed"}, candidates)
    assert failed["prediction"]["events"] == numerical
