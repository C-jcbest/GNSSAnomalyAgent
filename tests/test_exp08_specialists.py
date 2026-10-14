import importlib.util
import sys
from pathlib import Path

import numpy as np

from gnss_research.files import write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "exp08_specialists", Path(__file__).resolve().parents[1] / "scripts/experiment_exp08.py")
exp08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp08)


def event(kind, start, end):
    return {"kind": kind, "start": start, "end": end, "status": "confirmed",
            "evidence": "observed displacement", "phases": []}


def test_specialist_replaces_spikes_and_preserves_other_shapes_and_missing_mask():
    obs = {"dates": list(range(120)), "background_end": 89,
           "displacement_mm": np.zeros((120, 3)).tolist()}
    obs["displacement_mm"][105] = None
    original = {"prediction": {"events": [event("spike", 99, 99),
                event("transient", 101, 110), event("deformation", 95, 115)]},
                "type_scores": {"spike": [0] * 120}}
    mask = np.zeros(120, dtype=bool)
    mask[103] = True
    result = exp08.replace_spikes(obs, original, mask)
    assert {(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]} == {
        ("spike", 103, 103), ("transient", 101, 110), ("deformation", 95, 115)}
    assert not result["generic_mask"][105]
    assert result["generic_score"] is None
    assert "type_scores" not in result
    assert original["prediction"]["events"][0]["start"] == 99


def test_specialist_training_uses_only_training_stations_and_observed_analysis_days(tmp_path):
    values = np.random.default_rng(71).normal(0, .3, (120, 3))
    values[100] += [8, -5, 3]
    obs = {"dates": list(range(120)), "background_end": 89,
           "displacement_mm": values.tolist()}
    obs["displacement_mm"][101] = None
    write_json(tmp_path / "train-observation.json", obs)
    write_json(tmp_path / "train-truth.json", {"events": [event("spike", 100, 100)]})
    records = [{"station_id": "train", "split": "train", "observation": "train-observation.json",
                "truth": "train-truth.json"},
               {"station_id": "validation", "split": "validation",
                "observation": "never-read-observation.json", "truth": "never-read-truth.json"}]
    models, receipt = exp08.fit_spike_specialists(tmp_path, records, 71, ["binary_rf"])
    assert receipt["training_stations"] == ["train"]
    assert receipt["training_rows"] == 29
    assert receipt["positive_spike_rows"] == 1
    assert receipt["feature_count"] == 48
    assert models["binary_rf"].classes_.tolist() == [False, True]
