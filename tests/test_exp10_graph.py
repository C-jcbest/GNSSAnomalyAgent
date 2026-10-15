import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exp10_visual_graph import apply_graph, build_graph, select_training_references  # noqa: E402

from gnss_research.files import write_json


def event(kind, start, end):
    return {"kind": kind, "start": start, "end": end, "status": "confirmed",
            "evidence": "raw observed displacement", "phases": []}


def observation():
    result = {"dates": list(range(150)), "displacement_mm": np.zeros((150, 3)).tolist()}
    result["displacement_mm"][110] = None
    return result


def source():
    return {"jumps": [{"index": 100}, {"index": 105}, {"index": 120}],
            "prediction": {"events": [event("step", 100, 100), event("step", 105, 105),
                                      event("step", 120, 120), event("spike", 104, 104),
                                      event("deformation", 95, 130)]}}


def test_visual_reassigns_competing_pair_and_preserves_interior_step_spike_activity_and_gaps():
    numeric = source()
    graph = build_graph(numeric)
    response = {"status": "success", "graph_decision": {"decision": "replace_levels",
                "pair_ids": ["P2"], "step_ids": ["J2"], "evidence": "Observed return around an independent interior jump"}}
    result = apply_graph(observation(), numeric, graph, response)
    assert {(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]} == {
        ("transient", 100, 119), ("step", 105, 105), ("spike", 104, 104), ("deformation", 95, 130)}
    assert not result["generic_mask"][110]
    assert len(numeric["prediction"]["events"]) == 5


def test_failed_visual_retains_full_numerical_detection_and_conflicting_pair_is_rejected():
    numeric = source()
    graph = build_graph(numeric)
    result = apply_graph(observation(), numeric, graph, {"status": "failed"})
    assert result["prediction"] == numeric["prediction"]
    response = {"status": "success", "graph_decision": {"decision": "replace_levels",
                "pair_ids": ["P1", "P2"], "step_ids": [], "evidence": "conflicting pair hypotheses"}}
    with pytest.raises(ValueError, match="share a jump"):
        apply_graph(observation(), numeric, graph, response)


def test_reference_selection_reads_only_training_labels_and_prefers_interior_step(tmp_path):
    records = []
    for index, scenario in enumerate(("step", "transient", "deformation", "mixed", "mixed")):
        events = [event("transient", 100, 120), event("step", 100 if index == 3 else 110, 100 if index == 3 else 110)]
        name = f"train-{index}.json"
        write_json(tmp_path / name, {"scenario": scenario, "events": events})
        records.append({"station_id": f"t{index}", "split": "train", "truth": name, "observation": "unused"})
    records.append({"station_id": "v0", "split": "validation", "truth": "must-not-read.json"})
    selected = select_training_references(tmp_path, records)
    assert [reference["station_id"] for reference in selected] == ["t0", "t1", "t2", "t4"]
    assert all(reference["split"] == "train" for reference in selected)
