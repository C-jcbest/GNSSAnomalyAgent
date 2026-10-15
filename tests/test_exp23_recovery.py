"""Recovered segment starts must respect known jumps, events, and missing days."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exp14_fingerprint import apply  # noqa: E402
from experiment_exp23 import recovery_graph  # noqa: E402
from experiment_exp24 import gap_guard  # noqa: E402


def event(kind, start, end):
    return {"kind": kind, "start": start, "end": end, "status": "confirmed",
            "phases": [], "evidence": "observed structure"}


def observation():
    return {"dates": [str(i) for i in range(64)], "background_end": 9,
            "displacement_mm": [None if i in (19, 25) else [0, 0, 0] for i in range(64)]}


def numeric():
    return {"jumps": [{"index": 30}, {"index": 50}],
            "segments": [{"start": i} for i in (0, 20, 28, 50)]}


def test_recovers_first_observed_boundary_but_not_existing_jump_refinement():
    base = {"prediction": {"events": [event("step", 50, 50)]}}
    graph = recovery_graph(observation(), numeric(), base)
    assert [(pair["start"], pair["end"]) for pair in graph["pairs"]] == [(20, 49)]


def test_does_not_add_interval_over_existing_transient():
    base = {"prediction": {"events": [event("step", 50, 50), event("transient", 35, 40)]}}
    assert recovery_graph(observation(), numeric(), base)["pairs"] == []


def test_accepted_recovery_preserves_activity_other_steps_and_missing_mask():
    base = {"status": "success", "prediction": {"events": [
        event("deformation", 12, 18), event("step", 50, 50), event("step", 55, 55)]}}
    graph = recovery_graph(observation(), numeric(), base)
    response = {"status": "success", "pair_decision": {"pair_ids": ["R1"], "evidence": "observed return"}}
    result = apply(observation(), base, graph, response)
    assert [(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]] == [
        ("deformation", 12, 18), ("step", 55, 55), ("transient", 20, 49)]
    assert result["generic_mask"][20]
    assert not result["generic_mask"][25]
    assert len(base["prediction"]["events"]) == 3


def test_gap_guard_rejects_continuously_observed_bend_without_changing_response():
    graph = {"pairs": [{"id": "gap", "start": 20, "end": 35},
                       {"id": "bend", "start": 30, "end": 40}]}
    response = {"status": "success", "pair_decision": {
        "pair_ids": ["gap", "bend"], "evidence": "saved model response"}}
    guarded = gap_guard(observation(), graph, response)
    assert guarded["pair_decision"]["pair_ids"] == ["gap"]
    assert guarded["gap_guard_rejected_ids"] == ["bend"]
    assert response["pair_decision"]["pair_ids"] == ["gap", "bend"]
