import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_exp09 import consensus_repair  # noqa: E402


def event(kind, start, end, status="confirmed"):
    return {"kind": kind, "start": start, "end": end, "status": status,
            "evidence": "raw displacement", "phases": []}


def candidates():
    return {"numerical_prediction": {"prediction": {"events": [event("step", 105, 105),
            event("step", 115, 115), event("deformation", 100, 119)]}},
            "proposals": [{"id": "pair", **event("transient", 105, 114, "uncertain")}]}


def observation():
    result = {"dates": list(range(120)), "displacement_mm": np.zeros((120, 3)).tolist()}
    result["displacement_mm"][110] = None
    return result


def test_majority_maps_nearby_boundaries_to_same_pair_and_preserves_activity_and_missing():
    responses = [{"status": "success", "prediction": {"events": [event("transient", 104, 113)]}},
                 {"status": "success", "prediction": {"events": [event("transient", 106, 115)]}},
                 {"status": "failed"}]
    result = consensus_repair(observation(), responses, candidates())
    assert result["member_votes"] == {"pair": 2}
    assert {(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]} == {
        ("transient", 105, 114), ("deformation", 100, 119)}
    assert not result["generic_mask"][110]


def test_single_vote_with_two_failed_members_keeps_numerical_steps():
    responses = [{"status": "success", "prediction": {"events": [event("transient", 105, 114)]}},
                 {"status": "failed"}, {"status": "failed"}]
    result = consensus_repair(observation(), responses, candidates())
    assert result["semantic_repairs"] == []
    assert {(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]} == {
        ("step", 105, 105), ("step", 115, 115), ("deformation", 100, 119)}
