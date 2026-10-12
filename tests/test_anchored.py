import numpy as np

from gnss_research.fusion import anchored_review


def event(kind, start, end):
    return {"kind": kind, "start": start, "end": end, "status": "confirmed",
            "evidence": "observation", "phases": []}


def test_visual_review_cannot_extend_activity_to_platform_or_shift_numeric_point():
    observation = {"dates": list(range(100)), "displacement_mm": [[0., 0., 0.]] * 100}
    observation["displacement_mm"][15] = None
    numeric_events = [event("deformation", 10, 20), event("spike", 30, 30)]
    candidates = {"numerical_prediction": {"prediction": {"events": numeric_events}},
                  "windows": [{"inspection_kind": "transient", "start": 55, "end": 75}]}
    visual = {"status": "success", "prediction": {"events": [
        event("deformation", 0, 99), event("spike", 31, 31),
        event("transient", 60, 70), event("transient", 80, 90)]}}
    result = anchored_review(observation, visual, candidates)
    assert [(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]] == [
        ("deformation", 10, 20), ("transient", 60, 70)]
    assert not result["generic_mask"][15]
    assert not np.any(result["generic_mask"][21:60])


def test_uncertain_visual_morphology_does_not_confirm_numeric_activity():
    observation = {"dates": list(range(100)), "displacement_mm": [[0., 0., 0.]] * 100}
    proposal = event("deformation", 10, 20)
    uncertain = {**proposal, "status": "uncertain"}
    candidates = {"numerical_prediction": {"prediction": {"events": [proposal]}}, "windows": []}
    result = anchored_review(observation, {"status": "success", "prediction": {
        "events": [uncertain]}}, candidates)
    assert result["prediction"]["events"] == []
