import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exp11_activity_review import apply_activity  # noqa: E402


def event(kind, start, end):
    return {"kind": kind, "start": start, "end": end, "status": "confirmed", "phases": [],
            "evidence": "observed raw displacement"}


def test_presence_can_recover_unselected_fit_and_cannot_change_level_events_or_missing_labels():
    obs = {"dates": list(range(120)), "displacement_mm": np.zeros((120, 3)).tolist()}
    obs["displacement_mm"][100] = None
    source = {"prediction": {"events": [event("spike", 104, 104), event("step", 105, 105)]}}
    numeric = {"activity_fit": {"start": 95, "end": 115, "selected": False}}
    response = {"status": "success", "activity_decision": {"activity": "present", "evidence": "Sustained raw motion"}}
    result = apply_activity(obs, source, numeric, response)
    assert result["prediction"]["events"][:2] == source["prediction"]["events"]
    assert result["prediction"]["events"][-1]["start"] == 95
    assert result["prediction"]["events"][-1]["end"] == 115
    assert not result["generic_mask"][100]
    assert len(source["prediction"]["events"]) == 2


def test_absence_removes_only_activity_and_uncertain_or_failed_preserves_it():
    obs = {"dates": list(range(120)), "displacement_mm": np.zeros((120, 3)).tolist()}
    source = {"prediction": {"events": [event("transient", 100, 110), event("deformation", 90, 115)]}}
    numeric = {"activity_fit": {"start": 90, "end": 115, "selected": True}}
    absent = {"status": "success", "activity_decision": {"activity": "absent", "evidence": "Only abrupt shifts"}}
    assert apply_activity(obs, source, numeric, absent)["prediction"]["events"] == [source["prediction"]["events"][0]]
    uncertain = {"status": "success", "activity_decision": {"activity": "uncertain", "evidence": "Insufficient displacement evidence"}}
    assert apply_activity(obs, source, numeric, uncertain)["prediction"] == source["prediction"]
    assert apply_activity(obs, source, numeric, {"status": "failed"})["prediction"] == source["prediction"]
