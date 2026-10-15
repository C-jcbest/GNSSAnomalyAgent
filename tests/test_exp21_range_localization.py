"""VLM owns interval dates while numerical point events and observed masks persist."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exp21_range_localization import apply  # noqa: E402


def test_visual_ranges_replace_numerical_ranges_and_preserve_missing_mask():
    def event(kind, start, end):
        return {"kind": kind, "start": start, "end": end, "status": "confirmed", "phases": [], "evidence": "test"}

    observation = {"dates": [str(i) for i in range(60)], "background_end": 9,
                   "displacement_mm": [None if i == 25 else [0, 0, 0] for i in range(60)]}
    numeric = {"status": "success", "prediction": {"events": [event("step", 15, 15), event("deformation", 20, 40)]},
               "type_scores": {"deformation": [1] * 60}}
    decision = event("transient", 22, 30)
    decision.pop("phases")
    result = apply(observation, numeric, {"status": "success", "range_decision": {"events": [decision]}})
    assert [(e["kind"], e["start"], e["end"]) for e in result["prediction"]["events"]] == [
        ("step", 15, 15), ("transient", 22, 30)]
    assert not result["generic_mask"][25]
    assert result["generic_mask"][22]
    assert result["type_scores"] == {}
    assert numeric["prediction"]["events"][1]["kind"] == "deformation"


def test_failed_visual_retains_independent_numeric_prediction():
    numeric = {"status": "success", "prediction": {"events": []}}
    result = apply({}, numeric, {"status": "failed"})
    assert result["prediction"] == numeric["prediction"]
    assert result["range_visual_status"] == "failed"
