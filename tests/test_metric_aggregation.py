import numpy as np
import pytest

from gnss_research.metrics import aggregate, evaluate_record


def test_type_macro_keeps_point_tolerance_separate_from_daily_and_event_iou():
    def event(kind, start, end):
        return {"kind": kind, "start": start, "end": end, "status": "confirmed",
                "evidence": "observed", "phases": []}

    truth = {"events": [event("spike", 10, 10), event("deformation", 20, 25)]}
    mask = np.zeros(30, dtype=bool)
    mask[12] = True
    mask[20:24] = True
    result = {"prediction": {"events": [event("spike", 12, 12), event("deformation", 20, 23)]},
              "generic_mask": mask.tolist()}
    row = evaluate_record(truth, result, np.ones(30, dtype=bool), 0)
    summary = aggregate([row])
    assert summary["types_daily_macro_f1"] == pytest.approx(0.2)
    assert summary["types_event_macro_f1"]["iou_0.5"] == pytest.approx(0.25)
    spike = summary["types"]["spike"]
    assert spike["daily_micro"]["f1"] == 0
    assert spike["events_micro"]["iou_0.5"]["f1"] == 0
    assert spike["events_micro"]["tolerance_3_days_secondary"]["f1"] == 1
