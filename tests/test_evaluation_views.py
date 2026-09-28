from datetime import date, timedelta

import pytest

from gnss_sim.evaluation_views import evaluate_views, matched_count
from gnss_sim.schemas import CaseTruth, PointResult, RangeResult


def truth(events):
    typed = []
    for i, (kind, axis, start, end) in enumerate(events, 1):
        typed.append({"event_id": f"event_{i:03}", "type": kind, "axis": axis,
                      "start_index": start, "end_index": end,
                      "start_date": date(2025, 1, 1) + timedelta(days=start),
                      "end_date": date(2025, 1, 1) + timedelta(days=end),
                      "persistent": False, "source": "observation_artifact",
                      "parameters": {"duration_days": 1 if kind == "spike" else 14,
                                     "amplitude_mm": 4.5}})
    return CaseTruth(case_id="case", normal_background_mm=[], measurement_noise_mm=[],
                     injected_deformation_mm=[], observation_artifact_mm=[],
                     annual_phase_rad=(0, 0, 0), semiannual_phase_rad=(0, 0, 0),
                     component_seeds={"annual_phase": 1, "semiannual_phase": 2, "white_noise": 3},
                     events=typed, event_contributions=[
                         {"event_id": e["event_id"], "component": e["source"], "values_mm": []}
                         for e in typed])


def test_tolerance_boundary_and_competing_events_do_not_inflate_tp():
    assert matched_count({100, 104}, {102}, 3) == 1
    assert matched_count({100}, {97, 103, 104}, 3) == 1
    assert matched_count({100}, {104}, 3) == 0
    t = truth([("spike", "N", 100, 100)])
    p = PointResult(case_id="case", method="test", status="success",
                    predictions={"N": [103, 103], "E": [100], "U": []})
    r = evaluate_views({"case": t}, {"case": p}, "point")
    assert r["point_exact"]["tp"] == 0
    assert r["point_tolerance_3d"]["tp"] == 1
    assert r["point_tolerance_3d"]["fp"] == 1


def test_inclusive_iou_overlap_deduplication_and_negative_axis_penalty():
    t = truth([("transient_shift", "N", 100, 113)])
    p = RangeResult(case_id="case", method="test", status="success",
                    predictions={"N": [(100, 106), (103, 109)], "E": [(60, 69)], "U": []})
    r = evaluate_views({"case": t}, {"case": p}, "range")
    assert r["range_daily_positive_axes"]["mean_iou"] == 10 / 14
    assert r["range_daily_all_axes_micro"]["tp"] == 10
    assert r["range_daily_all_axes_micro"]["fp"] == 10
    assert r["negative_axes"]["far"] == 0.5


def test_failed_positive_kept_and_failed_normal_not_counted_clean():
    t = truth([("transient_shift", "N", 100, 113)])
    normal = truth([]).model_copy(update={"case_id": "normal"})
    r = evaluate_views({"case": t, "normal": normal}, {}, "range")
    assert r["cases"] == 2
    assert r["range_affiliation"]["f1"] == 0
    assert r["range_daily_positive_axes"]["mean_iou"] == 0
    assert r["normal"]["failure_rate"] == 1
    assert r["negative_axes"]["far"] is None


def test_empty_normal_metrics_are_na_but_alarm_is_reported():
    p = PointResult(case_id="case", method="test", status="success",
                    predictions={"N": [100], "E": [], "U": []})
    r = evaluate_views({"case": truth([])}, {"case": p}, "point")
    assert r["point_exact"]["f1"] is None
    assert r["point_exact"]["fp"] == 1
    assert r["normal"]["axis_far"] == pytest.approx(1 / 3)
