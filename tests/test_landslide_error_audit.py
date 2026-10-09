import numpy as np

from gnss_sim.landslide_error_audit import (
    activity_disagreements,
    raw_interval_evidence,
    regular_windows,
    stage_error_partition,
)
from gnss_sim.landslide_evaluation import ActivityPrediction


def rows(features, activity):
    return [{"feature": feature, "activity_label": label} for feature, label in zip(features, activity)]


def test_error_causes_are_disjoint_and_unknown_reference_is_not_negative():
    reference = rows(["acceleration"] * 5 + ["none"], [1] * 5 + [-1])
    prediction = ActivityPrediction(np.array([0, -1, 1, 1, 1, 1]), np.array([0, -1, -1, 3, 1, 1]))
    result = stage_error_partition(reference, prediction)
    assert result["counts"] == {"activity_output_failure": 0, "activity_not_confirmed": 1,
                                "activity_unknown": 1, "stage_output_failure": 0,
                                "stage_abstention": 1, "stage_wrong_class": 1, "correct": 1}
    assert result["false_stage_days_on_reference_negative"] == 0
    assert activity_disagreements(reference, prediction)["false_activity"]["days"] == 0


def test_stage_failure_separated_from_upstream_activity_miss():
    reference = rows(["acceleration"] * 3, [1, 1, 1])
    prediction = ActivityPrediction(np.array([0, 1, 1]), np.full(3, -1), stage_status="failed")
    result = stage_error_partition(reference, prediction)
    assert result["counts"]["activity_not_confirmed"] == 1
    assert result["counts"]["stage_output_failure"] == 2


def test_ungated_ablation_can_predict_correct_stage_outside_its_activity_gate():
    prediction = ActivityPrediction(np.array([0]), np.array([1]), gated=False)
    result = stage_error_partition(rows(["acceleration"], [1]), prediction)
    assert result["counts"]["correct"] == 1
    assert result["counts"]["activity_not_confirmed"] == 0


def test_fixed_windows_cover_entire_calendar_including_last_day():
    windows = regular_windows(1095)
    assert windows[0] == (0, 180)
    assert windows[-1] == (915, 1095)
    assert len(windows) == 12
    covered = {day for start, stop in windows for day in range(start, stop)}
    assert covered == set(range(1095))
    assert regular_windows(40) == [(0, 40)]


def test_raw_evidence_preserves_missing_days_and_translation_invariant_delta():
    values = np.column_stack([np.arange(60, dtype=float), np.zeros(60), np.zeros(60)])
    values[20:25] = np.nan
    evidence = raw_interval_evidence(values, 0, 60)
    translated = raw_interval_evidence(values + [100, -100, 5], 0, 60)
    assert evidence["observed_days"] == 55
    assert evidence["blocks"][1]["observed"] == 9
    assert evidence["endpoint_median_delta_mm"] == [46, 0, 0]
    assert evidence["endpoint_median_delta_mm"] == translated["endpoint_median_delta_mm"]
