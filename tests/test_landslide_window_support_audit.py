"""Regressions for calendar guards and non-classifying sensitivity diagnostics."""

import json
from datetime import date, timedelta

import numpy as np

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_window_support_audit import (
    audit_record,
    compare_windows,
    observation_guard,
    summarize,
)


def test_missing_run_at_window_edge_is_counted():
    observed = np.ones(150, dtype=bool)
    observed[20:28] = False
    guard = observation_guard(observed, 50, 61)
    assert guard["window_start"] == 20
    assert guard["longest_gap"] == 8
    assert guard["primary_reason"] == "long_gap"
    observed[20] = True
    assert observation_guard(observed, 50, 61)["primary_reason"] == "eligible"


def test_primary_reason_follows_actual_guard_order_with_overlap():
    observed = np.ones(150, dtype=bool)
    observed[20:40] = False
    guard = observation_guard(observed, 30, 61)
    assert guard["primary_reason"] == "center_missing"
    assert guard["overlapping_reasons"] == "center_missing|insufficient_count|long_gap"
    guard = observation_guard(observed, 15, 61)
    assert guard["primary_reason"] == "record_edge"
    assert "insufficient_side" in guard["overlapping_reasons"]


def test_bilateral_support_excludes_center_and_calendar_stop_is_exclusive():
    observed = np.ones(100, dtype=bool)
    observed[20:27] = False
    guard = observation_guard(observed, 35, 31)
    assert (guard["window_start"], guard["window_stop"]) == (20, 51)
    assert guard["valid_count"] == 24
    assert (guard["left_count"], guard["right_count"]) == (8, 15)
    assert guard["primary_reason"] == "insufficient_side"
    assert observation_guard(np.ones(100, dtype=bool), 84, 31)["primary_reason"] == "eligible"
    assert observation_guard(np.ones(100, dtype=bool), 85, 31)["primary_reason"] == "record_edge"


def motion_row(velocity, tangential):
    return {"velocity_available": velocity is not None, "tangential_available": tangential is not None,
            "velocity_mm_day": velocity,
            "speed_mm_day": float(np.linalg.norm(velocity)) if velocity is not None else None,
            "tangential_mm_day2": tangential}


def test_tiny_opposite_signs_are_reported_without_stage_or_significance_claim():
    comparison = compare_windows(motion_row([1., 0., 0.], 1e-14), motion_row([1., 0., 0.], -2e-14))
    assert comparison["tangential_opposite_sign"] is True
    assert comparison["opposite_min_abs_tangential"] == 1e-14
    assert comparison["velocity_cosine"] == 1.
    assert "stage" not in comparison
    assert "significant" not in comparison


def test_missing_and_zero_cannot_become_deceleration_or_velocity_reversal():
    comparison = compare_windows(motion_row([0., 0., 0.], None), motion_row([1., 0., 0.], 0.))
    assert comparison["common_velocity"] is True
    assert comparison["common_tangential"] is False
    assert comparison["velocity_cosine"] is None
    assert comparison["tangential_opposite_sign"] is False
    reversal = compare_windows(motion_row([1., 0., 0.], 0.), motion_row([-1., 0., 0.], .01))
    assert reversal["velocity_cosine"] == -1.
    assert reversal["tangential_opposite_sign"] is False


def test_full_observation_audit_exports_iso_dates_and_preserves_internal_gaps():
    observations = [[float(day), 0., 0.] for day in range(730)]
    observations[150:165] = [None] * 15
    case = LandslideInput(case_id="case_0001",
        dates=[date(2023,1,1) + timedelta(days=day) for day in range(730)],
        displacement_mm=observations)
    review = {"spans": [
        {"start":0,"stop":100,"state":"stationary"},
        {"start":100,"stop":200,"state":"activity"},
        {"start":200,"stop":730,"state":"unknown"},
    ]}
    motions = {window: derive_motion(case, window) for window in (31,61,91)}
    rows, pairs, spans = audit_record(case, review, motions)
    assert len(rows) == len(pairs) == 2190
    assert len(spans) == 1
    assert rows[0]["date"] == "2023-01-01"
    keyed = {(row["window_days"],row["day_index"]):row for row in rows}
    assert keyed[31,114]["contained_in_same_interior"] is False
    assert keyed[31,115]["contained_in_same_interior"] is True
    assert keyed[31,184]["contained_in_same_interior"] is True
    assert keyed[31,185]["contained_in_same_interior"] is False
    assert keyed[31,150]["activity_state"] == "activity"
    assert keyed[31,150]["observed"] is False
    assert keyed[31,150]["tangential_available"] is False
    summary = summarize(rows,pairs)
    assert summary["activity_calendar_days"] == 100
    assert summary["activity_observed_days"] == 85
    assert summary["activity_missing_days"] == 15
    json.dumps({"rows":rows,"pairs":pairs,"summary":summary}, allow_nan=False)
