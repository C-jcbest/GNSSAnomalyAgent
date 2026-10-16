"""Interval checks preserve weak accumulation, missing data and scoring exposure."""
from types import SimpleNamespace

import numpy as np
import pytest
from probe_landslide_interval_consistency import check_prediction

from gnss_sim.landslide_evaluation import ActivityPrediction, evaluate_case


def prediction_for(values):
    observed = np.isfinite(values).all(axis=1)
    activity = np.where(observed, 1, -1)
    stage = np.where(observed, 2, -1)
    return ActivityPrediction(activity, stage)


def case_for(values):
    return SimpleNamespace(displacement_mm=[
        row.tolist() if np.isfinite(row).all() else None for row in values
    ])


def test_weak_accumulation_survives_both_policies():
    values = np.arange(180)[:, None] * np.array([[0.04, -0.02, -0.01]])
    original = prediction_for(values)
    for policy in ("contradiction_only", "require_support"):
        result, details = check_prediction(case_for(values), np.ones(3), original, policy)
        np.testing.assert_array_equal(result.activity, original.activity)
        assert details[0]["evidence"]["consistent_accumulation"]


def test_clear_return_changes_to_unknown_without_fabricating_negative_or_stage():
    trajectory = np.r_[np.arange(90), np.arange(89, -1, -1)] * 0.1
    values = trajectory[:, None] * np.array([[1, 0, 0]])
    original = prediction_for(values)
    result, details = check_prediction(case_for(values), np.ones(3), original, "contradiction_only")
    assert np.all(result.activity == -1)
    assert np.all(result.stage == -1)
    assert np.all(original.activity == 1)
    assert details[0]["evidence"]["reversal"]
    rows = [{"activity_label": 1, "feature": "steady_motion", "episode_id": "one"}] * 180
    assert evaluate_case(rows, result)["daily"]["fn"] == 180


def test_large_vertical_noise_does_not_reverse_whitened_horizontal_motion():
    values = np.column_stack((np.arange(180) * 0.1, np.zeros(180),
                              np.r_[np.arange(90), np.arange(89, -1, -1)]))
    result, details = check_prediction(case_for(values), [1, 1, 100],
                                       prediction_for(values), "contradiction_only")
    assert details[0]["evidence"]["consistent_accumulation"]
    assert np.all(result.activity == 1)


def test_step_is_unsupported_but_not_claimed_as_confirmed_reversal():
    values = np.zeros((180, 3))
    values[40:, 0] = 20
    original = prediction_for(values)
    contradiction, _ = check_prediction(case_for(values), np.ones(3), original, "contradiction_only")
    strict, _ = check_prediction(case_for(values), np.ones(3), original, "require_support")
    assert np.all(contradiction.activity == 1)
    assert np.all(strict.activity == -1)


def test_insufficient_blocks_and_short_events_preserve_existing_decisions():
    values = np.zeros((90, 3))
    values[40:47] = np.nan
    original = prediction_for(values)
    result, details = check_prediction(case_for(values), np.ones(3), original, "require_support")
    np.testing.assert_array_equal(result.activity, original.activity)
    assert details[0]["evidence"]["status"] == "insufficient_blocks"
    short_values = np.zeros((35, 3))
    short_original = prediction_for(short_values)
    short, details = check_prediction(case_for(short_values), np.ones(3), short_original, "require_support")
    np.testing.assert_array_equal(short.activity, short_original.activity)
    assert details[0]["evidence"]["status"] == "too_short"


def test_negative_days_separate_events_and_invalid_noise_is_rejected():
    values = np.zeros((180, 3))
    original = prediction_for(values)
    original.activity[70:90] = 0
    original.stage[70:90] = 0
    result, details = check_prediction(case_for(values), np.ones(3), original, "require_support")
    assert len(details) == 2
    assert np.all(result.activity[70:90] == 0)
    assert np.all(result.stage[70:90] == 0)
    with pytest.raises(ValueError, match="finite positive"):
        check_prediction(case_for(values), [1, 0, np.nan], original, "require_support")
