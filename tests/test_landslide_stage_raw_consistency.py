from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_evaluation import ActivityPrediction
from gnss_sim.landslide_stage_raw_consistency import (
    StageContrastPolicy,
    abstain_on_stage_conflicts,
    raw_stage_contrasts,
    stage_conflict_mask,
)


def raw_case(values):
    return SimpleNamespace(displacement_mm=[
        None if not np.isfinite(value) else [float(value), 0.0, 0.0] for value in values
    ])


def test_decreasing_speed_rejects_acceleration_without_assigning_deceleration():
    time = np.arange(300)
    case = raw_case(3 * time - 0.003 * time ** 2)
    activity = np.ones(300, dtype=int)
    stage = np.ones(300, dtype=int)
    policy = StageContrastPolicy(relative_change=0.01)
    contrasts = {scale: raw_stage_contrasts(case, activity, np.ones(3) * 0.1, scale)
                 for scale in policy.separations}
    mask = stage_conflict_mask(stage, contrasts, policy)
    assert mask[100:200].all()
    result = abstain_on_stage_conflicts(ActivityPrediction(activity, stage), mask)
    assert np.array_equal(result.activity, activity)
    assert (result.stage[100:200] == -1).all()
    assert result.stage[0] == 1


def test_steady_signed_negative_motion_does_not_conflict_with_steady_stage():
    activity = np.ones(300, dtype=int)
    policy = StageContrastPolicy()
    contrasts = {scale: raw_stage_contrasts(raw_case(-np.arange(300)), activity,
                                           np.ones(3) * 0.1, scale)
                 for scale in policy.separations}
    assert not stage_conflict_mask(np.full(300, 2), contrasts, policy).any()


def test_evidence_cannot_cross_nonactive_or_observed_unknown_days():
    activity = np.ones(300, dtype=int)
    activity[:100] = 0
    activity[180] = -1
    evidence = raw_stage_contrasts(raw_case(np.arange(300)), activity, np.ones(3), 21)
    assert np.isnan(evidence[120]).all()
    assert np.isfinite(evidence[130]).all()
    assert np.isnan(evidence[170:200]).all()


def test_missing_gap_retains_calendar_and_cannot_receive_evidence():
    values = np.arange(300, dtype=float)
    values[100:110] = np.nan
    activity = np.ones(300, dtype=int)
    activity[100:110] = -1
    evidence = raw_stage_contrasts(raw_case(values), activity, np.ones(3), 21)
    assert np.isnan(evidence[105]).all()
    assert np.isnan(evidence[120]).all()
    assert np.isfinite(evidence[150]).all()


def test_one_scale_or_short_conflicts_do_not_trigger_main_policy():
    short = np.tile([2.0, 1.0, 0.01], (40, 1))
    steady = np.tile([1.0, 1.0, 0.01], (40, 1))
    stage = np.ones(40, dtype=int)
    assert not stage_conflict_mask(stage, {21: short, 42: steady}, StageContrastPolicy()).any()
    short[13:] = steady[13:]
    assert not stage_conflict_mask(stage, {21: short, 42: short}, StageContrastPolicy()).any()
