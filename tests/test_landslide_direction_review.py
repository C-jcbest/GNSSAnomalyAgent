from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_causal_direction_review import causal_direction_anchors
from gnss_sim.landslide_direction_review import (
    raw_block_changes,
    review_directional_additions,
    strong_direction_anchors,
)


def review(values, observed=None, variant="direction_and_motion", local_positive=True):
    days = len(values)
    observed = np.ones(days, dtype=bool) if observed is None else observed
    initial = np.zeros(days, dtype=int)
    initial[:100] = 1
    initial[~observed] = -1
    local = np.where(observed, int(local_positive), -1)
    scope = observed & (np.arange(days) >= 120)
    anchors = [{"start": 10, "stop": 40, "available_from": 89, "direction": [1., 0., 0.]}]
    return review_directional_additions(initial, local, scope, observed, values,
                                        np.ones(3) * .1, anchors, variant)


def test_weak_tail_requires_local_visual_and_raw_direction():
    values = np.column_stack((np.arange(220) * .04, np.zeros((220, 2))))
    activity, decisions = review(values)
    assert np.all(activity[:100] == 1)
    assert np.all(activity[120:] == 1)
    assert decisions[0]["accepted"]
    negative, _ = review(values, local_positive=False)
    assert not np.any(negative[120:] == 1)


def test_platform_and_turning_proposal_remain_unknown():
    plateau = np.zeros((220, 3))
    activity, _ = review(plateau)
    assert np.all(activity[120:] == -1)
    turning = np.column_stack((np.zeros(220), np.arange(220) * .04, np.zeros(220)))
    activity, decisions = review(turning)
    assert np.all(activity[120:] == -1)
    assert not decisions[0]["direction_pass"]
    local_direction, _ = review(turning, variant="without_direction")
    assert np.all(local_direction[120:] == 1)


def test_short_missing_gap_is_unlabelled_and_long_gap_refuses_evidence():
    values = np.column_stack((np.arange(220) * .04, np.zeros((220, 2))))
    observed = np.ones(220, dtype=bool)
    observed[170:173] = False
    values[~observed] = np.nan
    activity, _ = review(values, observed)
    assert np.all(activity[~observed] == -1)
    assert np.all(activity[120:][observed[120:]] == 1)
    observed[162:178] = False
    values[~observed] = np.nan
    activity, decisions = review(values, observed)
    assert np.all(activity[~observed] == -1)
    assert len(decisions) == 2
    # The proposal after the gap is exactly42days, independently checked.
    assert all(row["stop"] - row["start"] <= 42 for row in decisions)


def test_strong_raw_anchor_is_reference_invariant_and_not_a_step():
    values = np.column_stack((np.arange(240) * .2, np.zeros((240, 2))))
    observed = np.ones(240, dtype=bool)
    before, after = raw_block_changes(SimpleNamespace(displacement_mm=values.tolist()))[1:]
    anchors = strong_direction_anchors(before, after, observed, np.ones(3))
    assert len(anchors) == 1
    assert anchors[0]["direction"] == [1., 0., 0.]
    shifted = values + [500, -700, 250]
    shifted_before, shifted_after = raw_block_changes(SimpleNamespace(displacement_mm=shifted.tolist()))[1:]
    np.testing.assert_allclose(before, shifted_before, atol=1e-10)
    np.testing.assert_allclose(after, shifted_after, atol=1e-10)
    step = np.zeros((240, 3))
    step[120:, 0] = 100
    step_before, step_after = raw_block_changes(SimpleNamespace(displacement_mm=step.tolist()))[1:]
    assert not strong_direction_anchors(step_before, step_after, observed, np.ones(3))


def test_anchor_footprint_cannot_use_proposal_future():
    values = np.column_stack((np.arange(220) * .04, np.zeros((220, 2))))
    observed = np.ones(220, dtype=bool)
    initial = np.zeros(220, dtype=int)
    local = np.ones(220, dtype=int)
    anchors = [{"start": 100, "stop": 150, "available_from": 199, "direction": [1., 0., 0.]}]
    activity, decisions = review_directional_additions(initial, local, observed, observed, values,
                                                       np.ones(3) * .1, anchors)
    assert np.all(activity == -1)
    assert decisions[0]["reason"] == "no_preceding_anchor"


def test_causal_anchor_is_available_during_ongoing_strong_episode():
    values = np.column_stack((np.arange(240) * .2, np.zeros((240, 2))))
    case = SimpleNamespace(displacement_mm=values.tolist())
    before, after = raw_block_changes(case)[1:]
    anchors = causal_direction_anchors(before, after, np.ones(240, dtype=bool), np.ones(3))
    eligible = [row for row in anchors if row["available_from"] <= 150]
    assert eligible
    assert all(row["stop"] - row["start"] == 14 for row in eligible)
    # Changing only observations from150 onward cannot alter already available anchors.
    values[150:, 1] = np.arange(90) * 100
    before, after = raw_block_changes(SimpleNamespace(displacement_mm=values.tolist()))[1:]
    changed = causal_direction_anchors(before, after, np.ones(240, dtype=bool), np.ones(3))
    assert eligible == [row for row in changed if row["available_from"] <= 150]
