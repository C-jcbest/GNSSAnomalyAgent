"""Display decisions use the existing conflict threshold and no review answers."""
from types import SimpleNamespace

import numpy as np
import pytest

from gnss_sim.landslide_adaptive_review import ReviewPolicy
from gnss_sim.landslide_evidence_observation import evidence_display_choices


def inputs(motion_days):
    rate = np.zeros(40)
    rate[:motion_days] = .1
    features = SimpleNamespace(progressive_rate=rate, signal_to_noise=np.full(40, 3.),
                               observed=np.ones(40, dtype=bool))
    evidence = {"rate": np.zeros(40), "signal_to_noise": np.zeros(40)}
    target = {"index": 7, "start": 0, "stop": 40, "conflict_days": 20}
    return np.zeros(40, dtype=int), features, evidence, [target], ReviewPolicy()


def test_newly_reviewable_target_gets_blocks_but_threshold_target_keeps_raw():
    assert evidence_display_choices(*inputs(13))[0]["arm"] == "narrow_blocks"
    assert evidence_display_choices(*inputs(14))[0]["arm"] == "narrow_raw"


def test_missing_motion_days_do_not_inflate_original_conflict_support():
    initial, features, evidence, targets, policy = inputs(14)
    features.observed[0] = False
    initial[0] = -1
    choice = evidence_display_choices(initial, features, evidence, targets, policy)[0]
    assert choice["base42_conflict_days"] == 13
    assert choice["arm"] == "narrow_blocks"
    assert choice["index"] == 7


def test_display_policy_cannot_add_an_unselected_low_support_target():
    initial, features, evidence, targets, policy = inputs(0)
    targets[0]["conflict_days"] = 13
    with pytest.raises(ValueError, match="already selected84"):
        evidence_display_choices(initial, features, evidence, targets, policy)
