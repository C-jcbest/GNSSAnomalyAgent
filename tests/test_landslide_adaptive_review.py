"""Business guarantees of raw evidence, bounded selection and failure propagation."""
from types import SimpleNamespace

import numpy as np
import pytest

from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    numeric_review_summary,
    raw_motion_evidence,
    review_priorities,
)


def feature_fixture(days):
    return SimpleNamespace(observed=np.ones(days, dtype=bool), noise=np.ones(3),
                           progressive_rate=np.zeros(days), signal_to_noise=np.zeros(days))


def test_isolated_step_evidence_does_not_trigger_a_persistent_motion_review():
    days = 240
    features = feature_fixture(days)
    jump = np.zeros((days, 3))
    jump[120:, 0] = 20
    step_evidence = raw_motion_evidence(SimpleNamespace(displacement_mm=jump.tolist()), features)
    # A median that straddles the step can create one apparent progressive day.
    # The policy requires sustained conflict before spending a visual review.
    assert np.count_nonzero(step_evidence["rate"] > 0) == 1
    assert review_priorities(np.zeros(days), features, step_evidence,
                             [(0, 180), (60, 240)], ReviewPolicy())[1] == []
    ramp = np.arange(days)[:, None] * np.array([[0.08, -0.04, -0.03]])
    motion = raw_motion_evidence(SimpleNamespace(displacement_mm=ramp.tolist()), features)
    np.testing.assert_allclose(motion["rate"][49:-49], np.linalg.norm([0.08, -0.04, -0.03]))
    assert np.all(np.isnan(motion["rate"][:49]))


def test_disagreement_selection_is_bounded_and_avoids_duplicate_coverage():
    days = 400
    feature = feature_fixture(days)
    long_evidence = {"rate": np.zeros(days), "signal_to_noise": np.zeros(days)}
    long_evidence["rate"][40:160] = 0.1
    long_evidence["signal_to_noise"][40:160] = 4
    initial = np.zeros(days, dtype=int)
    initial[250:330] = 1
    priorities, selected = review_priorities(initial, feature, long_evidence,
        [(0, 180), (90, 270), (220, 400)], ReviewPolicy())
    assert priorities[0]["missed_motion_days"] == 120
    assert priorities[2]["unsupported_activity_days"] == 80
    assert [row["index"] for row in selected] == [0, 2]
    initial[:] = 0
    long_evidence["rate"][:] = 0
    assert review_priorities(initial, feature, long_evidence,
                             [(0, 180), (220, 400)], ReviewPolicy())[1] == []


def test_failed_review_replaces_target_with_unknown_and_missing_stays_unknown():
    initial = np.ones(20, dtype=int)
    observed = np.ones(20, dtype=bool)
    observed[5] = False
    review = {"start": 4, "stop": 10, "activity": [-1] * 20, "status": "failed"}
    result = combine_reviews(initial, [review], observed)
    assert np.all(result.activity[4:10] == -1)
    assert np.all(result.activity[:4] == 1)
    assert np.all(result.activity[10:] == 1)
    review["activity"][7] = 1
    with pytest.raises(ValueError, match="Failed review"):
        combine_reviews(initial, [review], observed)


def test_unknown_votes_are_not_removed_from_review_majority():
    initial = np.zeros(20, dtype=int)
    observed = np.ones(20, dtype=bool)
    positive = {"start": 2, "stop": 8, "activity": [1] * 20, "status": "success"}
    unknown = {"start": 2, "stop": 8, "activity": [-1] * 20, "status": "success"}
    result = combine_reviews(initial, [positive, unknown], observed)
    assert np.all(result.activity[2:8] == -1)
    assert np.all(result.activity[8:] == 0)


def test_numeric_summary_preserves_missing_support_without_zero_imputation():
    features = feature_fixture(60)
    features.observed[0:30] = False
    long_evidence = {"rate": np.ones(60), "signal_to_noise": np.ones(60)}
    rows = numeric_review_summary(features, long_evidence, 0, 60)
    assert rows[0]["observed_days"] == 0
    assert rows[0]["long_rate_mm_day"] == {"supported_days": 0, "median": None}
    assert rows[1]["long_rate_mm_day"] == {"supported_days": 30, "median": 1.0}
    with pytest.raises(ValueError):
        ReviewPolicy(maximum_reviews=0)
