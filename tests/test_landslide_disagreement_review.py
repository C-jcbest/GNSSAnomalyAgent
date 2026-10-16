import numpy as np

from gnss_sim.landslide_disagreement_review import (
    fit_review,
    review_benefit,
    review_features,
    revise_disagreements,
)


def test_revision_requires_disagreement_support_and_positive_predicted_benefit():
    visual = np.array([0.8, 0.2, 0.8, 0.2, 0.2])
    numeric = np.array([0.9, 0.9, 0.2, 0.9, 0.9])
    support = np.array([True, True, True, False, True])
    benefit = np.array([1, 1, -1, 1, 0])
    result, switched = revise_disagreements(visual, numeric, support, benefit, 0.5)
    assert switched.tolist() == [False, True, False, False, False]
    assert result.tolist() == [0.8, 0.9, 0.8, 0.2, 0.2]


def test_benefit_learns_actual_correction_and_keeps_identical_experts_unchanged():
    visual = np.array([0.1, 0.2, 0.8, 0.9])
    numeric = np.array([0.8, 0.7, 0.2, 0.1])
    actual = np.array([1, 1, 1, 1])
    features = review_features(visual, numeric, None, 0.5)
    model = fit_review(features, visual, numeric, actual, 0.5, 1)
    benefit = review_benefit(model, features)
    assert np.all(benefit[:2] > 0)
    assert np.all(benefit[2:] < 0)
    empty = fit_review(features, visual, visual, actual, 0.5, 1)
    assert empty["status"] == "no_training_disagreement"
    assert np.array_equal(review_benefit(empty, features), np.zeros(4))


def test_numerical_feedback_is_an_explicit_removable_source():
    visual = np.array([0.2, 0.8])
    numeric = np.array([0.7, 0.1])
    scores = review_features(visual, numeric, None, 0.5)
    with_raw = review_features(visual, numeric, np.ones((2, 36)), 0.5)
    assert scores.shape == (2, 6)
    assert with_raw.shape == (2, 42)
    assert np.array_equal(scores, with_raw[:, :6])
