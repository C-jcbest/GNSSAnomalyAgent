import numpy as np

from gnss_sim.landslide_partial_review import combine_partial_reviews


def local(activity, status="success"):
    return {"start": 0, "stop": len(activity), "activity": np.array(activity), "status": status}


def test_successful_abstention_keeps_the_prior_without_filling_unknown_or_missing():
    observed = np.array([True, True, True, False])
    initial = np.array([1, 0, -1, -1])
    prediction, no_new_decision = combine_partial_reviews(initial, [local([-1, -1, -1, -1])], observed)
    assert prediction.activity.tolist() == [1, 0, -1, -1]
    assert prediction.stage.tolist() == [-1, 0, -1, -1]
    assert no_new_decision.tolist() == [True, True, True, False]


def test_definite_negative_can_remove_prior_activity_and_positive_can_add_it():
    prediction, no_new_decision = combine_partial_reviews(
        np.array([1, 0]), [local([0, 1])], np.ones(2, dtype=bool))
    assert prediction.activity.tolist() == [0, 1]
    assert not no_new_decision.any()


def test_failure_does_not_restore_a_confident_prior():
    prediction, no_new_decision = combine_partial_reviews(
        np.array([1, 0]), [local([-1, -1], "failed")], np.ones(2, dtype=bool))
    assert prediction.activity.tolist() == [-1, -1]
    assert not no_new_decision.any()


def test_conflicting_definite_answers_are_unknown_despite_a_confident_prior():
    prediction, no_new_decision = combine_partial_reviews(
        np.array([1, 0]), [local([1, 0]), local([0, 1])], np.ones(2, dtype=bool))
    assert prediction.activity.tolist() == [-1, -1]
    assert not no_new_decision.any()


def test_partial_abstention_does_not_turn_a_single_vote_into_a_majority():
    prediction, no_new_decision = combine_partial_reviews(
        np.array([0]), [local([1]), local([-1])], np.ones(1, dtype=bool))
    assert prediction.activity.tolist() == [-1]
    assert not no_new_decision.any()
