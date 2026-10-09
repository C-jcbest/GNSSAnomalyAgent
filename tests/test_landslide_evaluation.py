from __future__ import annotations

import numpy as np
import pytest

from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    match_events,
    runs,
)


def annotation(activity, stages=None, episodes=None):
    stages = stages or ["none"] * len(activity)
    episodes = episodes or ["e01" if value == 1 else "" for value in activity]
    return [{"activity_label": value, "feature": stage, "episode_id": episode}
            for value, stage, episode in zip(activity, stages, episodes)]


def test_unknown_and_failure_are_not_true_negatives():
    rows = annotation([0, 0, 1, -1], ["none", "none", "acceleration", "none"])
    failed = ActivityPrediction(np.full(4, -1), np.full(4, -1), status="failed")
    result = aggregate_scores([evaluate_case(rows, failed)])
    assert result["daily"]["fn"] == 1
    assert result["stages"]["acceleration"]["fn"] == 1
    assert result["true_negative_days"] == 0
    assert result["coverage"] == 0
    assert result["negative_alarm_events_per_decided_station_year"] is None
    assert result["failed_cases"] == 1


def test_first_layer_miss_is_in_end_to_end_stage_score():
    rows = annotation([1, 1, 0], ["acceleration", "deceleration", "none"])
    prediction = ActivityPrediction(np.array([0, 1, 0]), np.array([0, 3, 0]))
    result = evaluate_case(rows, prediction)
    assert result["stages"]["acceleration"]["fn"] == 1
    assert result["stages"]["deceleration"]["tp"] == 1


def test_slow_displacement_is_not_relabelled_as_steady():
    rows = annotation([1], ["slow_displacement"])
    prediction = ActivityPrediction(np.array([1]), np.array([2]))
    result = aggregate_scores([evaluate_case(rows, prediction)])
    assert result["weak_recall"] == 1
    assert result["stages"]["steady_motion"]["tp"] == 0
    assert result["stage_macro_f1"] is None


def test_stage_outside_activity_rejected_except_explicit_ablation():
    with pytest.raises(ValueError, match="confirmed displacement"):
        ActivityPrediction(np.array([0]), np.array([1])).validate(1)
    prediction = ActivityPrediction(np.array([0]), np.array([1]), gated=False)
    result = evaluate_case(annotation([0]), prediction)
    assert result["false_acceleration_days"] == 1


def test_event_matching_does_not_reward_one_prediction_for_two_events():
    rows = annotation([1, 1, 0, 1, 1], episodes=["a", "a", "", "b", "b"])
    prediction = ActivityPrediction(np.ones(5, dtype=int), np.full(5, -1))
    result = evaluate_case(rows, prediction)
    assert result["events"]["tp"] == 1
    assert result["events"]["fn"] == 1


def test_event_matching_finds_maximum_cardinality_not_greedy_overlap():
    actual = [np.array([1, 1, 1, 0], dtype=bool), np.array([0, 0, 0, 1], dtype=bool)]
    predicted = [np.ones(4, dtype=bool), np.array([1, 1, 0, 0], dtype=bool)]
    assert match_events(actual, predicted, np.ones(4, dtype=bool), threshold=.25) == 2


def test_calendar_edges_and_missing_gap_do_not_split_reference_episode():
    assert runs(np.array([True, False, True])) == [(0, 1), (2, 3)]
    rows = annotation([1, -1, 1], episodes=["e", "e", "e"])
    prediction = ActivityPrediction(np.array([1, -1, 1]), np.full(3, -1))
    result = evaluate_case(rows, prediction)
    assert result["events"]["tp"] == 1
    assert result["daily"]["tp"] == 2
    assert result["unscored_label_days"] == 1


def test_predictions_in_uncertain_region_are_recorded_not_scored_as_negatives():
    rows = annotation([-1, -1, 0])
    prediction = ActivityPrediction(np.array([1, 1, 0]), np.full(3, -1))
    result = evaluate_case(rows, prediction)
    assert result["unscored_prediction_events"] == 1
    assert result["daily"]["fp"] == 0


def test_second_layer_failure_preserves_activity_but_counts_stage_misses():
    prediction = ActivityPrediction(np.array([1]), np.array([-1]), stage_status="failed")
    result = evaluate_case(annotation([1], ["acceleration"]), prediction)
    assert result["daily"]["tp"] == 1
    assert result["stages"]["acceleration"]["fn"] == 1
    assert result["stage_failed_cases"] == 1
    assert result["failed_cases"] == 0
