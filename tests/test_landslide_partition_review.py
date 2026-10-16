import numpy as np
import pytest

from gnss_sim.landslide_partition_review import (
    numeric_piece_activity,
    parse_piece_activity,
    partition_review_prompt,
    rate_partition,
    uniform_partition,
)


def test_piece_prompt_retains_provider_json_contract():
    class Features:
        observed = np.ones(6, bool)

    prompt = partition_review_prompt(Features(), None, uniform_partition(2, 0, 6), False)
    assert "JSON" in prompt
    assert "piece_states" in prompt
    assert '"activity":' not in prompt


def test_rate_proposals_preserve_unavailable_spans_without_predicting_activity():
    rates = np.array([np.nan, np.nan, 0.1, 0.1, 0.1, 0, 0, np.nan])
    assert rate_partition(rates, 1, 8) == [
        {"id": 0, "start": 1, "stop": 2},
        {"id": 1, "start": 2, "stop": 5},
        {"id": 2, "start": 5, "stop": 7},
        {"id": 3, "start": 7, "stop": 8},
    ]


def test_piece_states_preserve_mixed_unknowns_calendar_and_missing_dates():
    observed = np.array([True, True, False, True, True, True, True, True])
    pieces = uniform_partition(3, 1, 8)
    payload = {"piece_states": [[0, 0], [1, -1], [2, 1]], "evidence": "Plateau, transition, motion"}
    activity = parse_piece_activity(payload, observed, pieces)
    assert activity.tolist() == [-1, 0, -1, -1, -1, 1, 1, 1]


def test_long_proposals_use_existing_thirty_day_resolution_and_keep_rate_edges():
    rates = np.r_[np.full(61, 0.1), np.full(39, 0.02)]
    pieces = rate_partition(rates, 0, 100)
    assert [(row["start"], row["stop"]) for row in pieces] == [
        (0, 30),
        (30, 60),
        (60, 61),
        (61, 91),
        (91, 100),
    ]


@pytest.mark.parametrize(
    "states", [[[0, 1], [0, 1]], [[0, 1]], [[1, 1], [0, 1]], [[0, True], [1, 0]]]
)
def test_duplicate_omitted_reordered_and_boolean_decisions_are_rejected(states):
    with pytest.raises(ValueError):
        parse_piece_activity(
            {"piece_states": states, "evidence": "Raw view"},
            np.ones(6, bool),
            uniform_partition(2, 0, 6),
        )


def test_numeric_control_uses_observed_piece_mean_without_filling_a_missing_piece():
    observed = np.array([True, True, False, False, False, True])
    probability = np.array([0.7, 0.1, 0.99, 0.99, 0.99, 0.7])
    result = numeric_piece_activity(probability, observed, uniform_partition(3, 0, 6), 0.5)
    assert result.tolist() == [0, 0, -1, -1, -1, 1]


def test_noncontiguous_partition_cannot_silently_leave_unreviewed_days():
    pieces = [{"id": 0, "start": 0, "stop": 2}, {"id": 1, "start": 3, "stop": 5}]
    with pytest.raises(ValueError, match="contiguous"):
        parse_piece_activity(
            {"piece_states": [[0, 1], [1, 0]], "evidence": "Raw"}, np.ones(5, bool), pieces
        )
