from __future__ import annotations

import numpy as np
import pytest

from gnss_sim.landslide_candidate_confirmation import (
    calendar_information,
    confirmation_base,
    proposal_pool,
)
from gnss_sim.landslide_window_experiment import aggregate_windows


def test_one_positive_view_survives_as_proposal_but_not_confirmation():
    observed = np.ones(5, dtype=bool)
    rows = [
        {"start": 0, "stop": 5, "activity": [0, 1, 1, 1, 0], "status": "success"},
        {"start": 0, "stop": 5, "activity": [0, 0, -1, 0, 0], "status": "success"},
    ]
    strict = aggregate_windows(rows, observed)
    pool = proposal_pool(rows, observed)
    assert pool.spans == [(1, 4)]
    assert pool.positive_vote_days.tolist() == [False, True, True, True, False]
    assert confirmation_base(strict.activity, pool, observed).tolist() == [0, -1, -1, -1, 0]


def test_negative_unknown_conflict_is_reviewed_but_unanimous_unknown_is_not_invented():
    observed = np.ones(3, dtype=bool)
    rows = [
        {"start": 0, "stop": 3, "activity": [0, -1, 0], "status": "success"},
        {"start": 0, "stop": 3, "activity": [-1, -1, 0], "status": "success"},
    ]
    pool = proposal_pool(rows, observed)
    assert pool.spans == [(0, 1)]
    assert pool.negative_unknown_conflict_days.tolist() == [True, False, False]


def test_failure_does_not_create_conflict_or_discard_original_unknowns():
    observed = np.ones(3, dtype=bool)
    rows = [
        {"start": 0, "stop": 3, "activity": [0, 0, 0], "status": "success"},
        {"start": 0, "stop": 3, "activity": [-1, -1, -1], "status": "failed"},
    ]
    strict = aggregate_windows(rows, observed)
    pool = proposal_pool(rows, observed)
    assert pool.spans == []
    assert confirmation_base(strict.activity, pool, observed).tolist() == [-1, -1, -1]


def test_original_missing_gaps_join_proposals_but_observed_negatives_do_not():
    observed = np.array([True, False, True, True, True])
    rows = [{"start": 0, "stop": 5, "activity": [1, -1, 1, 0, 1], "status": "success"}]
    pool = proposal_pool(rows, observed)
    assert pool.spans == [(0, 3), (4, 5)]
    assert not pool.positive_vote_days[1]
    with pytest.raises(ValueError, match="missing-data"):
        proposal_pool([{**rows[0], "activity": [1, 1, 1, 0, 1]}], observed)


def test_calendar_prompt_reports_only_actual_missing_days_in_the_plot():
    text = calendar_information(np.array([False, True, False, True, True]), 1, 5)
    assert "indices are [2]" in text
    assert "days 1..4" in text
    assert "availability alone is not motion evidence" in text
