from __future__ import annotations

import numpy as np
import pytest

from gnss_sim.landslide_window_experiment import (
    aggregate_windows,
    apply_reviews,
    parse_window_activity,
    review_candidates,
)


def test_window_keeps_missing_calendar_and_rejects_context_labels():
    observed = np.array([True, True, False, True, True])
    payload = {"activity": [[1, 3]], "uncertain": [], "evidence": "continuing raw change"}
    assert parse_window_activity(payload, observed, 1, 4).tolist() == [-1, 1, -1, 1, -1]
    with pytest.raises(ValueError, match="beyond target"):
        parse_window_activity(payload, observed, 2, 4)


def test_shared_activity_uncertainty_endpoint_is_not_repaired():
    payload = {"activity": [[1, 3]], "uncertain": [[3, 4]], "evidence": "uncertain end"}
    with pytest.raises(ValueError, match="disjoint"):
        parse_window_activity(payload, np.ones(5, dtype=bool), 0, 5)


def test_majority_does_not_discard_failed_or_uncertain_views():
    observed = np.array([True, True, True, False, True])
    windows = [
        {"start": 0, "stop": 5, "activity": [1, 1, 0, 1, 0], "status": "success"},
        {"start": 0, "stop": 5, "activity": [1, 0, -1, 1, 0], "status": "success"},
        {"start": 0, "stop": 5, "activity": [-1] * 5, "status": "failed"},
    ]
    prediction = aggregate_windows(windows, observed)
    assert prediction.activity.tolist() == [1, -1, -1, -1, 0]
    assert prediction.stage.tolist() == [-1, -1, -1, -1, 0]
    assert prediction.status == "success"


def test_candidate_review_splits_platform_and_preserves_outside_unknowns():
    base = np.array([0, 1, 1, -1, 1, 1, 0, -1])
    assert review_candidates(base) == [(1, 6)]
    reviews = [{"start": 1, "stop": 6, "activity": [-1, 1, 0, 0, 0, 1, -1, -1]}]
    prediction = apply_reviews(base, reviews, np.ones(8, dtype=bool))
    assert prediction.activity.tolist() == [0, 1, 0, 0, 0, 1, 0, -1]
    assert prediction.stage.tolist() == [0, -1, 0, 0, 0, -1, 0, -1]


def test_candidates_do_not_bridge_known_platform_and_failed_windows_stay_unknown():
    assert review_candidates(np.array([1, 0, 1])) == [(0, 1), (2, 3)]
    prediction = aggregate_windows(
        [{"start": 0, "stop": 3, "activity": [-1, -1, -1], "status": "failed"}],
        np.ones(3, dtype=bool),
    )
    assert prediction.status == "failed"
    assert prediction.activity.tolist() == [-1, -1, -1]
