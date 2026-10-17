"""Coarse proposals can recover nearby observed boundaries without labeling gaps."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp42 import explore, search_edges  # noqa: E402


def test_search_recovers_both_edges_instead_of_testing_only_coarse_proposal_dates():
    values = np.random.default_rng(42).normal(0, .1, (730, 3))
    values[430:480] += [8, -3, 1]
    evidence, edges = explore(values, [{'id': 'R1', 'start': 433, 'end': 482}])
    result = evidence[0]
    assert result['onset_search'][0]['index'] == 430
    assert result['return_search'][0]['index'] == 480
    assert 430 in edges and 480 in edges
    assert any(pair['start'] == 430 and pair['end'] == 479
        and pair['vector_cancellation'] < .02 for pair in result['pair_evidence'])


def test_search_never_offers_unobserved_dates_or_invents_fit_evidence_for_empty_context():
    values = np.full((730, 3), np.nan)
    assert search_edges(values, 450, {}) == []
    values[430:440] = 1
    assert all(np.isfinite(values[item['index']]).all() for item in search_edges(values, 435, {}))
