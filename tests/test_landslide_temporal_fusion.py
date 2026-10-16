from itertools import product

import numpy as np
import pytest

from gnss_sim.landslide_temporal_fusion import decode_activity_scores, fused_motion_scores


@pytest.mark.parametrize("scores,cost", [([1, -0.2, 1, -2], 0.8), ([1, -1], 1), ([0, 0], 1)])
def test_decoder_matches_all_globally_optimal_paths(scores, cost):
    paths = list(product((0, 1), repeat=len(scores)))
    energies = [
        -sum((2 * state - 1) * value / 2 for state, value in zip(path, scores, strict=True))
        + cost * sum(left != right for left, right in zip(path[:-1], path[1:], strict=True))
        for path in paths
    ]
    optimum = min(energies)
    optimal = np.array(
        [
            path
            for path, energy in zip(paths, energies, strict=True)
            if abs(energy - optimum) < 1e-12
        ]
    )
    expected = [
        int(optimal[0, day]) if np.unique(optimal[:, day]).size == 1 else -1
        for day in range(len(scores))
    ]
    assert decode_activity_scores(scores, np.ones(len(scores), bool), cost).tolist() == expected


def test_missing_gap_does_not_bridge_stationary_and_active_evidence():
    result = decode_activity_scores(
        np.array([2, 2, 100, -2, -2]), np.array([True, True, False, True, True]), 100
    )
    assert result.tolist() == [1, 1, -1, 0, 0]


def test_jointly_unsupported_dates_stay_unknown_and_visual_ablation_removes_support():
    observed = np.array([True, True, True, False])
    supported = np.array([True, False, False, False])
    visual = np.array([-1, 1, -1, -1])
    scores, available = fused_motion_scores(
        np.array([0.8, 0.8, 0.8, 0.8]), supported, visual, observed, 0.5, 1
    )
    assert available.tolist() == [True, True, False, False]
    assert decode_activity_scores(scores, available, 0).tolist() == [1, 1, -1, -1]
    _, available_without_visual = fused_motion_scores(
        np.array([0.8] * 4), supported, visual, observed, 0.5, 0
    )
    assert available_without_visual.tolist() == [True, False, False, False]
