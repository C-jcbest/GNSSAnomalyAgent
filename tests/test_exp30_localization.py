"""Original index coverage and history boundaries for adaptive visual localization."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp30 import local_context, windows  # noqa: E402
from experiment_exp30_s1 import complete_windows  # noqa: E402


def test_scan_windows_cover_monitoring_horizon_without_history():
    observation = {'background_end': 364, 'dates': list(range(730))}
    coverage = np.zeros(730, dtype=int)
    for left, right in windows(observation):
        coverage[left:right] += 1
    assert not coverage[:365].any()
    assert (coverage[365:] >= 1).all()
    assert (coverage[485:545] == 2).all()


@pytest.mark.parametrize(('region', 'expected'), [
    ({'start': 365, 'end': 370}, (365, 406)),
    ({'start': 720, 'end': 729}, (685, 730)),
    ({'start': 500, 'end': 530}, (465, 566)),
])
def test_local_zoom_preserves_original_indices_and_observation_limits(region, expected):
    observation = {'background_end': 364, 'dates': list(range(730))}
    assert local_context(region, observation) == expected


def test_scan_stops_once_an_existing_window_covers_the_last_sample():
    observation = {'background_end': 364, 'dates': list(range(730))}
    assert complete_windows(observation) == [(365, 545), (485, 665), (605, 730)]
    # Do not add a five-day tail after an existing window already covers the horizon.
    assert all(right - left > 5 for left, right in complete_windows(observation))
