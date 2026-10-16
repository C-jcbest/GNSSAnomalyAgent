"""Original activity, missing dates and motion-coordinate phase evidence."""

import numpy as np
import pytest

from gnss_sim.landslide_phase_residual import phase_evidence_window, phase_residual_evidence


def test_window_shifts_inside_activity_and_never_crosses_known_negatives():
    activity = np.zeros(240, dtype=int)
    activity[50:150] = 1
    observed = np.ones(240, dtype=bool)
    assert phase_evidence_window(activity, observed, 56, 70) == (50, 148)
    activity[90:100] = 0
    assert phase_evidence_window(activity, observed, 56, 70) is None
    activity[90:100] = -1
    observed[90:100] = False
    assert phase_evidence_window(activity, observed, 56, 70) is None
    assert phase_evidence_window(activity, observed, 140, 154) is None


def test_constant_rate_has_zero_residual_and_missing_day_stays_missing():
    days = np.arange(98, dtype=float)
    values = days[:, None] * np.array([0.3, -0.7, 0.1]) + [2, -8, 12]
    activity = np.ones(98, dtype=int)
    evidence = phase_residual_evidence(values, [1, 1, 2], activity, 0, 98)
    np.testing.assert_allclose(evidence["residual"], 0, atol=1e-12)
    np.testing.assert_allclose(np.diff(evidence["progression"]), np.diff(evidence["linear_reference"]), atol=1e-12)
    values[40] = np.nan
    activity[40] = -1
    evidence = phase_residual_evidence(values, [1, 1, 2], activity, 0, 98)
    assert evidence["calendar"][40] == 40
    assert evidence["raw_progression"][40] is None
    assert evidence["raw_residual"][40] is None


def test_direction_projection_retains_curvature_under_axis_sign_changes():
    days = np.arange(98, dtype=float)
    values = (0.01 * days**2 + 0.2 * days)[:, None] * np.array([1, -2, 0.3])
    activity = np.ones(98, dtype=int)
    first = phase_residual_evidence(values, [1, 2, 3], activity, 0, 98)
    second = phase_residual_evidence(values * [-1, 1, -1] + [80, -40, 60], [1, 2, 3], activity, 0, 98)
    np.testing.assert_allclose(first["residual"], second["residual"], atol=1e-12)
    assert np.all(np.diff(first["adjacent_block_rates_proxy_per_day"]) > 0)
    activity[45] = 0
    with pytest.raises(ValueError, match="inactive"):
        phase_residual_evidence(values, [1, 2, 3], activity, 0, 98)
