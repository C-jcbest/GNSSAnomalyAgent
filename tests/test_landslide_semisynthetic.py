"""The semi-synthetic observation transformation must preserve known increments."""
import numpy as np
import pytest

from gnss_sim.landslide_semisynthetic import add_injection, residual_background


def test_observed_calendar_linear_trend_removed_without_filling_gaps():
    days = np.arange(100)
    coordinates = 1e6 + days[:, None] * np.array([.001, -.002, .003])
    coordinates[30:36] = np.nan
    residual, fit = residual_background(coordinates)
    assert np.isnan(residual[30:36]).all()
    assert np.max(np.abs(residual[np.isfinite(residual).all(axis=1)])) < 1e-6
    assert fit["observed_days"] == 94
    assert np.allclose(fit["rate_NEU_mm_day"], [1, -2, 3], atol=1e-7)


def test_injection_preserves_exact_known_increment_and_original_missing_mask():
    background = np.arange(90).reshape(30, 3).astype(float)
    background[8:11] = np.nan
    motion = np.arange(30)[:, None] * np.array([.04, .02, -.01])
    artifact = np.zeros_like(motion)
    artifact[20:] = [12, -4, 8]
    output = add_injection(background, motion, artifact)
    assert all(row is None for row in output[8:11])
    observed = np.isfinite(background).all(axis=1)
    actual = np.array([row for row in output if row is not None])
    assert np.allclose(actual - background[observed], (motion + artifact)[observed])
    with pytest.raises(ValueError, match="calendars"):
        add_injection(background, motion[:-1], artifact)
