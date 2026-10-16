import numpy as np
import pytest

from gnss_sim.landslide_local_persistence import local_offset_counterpart


def test_local_replacement_preserves_other_motion_errors_and_missing_dates():
    latent = np.arange(80, dtype=float)[:, None] * np.array([[0.5, -0.8, 0.2]])
    error = np.random.default_rng(71).normal(size=latent.shape)
    error[18:] += [6, -3, 1]
    error[61] += [12, 5, -8]
    observed = latent + error
    observed[38] = np.nan
    negative, replacement = local_offset_counterpart(observed, latent, 20, 50, 28)
    np.testing.assert_array_equal(negative[:20], observed[:20])
    np.testing.assert_array_equal(negative[51:], observed[51:])
    np.testing.assert_array_equal(replacement[:20], latent[:20])
    np.testing.assert_array_equal(replacement[51:], latent[51:])
    np.testing.assert_array_equal(negative[[20, 50]], observed[[20, 50]])
    np.testing.assert_array_equal(np.isnan(negative), np.isnan(observed))
    np.testing.assert_allclose(
        (negative - replacement)[np.isfinite(observed)],
        (observed - latent)[np.isfinite(observed)],
        atol=1e-13,
    )
    np.testing.assert_array_equal(np.diff(replacement[31:45], axis=0), 0)
    np.testing.assert_allclose(replacement[28] - replacement[27], latent[50] - latent[20])


@pytest.mark.parametrize("problem", ["zero", "missing_endpoint", "wrong_calendar"])
def test_invalid_training_interventions_are_rejected(problem):
    latent = np.arange(30, dtype=float)[:, None] * np.ones((1, 3))
    observed = latent.copy()
    if problem == "zero":
        latent[:] = 0
    elif problem == "missing_endpoint":
        observed[5] = np.nan
    else:
        latent = latent[:-1]
    with pytest.raises(ValueError):
        local_offset_counterpart(observed, latent, 5, 20, 10)
