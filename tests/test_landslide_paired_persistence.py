import numpy as np
import pytest

from gnss_sim.landslide_paired_persistence import offset_counterpart, render_encoder_chart


def test_equal_endpoints_preserve_noise_and_original_missing_dates():
    time = np.arange(183)
    latent = time[:, None] * np.array([[.07, -.04, -.02]])
    error = np.random.default_rng(42).normal(size=(183, 3))
    observations = latent + error
    observations[50:60] = np.nan
    counterpart, replacement = offset_counterpart(observations, latent, 0, 182, 91)
    np.testing.assert_array_equal(counterpart[[0, 182]], observations[[0, 182]])
    np.testing.assert_array_equal(np.isnan(counterpart), np.isnan(observations))
    observed = np.isfinite(observations).all(axis=1)
    np.testing.assert_allclose((counterpart - replacement)[observed], error[observed], atol=1e-12)
    assert np.count_nonzero(np.linalg.norm(np.diff(replacement, axis=0), axis=1)) == 1


def test_invalid_or_unobserved_endpoint_cannot_form_a_pair():
    observations = np.ones((183, 3))
    latent = np.arange(183)[:, None] * np.ones((1, 3))
    observations[0] = np.nan
    with pytest.raises(ValueError):
        offset_counterpart(observations, latent, 0, 182, 91)
    with pytest.raises(ValueError):
        offset_counterpart(np.ones((183, 3)), np.zeros((183, 3)), 0, 182, 91)


def test_missing_chart_slot_has_no_cross_gap_line():
    patch = np.zeros((6, 32, 61))
    patch[3:] = 1
    patch[:3, 16, :] = 1
    patch[:, :, 15] = 0
    image = np.asarray(render_encoder_chart(patch))
    assert image.shape == (256, 256, 3)
    # Slot15 lies away from the independent query marker at slot30.
    assert tuple(image[42, 83]) == (231, 231, 231)
    assert tuple(image[42, 79]) != (231, 231, 231)
