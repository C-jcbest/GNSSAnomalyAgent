from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_visual_invariance import (
    chart_numeric_descriptors,
    fit_numeric_scaling,
    numeric_input,
    raw_chart_patches,
    reflect_chart_axes,
)


def test_physical_translation_and_reflection_keep_calendar_and_chart_geometry():
    days = np.arange(260)
    displacement = np.clip(days - 70, 0, 110)[:, None] * np.array([0.2, -0.1, 0.05])
    original = raw_chart_patches(SimpleNamespace(displacement_mm=displacement.tolist()), np.ones(3))
    shifted = raw_chart_patches(SimpleNamespace(displacement_mm=(displacement + 1000).tolist()), np.ones(3))
    np.testing.assert_allclose(original, shifted, atol=1e-6)
    signs = np.tile(np.array([-1, 1, -1]), (len(days), 1))
    reflected = raw_chart_patches(SimpleNamespace(displacement_mm=(displacement * signs).tolist()), np.ones(3))
    np.testing.assert_allclose(reflected, reflect_chart_axes(original, signs), atol=1e-6)
    np.testing.assert_array_equal(reflected[:, 3:], original[:, 3:])
    numerical = chart_numeric_descriptors(original)
    reflected_numerical = chart_numeric_descriptors(reflected)
    expected = numerical.copy()
    expected[:, :183] = (expected[:, :183].reshape(len(days), 3, 61) * signs[:, :, None]).reshape(len(days), -1)
    np.testing.assert_allclose(reflected_numerical, expected, atol=1e-6)


def test_unavailable_columns_and_record_edges_have_no_filled_trace():
    rows = (np.arange(260)[:, None] * np.array([0.1, -0.1, 0.03])).tolist()
    rows[100:112] = [None] * 12
    charts = raw_chart_patches(SimpleNamespace(displacement_mm=rows), np.ones(3))
    # At center106, three-day block centered106 contains only missing raw days.
    assert not np.any(charts[106, :, :, 30])
    assert not np.any(charts[0, :, :, :30])
    assert np.any(charts[106, :3, :, 25])
    assert np.any(charts[106, :3, :, 35])


def test_numeric_scaling_uses_training_values_and_retains_missing_indicators():
    training = np.array([[1, np.nan, 5], [3, np.nan, 5]])
    scaling = fit_numeric_scaling(training)
    query = numeric_input(np.array([[100, np.nan, 5]]), scaling)
    np.testing.assert_allclose(query, [[98, 0, 0, 0, 1, 0]])
