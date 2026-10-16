"""Check physical context, missingness and no-clipping chart contracts."""

import numpy as np
import pytest

from gnss_sim.landslide_shared_frame import shared_frame_pair


def test_local_replacement_leaves_outer_pixels_identical():
    days = np.arange(240, dtype=float)
    original = np.column_stack((days, days * 0.1, -days * 0.2))
    negative = original.copy()
    negative[100:121] = original[99]
    negative[121:142] = original[141]
    original[80:87] = np.nan
    negative[80:87] = np.nan
    positive_chart, negative_chart, _ = shared_frame_pair(original, negative, 120, np.ones(3))
    outside = np.abs(np.arange(-90, 91, 3)) > 24
    np.testing.assert_array_equal(positive_chart[:, :, outside], negative_chart[:, :, outside])
    np.testing.assert_array_equal(positive_chart[3:], negative_chart[3:])
    assert not np.array_equal(positive_chart[:3], negative_chart[:3])
    assert np.any(positive_chart[3:, 0] == 0)


def test_negative_beyond_positive_span_is_never_clipped():
    original = np.zeros((240, 3))
    negative = original.copy()
    negative[120:140] = [900, -1200, 500]
    positive_chart, negative_chart, frame = shared_frame_pair(original, negative, 120, np.ones(3))
    assert np.all(frame["span"] >= [1800, 2400, 1000])
    for chart in (positive_chart, negative_chart):
        available = chart[3:, 0] > 0
        np.testing.assert_allclose(chart[:3].sum(axis=1)[available], 1, atol=1e-6)


def test_all_missing_observations_do_not_become_zero_trace():
    missing = np.full((240, 3), np.nan)
    left, right, frame = shared_frame_pair(missing, missing, 120, np.array([1, 2, 3]))
    assert not left.any()
    assert not right.any()
    np.testing.assert_array_equal(frame["span"], [6, 12, 18])


def test_changed_missing_mask_is_rejected():
    original = np.zeros((240, 3))
    negative = original.copy()
    negative[120] = np.nan
    with pytest.raises(ValueError, match="missing mask"):
        shared_frame_pair(original, negative, 120, np.ones(3))
