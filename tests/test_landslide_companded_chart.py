from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_companded_chart import (
    chart_ordinates,
    companded_chart_patches,
    raw_window_medians,
)
from gnss_sim.landslide_visual_invariance import raw_chart_patches


def case_for(values):
    return SimpleNamespace(
        displacement_mm=[None if not np.isfinite(row).all() else row.tolist() for row in values]
    )


def test_reference_translation_does_not_change_target_anchored_chart():
    values = np.column_stack([np.linspace(0, 20, 230)] * 3)
    medians, anchor = raw_window_medians(case_for(values))
    translated, translated_anchor = raw_window_medians(case_for(values + [500, -200, 100]))
    first = companded_chart_patches(medians, anchor, np.ones(3))
    second = companded_chart_patches(translated, translated_anchor, np.ones(3))
    assert np.allclose(first, second, atol=1e-5)


def test_missing_original_columns_are_not_filled_or_connected():
    values = np.ones((230, 3))
    values[112:120] = np.nan
    case = case_for(values)
    medians, anchor = raw_window_medians(case)
    patches = companded_chart_patches(medians, anchor, np.ones(3))
    original = raw_chart_patches(case, np.ones(3))
    # On observed query days the complete column calendar is exactly unchanged.
    observed = np.isfinite(values).all(axis=1)
    assert np.array_equal(patches[observed, 3:], original[observed, 3:])
    assert np.count_nonzero(patches[~observed]) == 0


def test_companding_preserves_motion_near_target_with_a_large_remote_offset():
    values = np.zeros((300, 3))
    values[35:] = 120
    values[100:150] += np.arange(50)[:, None] * 0.1
    values[150:] += 5
    case = case_for(values)
    medians, anchor = raw_window_medians(case)
    original = chart_ordinates(raw_chart_patches(case, np.ones(3)))
    companded = chart_ordinates(companded_chart_patches(medians, anchor, np.ones(3)))
    # Query110 sees the distant step and ongoing small movement. No removal or
    # interpolation is needed to make the nearby shape occupy more pixels.
    old_change = abs(original[110, 0, 34] - original[110, 0, 27])
    new_change = abs(companded[110, 0, 34] - companded[110, 0, 27])
    assert new_change > old_change * 2
    # A plateau remains flat even after a nonlinear monotone display map.
    assert np.ptp(companded[200, 0, 20:40]) < 1e-6
