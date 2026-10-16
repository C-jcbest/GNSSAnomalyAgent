"""Check the physical difference between spread motion and concentrated change."""

import numpy as np

from gnss_sim.landslide_motion_mass import motion_mass_descriptors


def test_continuous_motion_has_more_effective_edges_than_equal_displacement_step():
    days = np.arange(300, dtype=float)
    motion = np.column_stack((days, days * 0, days * 0))
    step = np.zeros_like(motion)
    step[149:] = [300, 0, 0]
    moving = motion_mass_descriptors(motion, np.ones(3))["global_distribution"][150]
    jumping = motion_mass_descriptors(step, np.ones(3))["global_distribution"][150]
    assert moving[4] > 3.9
    assert jumping[4] <= 2
    assert moving[3] < jumping[3]


def test_query_share_separates_same_shape_changes_at_different_dates():
    near = np.zeros((300, 3))
    distant = near.copy()
    near[149:] = [30, -10, 5]
    distant[79:] = [30, -10, 5]
    left = motion_mass_descriptors(near, np.ones(3))["query_distribution"][150]
    right = motion_mass_descriptors(distant, np.ones(3))["query_distribution"][150]
    assert left[-2] > right[-2]
    assert left[-1] < right[-1]


def test_offset_translation_preserves_descriptors_and_missing_dates_remain_unknown():
    days = np.arange(300, dtype=float)
    values = np.column_stack((days, days * 0.2, -days))
    values[149:154] = np.nan
    first = motion_mass_descriptors(values, np.ones(3))
    second = motion_mass_descriptors(values + [100, -50, 80], np.ones(3))
    np.testing.assert_allclose(
        first["query_distribution"], second["query_distribution"], atol=1e-10
    )
    assert np.isnan(first["query_distribution"][149:154]).all()


def test_long_missing_blocks_do_not_form_a_new_compacted_edge():
    values = np.zeros((300, 3))
    values[110:138] = np.nan
    result = motion_mass_descriptors(values, np.ones(3))
    # At query150 the unavailable original slots retain their positions.
    missing_blocks = ~np.isfinite(result["blocks"][150]).all(axis=1)
    assert missing_blocks.any()
    for index in np.flatnonzero(missing_blocks):
        if index > 0:
            assert not result["edge_available"][150, index - 1]
        if index < 12:
            assert not result["edge_available"][150, index]
