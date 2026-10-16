"""Dates, missing blocks and reference invariance in visual feature discovery."""

import numpy as np

from gnss_sim.landslide_feature_discovery import discovery_block_matrix


def test_blocks_use_the_original_fourteen_day_ranges_and_translation_is_irrelevant():
    days = np.arange(220, dtype=float)
    values = days[:, None] * [0.1, 0.3, -0.2]
    normalized, available, counts, medians = discovery_block_matrix(values, [1, 2, 3])
    np.testing.assert_allclose(medians[100, 6], np.median(values[93:107], axis=0))
    np.testing.assert_allclose(medians[100, 0], np.median(values[9:23], axis=0))
    assert available[100].all() and (counts[100] == 14).all()
    translated = discovery_block_matrix(values + [100, -30, 14], [1, 2, 3])[0]
    np.testing.assert_allclose(normalized, translated, atol=1e-12, equal_nan=True)


def test_record_edges_and_long_gaps_never_gain_filled_blocks():
    values = np.ones((220, 3))
    values[93:103] = np.nan
    normalized, available, counts, medians = discovery_block_matrix(values, [1, 2, 3])
    assert counts[100, 6] == 4
    assert not available[100, 6]
    assert np.isnan(medians[100, 6]).all()
    assert np.isnan(normalized[100, 6]).all()
    assert not available[0, :6].any()
    assert np.isnan(normalized[0, :6]).all()
    assert available[0, 7]
