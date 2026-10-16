"""Observation summaries preserve the original calendar and measured population."""
import numpy as np
import pytest

from gnss_sim.landslide_observation_view import observation_bounds, raw_block_summaries


def test_wider_context_clips_at_record_edges_without_changing_target():
    assert observation_bounds(20, 200, 730, 91) == (0, 291)
    assert observation_bounds(600, 730, 730, 91) == (509, 730)
    with pytest.raises(ValueError):
        observation_bounds(700, 731, 730, 91)


def test_missing_observations_are_neither_interpolated_nor_counted_as_support():
    values = np.column_stack([np.arange(42)] * 3).astype(float)
    values[:7] = np.nan
    values[14:20] = np.nan
    original = values.copy()
    blocks = raw_block_summaries(values, 0, 42)
    assert [block["start"] for block in blocks] == [14, 28]
    assert blocks[0]["observed_days"] == 8
    assert blocks[0]["median_mm"] == [23.5] * 3
    np.testing.assert_equal(values, original)


def test_calendar_anchor_keeps_shared_blocks_identical_and_excludes_partial_edges():
    values = np.column_stack([np.arange(70)] * 3).astype(float)
    narrow = raw_block_summaries(values, 15, 57)
    wide = raw_block_summaries(values, 0, 70)
    assert [block["start"] for block in narrow] == [28, 42]
    assert narrow == [block for block in wide if block["start"] in (28, 42)]


def test_single_step_stays_two_plateaus_instead_of_a_smoothed_daily_ramp():
    values = np.zeros((42, 3))
    values[14:] = 100
    blocks = raw_block_summaries(values, 0, 42)
    assert [block["median_mm"] for block in blocks] == [[0] * 3, [100] * 3, [100] * 3]
