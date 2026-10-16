from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_displacement_comparison_view import raw_block_comparisons


def test_ordered_comparisons_distinguish_plateau_and_step_from_continuing_displacement():
    values = np.zeros((84, 3))
    values[42:, 0] = 20
    facts = raw_block_comparisons(
        SimpleNamespace(displacement_mm=values.tolist()), np.ones(3), 0, 84
    )
    matrix = facts["differences"][:, :, 0]
    assert np.all(matrix[:3, :3] == 0)
    assert np.all(matrix[3:, 3:] == 0)
    assert np.all(matrix[:3, 3:] == 20)
    ramp = np.column_stack((np.arange(84), np.zeros(84), np.zeros(84)))
    continuing = raw_block_comparisons(
        SimpleNamespace(displacement_mm=ramp.tolist()), np.ones(3), 0, 84
    )
    assert np.all(np.diag(continuing["differences"][:, :, 0], 1) > 0)


def test_missing_blocks_are_unavailable_in_both_curve_and_comparisons():
    rows = np.column_stack((np.arange(84), np.zeros(84), np.zeros(84))).tolist()
    rows[28:42] = [None] * 14
    facts = raw_block_comparisons(SimpleNamespace(displacement_mm=rows), np.ones(3), 0, 84)
    assert facts["complete_days"][2] == 0
    assert np.isnan(facts["traces"][2]).all()
    assert np.isnan(facts["differences"][2]).all()
    assert np.isnan(facts["differences"][:, 2]).all()
    assert np.isfinite(facts["differences"][0, 5]).all()


def test_shared_facts_preserve_translation_and_axis_sign_without_changing_block_dates():
    raw = np.column_stack((np.arange(84), -np.arange(84), np.arange(84) * 2))
    first = raw_block_comparisons(
        SimpleNamespace(displacement_mm=raw.tolist()), np.array([1, 2, 3]), 7, 77
    )
    offset = raw_block_comparisons(
        SimpleNamespace(displacement_mm=(raw + 1000).tolist()), np.array([1, 2, 3]), 7, 77
    )
    reflected = raw_block_comparisons(
        SimpleNamespace(displacement_mm=(-raw).tolist()), np.array([1, 2, 3]), 7, 77
    )
    np.testing.assert_array_equal(first["starts"], [7, 21, 35, 49, 63])
    np.testing.assert_allclose(first["differences"], offset["differences"])
    np.testing.assert_allclose(first["traces"], offset["traces"])
    np.testing.assert_allclose(first["differences"], -reflected["differences"])
