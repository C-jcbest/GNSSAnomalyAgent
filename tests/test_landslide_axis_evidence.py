import numpy as np
import pytest

from gnss_sim.landslide_axis_evidence import axis_persistence_evidence, pool_axis_evidence


def test_single_offset_is_not_persistent_component_motion_away_from_transition():
    raw = np.zeros((240, 3))
    raw[120:] = [30, -10, 20]
    evidence = axis_persistence_evidence(raw, np.ones(3), (21,))
    assert np.array_equal(evidence[85], [0, 0, 0])
    assert np.array_equal(evidence[155], [0, 0, 0])


def test_noise_normalization_preserves_clear_horizontal_motion_with_vertical_reversal():
    days = np.arange(240)
    raw = np.column_stack((0.1 * days, np.zeros(240), np.abs(days - 120)))
    evidence = axis_persistence_evidence(raw, np.array([1, 1, 10]), (21,))
    assert evidence[120, 0] == pytest.approx(2.1)
    assert evidence[120, 1] == 0
    assert evidence[120, 2] < 0
    assert pool_axis_evidence(evidence)[120, 0] == pytest.approx(2.1)


def test_unobserved_query_and_insufficient_raw_blocks_remain_unavailable():
    raw = np.column_stack([np.arange(240)] * 3).astype(float)
    raw[110:132] = np.nan
    evidence = axis_persistence_evidence(raw, np.ones(3), (21,))
    assert np.isnan(evidence[120]).all()
    assert np.isnan(evidence[141]).all()
    assert np.isfinite(evidence[170]).all()
