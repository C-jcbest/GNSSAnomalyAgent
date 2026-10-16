from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_jump_view import jump_view_inputs
from gnss_sim.landslide_visual_invariance import raw_chart_patches


def test_single_jump_becomes_flat_but_sustained_change_remains_visible():
    days = np.arange(260)
    step = (days >= 130)[:, None] * np.array([20.0, -12.0, 8.0])
    case = SimpleNamespace(displacement_mm=step.tolist())
    raw, adaptive, _, evidence = jump_view_inputs(case, np.ones(3))
    np.testing.assert_array_equal(raw, raw_chart_patches(case, np.ones(3)))
    assert evidence["dominant"][130]
    stationary = raw_chart_patches(SimpleNamespace(displacement_mm=np.zeros((260, 3)).tolist()), np.ones(3))
    np.testing.assert_allclose(adaptive[130], stationary[130])
    ramp = days[:, None] * np.array([0.2, -0.1, 0.05])
    _, repaired, _, continued = jump_view_inputs(
        SimpleNamespace(displacement_mm=(ramp + step).tolist()), np.ones(3),
    )
    assert continued["dominant"][130]
    # Removing one three-day increment must not flatten the other 59 increments.
    assert np.count_nonzero(repaired[130, 0].sum(axis=1)) > 20


def test_continuous_ramp_and_two_sided_excursion_do_not_request_adaptive_view():
    days = np.arange(260)
    ramp = days[:, None] * np.array([0.2, -0.1, 0.05])
    raw, adaptive, _, evidence = jump_view_inputs(SimpleNamespace(displacement_mm=ramp.tolist()), np.ones(3))
    assert not np.any(evidence["dominant"])
    np.testing.assert_array_equal(raw, adaptive)
    excursion = np.zeros((260, 3))
    excursion[120:140] = [20, -12, 8]
    raw, adaptive, _, evidence = jump_view_inputs(SimpleNamespace(displacement_mm=excursion.tolist()), np.ones(3))
    assert not evidence["dominant"][130]
    np.testing.assert_array_equal(raw[130], adaptive[130])


def test_gap_never_creates_a_jump_and_reference_translation_preserves_views():
    values = np.zeros((260, 3))
    values[130:] = [20, -12, 8]
    rows = values.tolist()
    rows[124:136] = [None] * 12
    raw, adaptive, _, evidence = jump_view_inputs(SimpleNamespace(displacement_mm=rows), np.ones(3))
    assert not evidence["dominant"][130]
    assert not np.any(raw[130, :, :, 30])
    np.testing.assert_array_equal(raw[130], adaptive[130])
    shifted = [None if row is None else (np.array(row) + 1000).tolist() for row in rows]
    shifted_raw, shifted_adaptive, _, shifted_evidence = jump_view_inputs(
        SimpleNamespace(displacement_mm=shifted), np.ones(3),
    )
    np.testing.assert_allclose(raw, shifted_raw, atol=1e-6)
    np.testing.assert_allclose(adaptive, shifted_adaptive, atol=1e-6)
    np.testing.assert_array_equal(evidence["dominant"], shifted_evidence["dominant"])
