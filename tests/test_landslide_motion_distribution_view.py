import numpy as np

from gnss_sim.landslide_motion_distribution_view import (
    distribution_view_inputs,
    render_distribution_view,
    view_sequences,
)


def test_continuous_motion_agrees_while_spike_reverses_direction():
    days = np.arange(300, dtype=float)
    moving = np.column_stack((days, -days, days * 0.2))
    step = np.zeros((300, 3))
    step[145:159] = [30, 0, 0]
    movement = distribution_view_inputs(moving, np.ones(3))
    spike = distribution_view_inputs(step, np.ones(3))
    np.testing.assert_allclose(movement["agreement"][150], 1, atol=1e-12)
    assert np.nanmin(spike["agreement"][150]) < -0.99


def test_missing_blocks_remain_unavailable_and_are_not_connected():
    values = np.zeros((300, 3))
    values[138:165] = np.nan
    view = distribution_view_inputs(values, np.ones(3))
    assert np.isnan(view["magnitude"][150]).any()
    assert np.isnan(view["agreement"][150]).all()
    image = render_distribution_view(view["magnitude"][150], view["agreement"][150])
    assert image.size == (256, 256)
    assert image.mode == "RGB"


def test_order_ablation_preserves_values_and_unknown_counts():
    magnitude = np.arange(12, dtype=float)
    agreement = np.linspace(-1, 1, 11)
    magnitude[2] = np.nan
    agreement[5] = np.nan
    left, right = view_sequences(magnitude, agreement, True)
    np.testing.assert_array_equal(np.sort(left), np.sort(magnitude))
    np.testing.assert_array_equal(np.sort(right), np.sort(agreement))
    assert not np.array_equal(left, magnitude, equal_nan=True)
