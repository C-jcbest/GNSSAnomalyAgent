"""Contracts for the local, label-free visual grounding teacher."""

from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_query_shape_grounding import (
    grounded_chart_model,
    shape_grounding_targets,
)
from gnss_sim.landslide_visual_invariance import raw_chart_patches


def charts(values):
    case = SimpleNamespace(displacement_mm=values.tolist())
    return raw_chart_patches(case, np.ones(3))


def test_offset_outside_query_does_not_create_local_shape_motion():
    values = np.zeros((201, 3))
    values[40:, 0] = 12
    images = charts(values)[100:101]
    local = shape_grounding_targets(images)
    global_change = shape_grounding_targets(images, "global")
    np.testing.assert_allclose(local, 0, atol=1e-7)
    assert global_change[0, 0] > 0


def test_local_step_has_one_change_not_continuing_two_sided_motion():
    values = np.zeros((201, 3))
    values[90:, 0] = 12
    target = shape_grounding_targets(charts(values)[100:101])
    assert target[0, 0] > 0
    np.testing.assert_allclose(target[0, 3:], 0, atol=1e-7)


def test_missing_local_columns_remain_unavailable_not_stationary():
    images = charts(np.zeros((201, 3)))[100:101]
    images[:, :, :, 23:28] = 0
    target = shape_grounding_targets(images)
    assert np.isnan(target[0, :3]).all()
    np.testing.assert_allclose(target[0, 3:], 0)


def test_auxiliary_readout_cannot_use_numerical_answers():
    import torch

    torch.manual_seed(17)
    model = grounded_chart_model(16, 36)
    model.eval()
    images = torch.randn(2, 6, 32, 61)
    with torch.inference_mode():
        _, original = model(images, torch.zeros(2, 36))
        _, changed = model(images, torch.ones(2, 36) * 100)
    torch.testing.assert_close(original, changed)
