import pytest
import torch

from gnss_sim.landslide_spatial_readout import spatial_chart_readouts


def test_uniform_context_has_zero_contrast_and_preserves_global_projection():
    projection = torch.nn.Linear(1024, 512)
    maps = torch.ones(2, 1024, 8, 8)
    global_part, center_part, contrast = spatial_chart_readouts(maps, projection)
    expected = torch.nn.functional.normalize(projection(torch.ones(2, 1024)), dim=-1)
    assert torch.allclose(global_part, expected)
    assert torch.allclose(center_part, expected)
    assert torch.count_nonzero(contrast) == 0


def test_contrast_responds_to_center_change_and_rejects_wrong_encoder_geometry():
    projection = torch.nn.Linear(1024, 512, bias=False)
    with torch.no_grad():
        projection.weight.zero_()
        projection.weight[0, 0] = 1
        projection.weight[1, 1] = 1
    maps = torch.zeros(1, 1024, 8, 8)
    maps[:, 0] = 1
    maps[:, 1, :7, 3:5] = 1
    _, _, contrast = spatial_chart_readouts(maps, projection)
    assert contrast[0, 0] < 0
    assert contrast[0, 1] > 0
    with pytest.raises(ValueError, match="geometry"):
        spatial_chart_readouts(torch.zeros(1, 1024, 7, 8), projection)
