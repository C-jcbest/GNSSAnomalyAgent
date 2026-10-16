"""Business contracts of anchored visual correction and record cross-fitting."""

import math

import numpy as np
import pytest

from gnss_sim.landslide_visual_residual import anchor_logits, grouped_folds, residual_model


def test_cross_fitting_keeps_records_together():
    records = [f"record_{index}" for index in range(24)]
    folds = grouped_folds(records, 6)
    assert len(folds) == 4
    assert [record for fold in folds for record in fold] == records
    for held_out in folds:
        training = [record for fold in folds if fold != held_out for record in fold]
        assert set(training).isdisjoint(held_out)
    with pytest.raises(ValueError, match="two folds"):
        grouped_folds(["same", "same"], 1)


def test_zero_head_preserves_anchor_and_bounded_odds():
    torch = pytest.importorskip("torch")
    model = residual_model(16, 36)
    images = torch.zeros(5, 6, 32, 61)
    numbers = torch.zeros(5, 36)
    anchor = torch.from_numpy(anchor_logits([0, .1, .5, .9, 1]))
    assert torch.equal(model(images, numbers, anchor), anchor)
    with torch.no_grad():
        model.correction.classifier[-1].bias.fill_(100)
    change = model(images, numbers, anchor) - anchor
    assert torch.all(change.abs() <= math.log(4) + 1e-6)
    assert torch.all(torch.sigmoid(model(images, numbers, anchor)) >= torch.sigmoid(anchor))


def test_nonvisual_control_cannot_read_image_and_invalid_anchor_rejected():
    torch = pytest.importorskip("torch")
    model = residual_model(16, 36, use_image=False)
    with torch.no_grad():
        model.correction.classifier[-1].weight.fill_(.3)
    numbers = torch.ones(3, 36)
    anchor = torch.from_numpy(anchor_logits([.2, .4, .8]))
    assert torch.equal(model(torch.zeros(3, 6, 32, 61), numbers, anchor),
                       model(torch.ones(3, 6, 32, 61), numbers, anchor))
    assert np.isfinite(anchor_logits([0, 1])).all()
    with pytest.raises(ValueError, match="finite probabilities"):
        anchor_logits([np.nan])
