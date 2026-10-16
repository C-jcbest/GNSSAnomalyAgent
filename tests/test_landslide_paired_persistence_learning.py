import numpy as np
import pytest
import torch
from train_landslide_paired_persistence import modality_inputs

from gnss_sim.landslide_paired_persistence import pair_order_loss


def test_pair_loss_learns_motion_over_its_offset_and_depends_on_pair_identity():
    motion = torch.tensor([3., -2.], requires_grad=True)
    offset = torch.tensor([1., -4.], requires_grad=True)
    matched = pair_order_loss(motion, offset)
    shuffled = pair_order_loss(motion, offset.flip(0))
    assert matched < shuffled
    matched.backward()
    assert torch.all(motion.grad < 0)
    assert torch.all(offset.grad > 0)
    with pytest.raises(ValueError):
        pair_order_loss(motion, offset[:1])


def test_modality_ablation_preserves_original_order_and_only_removes_requested_information():
    embeddings = np.arange(12).reshape(3, 4)
    numbers = np.arange(6).reshape(3, 2)
    full = modality_inputs(embeddings, numbers, {"image": True, "numeric": True})
    numeric = modality_inputs(embeddings, numbers, {"image": False, "numeric": True})
    visual = modality_inputs(embeddings, numbers, {"image": True, "numeric": False})
    np.testing.assert_array_equal(full, np.column_stack((visual, numeric)))
    np.testing.assert_array_equal(numeric, numbers)
    np.testing.assert_array_equal(visual, embeddings)
    with pytest.raises(ValueError):
        modality_inputs(embeddings, numbers[:2], {"image": True, "numeric": True})
