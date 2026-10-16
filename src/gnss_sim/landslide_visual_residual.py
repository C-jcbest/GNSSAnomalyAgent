"""Learn a limited visual correction while retaining a fixed numerical anchor."""

from __future__ import annotations

import math

import numpy as np

from gnss_sim.landslide_visual_invariance import activity_model


def anchor_logits(probability):
    """Convert numerical probabilities without infinite logits at zero or one."""
    probability = np.asarray(probability, dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("The fixed anchor must provide finite probabilities in [0, 1]")
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return (np.log(clipped) - np.log1p(-clipped)).astype(np.float32)


def grouped_folds(record_ids, records_per_fold):
    """Partition complete records; adjacent days never enter another fold."""
    if records_per_fold <= 0 or len(record_ids) % records_per_fold:
        raise ValueError("Fold size must exactly partition the registered training records")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("A training record cannot appear in two folds")
    folds = [list(record_ids[start : start + records_per_fold])
             for start in range(0, len(record_ids), records_per_fold)]
    if len(folds) < 2:
        raise ValueError("Cross-fitting requires at least two complete-record folds")
    return folds


def residual_model(hidden_width, numeric_width, use_image=True, bounded=True):
    """Initialize at the anchor and learn one correction from existing inputs.

    The head also receives the anchor logit, equally in visual and nonvisual
    arms. A fixed log(4) bound limits odds changes; it does not guarantee that
    the final decision is correct or calibrated. No numerical anchor weights
    enter the optimizer.
    """
    import torch
    from torch import nn

    class VisualResidualModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.correction = activity_model(hidden_width, numeric_width + 1, use_image, True)
            nn.init.zeros_(self.correction.classifier[-1].weight)
            nn.init.zeros_(self.correction.classifier[-1].bias)

        def forward(self, images, numbers, anchor):
            observations = torch.cat((numbers, anchor[:, None]), dim=1)
            correction = self.correction(images, observations)
            if bounded:
                correction = math.log(4) * torch.tanh(correction)
            return anchor + correction

    return VisualResidualModel()
