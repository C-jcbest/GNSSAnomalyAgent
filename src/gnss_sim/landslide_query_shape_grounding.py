"""Ground visual activity learning in the shape around its queried date."""

from __future__ import annotations

import warnings

import numpy as np

from gnss_sim.landslide_visual_invariance import activity_model, chart_numeric_descriptors


def shape_grounding_targets(images, scope="query"):
    """Describe signed plotted changes, independently of activity labels.

    The teacher uses the actual chart ordinates, rather than physical magnitude
    erased by per-window raster scaling. Its three groups each require four
    available columns. An unavailable group never becomes a zero-change target.
    """
    if scope == "query":
        groups = ((23, 28), (28, 33), (33, 38))
    elif scope == "global":
        groups = ((0, 5), (28, 33), (56, 61))
    else:
        raise ValueError("Shape grounding scope must be query or global")
    descriptors = chart_numeric_descriptors(images)
    if descriptors.shape[1] != 366:
        raise ValueError("Shape grounding requires the original61 chart columns")
    # Image rows increase downward; negate to retain the displacement sign.
    positions = -descriptors[:, :183].reshape(len(images), 3, 61)
    medians = []
    for start, stop in groups:
        samples = positions[:, :, start:stop]
        complete = np.isfinite(samples).sum(axis=2) >= 4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(samples, axis=2)
        median[~complete] = np.nan
        medians.append(median)
    return np.concatenate((medians[1] - medians[0], medians[2] - medians[1]), axis=1)


def grounded_chart_model(hidden_width, numeric_width, use_image=True, use_numeric=True):
    """Reuse the chart classifier; numerical inputs cannot answer the teacher.

    Auxiliary supervision only reaches the image encoder. The activity head
    retains the original two-modality design and sees no generated answer.
    Every visual arm creates the same extra head, including the no-loss control.
    """
    import torch
    from torch import nn

    base = activity_model(hidden_width, numeric_width, use_image, use_numeric)

    class GroundedChartModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = base.image_encoder
            self.classifier = base.classifier
            if use_image:
                self.shape_head = nn.Sequential(
                    nn.Linear(160, hidden_width),
                    nn.ReLU(),
                    nn.Linear(hidden_width, 6),
                )

        def forward(self, images, numbers):
            modalities = []
            shape = None
            if use_image:
                visual = self.image_encoder(images)
                modalities.append(visual)
                shape = self.shape_head(visual)
            if use_numeric:
                modalities.append(numbers)
            logits = self.classifier(torch.cat(modalities, dim=1)).squeeze(1)
            return logits, shape

    return GroundedChartModel()
