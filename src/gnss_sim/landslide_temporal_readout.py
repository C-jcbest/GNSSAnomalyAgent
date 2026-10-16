"""Read visual motion at its original date rather than pooling the whole window."""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide_visual_invariance import numeric_input


def column_calendar(length):
    """Return the exact original day at each of the existing 61 chart columns."""
    if type(length) is not int or length <= 0:
        raise ValueError("The chart calendar must have a positive integer length")
    dates = np.arange(length)[:, None] + np.arange(-90, 91, 3)[None, :]
    available = (dates >= 0) & (dates < length)
    return np.clip(dates, 0, length - 1), available


def column_numeric_input(matrix, scaling):
    """Attach the actual column day's numerical evidence, without interpolation."""
    numbers = numeric_input(matrix, scaling)
    indices, available = column_calendar(len(numbers))
    aligned = numbers[indices]
    unavailable = numeric_input(np.full((1, matrix.shape[1]), np.nan), scaling)[0]
    aligned[~available] = unavailable
    return aligned


def column_supervision(activity, support):
    """Mask labels by the original calendar/support; keep unknown dates unknown.

    This function is for training only. A column uses its original center day's
    existing activity label; it does not infer activity from its three-day
    displacement median or propagate labels across unavailable observations.
    """
    activity = np.asarray(activity)
    support = np.asarray(support, dtype=bool)
    if activity.ndim != 1 or support.shape != activity.shape:
        raise ValueError("Original activity and support must share one calendar")
    if not np.isin(activity, [-1, 0, 1]).all():
        raise ValueError("Original activity must use unknown/stationary/motion labels")
    indices, calendar = column_calendar(len(activity))
    mask = calendar & support[indices] & (activity[indices] >= 0)
    target = np.where(mask, activity[indices], 0).astype(np.float32)
    return target, mask


def temporal_activity_model(hidden_width, numeric_width, readout="aligned", use_numeric=True):
    """Keep all 61 dates through the CNN, then predict each column's activity.

    The pooled control broadcasts one whole-window visual vector to the same
    dated numerical evidence and positional coordinate. Every model receives
    that coordinate; it identifies where a query lies inside this fixed chart
    and adds no new GNSS observation. Inference reads only original center30.
    """
    import torch
    from torch import nn

    if readout not in ("aligned", "pooled", "numeric"):
        raise ValueError("Readout must be aligned, pooled or numeric")
    if readout == "numeric" and not use_numeric:
        raise ValueError("Numerical-only readout requires numerical observations")

    class TemporalActivityModel(nn.Module):
        def __init__(self):
            super().__init__()
            if readout != "numeric":
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(6, 8, kernel_size=3, stride=(2, 1), padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 16, kernel_size=3, stride=(2, 1), padding=1),
                    nn.ReLU(),
                )
            if readout == "aligned":
                self.vertical_pool = nn.AdaptiveAvgPool2d((2, 61))
                self.temporal_encoder = nn.Sequential(
                    nn.Conv1d(32, 16, kernel_size=9, padding=4),
                    nn.ReLU(),
                    nn.Conv1d(16, 16, kernel_size=9, dilation=4, padding=16),
                    nn.ReLU(),
                )
                image_width = 16
            elif readout == "pooled":
                self.window_pool = nn.AdaptiveAvgPool2d((2, 5))
                image_width = 160
            else:
                image_width = 0
            input_width = image_width + (numeric_width if use_numeric else 0) + 1
            self.classifier = nn.Sequential(
                nn.Linear(input_width, hidden_width),
                nn.ReLU(),
                nn.Linear(hidden_width, 1),
            )
            self.register_buffer("relative_date", torch.linspace(-1, 1, 61).reshape(1, 61, 1))

        def forward(self, images, numbers):
            if images.shape[1:] != (6, 32, 61) or numbers.shape[1:] != (61, numeric_width):
                raise ValueError("Temporal readout requires original 32x61 charts and dated numerical columns")
            modalities = []
            if readout == "aligned":
                spatial = self.vertical_pool(self.image_encoder(images))
                by_date = spatial.reshape(len(images), 32, 61)
                modalities.append(self.temporal_encoder(by_date).transpose(1, 2))
            elif readout == "pooled":
                window = self.window_pool(self.image_encoder(images)).flatten(1)
                modalities.append(window[:, None, :].expand(-1, 61, -1))
            if use_numeric:
                modalities.append(numbers)
            modalities.append(self.relative_date.expand(len(images), -1, -1))
            return self.classifier(torch.cat(modalities, dim=2)).squeeze(2)

    return TemporalActivityModel()
