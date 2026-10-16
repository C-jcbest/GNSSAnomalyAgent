"""Shared observation representation with nonexclusive deformation/quality heads."""

from __future__ import annotations

import numpy as np


def query_quality_targets(rows, half_width=7):
    """Label observed artifact events within each original 14-day query.

    The canonical reference remains the transition/spike's original day. This
    auxiliary window target does not label a shifted platform as continuing
    deformation or a repeated instrument event. Insufficient observed query
    support remains unknown, including events hidden in a raw gap.
    """
    quality = np.array([int(row["quality_event_label"]) for row in rows])
    targets = np.full(len(rows), -1, dtype=int)
    for day in range(len(rows)):
        start = max(0, day - half_width)
        stop = min(len(rows), day + half_width)
        query = quality[start:stop]
        if np.sum(query >= 0) >= 8:
            targets[day] = int(np.any(query == 1))
    return targets


def deformation_quality_head(input_width, hidden_width):
    """Two independent sigmoid outputs permit motion and nuisance to coexist.

    Single-head ablation uses the identical network and initialization, with
    the quality loss disabled. Both arms get identical original observations;
    the only intervention is auxiliary quality supervision of the shared trunk.
    """
    from torch import nn

    class DeformationQualityHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.observation_trunk = nn.Sequential(nn.Linear(input_width, hidden_width), nn.ReLU())
            self.deformation = nn.Linear(hidden_width, 1)
            self.quality = nn.Linear(hidden_width, 1)

        def forward(self, observations):
            representation = self.observation_trunk(observations)
            return self.deformation(representation).squeeze(1), self.quality(
                representation
            ).squeeze(1)

    return DeformationQualityHead()
