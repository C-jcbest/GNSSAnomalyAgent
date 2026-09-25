"""Derived daily views of the canonical event truth; neither view is stored as labels."""

from __future__ import annotations

import numpy as np

from gnss_sim.generator import DAYS
from gnss_sim.schemas import CaseTruth

AXES = ("N", "E", "U")


def event_masks(truth: CaseTruth) -> tuple[np.ndarray, np.ndarray]:
    """Return active and effect masks, each shaped (365, 3), with inclusive bounds."""
    active = np.zeros((DAYS, len(AXES)), dtype=bool)
    effect = np.zeros_like(active)
    for event in truth.events:
        column = AXES.index(event.axis)
        active[event.start_index:event.end_index + 1, column] = True
        last = DAYS if event.persistent else event.end_index + 1
        effect[event.start_index:last, column] = True
    return active, effect
