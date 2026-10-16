"""Retrospective activity duration measured in supported observations."""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide_evaluation import runs


def activity_across_missing(mask, supported, observed, minimum_observations=14, maximum_gap=7):
    """Join positive evidence across short raw gaps without labeling missing days.

    An observed negative, or an observed day lacking feature support, separates
    events. Missing calendar days do not count toward the minimum duration.
    This is a common decoder for numerical and visual comparisons, not a
    method-specific correction.
    """
    mask = np.asarray(mask, dtype=bool)
    supported = np.asarray(supported, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    if mask.ndim != 1 or mask.shape != supported.shape or mask.shape != observed.shape:
        raise ValueError("Duration evidence must share one original-day calendar")
    if np.any(supported & ~observed):
        raise ValueError("Missing original days cannot have feature support")
    if minimum_observations < 1 or maximum_gap < 0:
        raise ValueError("Duration must be positive and the gap cap nonnegative")
    positive = mask & supported
    linked = positive.copy()
    for start, stop in runs(~observed):
        bounded = start > 0 and stop < len(linked)
        if bounded and stop - start <= maximum_gap:
            if positive[start - 1] and positive[stop]:
                linked[start:stop] = True
    activity = np.where(supported, 0, -1)
    for start, stop in runs(linked):
        if positive[start:stop].sum() >= minimum_observations:
            indices = np.flatnonzero(positive[start:stop]) + start
            activity[indices] = 1
    return activity
