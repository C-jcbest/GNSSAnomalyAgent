"""Make a raw direction anchor available from an already observed strong prefix."""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide_evaluation import runs


def causal_direction_anchors(before, after, observed, noise, rate=.12, snr=3.0):
    """Keep the most recent14 supported centers without waiting for episode end.

    Each anchor becomes available only after its complete42+7 raw footprint.
    Later observations may update the direction, but never revise an earlier
    anchor. This avoids retrospectively denying usable evidence merely because
    the same strong-motion run eventually continues past a proposed weak tail.
    """
    lengths = np.column_stack((np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)))
    consistency = np.sum(before * after, axis=1) / np.maximum(lengths.prod(axis=1), 1e-8)
    signal = np.minimum(np.linalg.norm(before / noise, axis=1), np.linalg.norm(after / noise, axis=1))
    strong = observed & (lengths.min(axis=1) / 42 >= rate) & (signal >= snr) & (consistency > 0)
    anchors = []
    for left, right in runs(strong):
        for stop in range(left + 14, right + 1):
            start = stop - 14
            changes = np.vstack((before[start:stop], after[start:stop]))
            unit_changes = changes / np.linalg.norm(changes, axis=1)[:, None]
            direction = np.median(unit_changes, axis=0)
            norm = np.linalg.norm(direction)
            if norm <= 1e-8:
                continue
            direction /= norm
            if np.median(unit_changes @ direction) < .5:
                continue
            anchors.append({"start": start, "stop": stop, "available_from": stop + 49,
                            "direction": direction.tolist()})
    return anchors
