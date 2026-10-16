"""Use accumulated raw motion to check a local denial of an existing event."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau

from gnss_sim.landslide_evaluation import event_masks, runs


def event_motion_support(case, noise, initial):
    """Return positive interval evidence, never a stationary or activity label.

    Rank agreement describes ordered accumulated movement within a visual
    event. Net displacement still needs the existing noise/rate criteria. A
    concentration check excludes a single dominant level change. These fixed
    descriptive rules are not confidence estimates or geological attribution.
    """
    raw = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    observed = np.isfinite(raw).all(axis=1)
    noise = np.asarray(noise)
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Event support needs positive three-axis observation noise")
    if initial.shape != observed.shape or not np.isin(initial, [-1, 0, 1]).all():
        raise ValueError("Initial activity must match the observed calendar")
    if np.any(initial[~observed] != -1):
        raise ValueError("Missing dates must remain unknown")
    support = np.zeros(len(raw), dtype=bool)
    details = []
    for event in event_masks(initial):
        days = np.flatnonzero(event)
        start, stop = int(days[0]), int(days[-1]) + 1
        row = {"start": start, "stop": stop, "supported": False, "reason": "insufficient_blocks"}
        medians, centers = [], []
        if any(right - left > 7 for left, right in runs(~observed[start:stop])):
            row["reason"] = "long_observation_gap"
            details.append(row)
            continue
        for left in range(start, stop - 13, 14):
            block = raw[left:left + 14]
            complete = block[np.isfinite(block).all(axis=1)]
            if len(complete) < 8:
                continue
            medians.append(np.median(complete, axis=0))
            centers.append(left + 6.5)
        row["block_count"] = len(medians)
        if len(medians) < 6:
            details.append(row)
            continue
        medians = np.array(medians)
        net = medians[-1] - medians[0]
        normalized_net = net / noise
        net_over_noise = float(np.linalg.norm(normalized_net))
        rate = float(np.linalg.norm(net) / (centers[-1] - centers[0]))
        row.update(net_over_noise=net_over_noise, rate_mm_per_day=rate)
        if net_over_noise < 2 or rate < .03:
            row["reason"] = "insufficient_net_motion"
            details.append(row)
            continue
        direction = normalized_net / net_over_noise
        projected = (medians / noise) @ direction
        ordering = float(kendalltau(centers, projected).statistic)
        changes = np.diff(projected)
        positive_changes = changes[changes > 0]
        concentration = float(positive_changes.max() / positive_changes.sum()) if len(positive_changes) else 1.0
        sustained = len(positive_changes) >= 3 and concentration <= .5
        confirmed = np.isfinite(ordering) and ordering >= .6 and sustained
        row.update(rank_agreement=None if not np.isfinite(ordering) else ordering,
                   positive_changes=int(len(positive_changes)), largest_positive_change_fraction=concentration,
                   block_centers=centers, projected_block_medians=projected.tolist(),
                   supported=bool(confirmed), reason="positive_motion" if confirmed else "unordered_or_concentrated")
        if confirmed:
            support |= event & observed & (initial == 1)
        details.append(row)
    return support, details


def protect_supported_denials(initial, reviewed, permission, observed, support):
    """Preserve a supported initial activity only against a definite local denial."""
    if any(values.shape != observed.shape for values in (initial, reviewed, permission, support)):
        raise ValueError("Protection inputs must use the same observed calendar")
    if permission.dtype != bool or support.dtype != bool or np.any(permission & ~observed):
        raise ValueError("Protection masks must contain observed dates only")
    if any(not np.isin(values, [-1, 0, 1]).all() or np.any(values[~observed] != -1)
           for values in (initial, reviewed)):
        raise ValueError("Missing and unknown prediction states must remain explicit")
    protected = permission & support & (initial == 1) & (reviewed == 0)
    activity = reviewed.copy()
    activity[~permission | protected] = initial[~permission | protected]
    return activity, protected
