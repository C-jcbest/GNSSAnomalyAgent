"""Preserve a global decision only when raw observations refute a definite edit."""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide_evaluation import runs


def raw_counterevidence(case, noise):
    """Separate observed plateaus from missing or weak persistence support.

    Both 21-day and 42-day contrasts must support the same state. A plateau uses the
    larger successive change below one existing noise unit; sustained motion
    uses the smaller change above the original .03/2 rule. These are descriptive
    proxies, not equivalence tests or calibrated probabilities. The region
    between the two conditions is deliberately inconclusive.
    A median exactly straddling a step can create isolated motion support;
    this evidence may veto an edit but cannot establish an activity interval.
    """
    values = np.asarray([row if row is not None else [np.nan] * 3
                         for row in case.displacement_mm])
    observed = np.isfinite(values).all(axis=1)
    noise = np.asarray(noise)
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Counterevidence needs positive three-axis observation noise")
    by_scale = {}
    for separation in (21, 42):
        plateau = np.zeros(len(values), dtype=bool)
        motion = np.zeros(len(values), dtype=bool)
        supported = np.zeros(len(values), dtype=bool)
        maximum_change = np.full(len(values), np.nan)
        minimum_change = np.full(len(values), np.nan)
        for day in range(separation + 7, len(values) - separation - 7):
            if not observed[day]:
                continue
            left, right = day - separation - 7, day + separation + 7
            if any(stop - start > 7 for start, stop in runs(~observed[left:right])):
                continue
            medians = []
            for center in (day - separation, day, day + separation):
                block = values[center - 7:center + 7]
                complete = block[np.isfinite(block).all(axis=1)]
                if len(complete) < 8:
                    break
                medians.append(np.median(complete, axis=0))
            if len(medians) != 3:
                continue
            before, after = medians[1] - medians[0], medians[2] - medians[1]
            normalized = (np.linalg.norm(before / noise), np.linalg.norm(after / noise))
            supported[day] = True
            maximum_change[day], minimum_change[day] = max(normalized), min(normalized)
            plateau[day] = maximum_change[day] <= 1
            rate = min(np.linalg.norm(before), np.linalg.norm(after)) / separation
            motion[day] = rate >= .03 and minimum_change[day] >= 2 and np.dot(before, after) > 0
        by_scale[separation] = {"supported": supported, "plateau": plateau, "motion": motion,
                                "maximum_change_over_noise": maximum_change,
                                "minimum_change_over_noise": minimum_change}
    plateau = by_scale[21]["plateau"] & by_scale[42]["plateau"]
    motion = by_scale[21]["motion"] & by_scale[42]["motion"]
    supported = by_scale[21]["supported"] & by_scale[42]["supported"]
    return {"plateau": plateau, "motion": motion, "supported": supported, "scales": by_scale}


def restrict_contradicted_edits(initial, reviewed, permission, observed, evidence,
                               check_plateau=True, check_motion=True):
    """Veto contradicted definite changes; unknown decisions keep their meaning."""
    for values in (initial, reviewed):
        if values.shape != observed.shape or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError("Review decisions must match the observed calendar")
        if np.any(values[~observed] != -1):
            raise ValueError("Missing days must remain unknown")
    if permission.shape != observed.shape or permission.dtype != bool or np.any(permission & ~observed):
        raise ValueError("Edit permission must contain observed dates only")
    plateau_conflict = permission & (initial == 0) & (reviewed == 1) & evidence["plateau"]
    motion_conflict = permission & (initial == 1) & (reviewed == 0) & evidence["motion"]
    rejected = np.zeros(len(initial), dtype=bool)
    if check_plateau:
        rejected |= plateau_conflict
    if check_motion:
        rejected |= motion_conflict
    activity = reviewed.copy()
    activity[~permission | rejected] = initial[~permission | rejected]
    return activity, rejected
