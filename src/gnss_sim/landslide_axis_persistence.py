"""Preserve each displacement axis when proposing weak-motion observations."""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide_evaluation import runs


def axis_persistence(case, noise):
    """Require concordant chronological progress on both halves of seven raw blocks.

    Fourteen-day blocks are disjoint and use the existing eight-observation
    support rule. Pairwise rank signs measure chronological consistency without
    treating a positive 3-D norm as movement. The daily-difference noise scale is
    only an amplitude proxy, not a standard error or significance calibration.
    """
    values = np.asarray(
        [row if row is not None else [np.nan] * 3 for row in case.displacement_mm],
        dtype=float,
    )
    noise = np.asarray(noise, dtype=float)
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Axis persistence requires three positive noise scales")
    medians = np.full(values.shape, np.nan)
    for center in range(7, len(values) - 7):
        sample = values[center - 7 : center + 7]
        complete = sample[np.isfinite(sample).all(axis=1)]
        if len(complete) >= 8:
            medians[center] = np.median(complete, axis=0)
    centers = np.arange(49, max(49, len(values) - 49))
    blocks = np.stack([medians[centers + offset] for offset in range(-42, 43, 14)], axis=1)
    valid = np.isfinite(blocks).all(axis=(1, 2))
    before_rank = np.zeros((len(centers), 3))
    after_rank = np.zeros((len(centers), 3))
    for left in range(4):
        for right in range(left + 1, 4):
            before_rank += np.sign(blocks[:, right] - blocks[:, left]) / 6
            after_rank += np.sign(blocks[:, right + 3] - blocks[:, left + 3]) / 6
    before_change = (blocks[:, 3] - blocks[:, 0]) / noise
    after_change = (blocks[:, 6] - blocks[:, 3]) / noise
    concordant = (before_rank * after_rank > 0) & (before_change * after_change > 0)
    strength = np.minimum(np.abs(before_change), np.abs(after_change))
    consistency = np.minimum(np.abs(before_rank), np.abs(after_rank))
    axis_score = np.where(concordant, strength * consistency, 0.0)
    axis_score[~valid] = np.nan
    before_rank[~valid] = np.nan
    after_rank[~valid] = np.nan
    axis_days = np.full(values.shape, np.nan)
    axis_days[centers] = axis_score
    rank_days = np.full((len(values), 6), np.nan)
    rank_days[centers] = np.column_stack((before_rank, after_rank))
    score = np.full(len(values), np.nan)
    score[centers[valid]] = axis_score[valid].max(axis=1)
    return {"score": score, "axis_score": axis_days, "rank_halves": rank_days}


def persistence_review_plan(initial, observed, evidence, windows, threshold):
    """Keep the existing two-view budget; replace only its raw evidence owner."""
    if initial.shape != observed.shape or evidence["score"].shape != observed.shape:
        raise ValueError("Persistence evidence and predictions must share the calendar")
    if not np.isin(initial, [-1, 0, 1]).all() or np.any(initial[~observed] != -1):
        raise ValueError("Invalid initial activity or missing-observation decision")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Persistence threshold must be positive")
    supported = observed & np.isfinite(evidence["score"])
    motion = supported & (evidence["score"] >= threshold)
    missed = motion & (initial != 1)
    unsupported = supported & ~motion & (initial == 1)
    conflict = missed | unsupported
    priorities = []
    for index, (start, stop) in enumerate(windows):
        priorities.append(
            {
                "index": index,
                "start": start,
                "stop": stop,
                "missed_motion_days": int(missed[start:stop].sum()),
                "unsupported_activity_days": int(unsupported[start:stop].sum()),
                "conflict_days": int(conflict[start:stop].sum()),
            }
        )
    selected = []
    for target in sorted(priorities, key=lambda item: (-item["conflict_days"], item["start"])):
        if target["conflict_days"] < 14:
            break
        if any(
            target["start"] < previous["stop"] and previous["start"] < target["stop"]
            for previous in selected
        ):
            continue
        selected.append(target)
        if len(selected) == 2:
            break
    neighbourhood = np.zeros(len(initial), dtype=bool)
    for start, stop in runs(conflict):
        neighbourhood[max(0, start - 49) : min(len(initial), stop + 49)] = True
    return priorities, sorted(selected, key=lambda item: item["start"]), neighbourhood & observed
