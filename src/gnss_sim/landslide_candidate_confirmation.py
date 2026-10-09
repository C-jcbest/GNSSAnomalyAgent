"""Preserve window proposals without treating them as confirmed displacement."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_sim.landslide_window_experiment import aggregate_windows, review_candidates


@dataclass
class CandidatePool:
    spans: list[tuple[int, int]]
    positive_vote_days: np.ndarray
    negative_unknown_conflict_days: np.ndarray

    def to_dict(self):
        return {
            "spans_half_open": self.spans,
            "positive_vote_days": self.positive_vote_days.tolist(),
            "negative_unknown_conflict_days": self.negative_unknown_conflict_days.tolist(),
            "interpretation": "Proposed review coverage only; not confirmed activity or stages",
        }


def proposal_pool(window_results, observed):
    """Retain any positive view and valid negative/unknown disagreement for review."""
    aggregate_windows(window_results, observed)
    positive = np.zeros(len(observed), dtype=int)
    negative = np.zeros(len(observed), dtype=int)
    unknown = np.zeros(len(observed), dtype=int)
    for result in window_results:
        values = np.asarray(result["activity"], dtype=int)
        if result["status"] == "failed":
            if np.any(values != -1):
                raise ValueError("Failed windows cannot supply a motion proposal")
            continue
        if result["status"] != "success" or np.any(values[~observed] != -1):
            raise ValueError("Window status or original missing-data mask is invalid")
        start, stop = result["start"], result["stop"]
        positive[start:stop] += values[start:stop] == 1
        negative[start:stop] += values[start:stop] == 0
        unknown[start:stop] += values[start:stop] == -1
    positive_days = (positive > 0) & observed
    conflict_days = (negative > 0) & (unknown > 0) & observed
    proposed = positive_days | conflict_days
    # Only short original missing gaps join proposals; observed non-proposals separate them.
    flags = np.where(proposed, 1, 0)
    flags[~observed] = -1
    return CandidatePool(review_candidates(flags), positive_days, conflict_days)


def confirmation_base(strict_activity, pool, observed):
    """Leave proposals unconfirmed until raw-displacement review has returned."""
    covered = np.zeros(len(observed), dtype=bool)
    for start, stop in pool.spans:
        covered[start:stop] = True
    if np.any((strict_activity == 1) & ~covered):
        raise ValueError("A confirmed source day is missing from the proposal coverage")
    activity = strict_activity.copy()
    activity[covered | ~observed] = -1
    return activity


def calendar_information(observed, plot_start, plot_stop):
    """Give actual calendar availability, without displacement or reference labels."""
    missing = np.flatnonzero(~observed[plot_start:plot_stop]) + plot_start
    return (
        "\nObserved calendar metadata only (no signal values or activity labels): "
        f"in the enlarged plot days {plot_start}..{plot_stop - 1}, the original missing "
        f"day indices are {missing.tolist()}. Only these days lack measurements; "
        "all other target days have observations. Do not infer gap lengths from "
        "broken lines or remove nearby observed days solely because a gap is visible. "
        "Missing days remain unlabelled; availability alone is not motion evidence.\n"
    )
