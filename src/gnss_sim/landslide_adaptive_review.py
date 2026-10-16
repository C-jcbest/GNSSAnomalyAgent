"""Bounded visual review selected by disagreement with raw sustained displacement."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from gnss_sim.landslide_evaluation import ActivityPrediction
from gnss_sim.landslide_window_experiment import WINDOW_TASK


@dataclass(frozen=True)
class ReviewPolicy:
    maximum_reviews: int = 2
    minimum_conflict_days: int = 14
    minimum_rate: float = 0.03
    minimum_signal_to_noise: float = 2.0

    def __post_init__(self):
        if type(self.maximum_reviews) is not int or not 1 <= self.maximum_reviews <= 2:
            raise ValueError("This policy permits one or two reviews per record")
        if self.minimum_conflict_days < 1 or self.minimum_rate <= 0:
            raise ValueError("Conflict support and motion rate must be positive")
        if self.minimum_signal_to_noise <= 0:
            raise ValueError("Noise proxy threshold must be positive")


def raw_motion_evidence(case, features, separation_days=42):
    """Compare three disjoint raw blocks, without centered derivatives or labels.

    The long scale can expose accumulation masked by the full-record image.
    Both successive changes must be present and point in consistent directions.
    A step straddling a median block can still produce isolated evidence; it
    does not establish activity and requires subsequent raw visual review.
    Ratios are noise proxies, not calibrated significance tests.
    """
    values = np.asarray([
        row if row is not None else [np.nan] * 3 for row in case.displacement_mm
    ])
    rate = np.full(len(values), np.nan)
    signal = np.full(len(values), np.nan)
    direction = np.full(len(values), np.nan)
    for day in range(separation_days + 7, len(values) - separation_days - 7):
        medians = []
        for center in (day - separation_days, day, day + separation_days):
            sample = values[center - 7:center + 7]
            complete = sample[np.isfinite(sample).all(axis=1)]
            if len(complete) < 8:
                break
            medians.append(np.median(complete, axis=0))
        if len(medians) != 3:
            continue
        before = medians[1] - medians[0]
        after = medians[2] - medians[1]
        lengths = (np.linalg.norm(before), np.linalg.norm(after))
        direction[day] = np.dot(before, after) / max(lengths[0] * lengths[1], 1e-8)
        rate[day] = min(lengths) / separation_days
        signal[day] = min(np.linalg.norm(before / features.noise),
                          np.linalg.norm(after / features.noise))
        if direction[day] <= 0:
            rate[day] = 0
    return {"rate": rate, "signal_to_noise": signal, "direction": direction}


def review_priorities(initial, features, long_evidence, windows, policy):
    """Rank every fixed window before any review; no feedback loop or label access."""
    short_supported = np.isfinite(features.progressive_rate) & features.observed
    long_supported = np.isfinite(long_evidence["rate"]) & features.observed
    short_motion = ((features.progressive_rate >= policy.minimum_rate)
                    & (features.signal_to_noise >= policy.minimum_signal_to_noise))
    long_motion = ((long_evidence["rate"] >= policy.minimum_rate)
                   & (long_evidence["signal_to_noise"] >= policy.minimum_signal_to_noise))
    supported = short_supported | long_supported
    raw_motion = (short_supported & short_motion) | (long_supported & long_motion)
    missed_motion = supported & raw_motion & (initial != 1)
    unsupported_activity = supported & ~raw_motion & (initial == 1)
    priorities = []
    for index, (start, stop) in enumerate(windows):
        missed = int(missed_motion[start:stop].sum())
        unsupported = int(unsupported_activity[start:stop].sum())
        priorities.append({"index": index, "start": start, "stop": stop,
                           "missed_motion_days": missed,
                           "unsupported_activity_days": unsupported,
                           "conflict_days": missed + unsupported})
    ranked = sorted(priorities, key=lambda row: (-row["conflict_days"], row["start"]))
    selected = []
    for row in ranked:
        if row["conflict_days"] < policy.minimum_conflict_days:
            break
        # Disjoint targets avoid spending the bounded budget twice on the same days.
        if any(row["start"] < item["stop"] and item["start"] < row["stop"]
               for item in selected):
            continue
        selected.append(row)
        if len(selected) == policy.maximum_reviews:
            break
    return priorities, sorted(selected, key=lambda row: row["start"])


def combine_reviews(initial, reviews, observed):
    """Strict majority where reviewed; retain the initial decision elsewhere.

    Unknowns and failed requests retain their denominator. A failed target is
    unknown, never automatically restored to a confident initial decision.
    """
    coverage = np.zeros(len(initial), dtype=int)
    positive = np.zeros(len(initial), dtype=int)
    negative = np.zeros(len(initial), dtype=int)
    for review in reviews:
        start, stop = review["start"], review["stop"]
        values = np.asarray(review["activity"], dtype=int)
        if not 0 <= start < stop <= len(initial) or values.shape != initial.shape:
            raise ValueError("Review does not match the original calendar")
        if not np.isin(values, [-1, 0, 1]).all() or np.any(values[~observed] != -1):
            raise ValueError("Review has invalid states or missing-data decisions")
        if review["status"] not in {"success", "failed"}:
            raise ValueError("Invalid review status")
        if review["status"] == "failed" and np.any(values != -1):
            raise ValueError("Failed review must remain unknown")
        coverage[start:stop] += 1
        positive[start:stop] += values[start:stop] == 1
        negative[start:stop] += values[start:stop] == 0
    activity = initial.copy()
    activity[coverage > 0] = -1
    activity[2 * positive > coverage] = 1
    activity[2 * negative > coverage] = 0
    activity[~observed] = -1
    prediction = ActivityPrediction(activity, np.where(activity == 0, 0, -1))
    prediction.validate(len(activity))
    return prediction


def numeric_review_summary(features, long_evidence, start, stop):
    """Provide raw persistence, not stage answers, in fixed 30-day reporting blocks."""
    rows = []
    arrays = {"short_rate_mm_day": features.progressive_rate,
              "short_change_over_noise": features.signal_to_noise,
              "long_rate_mm_day": long_evidence["rate"],
              "long_change_over_noise": long_evidence["signal_to_noise"]}
    for left in range(start, stop, 30):
        right = min(left + 30, stop)
        row = {"days_inclusive": [left, right - 1],
               "observed_days": int(features.observed[left:right].sum())}
        for name, array in arrays.items():
            sample = array[left:right]
            sample = sample[np.isfinite(sample) & features.observed[left:right]]
            row[name] = {"supported_days": len(sample),
                         "median": round(float(np.median(sample)), 4) if len(sample) else None}
        rows.append(row)
    return rows


def review_prompt(features, long_evidence, start, stop, numeric_feedback):
    prompt = WINDOW_TASK.format(start=start, end=stop - 1)
    if not numeric_feedback:
        return prompt
    return prompt + (
        "\nObservation-derived raw persistence summary, from the SAME N/E/U values. "
        "This is not independent sensor evidence, a prediction, or ground truth. "
        "Three non-overlapping 14-day raw median blocks are compared with successive "
        "center spacings 21 (short) and 42 (long) days. Both successive changes "
        "must have consistent directions; their smaller norm divided by spacing "
        "is the reported rate. The noise ratio uses full-record daily-difference "
        "MAD, not an uncertainty estimate. No movement/stage threshold or answer "
        "is supplied. Low or inconsistent changes can contradict continuing activity; "
        "long-scale persistence can motivate inspecting subtle accumulation. "
        "Estimates use surrounding observations and cannot locate exact boundaries. "
        "Reporting blocks are NOT activity boundaries. Check the raw plot for onset "
        "and cessation, and leave uncertain when indistinguishable. Null is unavailable.\n"
        + json.dumps(numeric_review_summary(features, long_evidence, start, stop))
    )
