"""Partition development disagreements without changing annotations or predictions."""
from __future__ import annotations

import numpy as np

from gnss_sim.landslide_evaluation import STAGES, ActivityPrediction, runs


def stage_error_partition(rows, prediction: ActivityPrediction):
    """Disjoint causes on reference A/S/D days; unknown boundaries are not negatives."""
    prediction.validate(len(rows))
    target = np.array([STAGES.get(row["feature"], -1) for row in rows])
    scored = target > 0
    remaining = scored.copy()
    causes = {}
    if prediction.status == "failed":
        causes["activity_output_failure"] = remaining.copy()
    else:
        causes["activity_output_failure"] = np.zeros(len(rows), dtype=bool)
    remaining &= ~causes["activity_output_failure"]
    causes["activity_not_confirmed"] = remaining & (prediction.activity == 0) & prediction.gated
    remaining &= ~causes["activity_not_confirmed"]
    causes["activity_unknown"] = remaining & (prediction.activity == -1) & prediction.gated
    remaining &= ~causes["activity_unknown"]
    causes["stage_output_failure"] = remaining & (prediction.stage_status == "failed")
    remaining &= ~causes["stage_output_failure"]
    causes["stage_abstention"] = remaining & (prediction.stage <= 0)
    remaining &= ~causes["stage_abstention"]
    causes["stage_wrong_class"] = remaining & (prediction.stage != target)
    causes["correct"] = remaining & (prediction.stage == target)
    counts = {name: int(mask.sum()) for name, mask in causes.items()}
    if sum(counts.values()) != int(scored.sum()):
        raise ValueError("Stage error partition does not conserve the reference support")
    # Prediction columns: unknown, none, A, S, D. Rows: reference A, S, D.
    confusion = [[int(np.count_nonzero((target == code) & (prediction.stage == value)))
                  for value in (-1, 0, 1, 2, 3)] for code in (1, 2, 3)]
    negative = np.array([int(row["activity_label"]) == 0 for row in rows])
    return {"stage_reference_days": int(scored.sum()), "activity_gate_applied": prediction.gated, "counts": counts,
            "confusion_rows": ["A", "S", "D"], "confusion_columns": ["unknown", "none", "A", "S", "D"],
            "confusion": confusion,
            "false_stage_days_on_reference_negative": int(np.count_nonzero(negative & (prediction.stage > 0)))}


def activity_disagreements(rows, prediction):
    actual = np.array([int(row["activity_label"]) for row in rows])
    masks = {"false_activity": (actual == 0) & (prediction.activity == 1),
             "missed_activity": (actual == 1) & (prediction.activity != 1),
             "weak_missed": np.array([row["feature"] == "slow_displacement" for row in rows]) & (prediction.activity != 1)}
    return {name: {"days": int(mask.sum()), "intervals_half_open": runs(mask)} for name, mask in masks.items()}


def regular_windows(days, length=180, stride=90):
    if days < 1 or length < 2 or not 1 <= stride <= length:
        raise ValueError("Invalid calendar/window specification")
    if days <= length:
        return [(0, days)]
    starts = list(range(0, days - length + 1, stride))
    if starts[-1] != days - length:
        starts.append(days - length)
    return [(start, start + length) for start in starts]


def raw_interval_evidence(values, start, stop):
    """Descriptive endpoint medians and coarse blocks, not a new activity label."""
    if not 0 <= start < stop <= len(values):
        raise ValueError("Invalid half-open evidence interval")
    sample = values[start:stop]
    width = min(14, max(1, len(sample) // 3))
    early = sample[:width]
    late = sample[-width:]
    enough = min(np.isfinite(early).all(axis=1).sum(), np.isfinite(late).all(axis=1).sum()) >= max(1, width // 2)
    delta = None
    if enough:
        delta = (np.nanmedian(late, axis=0) - np.nanmedian(early, axis=0)).tolist()
    blocks = []
    for left in range(start, stop, 14):
        right = min(left + 14, stop)
        block = values[left:right]
        observed = np.isfinite(block).all(axis=1)
        median = np.median(block[observed], axis=0).tolist() if observed.any() else None
        blocks.append({"start": left, "stop": right, "observed": int(observed.sum()), "median_mm": median})
    return {"start": start, "stop": stop, "observed_days": int(np.isfinite(sample).all(axis=1).sum()),
            "endpoint_median_delta_mm": delta, "blocks": blocks,
            "interpretation": "Descriptive observations only; does not establish physical landslide motion"}
