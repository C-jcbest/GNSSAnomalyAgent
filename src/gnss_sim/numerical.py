"""Input-only morphology detector; requires explicitly calibrated parameters."""
from __future__ import annotations

import numpy as np

from gnss_sim.schemas import CaseInput, PointResult, RangeResult

AXES = ("N", "E", "U")
PROTOCOL = "numerical-v1"
CONFIG = {"spike_half_window": 3, "jump_half_window": 7,
          "return_min_days": 2, "return_max_days": 60, "return_relative_error": 0.25,
          "trend_window": 31, "trend_max_gap": 3, "trend_min_days": 7,
          "peak_quantile": 0.95, "trend_low_quantile": 0.95}
BRANCHES = {"point": "point", "temporary": "range", "trend": "range", "range": "range"}


def series(values):
    x = np.asarray(values, dtype=float)
    if x.shape != (365,) or not np.isfinite(x).all():
        raise ValueError("expected 365 finite observations")
    return x


def runs(mask):
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return list(zip(starts.tolist(), ends.tolist()))


def spike_scores(values, config=CONFIG):
    x = series(values)
    h = config["spike_half_window"]
    score = np.zeros(len(x))
    for t in range(h, len(x) - h):
        left, right = x[t] - np.median(x[t-h:t]), x[t] - np.median(x[t+1:t+h+1])
        if left * right > 0:
            score[t] = min(abs(left), abs(right))
    return score


def without_spikes(values, threshold, config=CONFIG):
    x = series(values)
    points = np.flatnonzero(spike_scores(x, config) > threshold)
    good = np.ones(len(x), dtype=bool)
    good[points] = False
    cleaned = np.interp(np.arange(len(x)), np.flatnonzero(good), x[good])
    return points.tolist(), cleaned


def jump_scores(values, config=CONFIG):
    x = series(values)
    h = config["jump_half_window"]
    score = np.zeros(len(x))
    for t in range(h, len(x) - h + 1):
        score[t] = np.median(x[t:t+h]) - np.median(x[t-h:t])
    return score


def jumps(values, threshold, config=CONFIG):
    x = series(values)
    score = jump_scores(x, config)
    delta = np.r_[0., np.diff(x)]
    records = []
    h = config["jump_half_window"]
    for sign in (-1, 1):
        for start, end in runs(sign * score > threshold):
            t = int(start + np.argmax(sign * delta[start:end+1]))
            amplitude = float(np.median(x[t:t+h]) - np.median(x[t-h:t]))
            records.append({"day": t, "amplitude_mm": amplitude})
    return sorted(records, key=lambda r: r["day"])


def pair_returns(changes, config=CONFIG):
    points, temporary = [], []
    i = 0
    while i < len(changes):
        first = changes[i]
        if i + 1 < len(changes):
            second = changes[i+1]
            a, b = first["amplitude_mm"], second["amplitude_mm"]
            span = second["day"] - first["day"]
            if (a * b < 0 and config["return_min_days"] <= span <= config["return_max_days"]
                    and abs(a + b) <= config["return_relative_error"] * max(abs(a), abs(b))):
                temporary.append((first["day"], second["day"] - 1))
                i += 2
                continue
        points.append(first["day"])
        i += 1
    return points, temporary


def detrended_jumps(values, changes):
    cleaned = series(values).copy()
    for change in changes:
        cleaned[change["day"]:] -= change["amplitude_mm"]
    return cleaned


def trend_scores(values, config=CONFIG):
    x = series(values)
    window = config["trend_window"]
    left, right = np.triu_indices(window, 1)
    windows = np.lib.stride_tricks.sliding_window_view(x, window)
    slopes = np.median((windows[:, right] - windows[:, left]) / (right-left), axis=1)
    scores = np.zeros(len(x))
    center = window // 2
    scores[center:center+len(slopes)] = slopes
    return scores


def trend_ranges(scores, high, low, config=CONFIG):
    score = series(scores)
    intervals = []
    for sign in (-1, 1):
        spans = runs(sign * score > low)
        merged = []
        for start, end in spans:
            if merged and start - merged[-1][1] - 1 <= config["trend_max_gap"]:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        intervals.extend((s, e) for s, e in merged if e-s+1 >= config["trend_min_days"]
                         and np.any(sign * score[s:e+1] > high))
    return union_ranges(intervals)


def union_ranges(intervals):
    mask = np.zeros(365, dtype=bool)
    for start, end in intervals:
        mask[start:end+1] = True
    return runs(mask)


def calibrate(cases: list[CaseInput], config=CONFIG):
    if not cases:
        raise ValueError("empty calibration input")
    thresholds = {}
    q = config["peak_quantile"]
    for index, axis in enumerate(AXES):
        values = [series(np.asarray(c.displacement_mm)[:, index]) for c in cases]
        spike = float(np.quantile([spike_scores(x, config).max() for x in values], q))
        cleaned = [without_spikes(x, spike, config)[1] for x in values]
        jump = float(np.quantile([np.abs(jump_scores(x, config)).max() for x in cleaned], q))
        slopes = [trend_scores(detrended_jumps(x, jumps(x, jump, config)), config)
                  for x in cleaned]
        high = float(np.quantile([np.abs(s).max() for s in slopes], q))
        center = config["trend_window"] // 2
        low = float(np.quantile(np.abs(np.asarray(slopes)[:, center:-center]),
                                config["trend_low_quantile"]))
        thresholds[axis] = {"spike": spike, "jump": jump, "trend_high": high, "trend_low": low}
    return thresholds


def validate_thresholds(thresholds):
    if set(thresholds) != set(AXES):
        raise ValueError("expected thresholds for N/E/U")
    for t in thresholds.values():
        if (set(t) != {"spike", "jump", "trend_high", "trend_low"}
                or not all(np.isfinite(v) and v >= 0 for v in t.values())
                or t["trend_high"] < t["trend_low"]):
            raise ValueError("invalid calibrated thresholds")


def predict(case: CaseInput, thresholds, config=CONFIG):
    validate_thresholds(thresholds)
    outputs = {branch: {} for branch in BRANCHES}
    evidence = {}
    for index, axis in enumerate(AXES):
        t = thresholds[axis]
        isolated, cleaned = without_spikes(np.asarray(case.displacement_mm)[:, index],
                                           t["spike"], config)
        changes = jumps(cleaned, t["jump"], config)
        _, temporary = pair_returns(changes, config)
        trend = trend_ranges(trend_scores(detrended_jumps(cleaned, changes), config),
                             t["trend_high"], t["trend_low"], config)
        for branch, value in {"temporary": temporary, "trend": trend,
                              "point": isolated,
                              "range": union_ranges(temporary + trend)}.items():
            outputs[branch][axis] = value
        evidence[axis] = {"isolated_days": isolated, "jumps": changes,
                          "temporary_ranges": temporary, "trend_ranges": trend}
    results = {branch: (PointResult if task == "point" else RangeResult)(
        case_id=case.case_id, method=f"{PROTOCOL}/{branch}", status="success", predictions=value)
        for branch, value in outputs.items() for task in (BRANCHES[branch],)}
    return results, evidence


def failure(case_id):
    return {branch: (PointResult if task == "point" else RangeResult)(
        case_id=case_id, method=f"{PROTOCOL}/{branch}", status="failed",
        predictions={axis: [] for axis in AXES}) for branch, task in BRANCHES.items()}
