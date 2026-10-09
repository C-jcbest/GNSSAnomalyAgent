"""Observation-only boundary proposals; fixed fits, never injection metadata."""
from __future__ import annotations

import json

import numpy as np

from gnss_sim import numerical
from gnss_sim.report import intervals
from gnss_sim.schemas import RangeResult
from gnss_sim.visual import _unique_object

CONFIG = {"search_margin": 31, "context_margin": 31, "minimum_span": 2,
          "score": "SSE / independently calibrated sigma squared + k * log(n)"}


def noise_scales(cases):
    """Only observations from the dedicated Normal calibration boundary enter here."""
    values = np.concatenate([np.diff(np.asarray(c.displacement_mm), axis=0) for c in cases])
    center = np.median(values, axis=0)
    sigma = 1.4826 * np.median(np.abs(values - center), axis=0) / np.sqrt(2)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Nonpositive calibrated noise scale")
    return dict(zip(numerical.AXES, sigma.tolist()))


def _fit(x, basis, sigma, dimensions):
    """Fit intercept plus one shape coefficient, for every basis column."""
    centered = basis - basis.mean(axis=0)
    y = x - x.mean()
    denom = np.sum(centered * centered, axis=0)
    cross = np.sum(centered * y[:, None], axis=0)
    coefficient = np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 0)
    sse = np.maximum(0, y @ y - coefficient * cross)
    return sse / sigma**2 + dimensions * np.log(len(x)), coefficient


def _shape_fit(x, days, left, right, shape, sigma):
    scores, coefficients = [], []
    for offset in range(0, len(left), 4096):
        s, e = left[offset:offset+4096], right[offset:offset+4096]
        if shape == "level":
            basis = ((days[:, None] >= s) & (days[:, None] <= e)).astype(float)
        else:
            basis = np.clip((days[:, None]-s)/np.maximum(1, e-s), 0, 1)
        cost, amplitude = _fit(x, basis, sigma, 4)
        scores.append(cost)
        coefficients.append(amplitude)
    return np.concatenate(scores), np.concatenate(coefficients)


def propose(case, item, parameters, sigma, search_domain="endpoints"):
    if search_domain not in ("endpoints", "context"):
        raise ValueError("Unknown boundary search domain")
    axis, s, e = item["axis"], item["start"], item["end"]
    raw = np.asarray(case.displacement_mm)[:, numerical.AXES.index(axis)]
    removed, clean = numerical.without_spikes(
        raw, parameters["thresholds"][axis]["spike"], parameters["config"])
    margin = CONFIG["search_margin"]
    starts = np.arange(max(0, s-margin), min(364, s+margin)+1)
    ends = np.arange(max(0, e-margin), min(364, e+margin)+1)
    lo = max(0, int(starts.min())-CONFIG["context_margin"])
    hi = min(364, int(ends.max())+CONFIG["context_margin"])
    if search_domain == "context":
        starts = ends = np.arange(lo, hi+1)
    left, right = np.meshgrid(starts, ends, indexing="ij")
    valid = right-left+1 >= CONFIG["minimum_span"]
    left, right = left[valid], right[valid]
    days = np.arange(lo, hi+1)
    x = clean[lo:hi+1]
    normal_score = float(np.sum((x-x.mean())**2)/sigma**2 + np.log(len(x)))
    # A permanent step is a non-target alternative, not a Point or active Range.
    step_dates = np.arange(lo+1, hi+1)
    step_score, _ = _fit(x, (days[:, None] >= step_dates).astype(float), sigma, 3)
    null_score = min(normal_score, float(step_score.min()))
    options = {}
    for shape in ("level", "trend"):
        if shape == "level":
            original = ((days >= s) & (days <= e)).astype(float)
        else:
            original = np.clip((days-s)/max(1, e-s), 0, 1)
        scores, coefficients = _shape_fit(x, days, left, right, shape, sigma)
        best = int(np.argmin(scores))
        near = scores <= scores[best]+2
        options[shape] = {
            "start": int(left[best]), "end": int(right[best]),
            "score": round(float(scores[best]), 4),
            "improvement_over_non_target": round(float(null_score-scores[best]), 4),
            "amplitude_mm": round(float(coefficients[best]), 4),
            "near_best_start_span": [int(left[near].min()), int(left[near].max())],
            "near_best_end_span": [int(right[near].min()), int(right[near].max())],
        }
        original_score, _ = _fit(x, original[:, None], sigma, 4)
        if "original" not in options or original_score[0] < options["original"]["score"]:
            options["original"] = {"start": s, "end": e,
                "score": float(original_score[0]), "fit_shape": shape}
    selected = min(options, key=lambda name: options[name]["score"])
    samples = sorted({*range(max(lo, s-5), min(hi+1, s+6)),
                      *range(max(lo, e-5), min(hi+1, e+6)),
                      *np.linspace(lo, hi, 15, dtype=int).tolist()})
    return {"axis": axis, "context": [lo, hi], "search_start": [int(starts[0]), int(starts[-1])],
            "search_end": [int(ends[0]), int(ends[-1])], "sigma_mm": sigma,
            "options": options, "non_target_score": round(null_score, 4),
            "rule_choice": selected if options[selected]["score"] < null_score else None,
            "removed_point_days_for_fit": [d for d in removed if lo <= d <= hi],
            "raw_samples": [[int(d), round(float(raw[d]), 4)] for d in samples],
            "note": "Lower score is better. Near-best spans are score diagnostics, not confidence "
                    "intervals. Trend endpoints describe active evolution; its later plateau is "
                    "outside that range. Fits are proposals, not proof of an anomaly."}


def decisions(content, packages):
    value = json.loads(content, object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or set(value) != set(packages):
        raise ValueError("Return every candidate ID exactly once")
    for cid, selected in value.items():
        if selected is not None and (
            not isinstance(selected, str) or selected not in packages[cid]["options"]
        ):
            raise ValueError(f"{cid}: choose null, original, level or trend; never return dates")
    return value


def result(cid, method, packages, selected):
    decisions(json.dumps(selected), packages)
    prediction = {}
    for axis in numerical.AXES:
        days = set()
        for key, choice in selected.items():
            if choice is not None and packages[key]["axis"] == axis:
                option = packages[key]["options"][choice]
                days.update(range(option["start"], option["end"]+1))
        prediction[axis] = intervals(days)
    return RangeResult(case_id=cid, method=method, status="success", predictions=prediction)
