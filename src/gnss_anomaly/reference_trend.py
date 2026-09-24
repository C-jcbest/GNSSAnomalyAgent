"""Retrospective persistent departures from an initial Theil-Sen reference."""

import numpy as np

from .contracts import Prediction, mask_events
from .detectors import record_coverage, robust_scale, segments


def validate_reference_config(config):
    integer_keys = (
        "reference_days",
        "min_reference_samples",
        "min_reference_span_days",
        "confirmation_days",
        "min_confirmation_samples",
        "reset_gap_days",
    )
    if any(type(config.get(k)) is not int or config[k] < 1 for k in integer_keys):
        raise ValueError("reference_trend requires positive integer calendar settings")
    if not 3 <= config["min_reference_samples"] <= config["reference_days"]:
        raise ValueError("invalid reference sample minimum")
    if config["min_reference_span_days"] >= config["reference_days"]:
        raise ValueError("reference span exceeds reference period")
    if not 2 <= config["min_confirmation_samples"] <= config["confirmation_days"]:
        raise ValueError("invalid confirmation sample minimum")
    for key in ("threshold", "noise_floor_mm"):
        value = config.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError("reference threshold and noise floor must be finite and positive")


def fit_reference(indices, values):
    left, right = np.triu_indices(len(indices), 1)
    slope = float(np.median((values[right] - values[left]) / (indices[right] - indices[left])))
    intercept = float(np.median(values - slope * indices))
    return slope, intercept


def detect_reference_trend(window, config, diagnostics=None):
    if window.sampling_hours != 24:
        raise ValueError("reference_trend requires daily observations")
    validate_reference_config(config)
    values = window.array()
    n, _ = values.shape
    flags = np.zeros_like(values, dtype=bool)
    scorable = np.zeros_like(values, dtype=bool)
    details = []
    ref_end = config["reference_days"]
    width = config["confirmation_days"]
    minimum = config["min_confirmation_samples"]
    for c in range(3):
        x = values[:, c]
        indices = np.flatnonzero(np.isfinite(x[:ref_end]))
        if (
            len(indices) < config["min_reference_samples"]
            or indices[-1] - indices[0] < config["min_reference_span_days"]
        ):
            details.append({"status": "insufficient_reference"})
            continue
        slope, intercept = fit_reference(indices, x[indices])
        residual = x - (intercept + slope * np.arange(n))
        scale = max(robust_scale(residual[indices]), config["noise_floor_mm"])
        threshold = config["threshold"] * scale
        details.append(
            {
                "status": "fitted",
                "slope_mm_per_day": slope,
                "intercept_mm": intercept,
                "residual_scale_mm": scale,
                "threshold_mm": threshold,
                "reference_samples": len(indices),
            }
        )
        # Long gaps break confirmation windows, not the historical reference model.
        start = min(ref_end, n)
        parts = []
        for a, b in segments(~np.isfinite(x[start:])):
            if b - a > config["reset_gap_days"]:
                parts.append((start + a, start + b))
        intervals = []
        for a, b in parts:
            intervals.append((start, a))
            start = b
        intervals.append((start, n))
        for a, b in intervals:
            for end in range(a + width, b + 1):
                ids = np.arange(end - width, end)
                ids = ids[np.isfinite(x[ids])]
                if len(ids) < minimum:
                    continue
                scorable[ids, c] = True
                for sign in (-1, 1):
                    departed = ids[sign * residual[ids] > threshold]
                    if len(departed) >= minimum:
                        flags[departed, c] = True
    record_coverage(diagnostics, scorable, values)
    if diagnostics is not None:
        diagnostics.update(
            implementation="reference-trend-v1", reference_fits=details, retrospective=True
        )
    if not scorable.any():
        return Prediction(
            status="insufficient", reason="insufficient reference or confirmation observations"
        )
    return Prediction(
        events=mask_events(flags),
        reason="persistent departure from initial robust trend; retrospective",
    )
