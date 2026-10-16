"""Gate a bounded local review by compression caused by displacement history."""

from __future__ import annotations

import numpy as np


def historical_scale_gains(case, noise, windows):
    """Measure raw dynamic range relative to the existing local plot span.

    Global render_raw forces a minimum 60 mm span. That display floor also
    enlarges quiet residuals when cropped, so it cannot by itself justify a
    history-compression review. Use the observed global range instead. The
    denominator retains render_window's max(range, 6 * noise, 10 mm), making
    the gain invariant to an arbitrary displacement reference offset.
    """
    values = np.asarray([row if row is not None else [np.nan] * 3
                         for row in case.displacement_mm], dtype=float)
    noise = np.asarray(noise, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or noise.shape != (3,):
        raise ValueError("Scale review requires aligned NEU observations and noise")
    if not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Scale review requires a positive noise proxy")
    complete = np.isfinite(values).all(axis=1)
    global_range = np.zeros(3)
    if np.any(complete):
        global_range = np.ptp(values[complete], axis=0)
    gains = []
    for start, stop in windows:
        if not 0 <= start < stop <= len(values):
            raise ValueError("Scale review target lies outside the observation calendar")
        sample = values[start:stop][complete[start:stop]]
        local_range = np.ptp(sample, axis=0) if len(sample) else np.zeros(3)
        local_span = np.maximum.reduce([local_range, 6 * noise, np.full(3, 10.0)])
        axis_gains = global_range / local_span
        gains.append({"axis_gains": axis_gains.tolist(), "gain": float(np.max(axis_gains)),
                      "observed_days": len(sample)})
    return gains


def compressed_review_targets(targets, gains, minimum_missed_days=14, minimum_gain=4.0):
    """Keep only original bounded targets with a missed, compressed motion signal.

    This is an admission gate, not a second ranking or a new motion classifier.
    A fourfold gain represents two octaves of raw displacement contrast; it is
    a fixed engineering choice, not an optimal or calibrated perceptual limit.
    Numerical evidence never assigns an activity label.
    """
    if minimum_missed_days < 1 or minimum_gain <= 1:
        raise ValueError("Scale gate requires positive missed support and a magnification gain")
    return [target for target in targets
            if target["missed_motion_days"] >= minimum_missed_days
            and gains[target["index"]]["gain"] >= minimum_gain]
