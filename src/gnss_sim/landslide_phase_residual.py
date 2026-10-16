"""Activity-contained motion coordinates for a bounded visual phase probe."""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide_evaluation import event_masks, runs


def phase_evidence_window(activity, observed, start, stop):
    """Select the longest registered view contained in one predicted activity.

    Only original missing/unknown gaps of at most seven days can bridge an
    episode. A known negative never bridges. The queried dates and all observed
    evidence dates must already be active; phase evidence cannot add activity.
    """
    activity = np.asarray(activity)
    observed = np.asarray(observed, dtype=bool)
    if activity.ndim != 1 or observed.shape != activity.shape:
        raise ValueError("Activity and observations must share the original calendar")
    if not 0 <= start < stop <= len(activity) or stop - start != 14:
        raise ValueError("The phase query must contain fourteen original dates")
    if np.any(observed[start:stop] & (activity[start:stop] != 1)):
        return None
    for episode in event_masks(activity):
        left, right = np.flatnonzero(episode)[[0, -1]]
        right += 1
        if not left <= start < stop <= right:
            continue
        for length in (98, 70, 42):
            if right - left < length:
                continue
            window_start = int(np.clip((start + stop - length) // 2, left, right - length))
            window_stop = window_start + length
            if not window_start <= start < stop <= window_stop:
                continue
            if np.any(observed[window_start:window_stop] & (activity[window_start:window_stop] != 1)):
                continue
            gaps = runs(~observed[window_start:window_stop])
            if any(b - a > 7 for a, b in gaps):
                continue
            if any(observed[a : a + 14].sum() < 8 for a in range(window_start, window_stop, 14)):
                continue
            return window_start, window_stop
    return None


def phase_residual_evidence(values, noise, activity, start, stop):
    """Project observed displacement along its local noise-scaled net motion.

    The direction is a re-expression of observed endpoint block medians, not a
    latent slope direction or a new observation. Subtracting a fitted straight
    line exposes curvature while leaving original dates and missing gaps intact.
    Noise scales are descriptive daily-difference proxies, not confidence bounds.
    """
    values = np.asarray(values, dtype=float)
    noise = np.asarray(noise, dtype=float)
    activity = np.asarray(activity)
    if values.ndim != 2 or values.shape[1] != 3 or activity.shape != (len(values),):
        raise ValueError("Raw NEU and activity must share the original calendar")
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Three positive finite noise proxies are required")
    if not 0 <= start < stop <= len(values) or (stop - start) not in (42, 70, 98):
        raise ValueError("Evidence must use a registered original-calendar length")
    observed = np.isfinite(values).all(axis=1)
    if np.any(observed[start:stop] & (activity[start:stop] != 1)):
        raise ValueError("Phase evidence cannot cross an observed inactive day")
    if any(b - a > 7 for a, b in runs(~observed[start:stop])):
        raise ValueError("Phase evidence cannot cross a long missing gap")
    medians, times, counts = [], [], []
    for left in range(start, stop, 14):
        block = values[left : left + 14]
        complete = block[np.isfinite(block).all(axis=1)]
        if len(complete) < 8:
            raise ValueError("Each evidence block needs eight complete NEU observations")
        medians.append(np.median(complete, axis=0))
        times.append(left + 6.5)
        counts.append(len(complete))
    medians = np.asarray(medians)
    times = np.asarray(times)
    scaled = (medians - medians[0]) / noise
    net = scaled[-1] - scaled[0]
    magnitude = float(np.linalg.norm(net))
    if magnitude <= 1e-12:
        return None
    direction = net / magnitude
    progression = scaled @ direction
    centered_time = times - times.mean()
    design = np.column_stack((np.ones(len(times)), centered_time))
    coefficients = np.linalg.lstsq(design, progression, rcond=None)[0]
    linear = design @ coefficients
    calendar = np.arange(start, stop)
    raw_progression = ((values[start:stop] - medians[0]) / noise) @ direction
    raw_line = coefficients[0] + coefficients[1] * (calendar - times.mean())
    rates = np.diff(progression) / 14
    return {
        "start": start, "stop": stop, "block_centers": times.tolist(),
        "complete_days": counts, "median_neu_mm": medians.tolist(),
        "noise_proxy_mm": noise.tolist(), "scaled_direction": direction.tolist(),
        "progression": progression.tolist(), "linear_reference": linear.tolist(),
        "residual": (progression - linear).tolist(),
        "adjacent_block_rates_proxy_per_day": rates.tolist(),
        "linear_rate_proxy_per_day": float(coefficients[1]),
        "calendar": calendar.tolist(),
        "raw_progression": [float(x) if np.isfinite(x) else None for x in raw_progression],
        "raw_residual": [float(x) if np.isfinite(x) else None for x in raw_progression - raw_line],
        "is_independent_evidence": False,
    }
