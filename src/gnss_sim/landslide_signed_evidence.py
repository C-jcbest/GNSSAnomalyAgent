"""Axis-preserving raw facts for a bounded GNSS visual review."""

from __future__ import annotations

import json

import numpy as np

from gnss_sim.landslide_window_experiment import WINDOW_TASK


def signed_block_facts(case, noise, start, stop):
    """Describe disjoint raw blocks without deciding activity or interpolating gaps.

    The existing display uses 14-day blocks and requires eight complete NEU
    observations. Keep that support rule so this experiment changes evidence
    representation rather than adding another smoothing scale.
    """
    values = np.asarray(
        [row if row is not None else [np.nan] * 3 for row in case.displacement_mm], dtype=float
    )
    noise = np.asarray(noise, dtype=float)
    if not 0 <= start < stop <= len(values):
        raise ValueError("Raw fact window lies outside the observation calendar")
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Raw facts require three positive axis noise proxies")
    rows = []
    previous = None
    for left in range(start, stop, 14):
        right = min(left + 14, stop)
        sample = values[left:right]
        complete = sample[np.isfinite(sample).all(axis=1)]
        median = np.median(complete, axis=0) if len(complete) >= 8 else None
        change = median - previous if median is not None and previous is not None else None
        rows.append(
            {
                "days_inclusive": [left, right - 1],
                "complete_observed_days": len(complete),
                "median_NEU_mm": None if median is None else np.round(median, 4).tolist(),
                "change_from_previous_block_NEU_mm": None
                if change is None
                else np.round(change, 4).tolist(),
            }
        )
        # An unsupported block breaks the chain; never bridge it with a delta.
        previous = median
    return {
        "axis_order": ["N", "E", "U"],
        "daily_difference_noise_proxy_NEU_mm": np.round(noise, 4).tolist(),
        "blocks": rows,
    }


def signed_review_prompt(case, noise, start, stop):
    return WINDOW_TASK.format(start=start, end=stop - 1) + (
        "\nRaw displacement facts from the SAME plotted N/E/U observations, not "
        "independent evidence, predictions, or ground truth. Consecutive disjoint "
        "14-day blocks report per-axis medians only with at least eight complete "
        "observations. Signed changes compare adjacent supported blocks; null "
        "means unavailable and is never interpolated. Axis signs describe direction, "
        "not activity classes. The full-record daily-difference MAD is a noise "
        "proxy, not calibrated uncertainty. No activity or stage answer is supplied. "
        "Compare continuing raw changes with plateaus, isolated steps and noise "
        "in the plots; reporting blocks are not event boundaries. Leave uncertain "
        "when movement is indistinguishable from observation variability.\n"
        + json.dumps(signed_block_facts(case, noise, start, stop))
    )
