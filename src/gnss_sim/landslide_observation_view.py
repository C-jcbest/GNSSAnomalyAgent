"""Raw context and observation-only block summaries for a paired visual probe."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def observation_bounds(start, stop, days, context_days):
    if not 0 <= start < stop <= days or context_days < 0:
        raise ValueError("Target or context does not fit the observation calendar")
    return max(0, start - context_days), min(days, stop + context_days)


def raw_block_summaries(values, start, stop):
    """Summarize complete calendar-aligned blocks, never interpolate daily values.

    Eight complete observations is the existing raw evidence's minimum support.
    The calendar anchor is independent of the target or display width, so widening
    a view does not silently change the summaries in its overlapping interior.
    """
    if values.ndim != 2 or values.shape[1] != 3 or not 0 <= start < stop <= len(values):
        raise ValueError("Raw summaries require a three-component calendar")
    blocks = []
    first = ((start + 13) // 14) * 14
    for left in range(first, stop - 13, 14):
        sample = values[left:left + 14]
        complete = sample[np.isfinite(sample).all(axis=1)]
        if len(complete) < 8:
            continue
        blocks.append({"start": left, "stop": left + 14,
                       "observed_days": len(complete),
                       "median_mm": np.median(complete, axis=0).tolist()})
    return blocks


def render_observation_view(case, noise, start, stop, context_days, show_blocks, path):
    values = np.asarray([
        row if row is not None else [np.nan] * 3 for row in case.displacement_mm
    ], dtype=float)
    left, right = observation_bounds(start, stop, len(values), context_days)
    blocks = raw_block_summaries(values, left, right)
    fig, panels = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for component, name in enumerate(("N", "E", "U")):
        panel = panels[component]
        series = values[left:right, component]
        panel.plot(np.arange(left, right), series, linewidth=.8,
                   color=("#147d72", "#cc6d4b", "#a38a32")[component], label="Raw observations")
        # Match axis limits within each raw/block pair: overlays cannot rescale it.
        finite = series[np.isfinite(series)]
        if len(finite):
            center = (float(finite.min()) + float(finite.max())) / 2
            span = max(float(np.ptp(finite)), 6 * noise[component], 10)
            panel.set_ylim(center - .75 * span, center + .75 * span)
        if show_blocks:
            for index, block in enumerate(blocks):
                panel.hlines(block["median_mm"][component], block["start"], block["stop"] - 1,
                             color="#202020", linewidth=1.7,
                             label="14-day block median (>=8 observed days)" if index == 0 else None)
        panel.axvspan(start - .5, stop - .5, color="#386db5", alpha=.06)
        panel.axvline(start - .5, color="#386db5", linestyle="--", linewidth=.8)
        panel.axvline(stop - .5, color="#386db5", linestyle="--", linewidth=.8)
        panel.set_xlim(left, right - 1)
        panel.set_ylabel(name + " displacement (mm)")
        panel.grid(alpha=.2)
    panels[0].legend(loc="upper left", fontsize=8)
    boundary_ticks = np.unique([left, start, stop - 1, right - 1])
    regular_ticks = [int(day) for day in np.arange(left, right, 45)
                     if np.min(np.abs(boundary_ticks - day)) > 12]
    ticks = np.unique(np.r_[boundary_ticks, regular_ticks])
    panels[-1].set_xticks(ticks)
    panels[-1].set_xlabel("Original day index; missing raw observations remain gaps")
    display = "Raw + block summaries" if show_blocks else "Raw only"
    fig.suptitle(f"{case.case_id} | View {left}..{right - 1}; judge {start}..{stop - 1} | {display}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"view_start": left, "view_stop": right, "target_start": start, "target_stop": stop,
            "show_blocks": show_blocks, "raw_blocks": blocks,
            "axis_limits": "Raw-only range, noise floor and 10mm floor; identical within each pair",
            "missing_policy": "No daily interpolation; unsupported blocks absent, summaries are not labels"}
