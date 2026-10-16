"""Display chronological raw displacement comparisons without deciding activity."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def raw_block_comparisons(case, noise, start, stop):
    """Keep physical axes and unavailable blocks in the original calendar."""
    values = np.asarray(
        [row if row is not None else [np.nan] * 3 for row in case.displacement_mm],
        dtype=float,
    )
    noise = np.asarray(noise, dtype=float)
    if not 0 <= start < stop <= len(values):
        raise ValueError("Comparison view must stay inside the raw calendar")
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Comparison view requires three positive noise proxies")
    starts = np.arange(start, stop, 14)
    stops = np.minimum(starts + 14, stop)
    medians = np.full((len(starts), 3), np.nan)
    counts = []
    for index, (left, right) in enumerate(zip(starts, stops, strict=True)):
        sample = values[left:right]
        complete = sample[np.isfinite(sample).all(axis=1)]
        counts.append(len(complete))
        if len(complete) >= 8:
            medians[index] = np.median(complete, axis=0)
    normalized = medians / noise
    # Row is the reference block; column is the compared block. No temporal
    # interpolation or filling of unsupported blocks is performed.
    differences = normalized[None, :, :] - normalized[:, None, :]
    finite = np.flatnonzero(np.isfinite(medians).all(axis=1))
    traces = normalized - normalized[finite[0]] if len(finite) else normalized.copy()
    return {
        "starts": starts,
        "stops": stops,
        "complete_days": np.asarray(counts),
        "medians": medians,
        "traces": traces,
        "differences": differences,
    }


def render_comparison_supplement(case, noise, start, stop, arm, path):
    if arm not in {"comparison_map", "median_trace"}:
        raise ValueError("Unknown comparison supplement")
    facts = raw_block_comparisons(case, noise, start, stop)
    figure, panels = plt.subplots(1, 3, figsize=(12, 4.3))
    for index, axis in enumerate(("N", "E", "U")):
        panel = panels[index]
        if arm == "comparison_map":
            color = plt.get_cmap("RdBu_r").copy()
            color.set_bad("#b7b7b7")
            edges = np.r_[facts["starts"], stop]
            mesh = panel.pcolormesh(
                edges,
                edges,
                np.ma.masked_invalid(facts["differences"][:, :, index]),
                cmap=color,
                vmin=-3,
                vmax=3,
                shading="flat",
            )
            panel.set_ylim(start, stop)
            panel.set_ylabel("Reference block day (row)")
            figure.colorbar(mesh, ax=panel, extend="both", shrink=0.8)
        else:
            for left, right, value in zip(
                facts["starts"], facts["stops"], facts["traces"][:, index], strict=True
            ):
                if np.isfinite(value):
                    panel.hlines(value, left, right, color="#315f8b", linewidth=1.8)
                    panel.plot((left + right - 1) / 2, value, "o", color="#315f8b", markersize=3)
            # Six noise-proxy units is a display floor, not an activity threshold.
            finite = facts["traces"][:, index]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                lower, upper = float(finite.min()), float(finite.max())
                center = (lower + upper) / 2
                span = max(upper - lower, 6)
                panel.set_ylim(center - 0.6 * span, center + 0.6 * span)
            panel.set_ylabel("(Block median - first valid median) / noise proxy")
            panel.grid(alpha=0.2)
        panel.set_xlim(start, stop)
        panel.set_xticks(np.arange(start, stop + 1, 45))
        panel.set_xlabel(
            "Compared block day (column)" if arm == "comparison_map" else "Original block day"
        )
        panel.set_title(axis)
    if arm == "comparison_map":
        caption = "Column median - row median, divided by axis daily-difference MAD; display clipped at +/-3"
    else:
        caption = "Same block medians and noise units; disconnected blocks, no interpolation"
    figure.suptitle(
        f"{case.case_id} | Raw14-day block observations | Days {start}..{stop - 1}\n{caption}",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return facts
