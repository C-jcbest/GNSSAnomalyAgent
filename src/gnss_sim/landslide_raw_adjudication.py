"""Observation-only author review, with truthful provenance and no stage assignment."""
from __future__ import annotations

import hashlib

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from gnss_sim.landslide_reference_interface import parse_partition_draft


def raw_block_statistics(case) -> dict:
    """Fixed calendar blocks describe observed values; gaps are never interpolated."""
    blocks = []
    for start in range(0, len(case.dates), 30):
        stop = min(start + 30, len(case.dates))
        observed_values = [row for row in case.displacement_mm[start:stop] if row is not None]
        median = [None, None, None]
        interquartile_range = [None, None, None]
        if observed_values:
            values = np.asarray(observed_values)
            median = np.median(values, axis=0).tolist()
            quartiles = np.quantile(values, [0.25, 0.75], axis=0)
            interquartile_range = (quartiles[1] - quartiles[0]).tolist()
        blocks.append({"start": start, "stop": stop, "valid_days": len(observed_values),
                       "median_NEU_mm": median, "iqr_NEU_mm": interquartile_range})
    return {
        "input_sha256": hashlib.sha256(case.model_dump_json().encode("utf-8")).hexdigest(),
        "source": "non-overlapping raw 30-day blocks; descriptive only",
        "blocks": blocks,
    }


def compile_author_activity(payload, case, record):
    review, rows = parse_partition_draft(payload, case, record, "Codex")
    # The shared parser provides calendar validation, not the actual review identity.
    review["reviewer"] = "Codex / same-session AI author raw-displacement review"
    review["provenance"] = (
        "同会话AI作者复核；已见旧活动/阶段输出、部分辅助图及作者诊断；"
        "本轮重读四张原始位移图并查固定30日块统计；非盲标、非独立专家真值"
    )
    review["review_protocol"] = "landslide-raw-adjudication-v3"
    review["prior_outputs_seen"] = True
    review["independent_expert"] = False
    review["stage_status"] = "not_started"
    review["new_external_calls"] = 0
    review["raw_views_read"] = [view["image"] for view in record["views"]]
    review["block_statistics_source"] = "non-overlapping raw 30-day blocks; descriptive only"
    return review, rows


def render_author_activity(case, view, rows, path, plot):
    """Each view uses its original axes and the exact exported daily labels."""
    values = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    observed = np.asarray([row is not None for row in case.displacement_mm])
    colors = np.asarray([row["activity_label"] + 1 for row in rows])
    colors[~observed] = 3
    figure = Figure(figsize=(plot["figure_inches"][0], 9), dpi=plot["dpi"], facecolor="white")
    FigureCanvasAgg(figure)
    panels = figure.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [3, 3, 3, 1]})
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.91, hspace=0.18)
    geometry = view["audit"]
    for axis, name in enumerate(("N", "E", "U")):
        panel = panels[axis]
        panel.plot(np.arange(len(case.dates)), values[:, axis], linewidth=0.8,
                   color=("#147d72", "#cc6d4b", "#a38a32")[axis])
        panel.set_xlim(geometry["panels"][axis]["xlim"])
        panel.set_ylim(geometry["panels"][axis]["ylim"])
        panel.set_ylabel(name + " (mm)")
        panel.grid(alpha=0.18)
    panels[-1].imshow([colors], aspect="auto", interpolation="nearest", vmin=0, vmax=3,
                      cmap=ListedColormap(["#e5b65b", "#dce5de", "#427b9b", "#737373"]),
                      extent=(-0.5, len(case.dates) - 0.5, 0.5, -0.5))
    panels[-1].set_yticks([0], ["author v3"])
    panels[-1].set_xlim(geometry["panels"][0]["xlim"])
    panels[-1].set_xticks(geometry["tick_audit"]["ticks"])
    panels[-1].set_xlabel("Original day index; raw observations retain gaps")
    figure.suptitle(f"{case.case_id} | Days {view['view'][0]}..{view['view'][1]} | author activity review")
    figure.text(0.10, 0.035,
                "Blue: activity | pale green: stationary | amber: observed unknown | gray: missing\n"
                "Same raw axes; no stages assigned; same-session AI author review, NOT independent truth.",
                fontsize=9)
    figure.savefig(path, format="png")
    return {"label_colors": colors.tolist(), "xlim": geometry["panels"][0]["xlim"],
            "ylim_NEU": [panel["ylim"] for panel in geometry["panels"]]}
