"""Activity-local views of frozen estimates, without changing values or y scales."""
from __future__ import annotations

import hashlib

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from gnss_sim.landslide_stage_axis_experiment import _figure_audit, _legacy_figure, _png

ARMS = ("full-r1", "local-r1", "full-r2", "local-r2")
PAIRS = (("full-r1", "local-r1"), ("full-r2", "local-r2"),
         ("full-r1", "full-r2"), ("local-r1", "local-r2"))

LOCAL_CONTEXT_NOTE = """

OPTIONAL ACTIVITY-LOCAL AUXILIARY VIEWS
The first seven images retain the image order described above. Any additional
images are local views of exactly the same cached auxiliary estimates, ordered by
frozen activity start and then by 31/61/91-day window. Their titles identify the
HALF-OPEN activity interior and original day-index context. Y scales and units
remain those of the corresponding full-record figure; estimates are not recomputed.
Use additional images if present as supplementary views, not new independent data.
Context outside the allowed activity interior must not receive a definite stage.
The stage task, sparse JSON schema, uncertainty and frozen activity remain unchanged.
"""


def local_prompt(control_prompt):
    return control_prompt + LOCAL_CONTEXT_NOTE


def request_order(active_number):
    if type(active_number) is not int or active_number < 1:
        raise ValueError("Expected a positive active-record number")
    if active_number % 2:
        return (("full", 1), ("local", 1), ("local", 2), ("full", 2))
    return (("local", 1), ("full", 1), ("full", 2), ("local", 2))


def activity_windows(review, days, context_days):
    if type(context_days) is not int or context_days < 0:
        raise ValueError("Context must be a nonnegative integer")
    windows = []
    for span in review["spans"]:
        if span["state"] != "activity":
            continue
        start, stop = span["start"], span["stop"]
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= days:
            raise ValueError("Activity interior exceeds the observed calendar")
        windows.append({"start": start, "stop": stop,
                        "left": max(0, start - context_days),
                        "right": min(days - 1, stop - 1 + context_days)})
    return windows


def render_auxiliary_activity_window(case, diagnostics, window, full_ticks):
    input_digest = hashlib.sha256(case.model_dump_json().encode("utf-8")).hexdigest()
    if (diagnostics.case_id != case.case_id or diagnostics.input_sha256 != input_digest
            or len(diagnostics.velocity_mm_day) != len(case.dates)):
        raise ValueError("Local estimates do not match the record")
    figure, panels = _legacy_figure(case, diagnostics)
    panels[-1].set_xticks(full_ticks, [str(day) for day in full_ticks], fontsize=9)
    panels[-1].set_xlabel("Original day index")
    full_png = _png(figure)
    full_audit = _figure_audit(figure, panels)
    left, right = window["left"], window["right"]
    if not 0 <= left < right < len(case.dates):
        raise ValueError("Invalid local context bounds")
    if not left <= window["start"] < window["stop"] <= right + 1:
        raise ValueError("Local context must contain the entire activity interior")
    ticks = np.unique(np.linspace(left, right, 7, dtype=int))
    panels[-1].set_xticks(ticks, [str(day) for day in ticks], fontsize=9)
    for panel in panels:
        panel.set_xlim(left, right)
    figure.suptitle(
        f"{case.case_id} | {diagnostics.window_days}-day cached fit | "
        f"activity [{window['start']},{window['stop']}) | context [{left},{right}]",
        fontsize=12,
    )
    local_png = _png(figure)
    local_audit = _figure_audit(figure, panels)
    for before, after in zip(full_audit["panels"], local_audit["panels"], strict=True):
        for key in ("ylim", "rect_pixels", "line_sha256"):
            if before[key] != after[key]:
                raise ValueError(f"Local view changed frozen {key}")
    if full_audit["size_pixels"] != local_audit["size_pixels"]:
        raise ValueError("Local view changed image size")
    if local_audit["minimum_label_gap_pixels"] < 8:
        raise ValueError("Local day-index labels collide")
    return local_png, {"window": window, "full": full_audit, "local": local_audit,
                       "full_reference_sha256": hashlib.sha256(full_png).hexdigest(),
                       "values_y_limits_geometry_unchanged": True}


def render_local_comparison(case, view, rows_by_arm, path, plot):
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    observed = np.array([row is not None for row in case.displacement_mm])
    first = rows_by_arm[ARMS[0]]
    if len(first) != len(values):
        raise ValueError("Stage calendar does not match observations")
    activity_colors = [5 if not observed[i] else row["activity_label"] + 1 for i, row in enumerate(first)]
    label_colors = [activity_colors]
    codes = {"unknown": 0, "none": 1, "acceleration": 2, "steady_motion": 3, "deceleration": 4}
    calendar = [(r["case_id"], r["day_index"], r["date"], r["activity_label"]) for r in first]
    for arm in ARMS:
        rows = rows_by_arm[arm]
        if [(r["case_id"], r["day_index"], r["date"], r["activity_label"]) for r in rows] != calendar:
            raise ValueError("All stage strips must share the frozen calendar/activity")
        label_colors.append([5 if not observed[i] else codes[r["feature"]] for i, r in enumerate(rows)])
    geometry = view["audit"]
    figure = Figure(figsize=(plot["figure_inches"][0], 10), dpi=plot["dpi"], facecolor="white")
    FigureCanvasAgg(figure)
    panels = figure.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [3, 3, 3, 2.3]})
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.91, hspace=0.18)
    for axis, name in enumerate(("N", "E", "U")):
        panels[axis].plot(np.arange(len(values)), values[:, axis], linewidth=0.8,
                          color=("#147d72", "#cc6d4b", "#a38a32")[axis])
        panels[axis].set_xlim(geometry["panels"][axis]["xlim"])
        panels[axis].set_ylim(geometry["panels"][axis]["ylim"])
        panels[axis].set_ylabel(name + " (mm)")
        panels[axis].grid(alpha=0.18)
    panels[-1].imshow(label_colors, aspect="auto", interpolation="nearest", vmin=0, vmax=5,
                      cmap=ListedColormap(["#e5b65b", "#dce5de", "#d27545", "#557ab6", "#a98630", "#737373"]),
                      extent=(-0.5, len(values) - 0.5, 4.5, -0.5))
    panels[-1].set_yticks(range(5), ["activity", *ARMS])
    panels[-1].set_xlim(geometry["panels"][0]["xlim"])
    panels[-1].set_xticks(geometry["tick_audit"]["ticks"])
    panels[-1].set_xlabel("Original day index")
    figure.suptitle(f"{case.case_id} | added activity auxiliary views x two repeats | NOT stage truth")
    figure.text(0.12, 0.035,
                "Activity: orange=active; pale green=stationary; amber=unknown; gray=missing\n"
                "Stages: orange=A; blue=S; gold=D; amber=unknown; pale green=none; gray=missing. Same raw limits.", fontsize=9)
    figure.savefig(path, format="png")
    return {"label_colors": label_colors, "xlim": geometry["panels"][0]["xlim"],
            "ylim_NEU": [p["ylim"] for p in geometry["panels"]]}
