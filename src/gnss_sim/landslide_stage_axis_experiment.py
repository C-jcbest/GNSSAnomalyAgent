"""Paired auxiliary figures varying only the horizontal presentation."""
from __future__ import annotations

import hashlib
import io

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure


def _legacy_figure(case, diagnostics):
    """Reconstruct the frozen renderer locally; upstream source must stay immutable."""
    days = np.arange(len(case.dates))
    colors = ("#147d72", "#cc6d4b", "#a38a32")
    observed = np.array([row if row is not None else (np.nan,) * 3 for row in case.displacement_mm])
    fitted = np.array([row if row is not None else (np.nan,) * 3 for row in diagnostics.fitted_displacement_mm])
    velocity = np.array([row if row is not None else (np.nan,) * 3 for row in diagnostics.velocity_mm_day])
    acceleration = np.array([row if row is not None else (np.nan,) * 3 for row in diagnostics.acceleration_mm_day2])
    figure = Figure(figsize=(14, 11), dpi=150, facecolor="white")
    FigureCanvasAgg(figure)
    panels = figure.subplots(4, 1, sharex=True)
    for axis, color in enumerate(colors):
        panels[0].plot(days, observed[:, axis], color=color, alpha=0.4, linewidth=0.65)
        panels[0].plot(days, fitted[:, axis], color=color, linewidth=1.4, label=("N", "E", "U")[axis])
        panels[1].plot(days, velocity[:, axis], color=color, linewidth=1, label=("N", "E", "U")[axis])
        panels[2].plot(days, acceleration[:, axis], color=color, linewidth=1)
    panels[1].plot(days, diagnostics.speed_mm_day, color="#283847", linewidth=1.2,
                   linestyle="--", label="3D speed (positive noise bias near zero)")
    panels[3].plot(days, diagnostics.tangential_acceleration_mm_day2, color="#283847", linewidth=1.1)
    low, high = min(0, float(np.nanmin(observed))), max(0, float(np.nanmax(observed)))
    middle, span = (low + high) / 2, max(60, high - low)
    panels[0].set_ylim(middle - 0.75 * span, middle + 0.75 * span)
    labels = ("Displacement (mm)", "Velocity (mm/day)", "Component acceleration\n(mm/day^2)",
              "Tangential acceleration\n(mm/day^2)")
    for panel, label in zip(panels, labels, strict=True):
        panel.set_ylabel(label)
        panel.axhline(0, color="#9ca8a0", linewidth=0.6, linestyle="--")
        panel.grid(alpha=0.18)
        panel.set_xlim(0, len(days) - 1)
    panels[0].legend(loc="upper left", ncol=3)
    panels[1].legend(loc="upper left", ncol=4, fontsize=8)
    ticks = np.unique(np.linspace(0, len(days) - 1, 8, dtype=int))
    panels[-1].set_xticks(ticks, [f"{case.dates[i]}\nDay {i}" for i in ticks], fontsize=9)
    panels[-1].set_xlabel("Observation date / zero-based day index")
    figure.suptitle(f"{case.case_id} | Raw observations + {diagnostics.window_days}-day robust local quadratic fit",
                    fontsize=14)
    figure.text(0.10, 0.02,
                "Retrospective centered estimates; gaps/unsupported windows omitted. "
                "No generator labels. Tangential gate is heuristic.\n"
                f"Canonical input SHA-256: {diagnostics.input_sha256}", fontsize=8, color="#536459")
    figure.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.12, hspace=0.20)
    return figure, panels


def _figure_audit(figure, panels):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = [label.get_window_extent(renderer) for label in panels[-1].get_xticklabels()]
    return {"size_pixels": list(figure.canvas.get_width_height()),
            "panels": [{"rect_pixels": list(panel.get_window_extent(renderer).bounds),
                        "xlim": list(panel.get_xlim()), "ylim": list(panel.get_ylim()),
                        "line_sha256": [hashlib.sha256(line.get_xydata().tobytes()).hexdigest()
                                        for line in panel.lines]} for panel in panels],
            "ticks": panels[-1].get_xticks().tolist(),
            "tick_labels": [label.get_text() for label in panels[-1].get_xticklabels()],
            "label_bounds_pixels": [[box.x0, box.x1] for box in bounds],
            "minimum_label_gap_pixels": min(bounds[i + 1].x0 - bounds[i].x1 for i in range(len(bounds) - 1))}


def _png(figure):
    output = io.BytesIO()
    figure.savefig(output, format="png")
    return output.getvalue()


def render_auxiliary_axis_pair(case, diagnostics, day_ticks):
    if not day_ticks or day_ticks != sorted(set(day_ticks)) or day_ticks[0] != 0 or day_ticks[-1] != len(case.dates) - 1:
        raise ValueError("Expected sorted unique full-record frozen day ticks")
    if any(type(tick) is not int for tick in day_ticks):
        raise ValueError("Day ticks must be integer observation indices")
    figure, panels = _legacy_figure(case, diagnostics)
    original = _png(figure)
    original_audit = _figure_audit(figure, panels)
    panels[-1].set_xticks(day_ticks, [str(day) for day in day_ticks], fontsize=9)
    panels[-1].set_xlabel("Original day index")
    day_index = _png(figure)
    day_audit = _figure_audit(figure, panels)
    if original_audit["panels"] != day_audit["panels"] or original_audit["size_pixels"] != day_audit["size_pixels"]:
        raise ValueError("Changing horizontal annotations changed curve geometry or values")
    if day_audit["minimum_label_gap_pixels"] < 8:
        raise ValueError("Day-index labels collide; reject before inference")
    return original, day_index, {"dates": original_audit, "day_index": day_audit,
                                 "curve_geometry_unchanged": True,
                                 "intervention": "Ticks, tick labels, xlabel and corresponding vertical grids only"}


def render_stage_axis_comparison(case, view, rows_by_arm, path, plot):
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    observed = np.array([row is not None for row in case.displacement_mm])
    first = rows_by_arm["dates-r1"]
    colors = [[5 if not observed[i] else row["activity_label"] + 1 for i, row in enumerate(first)]]
    codes = {"unknown": 0, "none": 1, "acceleration": 2, "steady_motion": 3, "deceleration": 4}
    arms = ("dates-r1", "day_index-r1", "dates-r2", "day_index-r2")
    for arm in arms:
        rows = rows_by_arm[arm]
        if [(r["day_index"], r["date"], r["activity_label"]) for r in rows] != [
                (r["day_index"], r["date"], r["activity_label"]) for r in first]:
            raise ValueError("All comparison strips must share the frozen calendar/activity")
        colors.append([5 if not observed[i] else codes[r["feature"]] for i, r in enumerate(rows)])
    figure = Figure(figsize=(plot["figure_inches"][0], 10), dpi=plot["dpi"], facecolor="white")
    FigureCanvasAgg(figure)
    panels = figure.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [3, 3, 3, 2.3]})
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.91, hspace=0.18)
    geometry = view["audit"]
    for axis, name in enumerate(("N", "E", "U")):
        panels[axis].plot(np.arange(len(values)), values[:, axis], linewidth=0.8,
                          color=("#147d72", "#cc6d4b", "#a38a32")[axis])
        panels[axis].set_xlim(geometry["panels"][axis]["xlim"])
        panels[axis].set_ylim(geometry["panels"][axis]["ylim"])
        panels[axis].set_ylabel(name + " (mm)")
        panels[axis].grid(alpha=0.18)
    panels[-1].imshow(colors, aspect="auto", interpolation="nearest", vmin=0, vmax=5,
                      cmap=ListedColormap(["#e5b65b", "#dce5de", "#d27545", "#557ab6", "#a98630", "#737373"]),
                      extent=(-0.5, len(values) - 0.5, 4.5, -0.5))
    panels[-1].set_yticks(range(5), ["activity", *arms])
    panels[-1].set_xlim(geometry["panels"][0]["xlim"])
    panels[-1].set_xticks(geometry["tick_audit"]["ticks"])
    panels[-1].set_xlabel("Original day index")
    figure.suptitle(f"{case.case_id} | two horizontal presentations x two repeats | NOT stage truth")
    figure.text(0.12, 0.035,
                "Activity: orange=active; pale green=stationary; amber=unknown; gray=missing\n"
                "Stages: orange=A; blue=S; gold=D; amber=unknown; pale green=none; gray=missing. Same raw limits.", fontsize=9)
    figure.savefig(path, format="png")
    return {"label_colors": colors, "xlim": geometry["panels"][0]["xlim"],
            "ylim_NEU": [p["ylim"] for p in geometry["panels"]]}
