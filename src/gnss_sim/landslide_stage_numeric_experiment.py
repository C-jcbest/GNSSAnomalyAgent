"""Fixed observation summaries for a paired stage experiment; no stage decisions."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

ARMS = ("image-r1", "numeric-r1", "image-r2", "numeric-r2")
PAIRS = (("image-r1", "numeric-r1"), ("image-r2", "numeric-r2"),
         ("image-r1", "image-r2"), ("numeric-r1", "numeric-r2"))

NUMERIC_NOTE = """

SAME-SOURCE DESCRIPTIVE NUMERIC SUMMARY
The JSON below describes observed data, not reference stages or recommendations.
Bins are calendar-aligned 30-day blocks clipped to each allowed activity interior;
their boundaries are NOT stage boundaries. Raw medians describe NEU displacement
in mm, not instantaneous endpoints. Use their time order together with raw plots.
Each fit has its window in days. v is componentwise median velocity NEU (mm/day),
q is speed [25th,50th,75th percentile] (mm/day), and t is median tangential
acceleration (mm/day^2). Median vector norm need not equal median speed.
n=[observed days, finite velocity AND speed days, finite tangential days]; all
fit summaries use only observed days in this bin, excluding missing/unsupported
values. Tangential days also require velocity and speed. c is median cached
window observation count across the observed bin days (including unsupported fits).
Counts indicate availability, NOT confidence, nonzero evidence or independent data.
null means unavailable, never zero. Quantiles are within-bin spread, NOT confidence
intervals. Fits are centered, overlapping and retrospective; their support may
include days outside the bin/activity. Edge gaps and smoothing are not stage changes.
3D speed has a noise-related positive bias, especially in weak movement: a positive
q alone does not establish nonzero steady motion. Raw sustained displacement still
controls activity; stage uncertainty must not overturn the supplied activity.
Assess rate evolution across bins/windows and the plots; do not convert a single
t sign, a direction sign, or each bin into a stage. The original JSON schema and
all stage definitions remain unchanged. This is not an online prediction task.
"""


def request_order(active_number):
    if type(active_number) is not int or active_number < 1:
        raise ValueError("Expected a positive active-record number")
    if active_number % 2:
        return (("image", 1), ("numeric", 1), ("numeric", 2), ("image", 2))
    return (("numeric", 1), ("image", 1), ("image", 2), ("numeric", 2))


def _rounded_median(values):
    if not values:
        return None
    return np.round(np.median(values, axis=0), 6).tolist()


def numeric_summary(case, review, motions):
    """Aggregate existing caches only; never interpolate or refit observations."""
    days = len(case.dates)
    input_digest = hashlib.sha256(case.model_dump_json().encode("utf-8")).hexdigest()
    if set(motions) != {31, 61, 91}:
        raise ValueError("Expected exactly three frozen fit windows")
    for window, motion in motions.items():
        if (motion.case_id != case.case_id or motion.input_sha256 != input_digest
                or motion.window_days != window):
            raise ValueError("Numeric cache provenance does not match record")
        for name in ("valid_counts", "velocity_mm_day", "speed_mm_day", "tangential_acceleration_mm_day2"):
            if len(getattr(motion, name)) != days:
                raise ValueError("Numeric cache calendar differs")
    activities = []
    previous_stop = 0
    for span in review["spans"]:
        if span["state"] != "activity":
            continue
        start, stop = span["start"], span["stop"]
        if (type(start) is not int or type(stop) is not int
                or not previous_stop <= start < stop <= days):
            raise ValueError("Invalid activity calendar")
        previous_stop = stop
        blocks = []
        for calendar_start in range(start // 30 * 30, stop, 30):
            left, right = max(start, calendar_start), min(stop, calendar_start + 30)
            observed = [i for i in range(left, right) if case.displacement_mm[i] is not None]
            raw = [case.displacement_mm[i] for i in observed]
            fits = {}
            for window, motion in motions.items():
                supported = [i for i in observed
                             if motion.velocity_mm_day[i] is not None
                             and np.all(np.isfinite(motion.velocity_mm_day[i]))
                             and motion.speed_mm_day[i] is not None
                             and np.isfinite(motion.speed_mm_day[i])]
                tangential = [i for i in supported
                              if motion.tangential_acceleration_mm_day2[i] is not None
                              and np.isfinite(motion.tangential_acceleration_mm_day2[i])]
                speed = [motion.speed_mm_day[i] for i in supported]
                quantiles = None
                if speed:
                    quantiles = np.round(np.quantile(speed, [0.25, 0.5, 0.75]), 6).tolist()
                fits[str(window)] = {
                    "n": [len(observed), len(supported), len(tangential)],
                    "v": _rounded_median([motion.velocity_mm_day[i] for i in supported]),
                    "q": quantiles,
                    "t": _rounded_median([motion.tangential_acceleration_mm_day2[i] for i in tangential]),
                    "c": _rounded_median([motion.valid_counts[i] for i in observed]),
                }
            blocks.append({"bin": [left, right], "observed": len(observed),
                           "raw_NEU_mm": _rounded_median(raw), "fits": fits})
        activities.append({"interior": [start, stop], "blocks": blocks})
    return {"case_id": case.case_id, "input_sha256": input_digest,
            "cache_sha256": {str(w): hashlib.sha256(m.model_dump_json().encode()).hexdigest()
                             for w, m in motions.items()},
            "bin_days": 30, "activities": activities}


def numeric_prompt(original, summary):
    return original + NUMERIC_NOTE + json.dumps(summary, separators=(",", ":"), allow_nan=False)


def render_numeric_comparison(case, view, rows_by_arm, path, plot):
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
    figure.suptitle(f"{case.case_id} | same-source numeric summary x two repeats | NOT stage truth")
    figure.text(0.12, 0.035,
                "Activity: orange=active; pale green=stationary; amber=unknown; gray=missing\n"
                "Stages: orange=A; blue=S; gold=D; amber=unknown; pale green=none; gray=missing. Same raw limits.", fontsize=9)
    figure.savefig(path, format="png")
    return {"label_colors": label_colors, "xlim": geometry["panels"][0]["xlim"],
            "ylim_NEU": [p["ylim"] for p in geometry["panels"]]}
