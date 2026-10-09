"""A fixed prompt intervention and presentation for per-activity stage evidence."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

ARMS = ("control-r1", "structured-r1", "control-r2", "structured-r2")
PAIRS = (("control-r1", "structured-r1"), ("control-r2", "structured-r2"),
         ("control-r1", "control-r2"), ("structured-r1", "structured-r2"))

EVIDENCE_INSTRUCTION = """

PER-ACTIVITY OBSERVABLE EVIDENCE PROCEDURE
Apply the following observation sequence separately inside EACH frozen activity
interior. Do not re-decide activity, and do not use the strongest episode as the
minimum rate required for weaker episodes. The activity review remains fixed.

1. Read the raw local displacement first: describe sustained direction and how
   slope steepness changes, keeping N/E/U directions separate from motion speed.
   An offset jump or isolated spike is not a sustained rate-change stage.
2. Compare the SIZE of the motion rate in earlier, middle and later parts of that
   interior using the existing auxiliary windows and the raw slopes. These are
   qualitative observations, not fixed thirds or mandatory stage boundaries.
   A signed velocity becoming more negative may mean speed is INCREASING, whereas
   returning toward zero may mean speed is DECREASING. Use magnitude evolution,
   not the sign of displacement, to distinguish A from D.
3. Inspect INTERNAL changes in that evolution, including peaks and flattening.
   If rise and fall coexist, propose separate supported stages or omit conflicting
   transition dates; do not label the whole interior by its dominant trend alone.
   If the windows disagree or do not support a turn, report the uncertainty.
4. Assign stages only after those observations. For S, state evidence for a
   sustained NONZERO rate AND approximately unchanged rate. No visible velocity
   or a near-zero noise-biased speed is not evidence for S. Weak motion can stay
   stage-unknown while the frozen displacement activity remains confirmed.
   Separate a real slow trend from derivative response around a jump. Missing
   auxiliary estimates or truncated edges limit stage evidence, not activity.

For each reported stage, keep evidence concise and observable: raw direction /
slope evolution; rate magnitude evolution and internal turns; window support or
conflict. Avoid invented precision or claiming a trend that the plots do not show.
Use stage_notes for a brief entry for each activity with omitted dates, explaining
the unresolved stage or unavailable support without declaring the activity absent.
Keep the existing JSON schema and sparse stages; no additional keys or plots.
"""


def evidence_prompt(control_prompt, condition):
    if condition == "control":
        return control_prompt
    if condition == "structured":
        return control_prompt + EVIDENCE_INSTRUCTION
    raise ValueError("Unknown stage-evidence condition")


def request_order(active_number):
    if type(active_number) is not int or active_number < 1:
        raise ValueError("Expected a positive active-record number")
    if active_number % 2:
        return (("control", 1), ("structured", 1), ("structured", 2), ("control", 2))
    return (("structured", 1), ("control", 1), ("control", 2), ("structured", 2))


def render_evidence_comparison(case, view, rows_by_arm, path, plot):
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
    figure.suptitle(f"{case.case_id} | stage evidence instructions x two repeats | NOT stage truth")
    figure.text(0.12, 0.035,
                "Activity: orange=active; pale green=stationary; amber=unknown; gray=missing\n"
                "Stages: orange=A; blue=S; gold=D; amber=unknown; pale green=none; gray=missing. Same raw limits.", fontsize=9)
    figure.savefig(path, format="png")
    return {"label_colors": label_colors, "xlim": geometry["panels"][0]["xlim"],
            "ylim_NEU": [p["ylim"] for p in geometry["panels"]]}
