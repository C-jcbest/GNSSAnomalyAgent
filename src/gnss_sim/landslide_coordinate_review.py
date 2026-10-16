"""Check activity decisions under a sign change of the displacement coordinates."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def coordinate_values(case, sign):
    """Change coordinate orientation without changing time, units or missingness."""
    if sign not in (-1, 1):
        raise ValueError("Only an exact coordinate sign reflection is permitted")
    values = np.asarray([
        row if row is not None else [np.nan] * 3
        for row in case.displacement_mm
    ], dtype=float)
    return sign * values


def render_coordinate_view(case, noise, start, stop, sign, path):
    values = coordinate_values(case, sign)
    if not 0 <= start < stop <= len(values):
        raise ValueError("Coordinate view must use the original calendar")
    fig, panels = plt.subplots(3, 1, figsize=(10.24, 6.3), sharex=True)
    limits = []
    for axis, name in enumerate(("N", "E", "U")):
        panel = panels[axis]
        series = values[start:stop, axis]
        panel.plot(np.arange(start, stop), series, linewidth=.8,
                   color=("#147d72", "#cc6d4b", "#a38a32")[axis])
        finite = series[np.isfinite(series)]
        if len(finite):
            center = (float(finite.min()) + float(finite.max())) / 2
            span = max(float(np.ptp(finite)), 6 * noise[axis], 10)
            panel.set_ylim(center - .75 * span, center + .75 * span)
        panel.set_xlim(start, stop - 1)
        panel.set_ylabel(name + " displacement (mm)")
        panel.grid(alpha=.2)
        limits.append(list(panel.get_ylim()))
    panels[-1].set_xticks(np.unique(np.r_[np.arange(start, stop, 30), stop - 1]))
    panels[-1].set_xlabel("Original day index; missing observations remain gaps")
    fig.suptitle(f"{case.case_id} | Days {start}..{stop - 1} | Raw only; NO labels\n"
                 f"Coordinate convention: displayed N/E/U = {sign:+d} times original N/E/U")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"sign": sign, "start": start, "stop": stop, "axis_limits": limits}


def agree_or_abstain(initial, first, second, scope, observed):
    """Allow a definite revision only when two visual readings agree.

    This is a reliability hypothesis, not a correctness guarantee: systematic
    visual errors can agree. A failed or inconclusive reading cannot certify a
    definite decision. The same rule is used for the ordinary-repeat control.
    """
    for values in (initial, first, second):
        if values.shape != observed.shape or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError("Coordinate review requires aligned activity states")
        if np.any(values[~observed] != -1):
            raise ValueError("Missing observations must remain unknown")
    if scope.shape != observed.shape or scope.dtype != bool or np.any(scope & ~observed):
        raise ValueError("Revision authority must contain observed days only")
    activity = initial.copy()
    activity[scope] = -1
    agreement = scope & (first == second) & (first >= 0)
    activity[agreement] = first[agreement]
    return activity
