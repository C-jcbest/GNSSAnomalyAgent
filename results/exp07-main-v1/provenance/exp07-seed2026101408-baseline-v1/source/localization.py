"""Estimate a step date inside an inspection window, without deciding its morphology."""
from __future__ import annotations

import numpy as np

from gnss_research.features import Evidence


def locate_step(evidence: Evidence, window: dict, minimum_side: int = 5) -> dict:
    """Fit a common local trend plus a level jump on station-standardized residuals.

    Each observed, adjacent day pair is a possible onset. Five observed samples per
    side prevent an edge fit from explaining one isolated point. This is a declared
    sample-support rule, not a physical alarm threshold. Visual review decides whether
    a persistent step exists; the lowest loss alone must never emit an anomaly.
    """
    left, right = window["start"], window["end"]
    days = np.arange(left, right + 1)
    days = days[evidence.observed[left:right + 1]]
    result = {"window_id": window["id"], "start": left, "end": right,
              "minimum_observed_per_side": minimum_side, "index": None}
    if len(days) < 2 * minimum_side:
        return result
    values = evidence.residual[days] / evidence.noise
    time = (days - days.mean()) / max(1, right - left)
    background_design = np.column_stack([np.ones(len(days)), time])
    background_fit = np.linalg.lstsq(background_design, values, rcond=None)[0]
    background_loss = float(np.square(values - background_design @ background_fit).sum())
    best_loss = np.inf
    for index in days:
        if index == 0 or not evidence.observed[index - 1]:
            continue
        before = int((days < index).sum())
        after = len(days) - before
        if min(before, after) < minimum_side:
            continue
        design = np.column_stack([background_design, days >= index])
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        loss = float(np.square(values - design @ coefficients).sum())
        if loss < best_loss:
            best_loss = loss
            result.update(index=int(index), standardized_jump=coefficients[-1].tolist(),
                          observed_before=before, observed_after=after,
                          residual_sum_squares=loss, no_jump_sum_squares=background_loss)
    return result
