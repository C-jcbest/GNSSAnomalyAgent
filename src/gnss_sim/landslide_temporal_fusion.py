"""Combine numerical motion scores and visual intervals before temporal decoding."""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide_evaluation import runs


def decode_activity_scores(scores, available, transition_cost):
    """Minimize unary evidence plus a cost per stationary/activity transition.

    Missing or jointly unsupported dates split the sequence. A tie between
    globally optimal states stays unknown; no confidence threshold is inferred.
    Scores are evidence weights, not calibrated posterior probabilities.
    """
    scores = np.asarray(scores, dtype=float)
    available = np.asarray(available)
    if scores.ndim != 1 or available.shape != scores.shape or available.dtype != bool:
        raise ValueError("Temporal evidence requires an aligned boolean calendar")
    if not np.isfinite(scores).all() or not np.isfinite(transition_cost) or transition_cost < 0:
        raise ValueError("Temporal costs must be finite and nonnegative")
    activity = np.full(len(scores), -1, dtype=int)
    switching = np.array([[0.0, transition_cost], [transition_cost, 0.0]])
    for start, stop in runs(available):
        evidence = scores[start:stop]
        unary = np.column_stack((evidence / 2, -evidence / 2))
        forward = np.empty_like(unary)
        backward = np.empty_like(unary)
        forward[0] = unary[0]
        for day in range(1, len(unary)):
            forward[day] = unary[day] + (forward[day - 1, :, None] + switching).min(axis=0)
        backward[-1] = 0
        for day in range(len(unary) - 2, -1, -1):
            backward[day] = (switching + unary[day + 1] + backward[day + 1]).min(axis=1)
        marginal_cost = forward + backward
        tied = np.isclose(marginal_cost[:, 0], marginal_cost[:, 1], rtol=0, atol=1e-12)
        result = marginal_cost.argmin(axis=1)
        result[tied] = -1
        activity[start:stop] = result
    return activity


def fused_motion_scores(
    probability, numeric_supported, visual_activity, observed, threshold, visual_weight
):
    """Keep VLM states as signed votes, without pretending they are probabilities."""
    probability = np.asarray(probability, dtype=float)
    visual_activity = np.asarray(visual_activity)
    observed = np.asarray(observed)
    numeric_supported = np.asarray(numeric_supported)
    if observed.ndim != 1 or observed.dtype != bool or numeric_supported.dtype != bool:
        raise ValueError("Fusion requires boolean observation and numerical support masks")
    if any(
        values.shape != observed.shape
        for values in (probability, visual_activity, numeric_supported)
    ):
        raise ValueError("Numerical and visual evidence must share the original calendar")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Numerical evidence requires finite probabilities in [0,1]")
    if not np.isin(visual_activity, [-1, 0, 1]).all() or np.any(visual_activity[~observed] != -1):
        raise ValueError("Missing dates must not have a visual activity decision")
    if not 0 < threshold < 1 or not np.isfinite(visual_weight) or visual_weight < 0:
        raise ValueError("Fusion needs a valid numerical threshold and nonnegative visual weight")
    usable_numeric = observed & numeric_supported
    usable_visual = observed & (visual_activity >= 0) & (visual_weight > 0)
    bounded = np.clip(probability, 1e-6, 1 - 1e-6)
    numeric_score = np.log(bounded / (1 - bounded)) - np.log(threshold / (1 - threshold))
    scores = np.where(usable_numeric, numeric_score, 0.0)
    scores += np.where(usable_visual, visual_weight * (2 * visual_activity - 1), 0.0)
    return scores, usable_numeric | usable_visual
