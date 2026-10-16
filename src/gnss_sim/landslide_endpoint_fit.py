"""Repair proposed activity endpoints only when raw transition fits beat rivals."""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide_displacement_comparison_view import raw_block_comparisons
from gnss_sim.landslide_evaluation import event_masks, runs
from gnss_sim.landslide_motion_competition import best_model_fit


def endpoint_observation(case, noise, activity_start, activity_stop, boundary, side, limits):
    """Fit a calendar-preserving local stationary/motion transition.

    The proposed event supplies topology, not a numerical class label. Other
    proposed events bound the evidence window; the opposite endpoint truncates
    the moving shoulder so it cannot include a second transition.
    """
    length = len(case.displacement_mm)
    if side not in {"onset", "offset"}:
        raise ValueError("Endpoint side must be onset or offset")
    if boundary in {0, length}:
        return {"status": "unavailable", "reason": "Record-censored endpoint"}
    lower, upper = limits
    start = max(lower, boundary - 98)
    stop = min(upper, boundary + 98)
    if side == "onset":
        stop = min(stop, activity_stop)
    else:
        start = max(start, activity_start)
    if stop <= start:
        return {"status": "unavailable", "reason": "No local endpoint observations"}
    values = np.asarray(
        [row if row is not None else [np.nan] * 3 for row in case.displacement_mm],
        dtype=float,
    )
    complete = np.isfinite(values[start:stop]).all(axis=1)
    observed_edges = np.flatnonzero(np.r_[True, complete, True])
    if np.max(np.diff(observed_edges) - 1) > 7:
        return {"status": "unavailable", "reason": "Long raw gap in endpoint evidence"}
    facts = raw_block_comparisons(case, noise, start, stop)
    supported = np.isfinite(facts["medians"]).all(axis=1)
    times = (facts["starts"][supported] + facts["stops"][supported] - 1) / 2
    medians = facts["medians"][supported]
    if len(times) < 6:
        return {"status": "unavailable", "reason": "Fewer than six supported raw blocks"}
    candidates = []
    for knot in range(max(lower, boundary - 49), min(upper, boundary + 49) + 1):
        if side == "onset" and knot > activity_stop - 28:
            continue
        if side == "offset" and knot < activity_start + 28:
            continue
        # Two blocks on each side make a shoulder observable. This is support,
        # not a minimum velocity or a confidence interval.
        if (times < knot).sum() < 2 or (times >= knot).sum() < 2:
            continue
        candidates.append(knot)
    if not candidates:
        return {"status": "unavailable", "reason": "No two-sided supported search knot"}
    transition = []
    steps = []
    for knot in candidates:
        shape = np.maximum(times - knot, 0) if side == "onset" else np.minimum(times - knot, 0)
        transition.append({"values": shape, "start": float(knot), "stop": None})
        steps.append({"values": (times >= knot).astype(float), "start": float(knot), "stop": None})
    templates = {
        "transition": transition,
        "stationary": [{"values": np.zeros(len(times)), "start": None, "stop": None}],
        "linear_drift": [{"values": times - times[0], "start": float(times[0]), "stop": None}],
        "single_step": steps,
    }
    full = np.ones(len(times), dtype=bool)
    full_fit = best_model_fit(medians, transition, full, noise)
    folds = []
    for parity in (0, 1):
        training = np.arange(len(times)) % 2 == parity
        held_out = ~training
        errors, knots = {}, {}
        for name, models in templates.items():
            fitted = best_model_fit(medians, models, training, noise)
            if fitted is None:
                return {"status": "unavailable", "reason": "Unidentifiable alternating fit"}
            prediction = np.asarray(fitted["fitted_medians_mm"])
            errors[name] = float(np.mean(((medians[held_out] - prediction[held_out]) / noise) ** 2))
            knots[name] = fitted["start"]
        folds.append({"training_parity": parity, "held_out_mse": errors, "knots": knots})
    selected = int(full_fit["start"])
    fitted_knots = [selected] + [int(fold["knots"]["transition"]) for fold in folds]
    return {
        "status": "available",
        "side": side,
        "original_boundary": boundary,
        "selected_boundary": selected,
        "evidence_start": start,
        "evidence_stop": stop,
        "block_times": times.tolist(),
        "block_medians_mm": medians.tolist(),
        "full_transition_fit": full_fit,
        "folds": folds,
        "knot_span_days": max(fitted_knots) - min(fitted_knots),
        "interpretation": "Same raw displacement; descriptive validation, not independent blocks, calibrated confidence, or geological attribution",
    }


def accept_endpoint(diagnostic, rule):
    if rule not in {"selective", "no_step_rival", "unconditional_fit"}:
        raise ValueError("Unknown endpoint repair rule")
    if diagnostic["status"] != "available":
        return False
    if rule == "unconditional_fit":
        return True
    if diagnostic["knot_span_days"] > 14:
        return False
    rivals = ["stationary", "linear_drift"]
    if rule == "selective":
        rivals.append("single_step")
    for fold in diagnostic["folds"]:
        errors = fold["held_out_mse"]
        for rival in rivals:
            # Arithmetic ties do not establish new transition evidence.
            tolerance = 1e-12 * max(1, errors[rival], errors["transition"])
            if errors["transition"] >= errors[rival] - tolerance:
                return False
    return True


def fit_activity_endpoints(case, noise, activity):
    """Fit the same raw observations once for all acceptance-rule comparisons."""
    activity = np.asarray(activity)
    observed = np.asarray([row is not None for row in case.displacement_mm])
    if activity.shape != observed.shape or not np.isin(activity, [-1, 0, 1]).all():
        raise ValueError("Endpoint repair requires aligned activity/unknown states")
    if np.any(activity[~observed] != -1):
        raise ValueError("Missing raw dates must remain unknown")
    events = event_masks(activity)
    extents = [(int(np.flatnonzero(mask)[0]), int(np.flatnonzero(mask)[-1]) + 1) for mask in events]
    limits = [[0, len(activity)] for _ in extents]
    for index in range(len(extents) - 1):
        stop, next_start = extents[index][1], extents[index + 1][0]
        stationary = np.flatnonzero(activity[stop:next_start] == 0) + stop
        if len(stationary):
            # Both events share a fixed observed separator. Independent outward
            # repairs must not consume the same gap and merge event identities.
            midpoint = (stop + next_start) / 2
            separator = int(stationary[np.argmin(np.abs(stationary - midpoint))])
            limits[index][1] = separator
            limits[index + 1][0] = separator + 1
        else:
            limits[index][1] = stop
            limits[index + 1][0] = next_start
    long_unknown = [(start, stop) for start, stop in runs(activity == -1) if stop - start > 7]
    fits = []
    for index, (start, stop) in enumerate(extents):
        lower, upper = limits[index]
        for gap_start, gap_stop in long_unknown:
            if gap_stop <= start:
                lower = max(lower, gap_stop)
            elif gap_start >= stop:
                upper = min(upper, gap_start)
        endpoints = {}
        for side, boundary in (("onset", start), ("offset", stop)):
            diagnostic = endpoint_observation(
                case, noise, start, stop, boundary, side, (lower, upper)
            )
            endpoints[side] = {"side": side, "original_boundary": boundary, **diagnostic}
        fits.append({"event_index": index, "start": start, "stop": stop, "endpoints": endpoints})
    return fits


def apply_endpoint_fits(activity, observed, fits, rule):
    """Keep event identity and all unknown dates; edit only endpoint bands."""
    activity = np.asarray(activity)
    observed = np.asarray(observed)
    if activity.shape != observed.shape or observed.dtype != bool:
        raise ValueError("Endpoint fits require the original boolean observation calendar")
    if not np.isin(activity, [-1, 0, 1]).all() or np.any(activity[~observed] != -1):
        raise ValueError("Endpoint fits require valid activity and raw missing states")
    repaired = activity.copy()
    diagnostics = []
    for fitted in fits:
        start, stop = fitted["start"], fitted["stop"]
        proposals = {}
        for side, diagnostic in fitted["endpoints"].items():
            accepted = accept_endpoint(diagnostic, rule)
            proposals[side] = (
                diagnostic["selected_boundary"] if accepted else diagnostic["original_boundary"]
            )
            diagnostics.append(
                {"event_index": fitted["event_index"], "accepted": accepted, **diagnostic}
            )
        new_start, new_stop = proposals["onset"], proposals["offset"]
        # Independently fitted endpoints may cross on a short event. In that
        # case the observations do not support a coherent repair of this event.
        if new_stop - new_start < 28:
            for diagnostic in diagnostics[-2:]:
                diagnostic["accepted"] = False
                diagnostic["joint_rejection"] = "Repaired event shorter than two blocks"
            continue
        if new_start < start:
            changed = repaired[new_start:start]
            changed[changed == 0] = 1
        elif new_start > start:
            changed = repaired[start:new_start]
            changed[changed == 1] = 0
        if new_stop > stop:
            changed = repaired[stop:new_stop]
            changed[changed == 0] = 1
        elif new_stop < stop:
            changed = repaired[new_stop:stop]
            changed[changed == 1] = 0
    repaired[~observed] = -1
    return repaired, diagnostics
