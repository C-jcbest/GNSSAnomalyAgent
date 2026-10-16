"""Compare continuous deformation with a background that also allows an offset.

The evidence is a descriptive, calibrated input, never a geological attribution,
probability, activity boundary, or replacement for the original displacement.
"""

from __future__ import annotations

import numpy as np

from gnss_sim.landslide_observation_view import raw_block_summaries


def huber_loss(residual, delta=1.5):
    absolute = np.abs(residual)
    return np.where(absolute <= delta, residual**2 / 2, delta * (absolute - delta / 2))


def fit_candidates(values, noise, designs, metadata):
    """Fit equal-noise NEU explanations with bounded, axis-specific Huber IRLS.

    All candidates in a batch have the same coefficient count. Block medians
    resist isolated spikes; Huber fitting limits the remaining block influence.
    The complexity penalty includes axis coefficients and searched knot counts.
    """
    if not designs:
        return None
    design = np.asarray(designs, dtype=float)
    identifiable = np.linalg.matrix_rank(design) == design.shape[-1]
    if not identifiable.any():
        return None
    indices = np.flatnonzero(identifiable)
    design = design[indices]
    normalized = values / noise
    coefficients = np.einsum("cpn,na->cpa", np.linalg.pinv(design), normalized)
    iterations = 0
    # This is a finite numerical solver, not an agent observation loop.
    for iterations in range(1, 21):
        residual = normalized[None] - np.einsum("cnp,cpa->cna", design, coefficients)
        weights = np.minimum(1.0, 1.5 / np.maximum(np.abs(residual), 1e-12))
        gram = np.einsum("cnp,cna,cnq->capq", design, weights, design)
        right = np.einsum("cnp,cna,na->cap", design, weights, normalized)
        updated = np.linalg.solve(gram, right[..., None])[..., 0].transpose(0, 2, 1)
        change = np.max(np.abs(updated - coefficients), axis=(1, 2))
        coefficients = updated
        if np.all(change <= 1e-5):
            break
    predicted = np.einsum("cnp,cpa->cna", design, coefficients) * noise
    costs = np.sum(huber_loss((values[None] - predicted) / noise), axis=(1, 2))
    for index, source in enumerate(indices):
        parameters = 3 * design.shape[-1] + metadata[source]["searched_knots"]
        costs[index] += parameters * np.log(values.size)
    selected = int(np.argmin(costs))
    return {
        **metadata[indices[selected]],
        "penalized_cost": float(costs[selected]),
        "coefficients_mm": (coefficients[selected] * noise).tolist(),
        "fitted_medians_mm": predicted[selected].tolist(),
        "irls_iterations": iterations,
        "selected_coefficient_change": float(change[selected]),
    }


def best_explanation(fits):
    available = [fit for fit in fits if fit is not None]
    if not available:
        raise ValueError("No identifiable observation explanation")
    return min(available, key=lambda fit: fit["penalized_cost"])


def nuisance_evidence(values, noise, start, stop, query_day):
    """Compare H0=constant(+step), H1=constant(+step)+continuous ramp.

    Ramp intervals must contain the query date. Their fitted endpoints are only
    local model parameters; they do not become exported activity boundaries.
    The exclusive-rivals ablation excludes a step from H1, but retains it in H0.
    """
    values = np.asarray(values, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Evidence requires three raw displacement components")
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Evidence requires three positive observation-noise proxies")
    if not 0 <= start <= query_day < stop <= len(values):
        raise ValueError("Evidence and query must use the original calendar")
    result = {"start": start, "stop": stop, "query_day": query_day, "status": "unavailable"}
    observed = np.isfinite(values[start:stop]).all(axis=1)
    valid_edges = np.flatnonzero(np.r_[True, observed, True])
    if np.max(np.diff(valid_edges) - 1) > 7:
        return {**result, "reason": "More than seven consecutive missing dates"}
    blocks = raw_block_summaries(values, start, stop)
    if len(blocks) < 6:
        return {**result, "reason": "Fewer than six supported 14-day observation blocks"}
    times = np.array([(row["start"] + row["stop"] - 1) / 2 for row in blocks])
    medians = np.array([row["median_mm"] for row in blocks])
    ones = np.ones(len(times))
    constant = fit_candidates(medians, noise, [ones[:, None]], [{"kind": "constant", "searched_knots": 0}])
    steps = []
    step_metadata = []
    for knot in (times[:-1] + times[1:]) / 2:
        steps.append((times >= knot).astype(float))
        step_metadata.append({"kind": "step", "step_day": float(knot), "searched_knots": 1})
    offset = fit_candidates(medians, noise, [np.column_stack((ones, step)) for step in steps], step_metadata)
    ramp_designs = []
    ramp_metadata = []
    joint_designs = []
    joint_metadata = []
    for onset in times:
        for offset_day in times:
            if not onset <= query_day < offset_day or offset_day - onset < 28:
                continue
            ramp = np.clip((times - onset) / (offset_day - onset), 0, 1)
            metadata = {"kind": "ramp", "ramp_start": float(onset), "ramp_stop": float(offset_day), "searched_knots": 2}
            ramp_designs.append(np.column_stack((ones, ramp)))
            ramp_metadata.append(metadata)
            for step, step_info in zip(steps, step_metadata, strict=True):
                joint_designs.append(np.column_stack((ones, step, ramp)))
                joint_metadata.append({**metadata, "kind": "step_and_ramp", "step_day": step_info["step_day"], "searched_knots": 3})
    if not ramp_designs:
        return {**result, "reason": "No supported continuous ramp containing the query"}
    ramp = fit_candidates(medians, noise, ramp_designs, ramp_metadata)
    joint = fit_candidates(medians, noise, joint_designs, joint_metadata)
    null = best_explanation([constant, offset])
    deformation = best_explanation([ramp, joint])
    scale = float(medians.size)
    return {
        **result, "status": "available", "blocks": blocks, "noise_proxy_mm": noise.tolist(),
        "null": null, "deformation": deformation, "exclusive_deformation": ramp,
        "deformation_evidence": (null["penalized_cost"] - deformation["penalized_cost"]) / scale,
        "exclusive_evidence": (null["penalized_cost"] - ramp["penalized_cost"]) / scale,
        "interpretation": "Descriptive penalized Huber fit; shared observational offset; no probability, causal attribution or activity-boundary assertion",
    }


def record_nuisance_evidence(values, noise):
    """Evaluate once per 14-day target using a bounded 168-day context.

    Unavailable evidence remains NaN. Broadcast only within the original target
    block; individual missing observations always retain unavailable evidence.
    Existing downstream observation/activity support is not expanded.
    """
    values = np.asarray(values, dtype=float)
    matrix = np.full((len(values), 2), np.nan)
    diagnostics = []
    for target_start in range(0, len(values), 14):
        target_stop = min(target_start + 14, len(values))
        center = (target_start + target_stop - 1) // 2
        start = max(0, target_start - 70)
        stop = min(len(values), target_start + 98)
        diagnostic = nuisance_evidence(values, noise, start, stop, center)
        diagnostics.append({"target_start": target_start, "target_stop": target_stop, **diagnostic})
        if diagnostic["status"] == "available":
            matrix[target_start:target_stop] = [diagnostic["deformation_evidence"], diagnostic["exclusive_evidence"]]
    matrix[~np.isfinite(values).all(axis=1)] = np.nan
    return matrix, diagnostics


def render_nuisance_evidence(values, diagnostic, path):
    """Show original raw gaps and joint explanations without activity labels."""
    import matplotlib.pyplot as plt

    if diagnostic["status"] != "available":
        raise ValueError("Cannot render an unavailable evidence fit")
    times = np.array([(row["start"] + row["stop"] - 1) / 2 for row in diagnostic["blocks"]])
    start, stop = diagnostic["start"], diagnostic["stop"]
    calendar = np.arange(start, stop)
    figure, panels = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis, panel in enumerate(panels):
        panel.plot(calendar, values[start:stop, axis], color="#78909c", linewidth=.7, label="Original observations")
        panel.scatter(times, [row["median_mm"][axis] for row in diagnostic["blocks"]], color="#263238", s=12, label="Raw block medians")
        for hypothesis, color in (("null", "#ba5b37"), ("deformation", "#167968")):
            fit = diagnostic[hypothesis]
            coefficients = np.array(fit["coefficients_mm"])
            curve = np.full(len(calendar), coefficients[0, axis])
            column = 1
            if fit["kind"] in {"step", "step_and_ramp"}:
                curve += coefficients[column, axis] * (calendar >= fit["step_day"])
                column += 1
            if fit["kind"] in {"ramp", "step_and_ramp"}:
                curve += coefficients[column, axis] * np.clip((calendar - fit["ramp_start"]) / (fit["ramp_stop"] - fit["ramp_start"]), 0, 1)
            curve[~np.isfinite(values[start:stop, axis])] = np.nan
            panel.plot(calendar, curve, color=color, label=f"{hypothesis}: {fit['kind']}")
        panel.axvspan(diagnostic["target_start"], diagnostic["target_stop"] - 1, color="#b7d4ed", alpha=.25)
        panel.set_ylabel(("N", "E", "U")[axis] + " (mm)")
        panel.grid(alpha=.2)
    panels[0].legend(fontsize=8, ncol=2)
    panels[-1].set_xlabel("Original day index; shaded query block; fits are explanations, not labels")
    figure.suptitle(f"Shared nuisance evidence = {diagnostic['deformation_evidence']:.3f}; exclusive rivals = {diagnostic['exclusive_evidence']:.3f}")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
