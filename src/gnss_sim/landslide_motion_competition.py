"""Compare stationary, single-step and continuous-ramp raw displacement models.

These are competing descriptions of the same observations, not independent
measurements or activity labels. Alternating-block prediction error exposes
overfitting without treating correlated blocks as an independent statistical test.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from gnss_sim.landslide_observation_view import raw_block_summaries

MODEL_NAMES = ("stationary", "single_step", "continuous_ramp")


def model_templates(times):
    """Use observation-block anchors and the existing 14-day block duration."""
    templates = {"stationary": [{"values": np.zeros(len(times)), "start": None, "stop": None}]}
    cuts = (times[:-1] + times[1:]) / 2
    templates["single_step"] = [{"values": (times >= cut).astype(float), "start": float(cut), "stop": None}
                                for cut in cuts]
    templates["continuous_ramp"] = []
    for start in times:
        for stop in times:
            # A single abrupt change is represented by the step competitor.
            # A ramp needs at least two complete block widths of accumulation.
            if stop - start < 28:
                continue
            fraction = np.clip((times - start) / (stop - start), 0, 1)
            templates["continuous_ramp"].append({"values": fraction,
                "start": float(start), "stop": float(stop)})
    return templates


def fit_template(values, template, training, noise):
    design = np.column_stack((np.ones(len(values)), template["values"]))
    if template["start"] is None:
        intercept = np.mean(values[training], axis=0)
        coefficients = np.vstack((intercept, np.zeros(3)))
    else:
        if np.linalg.matrix_rank(design[training]) < 2:
            return None
        coefficients = np.linalg.lstsq(design[training], values[training], rcond=None)[0]
    predicted = design @ coefficients
    residual = (values[training] - predicted[training]) / noise
    return {"start": template["start"], "stop": template["stop"],
            "offset_mm": coefficients[0].tolist(), "net_change_mm": coefficients[1].tolist(),
            "fitted_medians_mm": predicted.tolist(), "training_mse_noise_units": float(np.mean(residual ** 2))}


def best_model_fit(values, candidates, training, noise):
    best = None
    for candidate in candidates:
        fitted = fit_template(values, candidate, training, noise)
        if fitted is None:
            continue
        if best is None or fitted["training_mse_noise_units"] < best["training_mse_noise_units"]:
            best = fitted
    return best


def motion_competition(case, noise, start, stop):
    """Return a finite diagnostic; never infer an activity interval or stage."""
    values = np.asarray([row if row is not None else [np.nan] * 3
                         for row in case.displacement_mm], dtype=float)
    noise = np.asarray(noise, dtype=float)
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Competition requires three positive observation-noise proxies")
    blocks = raw_block_summaries(values, start, stop)
    result = {"start": start, "stop": stop, "blocks": blocks,
        "noise_proxy_mm": noise.tolist(), "models": {}, "status": "unavailable",
        "interpretation": "Same raw observations. Model nodes are descriptive, never activity boundaries. "
                          "Block errors are not independent tests, probabilities or calibrated uncertainties. "
                          "Hiding blocks can make a step location ambiguous: held-out error alone cannot "
                          "identify the physical cause or rule out an instrument step."}
    # Each alternating fold must have at least three supported raw block medians.
    if len(blocks) < 6:
        result["unavailable_reason"] = "Fewer than six supported complete raw blocks"
        return result
    times = np.array([(row["start"] + row["stop"] - 1) / 2 for row in blocks])
    medians = np.array([row["median_mm"] for row in blocks])
    templates = model_templates(times)
    full = np.ones(len(times), dtype=bool)
    for name in MODEL_NAMES:
        fitted = best_model_fit(medians, templates[name], full, noise)
        if fitted is None:
            result["unavailable_reason"] = f"No identifiable {name} competitor"
            return result
        folds = []
        errors = []
        for parity in (0, 1):
            training = np.arange(len(times)) % 2 == parity
            held_out = ~training
            fold = best_model_fit(medians, templates[name], training, noise)
            if fold is None:
                result["unavailable_reason"] = f"Unidentifiable alternating-block {name} fit"
                return result
            predicted = np.array(fold["fitted_medians_mm"])
            squared = ((medians[held_out] - predicted[held_out]) / noise) ** 2
            errors.extend(squared.tolist())
            folds.append({"training_block_indices": np.flatnonzero(training).tolist(),
                "held_out_block_indices": np.flatnonzero(held_out).tolist(),
                "fitted_start": fold["start"], "fitted_stop": fold["stop"],
                "held_out_rmse_noise_units": float(np.sqrt(np.mean(squared)))})
        fitted["alternating_block_rmse_noise_units"] = float(np.sqrt(np.mean(errors)))
        fitted["folds"] = folds
        result["models"][name] = fitted
    result["status"] = "available"
    return result


def render_competition(case, diagnostic, show_errors, path):
    """Keep raw pixels and candidate curves identical within the feedback pair."""
    start, stop = diagnostic["start"], diagnostic["stop"]
    values = np.asarray([row if row is not None else [np.nan] * 3
                         for row in case.displacement_mm], dtype=float)
    fig, panels = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = {"stationary": "#667788", "single_step": "#b66a32", "continuous_ramp": "#773c9a"}
    block_times = np.array([(row["start"] + row["stop"] - 1) / 2 for row in diagnostic["blocks"]])
    for component, name in enumerate(("N", "E", "U")):
        panel = panels[component]
        panel.plot(np.arange(start, stop), values[start:stop, component], color="#24887b",
                   linewidth=.65, alpha=.65, label="Raw observations")
        for model, fitted in diagnostic["models"].items():
            # Curves describe models only. Missing raw dates remain gaps, including
            # in the fitted overlay, and no prediction is extrapolated beyond blocks.
            if not len(block_times):
                continue
            times = np.arange(int(np.ceil(block_times[0])), int(np.floor(block_times[-1])) + 1)
            if model == "stationary":
                fraction = np.zeros(len(times))
            elif model == "single_step":
                fraction = (times >= fitted["start"]).astype(float)
            else:
                fraction = np.clip((times - fitted["start"]) / (fitted["stop"] - fitted["start"]), 0, 1)
            curve = fitted["offset_mm"][component] + fraction * fitted["net_change_mm"][component]
            curve[~np.isfinite(values[times, component])] = np.nan
            panel.plot(times, curve, color=colors[model], linewidth=1.6,
                       linestyle="--", label=model.replace("_", " "))
        panel.set_ylabel(name + " displacement (mm)")
        panel.grid(alpha=.2)
        panel.set_xlim(start, stop - 1)
    panels[0].legend(loc="best", fontsize=8, ncol=2)
    panels[-1].set_xticks(np.unique([start, start + 45, start + 90, start + 135, stop - 1]))
    panels[-1].set_xlabel("Original day index; dashed curves are competing fits, never labels")
    fig.suptitle(f"{case.case_id} | Same observed target {start}..{stop - 1} | Rival motion descriptions")
    fig.subplots_adjust(top=.91, bottom=.22, left=.10, right=.97, hspace=.14)
    if show_errors and diagnostic["status"] == "available":
        rows = [[name.replace("_", " "),
                 f"{np.sqrt(diagnostic['models'][name]['training_mse_noise_units']):.3f}",
                 f"{diagnostic['models'][name]['alternating_block_rmse_noise_units']:.3f}"] for name in MODEL_NAMES]
        table = fig.add_axes([.12, .01, .77, .08])
        table.axis("off")
        table.table(cellText=rows, colLabels=["Rival description", "All-block fit RMSE", "Held-out-block RMSE"],
                    bbox=[0, 0, 1, 1], cellLoc="center")
        fig.text(.12, .145, "Errors / raw daily-difference MAD noise proxy; smaller = closer prediction of block medians.", fontsize=8)
        fig.text(.12, .125, "Held-out step location can be ambiguous; error ranking does not identify a physical cause.", fontsize=8)
        fig.text(.12, .105, "Blocks remain correlated; errors are not significance, probabilities or activity labels.", fontsize=8)
    elif diagnostic["status"] != "available":
        fig.text(.12, .1, "Rival-fit evidence unavailable; insufficient supported raw blocks.", fontsize=9)
    fig.savefig(path, dpi=120)
    plt.close(fig)
