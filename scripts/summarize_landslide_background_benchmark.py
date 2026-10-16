"""Verify frozen background results and export complete-record diagnostics."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_background_benchmark import verify_data
from analyze_landslide_observation_probe import csv_rows
from fit_landslide_raw_persistence_baselines import raw_gate
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, load_labels, predict_baselines, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import STAGES
from gnss_sim.landslide_local_experiment import learned_stages


def load_head(source, name):
    model = XGBClassifier()
    model.load_model(source / name)
    return model


def recompute_numerical(source, records):
    """Use this benchmark's fitted models, without retraining or old-model reuse."""
    head = load_head(source, "stage-xgboost.json")
    features = {cid: pair[1] for cid, pair in records.items()}
    predictions = predict_baselines(features, list(records), read_json(source / "frozen-selection.json"),
                                    load_head(source, "activity-xgboost.json"), head)
    predictions.pop("ungated_rule")
    choices = read_json(source / "long-selection.json")
    models = {name: load_head(source, f"{name}.json") for name in choices if name.startswith("xgboost")}
    for cid, (case, feature) in records.items():
        h42 = raw_motion_evidence(case, feature)
        h84 = raw_motion_evidence(case, feature, 84)
        support = feature.observed & (np.isfinite(feature.progressive_rate)
            | np.isfinite(h42["rate"]) | np.isfinite(h84["rate"]))
        matrix42 = np.column_stack((feature.matrix, h42["rate"], h42["signal_to_noise"], h42["direction"]))
        matrix84 = np.column_stack((matrix42, h84["rate"], h84["signal_to_noise"], h84["direction"]))
        codes = head.predict(feature.matrix).astype(int) + 1
        for name, choice in choices.items():
            if name == "raw_multiscale_rule":
                gate = raw_gate(feature, h42, choice["rate"], choice["snr"])
            elif name == "raw_rule_84":
                positive = (feature.progressive_rate >= choice["rate"]) & (feature.signal_to_noise >= choice["snr"])
                positive |= (h42["rate"] >= choice["rate"]) & (h42["signal_to_noise"] >= choice["snr"])
                positive |= (h84["rate"] >= choice["rate"]) & (h84["signal_to_noise"] >= choice["snr"])
                gate = activity_from_mask(positive, support)
            else:
                matrix = matrix84 if name == "xgboost_84" else matrix42
                eligible = feature.supported if name == "xgboost_raw_persistence" else support
                gate = activity_from_mask(models[name].predict_proba(matrix)[:, 1] >= choice["threshold"], eligible)
            predictions.setdefault(name, {})[cid] = learned_stages(codes, gate, feature.supported)
    return {name: {cid: value.to_dict() for cid, value in cases.items()} for name, cases in predictions.items()}


def render_record(case, truth, reference, predictions, path):
    from matplotlib.colors import ListedColormap

    methods = [name for rep in (0, 1) for name in (
        f"direct_visual_r{rep}", f"fixed_image_review_r{rep}__conflict_scope",
        f"adaptive_no_feedback_r{rep}__whole_target", f"adaptive_no_feedback_r{rep}__conflict_scope",
        f"adaptive_numeric_feedback_r{rep}__conflict_scope")]
    methods += ["raw_multiscale_rule", "xgboost_42_matched_support", "xgboost_84"]
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    activity = np.array([int(row["activity_label"]) for row in reference])
    stage = np.array([STAGES[row["feature"]] if int(row["activity_label"]) == 1
                      else int(row["activity_label"]) for row in reference])
    matrix = np.array([activity] + [predictions[name][case.case_id]["activity"] for name in methods])
    fig, panels = plt.subplots(5, 1, figsize=(13, 11), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1, .35, 3]})
    for component, name in enumerate(("N", "E", "U")):
        panels[component].plot(values[:, component], color="#24887b", linewidth=.8)
        panels[component].set_ylabel(name + " (mm)")
        panels[component].grid(alpha=.2)
    extent = [-.5, len(activity) - .5, .5, -.5]
    panels[3].imshow(stage[None, :], aspect="auto", interpolation="nearest", vmin=-1, vmax=3,
        cmap=ListedColormap(["#c5c9cc", "#fff4dc", "#c9584f", "#377dab", "#56a477"]), extent=extent)
    panels[3].set_yticks([0], ["Injected ASD"], fontsize=8)
    panels[4].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
        cmap=ListedColormap(["#c5c9cc", "#fff4dc", "#377dab"]),
        extent=[-.5, len(activity) - .5, len(matrix) - .5, -.5])
    panels[4].set_yticks(range(len(matrix)), ["Injected activity"] + methods, fontsize=8)
    panels[4].set_xlabel("Original day index | activity blue; quiet cream; missing/unknown gray | ASD red/blue/green")
    panels[4].set_xlim(0, len(activity) - 1)
    fig.suptitle(f"{case.case_id} | {truth['family']} | P227 background {truth['background_group']}\n"
                 "Known injection recovery; residual motion is not field ground truth")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summarize(run, analysis, out):
    verify_freeze(run)
    extra = read_json(run / "extra-source-freeze.json")
    for name, expected in extra.items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Extra source binding changed: {name}")
        path = Path(name)
        if "extra-source" in path.parts:
            relative = Path(*path.parts[path.parts.index("extra-source") + 1:])
            if sha(ROOT / relative) != expected:
                raise ValueError(f"Live source differs from extra snapshot: {relative}")
    for name, expected in read_json(analysis / "manifest.json")["files"].items():
        if sha(analysis / name) != expected:
            raise ValueError(f"Scored output changed: {name}")
    config = read_json(run / "protocol.json")
    source = ROOT / config["source_bundle"]
    verify_data(source)
    records = {cid: load_case_features(run, cid) for cid in config["case_ids"]}
    labels = load_labels(analysis / "reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    predictions = read_json(analysis / "predictions.json")
    results = read_json(analysis / "results.json")
    started = time.monotonic()
    numeric = recompute_numerical(source, records)
    numeric_seconds = time.monotonic() - started
    for name, cases in numeric.items():
        for cid, value in cases.items():
            if value != predictions[name][cid]:
                raise ValueError(f"Frozen numerical recomputation differs: {name} {cid}")
    weak_checks, changes, denominators = {}, [], {"known_days": 0, "active_days": 0, "quiet_days": 0,
                                                 "weak_days": 0, "stage_ASD": [0, 0, 0]}
    for rows in labels.values():
        for row in rows:
            actual = int(row["activity_label"])
            denominators["known_days"] += int(actual >= 0)
            denominators["active_days"] += int(actual == 1)
            denominators["quiet_days"] += int(actual == 0)
            denominators["weak_days"] += int(row["weak_signal"])
            if actual == 1:
                denominators["stage_ASD"][STAGES[row["feature"]] - 1] += 1
    for name, cases in predictions.items():
        total, detected = 0, 0
        for cid, value in cases.items():
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
            if np.any(weak & (actual != 1)):
                raise ValueError("Weak reference includes nonactive or missing dates")
            counts = int(weak.sum()), int(np.count_nonzero(weak & (np.array(value["activity"]) == 1)))
            measured = results[name]["per_case"][cid]
            if counts != (measured["weak_days"], measured["weak_detected_days"]):
                raise ValueError("Independent weak counts differ")
            total += counts[0]
            detected += counts[1]
        aggregate = results[name]["aggregate"]
        if (total, detected) != (aggregate["weak_days"], aggregate["weak_detected_days"]):
            raise ValueError("Independent weak population differs")
        weak_checks[name] = {"reference_days": total, "detected_days": detected, "recall": detected / total}
    selections = read_json(run / "selections.json")
    for rep in (0, 1):
        candidate = f"adaptive_no_feedback_r{rep}__conflict_scope"
        controls = [f"direct_visual_r{rep}", f"fixed_image_review_r{rep}__conflict_scope",
                    f"adaptive_no_feedback_r{rep}__whole_target",
                    f"adaptive_no_feedback_r{rep}__matched_center_scope",
                    f"adaptive_numeric_feedback_r{rep}__conflict_scope"]
        for cid, (case, feature) in records.items():
            initial = np.array(predictions[f"direct_visual_r{rep}"][cid]["activity"])
            _, neighbourhood = conflict_neighbourhood(initial, feature, raw_motion_evidence(case, feature), ReviewPolicy())
            windows = regular_windows(len(initial), 180, 90)
            selected = selections[f"{cid}-r{rep}"]
            for arm in ("fixed_image_review", "adaptive_no_feedback", "adaptive_numeric_feedback"):
                targets = ([{"start": windows[i][0], "stop": windows[i][1], "index": i}
                            for i in selected["fixed_indices"]] if arm == "fixed_image_review" else selected["selected"])
                scope, center, _ = target_scopes(neighbourhood, feature.observed, targets)
                if int(scope.sum()) != int(center.sum()):
                    raise ValueError("Matched center has different editable-day count")
                scoped = np.array(predictions[f"{arm}_r{rep}__conflict_scope"][cid]["activity"])
                if np.any(scoped[~scope] != initial[~scope]):
                    raise ValueError("Scoped local answer overwrote outside revision permission")
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            new = np.array(predictions[candidate][cid]["activity"])
            for control in controls:
                old = np.array(predictions[control][cid]["activity"])
                added, removed = (new == 1) & (old != 1), (new != 1) & (old == 1)
                counts = {"gained_tp": int(np.count_nonzero(added & (actual == 1))),
                          "lost_tp": int(np.count_nonzero(removed & (actual == 1))),
                          "gained_fp": int(np.count_nonzero(added & (actual == 0))),
                          "removed_fp": int(np.count_nonzero(removed & (actual == 0)))}
                for actual_code, gained, lost in ((1, "gained_tp", "lost_tp"), (0, "gained_fp", "removed_fp")):
                    delta = np.count_nonzero((new == 1) & (actual == actual_code)) - np.count_nonzero((old == 1) & (actual == actual_code))
                    if delta != counts[gained] - counts[lost]:
                        raise ValueError("Correction decomposition does not conserve totals")
                changes.append({"repeat": rep + 1, "case_id": cid, "control": control, **counts})
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    save_json(out / "denominators.json", denominators)
    save_json(out / "independent-weak-verification.json", weak_checks)
    csv_rows(out / "corrections.csv", changes)
    for cid, (case, _) in records.items():
        truth = read_json(source / "mechanism-reference" / cid / "truth.json")
        render_record(case, truth, labels[cid], predictions, out / f"{cid}.png")
    save_json(out / "verification.json", {"numerical_recomputed_records": len(numeric) * len(records),
        "weak_metric_groups": len(weak_checks), "conserved_correction_rows": len(changes),
        "scoped_authority_checks": len(records) * 6, "complete_record_figures": len(records),
        "extra_frozen_bindings": len(extra), "numerical_model_load_and_inference_seconds": numeric_seconds,
        "timing_excludes_feature_extraction_training_plotting_and_transport": True,
        "no_new_requests": True, "no_test_tuning": True, "field_reference": False})
    save_json(out / "manifest.json", {"analysis_manifest_sha256": sha(analysis / "manifest.json"),
        "files": {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Background supplement verified", len(numeric) * len(records), len(changes), denominators, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run, args.analysis, args.out)


if __name__ == "__main__":
    main()
