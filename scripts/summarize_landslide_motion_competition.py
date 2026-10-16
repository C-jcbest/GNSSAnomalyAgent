"""Export frozen target diagnostics and apply the existing nine numerical controls."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_observation_probe import csv_rows, evaluate_population
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_evidence_display_confirmation import numerical_predictions
from run_landslide_motion_competition_probe import BASE, verify
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import precision_recall_fscore_support
from verify_landslide_evidence_display_confirmation import verify_live_sources

from gnss_sim.landslide_evaluation import STAGES, ActivityPrediction


def verify_metrics(predictions, results, labels, records):
    """Check scores while retaining undefined precision for all-negative outputs."""
    verified = {}
    for method, cases in predictions.items():
        actual_activity, detected_activity, actual_stage, predicted_stage = [], [], [], []
        internal_actual, internal_predicted = [], []
        for cid, value in cases.items():
            activity = np.array(value["activity"])
            stage = np.array(value["stage"])
            observed = records[cid][1].observed
            ActivityPrediction(activity, stage).validate(len(observed))
            if np.any(activity[~observed] != -1) or np.any(stage[~observed] != -1):
                raise ValueError("Numerical target invented a missing-date prediction")
            for day, row in enumerate(labels[cid]):
                actual = int(row["activity_label"])
                if actual < 0:
                    continue
                expected_stage = STAGES[row["feature"]] if actual == 1 else 0
                actual_activity.append(actual)
                detected_activity.append(int(activity[day] == 1))
                actual_stage.append(expected_stage)
                predicted_stage.append(int(stage[day]))
                if actual == 1:
                    internal_actual.append(expected_stage)
                    internal_predicted.append(int(stage[day]))
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_activity, detected_activity, labels=[1], zero_division=0)
        daily = results[method]["aggregate"]["daily"]
        for name, score, denominator in (("precision", precision[0], sum(detected_activity)),
                ("recall", recall[0], sum(actual_activity)),
                ("f1", f1[0], sum(detected_activity) + sum(actual_activity))):
            if denominator == 0:
                if daily[name] is not None:
                    raise ValueError(f"Undefined {name} was replaced by a score: {method}")
            elif not np.isclose(daily[name], score):
                raise ValueError(f"Independent {name} differs: {method}")
        _, _, end_f1, _ = precision_recall_fscore_support(
            actual_stage, predicted_stage, labels=[1, 2, 3], zero_division=0)
        defined = [code for code in (1, 2, 3) if code in actual_stage or code in predicted_stage]
        if not np.isclose(np.mean([end_f1[code - 1] for code in defined]),
                          results[method]["aggregate"]["stage_macro_f1"]):
            raise ValueError("Independent defined-class stage macro differs")
        precision, recall, f1, support = precision_recall_fscore_support(
            internal_actual, internal_predicted, labels=[1, 2, 3], zero_division=0)
        verified[method] = {"scope": "True activity in selected targets, missed and unknown retained",
            "precision_ASD": precision.tolist(), "recall_ASD": recall.tolist(), "f1_ASD": f1.tolist(),
            "support_ASD": support.tolist(), "macro_f1": float(np.mean(f1)),
            "end_to_end_fixed_ASD_macro_f1": float(np.mean(end_f1)),
            "repository_defined_stage_classes": defined,
            "precision_undefined": sum(detected_activity) == 0}
    return verified

def target_predictions(predictions, targets):
    """Match the probe's scoring calendar without inventing decisions outside it."""
    masked = {}
    for method, cases in predictions.items():
        masked[method] = {}
        for target in targets:
            cid = target["case_id"]
            value = cases[cid]
            activity = np.array(value["activity"])
            stage = np.array(value["stage"])
            outside = np.ones(len(activity), dtype=bool)
            outside[target["start"]:target["stop"]] = False
            activity[outside] = -1
            stage[outside] = -1
            masked[method][cid] = {**value, "activity": activity.tolist(), "stage": stage.tolist()}
    return masked


def render_target(case, target, labels, predictions, path):
    start, stop = target["start"], target["stop"]
    days = np.arange(start, stop)
    values = np.asarray([row if row is not None else [np.nan] * 3
                         for row in case.displacement_mm], dtype=float)
    methods = [f"{arm}_r{rep}" for rep in (0, 1)
               for arm in ("raw", "rival_curves", "rival_feedback")]
    methods += ["xgboost_42_matched_support", "xgboost_84"]
    truth = np.array([int(row["activity_label"]) for row in labels])
    matrix = np.array([truth[start:stop]] + [predictions[name][case.case_id]["activity"][start:stop]
                                           for name in methods])
    fig, panels = plt.subplots(4, 1, figsize=(11, 9), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1, 2]})
    for component, name in enumerate(("N", "E", "U")):
        panels[component].plot(days, values[start:stop, component], linewidth=.8, color="#24887b")
        panels[component].set_ylabel(name + " (mm)")
        panels[component].grid(alpha=.2)
    from matplotlib.colors import ListedColormap
    panels[3].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
                     cmap=ListedColormap(["#c5c9cc", "#fff4dc", "#377dab"]),
                     extent=[start - .5, stop - .5, len(matrix) - .5, -.5])
    panels[3].set_yticks(range(len(matrix)), ["Design reference"] + methods, fontsize=8)
    panels[3].set_xlabel("Original day index | blue: activity; cream: quiet; gray: unknown/missing")
    panels[3].set_xlim(start, stop - 1)
    fig.suptitle(f"{case.case_id} | {target['role']} | Viewed selected-target development")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summarize(run, out):
    verify(run)
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Finish and freeze all visual arms before diagnostic export")
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError("Frozen visual predictions changed")
    training = BASE / "evidence-display-confirmation-v2"
    verify_live_sources(training)
    for name, expected in read_json(training / "training-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Existing common-training artifact changed")
    targets = read_json(run / "protocol.json")["targets"]
    records = {target["case_id"]: load_case_features(run, target["case_id"]) for target in targets}
    labels = load_labels(run / "reference-subset.csv", {cid: row[0] for cid, row in records.items()})
    started = time.monotonic()
    numerical = numerical_predictions(training, records)
    numerical_seconds = time.monotonic() - started
    # Recompute on the original complete record before masking to the selected
    # target. This checks reuse of the same train/validation models and parameters.
    for target in targets:
        source = BASE / target["source"]
        name = "prior-comparison-predictions.json" if target["source"].startswith("block-") else "numeric-predictions.json"
        cached = read_json(source / name)
        for method, cases in numerical.items():
            if cases[target["case_id"]] != cached[method][target["case_id"]]:
                raise ValueError(f"Common numerical control differs: {method} {target['case_id']}")
    all_predictions = {**read_json(run / "predictions.json"), **target_predictions(numerical, targets)}
    results, summary, per_target = evaluate_population(all_predictions, labels)
    stages = verify_metrics(all_predictions, results, labels, records)
    for row in summary:
        row["true_activity_stage_macro_f1"] = stages[row["method"]]["macro_f1"]
    changes = []
    for rep in (0, 1):
        for candidate, control in (("rival_curves", "raw"), ("rival_feedback", "raw"),
                                   ("rival_feedback", "rival_curves")):
            for cid in records:
                truth = np.array([int(row["activity_label"]) for row in labels[cid]])
                new = np.array(all_predictions[f"{candidate}_r{rep}"][cid]["activity"])
                old = np.array(all_predictions[f"{control}_r{rep}"][cid]["activity"])
                added = (new == 1) & (old != 1)
                removed = (new != 1) & (old == 1)
                counts = {"gained_tp": int(np.count_nonzero(added & (truth == 1))),
                          "lost_tp": int(np.count_nonzero(removed & (truth == 1))),
                          "gained_fp": int(np.count_nonzero(added & (truth == 0))),
                          "removed_fp": int(np.count_nonzero(removed & (truth == 0)))}
                for actual, gained, lost in ((1, "gained_tp", "lost_tp"), (0, "gained_fp", "removed_fp")):
                    delta = int(np.count_nonzero((new == 1) & (truth == actual))
                                - np.count_nonzero((old == 1) & (truth == actual)))
                    if delta != counts[gained] - counts[lost]:
                        raise ValueError("Target correction counts do not conserve predictions")
                changes.append({"repeat": rep + 1, "candidate": candidate, "control": control,
                                "case_id": cid, **counts})
    receipts = [read_json(path) for path in sorted((run / "requests").glob("*.response.json"))]
    denominators = {"known_days": 0, "quiet_days": 0, "active_days": 0, "weak_days": 0, "stage_ASD": [0, 0, 0]}
    for rows in labels.values():
        for row in rows:
            actual = int(row["activity_label"])
            denominators["known_days"] += int(actual >= 0)
            denominators["quiet_days"] += int(actual == 0)
            denominators["active_days"] += int(actual == 1)
            denominators["weak_days"] += int(row["weak_signal"])
            if actual == 1:
                denominators["stage_ASD"][STAGES[row["feature"]] - 1] += 1
    for row in summary:
        if denominators["weak_days"] != row["weak_days"]:
            raise ValueError("Weak reference population differs from the target scoring denominator")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    save_json(out / "predictions.json", all_predictions)
    save_json(out / "results.json", results)
    save_json(out / "stage-metrics.json", stages)
    csv_rows(out / "summary.csv", summary)
    csv_rows(out / "per-target.csv", per_target)
    csv_rows(out / "changes.csv", changes)
    save_json(out / "denominators.json", denominators)
    save_json(out / "costs.json", {"new_requests": 0, "numerical_models_and_inference_seconds": numerical_seconds,
        "visual": {f"{arm}_r{rep}": {"requests": 8,
            "tokens": sum(row["usage"]["total_tokens"] for row in receipts if row["request_id"].endswith(f"-{arm}-r{rep}")),
            "seconds": sum(row["seconds"] for row in receipts if row["request_id"].endswith(f"-{arm}-r{rep}"))}
            for arm in ("raw", "rival_curves", "rival_feedback") for rep in (0, 1)},
        "scope": "Local probes only; features and global context preparation excluded. No full-method cost claim"})
    for target in targets:
        cid = target["case_id"]
        render_target(records[cid][0], target, labels[cid], all_predictions, out / f"{cid}.png")
    save_json(out / "verification.json", {"common_training_complete_record_recomputations": 72,
        "independent_activity_and_stage_groups": 15, "correction_conservation_groups": len(changes),
        "no_new_fitting_or_tuning": True, "target_only_not_whole_method": True})
    save_json(out / "manifest.json", {"run_freeze_sha256": sha(run / "freeze.json"),
        "prediction_freeze_sha256": sha(run / "prediction-freeze.json"),
        "common_training_bindings_sha256": sha(training / "training-bindings.json"),
        "files": {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Nine numerical controls, 15 target configurations and 48 correction comparisons verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run, args.out)


if __name__ == "__main__":
    main()
