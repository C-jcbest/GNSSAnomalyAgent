"""Score frozen new confirmation, matched controls and the executed TAMA adaptation."""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors, paired_record_interval
from finalize_landslide_adaptive_evidence import verify_metrics, verify_request_receipts
from probe_landslide_interval_consistency import POLICIES, cached_predictions, check_prediction
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_tama_adapter import verify as verify_tama
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    score_counts,
)


def request_costs(paths):
    receipts = [read_json(path) for path in paths]
    return {
        "requests": len(receipts),
        "tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts),
        "seconds": sum(row["seconds"] for row in receipts),
        "request_ids": [row["request_id"] for row in receipts],
        "scope": "Actual receipts; shared reference and detection calls are not additive across arms",
    }


def analyze(run, tama, out):
    if read_json(run / "state.json")["status"] != "complete":
        raise ValueError("Main inference and scoring must be completed")
    if read_json(tama / "state.json")["status"] != "predictions_complete":
        raise ValueError("All TAMA adaptation predictions must be frozen")
    verify_tama(tama)
    receipt_groups = {"main": verify_request_receipts(run), "tama": verify_request_receipts(tama)}
    for file, expected in read_json(tama / "prediction-freeze.json").items():
        if sha(tama / file) != expected:
            raise ValueError("TAMA adaptation predictions changed")
    data = ROOT / read_json(run / "protocol.json")["source_bundle"]
    for file, expected in read_json(data / "manifest.json")["files"].items():
        if sha(data / file) != expected:
            raise ValueError("New confirmation generation evidence changed")
    candidate_source = ROOT / "scripts/probe_landslide_interval_consistency.py"
    if sha(candidate_source) != read_json(data / "protocol.json")["candidate_source_sha256"]:
        raise ValueError("Candidate source changed after freezing new data")
    methods = cached_predictions(run, data / "additional-baselines/predictions.json")
    methods.update(read_json(tama / "predictions.json"))
    case_ids = read_json(run / "protocol.json")["case_ids"]
    if read_json(tama / "protocol.json")["case_ids"] != case_ids:
        raise ValueError("Comparison must use exactly the same records")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    cases, features = {}, {}
    for cid in case_ids:
        if sha(run / "inputs" / cid / "input.json") != sha(tama / "inputs" / cid / "input.json"):
            raise ValueError("TAMA adaptation input differs from the primary comparison")
        cases[cid], features[cid] = load_case_features(run, cid)
    predictions, diagnostics, gate_times = {}, {}, {}
    for method, records in methods.items():
        for policy in ("original", *POLICIES):
            arm = f"{method}__{policy}"
            predictions[arm], diagnostics[arm] = {}, {}
            started = time.perf_counter()
            for cid in case_ids:
                value = records[cid]
                original = ActivityPrediction(np.array(value["activity"]), np.array(value["stage"]),
                                              status=value["status"], gated=value["gated"],
                                              stage_status=value["stage_status"])
                if policy == "original":
                    prediction, detail = original, []
                else:
                    prediction, detail = check_prediction(
                        cases[cid], features[cid].noise, original, policy)
                predictions[arm][cid] = prediction.to_dict()
                diagnostics[arm][cid] = detail
            gate_times[arm] = time.perf_counter() - started
    save_json(out / "predictions.json", predictions)
    save_json(out / "diagnostics.json", diagnostics)
    save_json(out / "prediction-freeze.json", {
        file: sha(out / file) for file in ("predictions.json", "diagnostics.json")
    })
    labels = load_labels(run / "reference-subset.csv", cases)
    costs = read_json(run / "costs.json")
    for rep in range(2):
        reference = [tama / "requests" / f"normal-pattern-r{rep}.response.json"]
        detect = sorted((tama / "requests").glob(f"*-detect-r{rep}.response.json"))
        reviews = sorted((tama / "requests").glob(f"*-review-r{rep}.response.json"))
        costs[f"tama_gnss_no_review_r{rep}"] = request_costs(reference + detect)
        costs[f"tama_gnss_positive_review_r{rep}"] = request_costs(reference + detect + reviews)
    summary, changes, results = [], [], {}
    for arm, records in predictions.items():
        method, policy = arm.split("__")
        per_case, boundaries, boundaries05 = {}, {}, {}
        weak_total, weak_detected, lost_tp, removed_fp = 0, 0, 0, 0
        for cid, value in records.items():
            activity, stage = np.array(value["activity"]), np.array(value["stage"])
            prediction = ActivityPrediction(activity, stage, status=value["status"],
                                            gated=value["gated"], stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            boundaries[cid] = boundary_errors(labels[cid], activity, .3)
            boundaries05[cid] = boundary_errors(labels[cid], activity, .5)
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
            changed = (np.array(methods[method][cid]["activity"]) == 1) & (activity == -1)
            tp_loss = int(np.count_nonzero(changed & (actual == 1)))
            fp_removed = int(np.count_nonzero(changed & (actual == 0)))
            lost_tp += tp_loss
            removed_fp += fp_removed
            weak_total += int(weak.sum())
            weak_detected += int(np.count_nonzero(weak & (activity == 1)))
            changes.append({"method": method, "policy": policy, "case_id": cid,
                            "lost_tp": tp_loss, "removed_fp": fp_removed,
                            "weak_detected": int(np.count_nonzero(weak & (activity == 1))),
                            "changed_positive_days": int(changed.sum())})
        score = aggregate_scores(list(per_case.values()))
        pairs = [pair for detail in boundaries.values() for pair in detail["matched_pairs"]]
        if len(pairs) != score["events"]["tp"]:
            raise ValueError("Event matching and boundary analysis disagree")
        event05 = score_counts(
            sum(len(detail["matched_pairs"]) for detail in boundaries05.values()),
            sum(detail["unmatched_prediction_events"] for detail in boundaries05.values()),
            sum(detail["missed_events"] for detail in boundaries05.values()),
        )
        results[arm] = {"aggregate": score, "per_case": per_case,
                        "boundaries_iou03": boundaries, "boundaries_iou05": boundaries05}
        cost = costs.get(method, {})
        summary.append({
            "method": method, "policy": policy, **score["daily"],
            "event_f1_iou03": score["events"]["f1"], "event_f1_iou05": event05["f1"],
            "onset_mae_matched_days": float(np.mean([abs(pair["onset_error_days"]) for pair in pairs]))
            if pairs else None,
            "offset_mae_matched_days": float(np.mean([abs(pair["offset_error_days"]) for pair in pairs]))
            if pairs else None,
            "weak_recall": weak_detected / weak_total, "weak_days": weak_total,
            "stage_macro_f1": score["stage_macro_f1"], "coverage": score["coverage"],
            "abstained_known_days": score["abstained_known_days"],
            "lost_tp": lost_tp, "removed_fp": removed_fp,
            "requests": cost.get("requests"), "tokens": cost.get("tokens"),
            "seconds_model_receipts": cost.get("seconds"),
            "seconds_deterministic_replay": gate_times[arm],
        })
    save_json(out / "results.json", results)
    paired_intervals = {}
    for rep in range(2):
        treatment = f"adaptive_no_feedback_r{rep}__original"
        for control in (f"direct_visual_r{rep}__original", f"fixed_image_review_r{rep}__original",
                        f"tama_gnss_positive_review_r{rep}__original", "raw_multiscale_rule__original",
                        "xgboost_raw_persistence__original"):
            paired_intervals[f"{treatment}-minus-{control}"] = paired_record_interval(
                results, control, treatment)
        candidate = f"adaptive_no_feedback_r{rep}__require_support"
        paired_intervals[f"{candidate}-minus-{treatment}"] = paired_record_interval(
            results, treatment, candidate)
    save_json(out / "paired-intervals.json", paired_intervals)
    internal_stages = verify_metrics(predictions, results, run / "reference-subset.csv", run)
    save_json(out / "true-activity-stage-scores.json", internal_stages)
    for row in summary:
        row["true_activity_stage_macro_f1"] = internal_stages[
            f"{row['method']}__{row['policy']}"
        ]["macro_f1"]
    for file, rows in (("summary.csv", summary), ("per-record-changes.csv", changes)):
        with (out / file).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    unique = {}
    for name, folder in (("main", run), ("tama_adapter", tama)):
        unique[name] = request_costs(sorted((folder / "requests").glob("*.response.json")))
    unique["total_requests"] = unique["main"]["requests"] + unique["tama_adapter"]["requests"]
    unique["total_tokens"] = unique["main"]["tokens"] + unique["tama_adapter"]["tokens"]
    save_json(out / "costs.json", {"per_method": costs, "unique_actual": unique})
    decision = {"frozen_rule": read_json(data / "protocol.json")["decision_rule"], "repetitions": []}
    for rep in range(2):
        rows = {row["policy"]: row for row in summary
                if row["method"] == f"adaptive_no_feedback_r{rep}"}
        original, candidate = rows["original"], rows["require_support"]
        decision["repetitions"].append({
            "repeat": rep + 1, "f1_difference": candidate["f1"] - original["f1"],
            "weak_recall_difference": candidate["weak_recall"] - original["weak_recall"],
            "lost_tp": candidate["lost_tp"], "removed_fp": candidate["removed_fp"],
        })
    decision["passes_frozen_screen"] = all(
        row["f1_difference"] >= 0 and row["weak_recall_difference"] >= 0 and row["removed_fp"] > 0
        for row in decision["repetitions"]
    )
    decision["scope"] = "Descriptive small synthetic confirmation screen; not a significance test, "
    decision["scope"] += "field reliability certificate or automatic change of repository default"
    save_json(out / "candidate-decision.json", decision)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    visual = (("direct_visual", "original"), ("fixed_image_review", "original"),
              ("adaptive_no_feedback", "original"), ("adaptive_no_feedback", "require_support"),
              ("tama_gnss_no_review", "original"), ("tama_gnss_positive_review", "original"))
    for index, metric in enumerate(("f1", "weak_recall")):
        for rep in range(2):
            values = [next(row[metric] for row in summary
                           if row["method"] == f"{method}_r{rep}" and row["policy"] == policy)
                      for method, policy in visual]
            positions = np.arange(len(visual)) + (rep - .5) * .36
            axes[index].bar(positions, values, width=.34, label=f"repeat {rep + 1}")
        axes[index].set_xticks(range(len(visual)), [
            "direct", "fixed", "adaptive", "adaptive\n+check", "TAMA ref", "TAMA\nreview",
        ])
        axes[index].set_ylabel("Activity F1" if metric == "f1" else "Weak-motion recall")
        axes[index].set_ylim(0, 1.02)
        axes[index].grid(axis="y", alpha=.2)
        axes[index].legend()
    fig.suptitle("New 12 controlled records; known latent process, no field accuracy claim")
    fig.tight_layout()
    fig.savefig(out / "new-confirmation.png", dpi=180)
    plt.close(fig)
    save_json(out / "verification.json", {
        "input_and_prediction_freezes": "passed", "same_tama_and_main_inputs": len(case_ids),
        "independent_sklearn_activity_and_stage_metrics": len(predictions),
        "common_event_boundary_agreement": len(predictions),
        "candidate_source_unchanged_since_data_freeze": True,
        "actual_payloads_verified": {name: len(values) for name, values in receipt_groups.items()},
        "test_label_fitting_or_tuning": False,
    })
    save_json(out / "manifest.json", {
        "source_bindings": {str(path): sha(path) for path in (
            data / "manifest.json", run / "freeze.json", run / "prediction-freeze.json",
            tama / "freeze.json", tama / "prediction-freeze.json",
        )},
        "files": {str(path.relative_to(out)): sha(path)
                  for path in out.rglob("*") if path.is_file()},
    })
    print("New confirmation decision:", decision, flush=True)
    print("Actual unique requests/tokens:", unique["total_requests"], unique["total_tokens"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tama", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.tama, args.out)


if __name__ == "__main__":
    main()
