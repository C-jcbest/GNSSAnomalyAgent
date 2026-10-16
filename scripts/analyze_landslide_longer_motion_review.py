"""Score the small longer-scale development test after freezing all answers."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors, paired_record_interval
from finalize_landslide_adaptive_evidence import verify_metrics, verify_request_receipts
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_longer_motion_review import verify
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    score_counts,
)


def analyze(run, out, additional_baselines=None):
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Freeze all longer-scale predictions first")
    verify(run)
    for file, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / file) != expected:
            raise ValueError("Longer-scale output changed")
    config = read_json(run / "protocol.json")
    source = Path(config["source_bundle"])
    original_analysis = source / "analysis-v1"
    for file, expected in read_json(original_analysis / "manifest.json")["files"].items():
        if sha(original_analysis / file) != expected:
            raise ValueError("Original unified comparison changed")
    case_ids = config["case_ids"]
    cases = {cid: load_case_features(run, cid)[0] for cid in case_ids}
    for cid in case_ids:
        if sha(run / "inputs" / cid / "input.json") != sha(source / "inputs" / cid / "input.json"):
            raise ValueError("Comparison observation differs")
    if sha(run / "stage-xgboost.json") != sha(source / "stage-xgboost.json"):
        raise ValueError("Comparison stage head differs")
    new_receipts = verify_request_receipts(run)
    predictions = read_json(original_analysis / "predictions.json")
    predictions.update(read_json(run / "predictions.json"))
    expected_methods = 38
    if additional_baselines is not None:
        for file, expected in read_json(additional_baselines / "manifest.json")["files"].items():
            if sha(additional_baselines / file) != expected:
                raise ValueError("Additional numerical control changed")
        for file, expected in read_json(additional_baselines / "manifest.json")["source_bindings"].items():
            if sha(Path(file)) != expected:
                raise ValueError("Additional control training or source binding changed")
        extra = read_json(additional_baselines / "predictions.json")
        if set(extra) & set(predictions):
            raise ValueError("Additional control must not replace an existing method")
        predictions.update(extra)
        expected_methods += len(extra)
    if len(predictions) != expected_methods or any(set(value) != set(case_ids) for value in predictions.values()):
        raise ValueError("Every method must use the same twelve records")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "predictions.json", predictions)
    save_json(out / "prediction-freeze.json", {"predictions.json": sha(out / "predictions.json")})
    labels = load_labels(run / "reference-subset.csv", cases)
    original_costs = read_json(original_analysis / "costs.json")["per_method"]
    selection = read_json(run / "selections.json")
    costs = {}
    for rep in range(2):
        receipts = [read_json(source / "requests" / f"{cid}-direct-r{rep}.response.json") for cid in case_ids]
        for cid in case_ids:
            for target in selection[f"{cid}-r{rep}"]["selected"]:
                request_id = f"{cid}-window-{target['index']:02d}-image-r{rep}"
                path = run / "cached-responses" / f"{request_id}.response.json"
                if not path.exists():
                    path = run / "requests" / path.name
                receipts.append(read_json(path))
        costs[rep] = {"requests": len(receipts),
                      "tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts),
                      "seconds": sum(row["seconds"] for row in receipts),
                      "scope": "End-to-end hypothetical method receipt sum; initial/cached queries "
                               "were actually executed in the original run, never additive across methods"}
    results, summary, records = {}, [], []
    for method, values in predictions.items():
        per_case, boundaries, boundaries05 = {}, {}, {}
        weak_detected, weak_total = 0, 0
        changes = dict.fromkeys(("lost_tp", "gained_tp", "removed_fp", "gained_fp"), 0)
        for cid, value in values.items():
            activity = np.array(value["activity"])
            prediction = ActivityPrediction(activity, np.array(value["stage"]),
                                           status=value["status"], gated=value["gated"], stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            boundaries[cid] = boundary_errors(labels[cid], activity, .3)
            boundaries05[cid] = boundary_errors(labels[cid], activity, .5)
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
            detected = int(np.count_nonzero(weak & (activity == 1)))
            weak_detected += detected
            weak_total += int(weak.sum())
            change = dict.fromkeys(changes, 0)
            if method.startswith("extended84_"):
                rep = int(method[-1])
                control = np.array(predictions[f"adaptive_no_feedback_r{rep}__conflict_scope"][cid]["activity"])
                removed = (control == 1) & (activity != 1)
                gained = (control != 1) & (activity == 1)
                change = {"lost_tp": int(np.count_nonzero(removed & (actual == 1))),
                          "gained_tp": int(np.count_nonzero(gained & (actual == 1))),
                          "removed_fp": int(np.count_nonzero(removed & (actual == 0))),
                          "gained_fp": int(np.count_nonzero(gained & (actual == 0)))}
            for key in changes:
                changes[key] += change[key]
            records.append({"method": method, "case_id": cid, **per_case[cid]["daily"],
                            "weak_detected": detected, **change})
        aggregate = aggregate_scores(list(per_case.values()))
        pairs = [pair for result in boundaries.values() for pair in result["matched_pairs"]]
        if len(pairs) != aggregate["events"]["tp"]:
            raise ValueError("Event and boundary matching disagree")
        event05 = score_counts(sum(len(d["matched_pairs"]) for d in boundaries05.values()),
                              sum(d["unmatched_prediction_events"] for d in boundaries05.values()),
                              sum(d["missed_events"] for d in boundaries05.values()))
        results[method] = {"aggregate": aggregate, "per_case": per_case,
                           "boundaries_iou03": boundaries, "boundaries_iou05": boundaries05}
        cost = original_costs.get(method.split("__")[0], {})
        if method.startswith("extended84_"):
            cost = costs[int(method[-1])]
        summary.append({"method": method, **aggregate["daily"],
            "event_f1_iou03": aggregate["events"]["f1"], "event_f1_iou05": event05["f1"],
            "onset_mae_matched_days": float(np.mean([abs(p["onset_error_days"]) for p in pairs])) if pairs else None,
            "offset_mae_matched_days": float(np.mean([abs(p["offset_error_days"]) for p in pairs])) if pairs else None,
            "weak_recall": weak_detected / weak_total, "weak_days": weak_total,
            "coverage": aggregate["coverage"], "stage_macro_f1": aggregate["stage_macro_f1"],
            **changes, "requests": cost.get("requests", 0), "tokens": cost.get("tokens", 0),
            "seconds_model_receipts": cost.get("seconds", 0)})
    internal = verify_metrics(predictions, results, run / "reference-subset.csv", run)
    save_json(out / "true-activity-stage-scores.json", internal)
    save_json(out / "results.json", results)
    for row in summary:
        row["true_activity_stage_macro_f1"] = internal[row["method"]]["macro_f1"]
    indexed = {row["method"]: row for row in summary}
    intervals, decisions = {}, []
    for rep in range(2):
        candidate, control = f"extended84_evidence_scope_r{rep}", f"adaptive_no_feedback_r{rep}__conflict_scope"
        intervals[f"{candidate}-minus-{control}"] = paired_record_interval(results, control, candidate)
        a, b = indexed[candidate], indexed[control]
        decision = {"repeat": rep + 1, "f1_difference": a["f1"] - b["f1"],
                    "weak_recall_difference": a["weak_recall"] - b["weak_recall"],
                    **{key: a[key] for key in changes}, "additional_method_requests": a["requests"] - b["requests"]}
        decision["passes"] = decision["f1_difference"] > 0 and decision["weak_recall_difference"] > 0
        decisions.append(decision)
    decision = {"frozen_stop": config["stop"], "repetitions": decisions,
                "retain_extension_for_new_confirmation": all(row["passes"] for row in decisions),
                "scope": "Viewed development screen, not new independent or field confirmation"}
    save_json(out / "candidate-decision.json", decision)
    save_json(out / "paired-intervals.json", intervals)
    for file, rows in (("summary.csv", summary), ("per-record.csv", records)):
        with (out / file).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    numerical_methods = ["raw_multiscale_rule", "raw_rule_84", "xgboost_raw_persistence"]
    plot_labels = ["base scope", "84 whole", "84 +49", "84 physical", "rule42", "rule84", "XGB42"]
    if "xgboost_42_matched_support" in indexed:
        numerical_methods.append("xgboost_42_matched_support")
        plot_labels.append("XGB42 same support")
    numerical_methods.append("xgboost_84")
    plot_labels.append("XGB84 same support")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for index, metric in enumerate(("f1", "weak_recall")):
        for rep in range(2):
            methods = (f"adaptive_no_feedback_r{rep}__conflict_scope", f"extended84_whole_r{rep}",
                       f"extended84_same49_r{rep}", f"extended84_evidence_scope_r{rep}",
                       *numerical_methods)
            axes[index].bar(np.arange(len(methods)) + (rep - .5) * .36,
                            [indexed[method][metric] for method in methods], width=.34, label=f"repeat {rep + 1}")
        axes[index].set_xticks(range(len(plot_labels)), plot_labels, rotation=30, ha="right")
        axes[index].set_ylim(0, 1.02)
        axes[index].set_ylabel("Activity F1" if metric == "f1" else "Weak-motion recall")
        axes[index].grid(axis="y", alpha=.2)
        axes[index].legend()
    fig.suptitle("Viewed development records: evidence extension; numerical bars repeat the same fitted model")
    fig.tight_layout()
    fig.savefig(out / "longer-evidence-development.png", dpi=180)
    plt.close(fig)
    save_json(out / "costs.json", {
        "per_extended_repetition": costs, "new_unique_requests": len(new_receipts),
        "new_unique_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in new_receipts),
        "cached_selected_receipts": len(list((run / "cached-responses").glob("*.response.json"))),
        "scope": "No new initial inference; shared cached receipts do not add to the mainline actual ledger"})
    save_json(out / "verification.json", {"same_inputs_and_stage_head": len(case_ids),
        "independent_sklearn_and_event_boundary_groups": len(predictions), "new_actual_payloads": len(new_receipts),
        "original_train24_validation12": True, "validation_trials": len(read_json(run / "validation-trials.json")),
        "viewed_test_labels_used_for_fitting": False, "all_repetitions_retained": True})
    source_bindings = {str(run / "freeze.json"): sha(run / "freeze.json"),
              str(run / "prediction-freeze.json"): sha(run / "prediction-freeze.json"),
              str(original_analysis / "manifest.json"): sha(original_analysis / "manifest.json")}
    if additional_baselines is not None:
        source_bindings[str(additional_baselines / "manifest.json")] = sha(additional_baselines / "manifest.json")
    save_json(out / "manifest.json", {"source_bindings": source_bindings,
              "files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Longer-scale development decision:", decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--additional-baselines", type=Path)
    args = parser.parse_args()
    analyze(args.run, args.out, args.additional_baselines)


if __name__ == "__main__":
    main()
