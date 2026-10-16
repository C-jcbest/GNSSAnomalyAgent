"""Verify new conflict-scope confirmation and compare all frozen same-data arms."""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors, paired_record_interval
from finalize_landslide_adaptive_evidence import (
    verify_manifest,
    verify_metrics,
    verify_request_receipts,
)
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_tama_adapter import verify as verify_tama
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    score_counts,
)


def receipt_cost(paths):
    receipts = [read_json(path) for path in paths]
    return {
        "requests": len(receipts),
        "tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts),
        "seconds": sum(row["seconds"] for row in receipts),
        "scope": "Actual saved receipts; shared calls are not additive across methods",
    }


def analyze(run, tama, scoped, out):
    if read_json(run / "state.json")["status"] != "complete":
        raise ValueError("Complete and freeze all main inference before analysis")
    if read_json(tama / "state.json")["status"] != "predictions_complete":
        raise ValueError("Complete and freeze all TAMA inference before analysis")
    verify_freeze(run)
    verify_tama(tama)
    data = ROOT / read_json(run / "protocol.json")["source_bundle"]
    protocol = read_json(data / "protocol.json")
    verified_files = verify_manifest(data) + verify_manifest(scoped)
    candidate_source = ROOT / "scripts/probe_landslide_revision_scope.py"
    if sha(candidate_source) != protocol["candidate_source_sha256"]:
        raise ValueError("Candidate changed after generation freeze")
    if sha(scoped / "source.py") != protocol["candidate_source_sha256"]:
        raise ValueError("Scope replay did not execute the pre-generation candidate")
    for file, expected in read_json(scoped / "source-bindings.json")[run.name].items():
        path = Path(file) if "/" in file or "\\" in file else run / file
        if sha(path) != expected:
            raise ValueError("Scope source no longer matches the frozen comparison")
    for file, expected in read_json(data / "training-bindings.json").items():
        if sha(Path(file)) != expected:
            raise ValueError("Common train/validation binding changed")
    scope_predictions = scoped / run.name
    for folder in (run, tama, scope_predictions):
        for file, expected in read_json(folder / "prediction-freeze.json").items():
            if sha(folder / file) != expected:
                raise ValueError("Frozen inference or revision changed")
    case_ids = protocol["case_ids"]
    for folder in (run, tama):
        if read_json(folder / "protocol.json")["case_ids"] != case_ids:
            raise ValueError("All arms must share the frozen new record order")
        if sha(folder / "stage-xgboost.json") != sha(data / "stage-xgboost.json"):
            raise ValueError("Stage model differs from the common trained head")
        for cid in case_ids:
            for file in ("input.json", "motion-31.json", "motion-61.json", "motion-91.json"):
                if sha(folder / "inputs" / cid / file) != sha(data / "inputs" / cid / file):
                    raise ValueError("Numerical and visual comparison input differs")
    receipts = {"main": verify_request_receipts(run), "tama": verify_request_receipts(tama)}
    predictions = read_json(scope_predictions / "predictions.json")
    predictions.update(read_json(tama / "predictions.json"))
    if len(predictions) != 30 or any(set(records) != set(case_ids) for records in predictions.values()):
        raise ValueError("Expected 26 scope/baseline plus four TAMA same-data arms")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "predictions.json", predictions)
    save_json(out / "prediction-freeze.json", {"predictions.json": sha(out / "predictions.json")})
    cases, features = {}, {}
    for cid in case_ids:
        cases[cid], features[cid] = load_case_features(run, cid)

    # Time the deterministic candidate at record level with inputs already loaded.
    # It includes raw evidence, mask construction and merge, but excludes the
    # shared initial/local inference, rendering and learned stage head.
    cache = read_json(run / "predictions.json")
    selections = read_json(run / "selections.json")
    policy = ReviewPolicy(**protocol["candidate_policy"]["policy"])
    scope_times = []
    for rep in range(2):
        for cid in case_ids:
            started = time.perf_counter()
            initial = np.array(cache[f"direct_visual_r{rep}"][cid]["activity"])
            evidence = raw_motion_evidence(cases[cid], features[cid])
            _, neighbourhood = conflict_neighbourhood(initial, features[cid], evidence, policy)
            scope, _, _ = target_scopes(neighbourhood, features[cid].observed,
                                       selections[f"{cid}-r{rep}"]["selected"])
            reviewed = np.array(cache[f"adaptive_no_feedback_r{rep}"][cid]["activity"])
            activity = scoped_activity(initial, reviewed, scope, features[cid].observed)
            seconds = time.perf_counter() - started
            expected = predictions[f"adaptive_no_feedback_r{rep}__conflict_scope"][cid]["activity"]
            if activity.tolist() != expected:
                raise ValueError("Timed candidate differs from frozen scope output")
            scope_times.append({"repeat": rep + 1, "case_id": cid, "seconds": seconds})
    save_json(out / "scope-costs.json", {"record_timings": scope_times,
              "scope": "One local measurement per record/repetition, not a performance benchmark; "
                       "raw evidence + masks + merge, excludes VLM, rendering and shared stage head"})
    labels = load_labels(run / "reference-subset.csv", cases)
    costs = read_json(run / "costs.json")
    for rep in range(2):
        reference = [tama / "requests" / f"normal-pattern-r{rep}.response.json"]
        detect = sorted((tama / "requests").glob(f"*-detect-r{rep}.response.json"))
        reviews = sorted((tama / "requests").glob(f"*-review-r{rep}.response.json"))
        costs[f"tama_gnss_no_review_r{rep}"] = receipt_cost(reference + detect)
        costs[f"tama_gnss_positive_review_r{rep}"] = receipt_cost(reference + detect + reviews)
    summary, records, results = [], [], {}
    for arm, values in predictions.items():
        method, _, variant = arm.partition("__")
        per_case, boundaries, boundaries05 = {}, {}, {}
        weak_total, weak_detected = 0, 0
        totals = {key: 0 for key in ("lost_tp", "gained_tp", "removed_fp", "gained_fp")}
        for cid, value in values.items():
            activity = np.array(value["activity"])
            prediction = ActivityPrediction(activity, np.array(value["stage"]),
                status=value["status"], gated=value["gated"], stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            boundaries[cid] = boundary_errors(labels[cid], activity, .3)
            boundaries05[cid] = boundary_errors(labels[cid], activity, .5)
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
            weak_total += int(weak.sum())
            detected = int(np.count_nonzero(weak & (activity == 1)))
            weak_detected += detected
            change = dict.fromkeys(totals, 0)
            if variant:
                original = np.array(cache[method][cid]["activity"])
                removed = (original == 1) & (activity != 1)
                gained = (original != 1) & (activity == 1)
                change = {
                    "lost_tp": int(np.count_nonzero(removed & (actual == 1))),
                    "gained_tp": int(np.count_nonzero(gained & (actual == 1))),
                    "removed_fp": int(np.count_nonzero(removed & (actual == 0))),
                    "gained_fp": int(np.count_nonzero(gained & (actual == 0))),
                }
            for key in totals:
                totals[key] += change[key]
            records.append({"arm": arm, "case_id": cid, **per_case[cid]["daily"],
                            "weak_detected": detected, **change})
        aggregate = aggregate_scores(list(per_case.values()))
        pairs = [pair for detail in boundaries.values() for pair in detail["matched_pairs"]]
        if len(pairs) != aggregate["events"]["tp"]:
            raise ValueError("Event and boundary matching disagree")
        event05 = score_counts(sum(len(d["matched_pairs"]) for d in boundaries05.values()),
                              sum(d["unmatched_prediction_events"] for d in boundaries05.values()),
                              sum(d["missed_events"] for d in boundaries05.values()))
        results[arm] = {"aggregate": aggregate, "per_case": per_case,
                       "boundaries_iou03": boundaries, "boundaries_iou05": boundaries05}
        cost = costs.get(method, {})
        summary.append({"arm": arm, "method": method, "variant": variant or "original",
            **aggregate["daily"], "event_f1_iou03": aggregate["events"]["f1"],
            "event_f1_iou05": event05["f1"], "matched_events_iou03": len(pairs),
            "onset_mae_matched_days": float(np.mean([abs(p["onset_error_days"]) for p in pairs]))
            if pairs else None,
            "offset_mae_matched_days": float(np.mean([abs(p["offset_error_days"]) for p in pairs]))
            if pairs else None,
            "weak_recall": weak_detected / weak_total, "weak_days": weak_total,
            "stage_macro_f1": aggregate["stage_macro_f1"], "coverage": aggregate["coverage"],
            "abstained_known_days": aggregate["abstained_known_days"], **totals,
            "requests": cost.get("requests", 0), "tokens": cost.get("tokens", 0),
            "seconds_model_receipts": cost.get("seconds", 0),
        })
    internal = verify_metrics(predictions, results, run / "reference-subset.csv", run)
    save_json(out / "true-activity-stage-scores.json", internal)
    for row in summary:
        row["true_activity_stage_macro_f1"] = internal[row["arm"]]["macro_f1"]
    save_json(out / "results.json", results)
    intervals = {}
    indexed = {row["arm"]: row for row in summary}
    decision = {"frozen_rule": protocol["decision_rule"], "repetitions": []}
    for rep in range(2):
        treatment = f"adaptive_no_feedback_r{rep}__conflict_scope"
        whole = f"adaptive_no_feedback_r{rep}__whole_target"
        controls = (whole, f"adaptive_no_feedback_r{rep}__matched_center_scope",
                    f"direct_visual_r{rep}", f"fixed_image_review_r{rep}__whole_target",
                    f"adaptive_numeric_feedback_r{rep}__conflict_scope",
                    f"tama_gnss_positive_review_r{rep}", "raw_multiscale_rule", "xgboost_raw_persistence")
        for control in controls:
            intervals[f"{treatment}-minus-{control}"] = paired_record_interval(results, control, treatment)
        original, candidate = indexed[whole], indexed[treatment]
        change = {"repeat": rep + 1, "f1_difference": candidate["f1"] - original["f1"],
                  "weak_recall_difference": candidate["weak_recall"] - original["weak_recall"],
                  "fp_difference": candidate["fp"] - original["fp"],
                  **{key: candidate[key] for key in totals}}
        change["passes"] = (change["lost_tp"] == 0 and change["weak_recall_difference"] >= 0
                            and change["fp_difference"] < 0 and change["f1_difference"] >= 0)
        decision["repetitions"].append(change)
    decision["passes_frozen_screen"] = all(row["passes"] for row in decision["repetitions"])
    decision["scope"] = "Small same-family latent-mechanism confirmation, not independent "
    decision["scope"] += "field accuracy, significance or automatic deployment/default change"
    save_json(out / "candidate-decision.json", decision)
    save_json(out / "paired-intervals.json", intervals)
    for file, rows in (("summary.csv", summary), ("per-record.csv", records)):
        with (out / file).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    unique = {name: receipt_cost(sorted((folder / "requests").glob("*.response.json")))
              for name, folder in (("main", run), ("tama", tama))}
    unique["total_requests"] = unique["main"]["requests"] + unique["tama"]["requests"]
    unique["total_tokens"] = unique["main"]["tokens"] + unique["tama"]["tokens"]
    save_json(out / "costs.json", {"per_method": costs, "unique_actual": unique})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for index, metric in enumerate(("f1", "weak_recall")):
        for rep in range(2):
            arms = (f"direct_visual_r{rep}", f"fixed_image_review_r{rep}__whole_target",
                    f"adaptive_no_feedback_r{rep}__whole_target",
                    f"adaptive_no_feedback_r{rep}__conflict_scope",
                    f"adaptive_no_feedback_r{rep}__matched_center_scope",
                    f"adaptive_numeric_feedback_r{rep}__conflict_scope",
                    f"tama_gnss_positive_review_r{rep}", "raw_multiscale_rule", "xgboost_raw_persistence")
            axes[index].bar(np.arange(len(arms)) + (rep - .5) * .36,
                            [indexed[arm][metric] for arm in arms], width=.34, label=f"repeat {rep + 1}")
        axes[index].set_xticks(range(9), ["direct", "fixed", "whole", "conflict", "centre",
                                        "numeric", "TAMA", "raw rule", "raw XGB"], rotation=30)
        axes[index].set_ylabel("Activity F1" if metric == "f1" else "Weak-motion recall")
        axes[index].set_ylim(0, 1.02)
        axes[index].grid(axis="y", alpha=.2)
        axes[index].legend()
    fig.suptitle("Pre-frozen scope candidate: 12 new controlled records, no field accuracy claim")
    fig.tight_layout()
    fig.savefig(out / "scope-confirmation.png", dpi=180)
    plt.close(fig)
    save_json(out / "verification.json", {
        "candidate_source_unchanged_since_generation": True,
        "data_and_scope_manifest_files": verified_files, "same_inputs_and_stage_head": len(case_ids),
        "independent_sklearn_metric_groups": len(predictions), "common_event_boundary_agreement": len(predictions),
        "request_payload_bindings": {name: len(rows) for name, rows in receipts.items()},
        "test_fitting_or_tuning": False,
    })
    save_json(out / "manifest.json", {
        "source_bindings": {str(path): sha(path) for path in (
            data / "manifest.json", run / "freeze.json", run / "prediction-freeze.json",
            tama / "freeze.json", tama / "prediction-freeze.json", scoped / "manifest.json")},
        "files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    })
    print("Scope confirmation decision:", decision, flush=True)
    print("Unique actual requests/tokens:", unique["total_requests"], unique["total_tokens"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tama", type=Path, required=True)
    parser.add_argument("--scoped", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.tama, args.scoped, args.out)


if __name__ == "__main__":
    main()
