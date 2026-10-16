"""Score the pre-frozen original method on a new seed, preserving all tradeoffs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors, paired_record_interval
from close_landslide_utility_round import comparison_table
from finalize_landslide_adaptive_evidence import (
    verify_manifest,
    verify_metrics,
    verify_request_receipts,
)
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_evidence_display_confirmation import numerical_predictions
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_missed_motion_repair import write_rows

from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    score_counts,
)
from gnss_sim.landslide_review_utility import correction_utility


def analyze(run, scoped, out):
    verify_freeze(run)
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError("Main inference changed")
    data = ROOT / read_json(run / "protocol.json")["source_bundle"]
    verified_files = verify_manifest(data) + verify_manifest(scoped)
    for name, expected in read_json(data / "pre-generation-freeze.json").items():
        if name.startswith("source/") and sha(ROOT / name.removeprefix("source/")) != expected:
            raise ValueError("Live pre-generation source changed")
    for name, expected in read_json(data / "training-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Original training or validation artifact changed")
    protocol = read_json(data / "protocol.json")
    if sha(scoped / "source.py") != protocol["candidate_source_sha256"]:
        raise ValueError("Scope execution differs from the registered original method")
    folder = scoped / run.name
    for name, expected in read_json(folder / "prediction-freeze.json").items():
        if sha(folder / name) != expected:
            raise ValueError("Scope predictions changed")
    predictions = read_json(folder / "predictions.json")
    ids = protocol["case_ids"]
    if len(predictions) != 29 or any(set(values) != set(ids) for values in predictions.values()):
        raise ValueError("Expected twenty visual scope arms and nine common numerical controls")
    cases = {cid: load_case_features(run, cid) for cid in ids}
    recomputed = numerical_predictions(data, cases)
    for method, values in recomputed.items():
        if predictions[method] != values:
            raise ValueError("Common numerical predictions do not reconstruct")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "predictions.json", predictions)
    save_json(out / "prediction-freeze.json", {"predictions.json": sha(out / "predictions.json")})
    labels = load_labels(run / "reference-subset.csv", {cid: pair[0] for cid, pair in cases.items()})
    costs = read_json(run / "costs.json")
    receipts = verify_request_receipts(run)
    results, summaries, daily, events, changes, groups = {}, [], [], [], {}, []
    families = {cid: read_json(data / "mechanism-reference" / cid / "truth.json")["family"] for cid in ids}
    for method, values in predictions.items():
        per_case, pairs = {}, []
        weak_total, weak_detected = 0, 0
        event05 = np.zeros(3, dtype=int)
        for cid, value in values.items():
            activity = np.array(value["activity"])
            prediction = ActivityPrediction(activity, np.array(value["stage"]),
                status=value["status"], gated=value["gated"], stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
            weak_total += int(weak.sum())
            weak_detected += int((weak & (activity == 1)).sum())
            for iou in (.3, .5):
                detail = boundary_errors(labels[cid], activity, iou)
                events.append({"method": method, "case_id": cid, "iou": iou,
                    "matched": len(detail["matched_pairs"]),
                    "unmatched_predictions": detail["unmatched_prediction_events"], "missed": detail["missed_events"]})
                if iou == .3:
                    pairs.extend(detail["matched_pairs"])
                else:
                    event05 += [len(detail["matched_pairs"]), detail["unmatched_prediction_events"], detail["missed_events"]]
            if method.startswith("adaptive_no_feedback") and "__conflict_scope" in method:
                stem = method.partition("__")[0]
                for control in (stem + "__whole_target", "direct_visual_" + stem.rsplit("_", 1)[1]):
                    before = np.array(predictions[control][cid]["activity"])
                    change = correction_utility(actual, before, activity, actual >= 0)
                    change.update({
                        "gained_tp": int(((actual == 1) & (before != 1) & (activity == 1)).sum()),
                        "lost_tp": int(((actual == 1) & (before == 1) & (activity != 1)).sum()),
                        "added_fp": int(((actual == 0) & (before != 1) & (activity == 1)).sum()),
                        "removed_fp": int(((actual == 0) & (before == 1) & (activity != 1)).sum()),
                    })
                    changes.setdefault(method, {}).setdefault(control, {})[cid] = change
            for day, row in enumerate(labels[cid]):
                daily.append({"method": method, "case_id": cid, "day_index": day,
                    "actual_activity": row["activity_label"], "actual_stage": row["feature"],
                    "predicted_activity": value["activity"][day], "predicted_stage": value["stage"][day]})
        aggregate = aggregate_scores(list(per_case.values()))
        results[method] = {"aggregate": aggregate, "per_case": per_case}
        stem = method.partition("__")[0]
        cost = costs.get(stem, {})
        summaries.append({"method": method, **aggregate["daily"], "weak_recall": weak_detected / weak_total,
            "weak_days": weak_total, "coverage": aggregate["coverage"],
            "event_f1_iou03": aggregate["events"]["f1"], "event_f1_iou05": score_counts(*event05)["f1"],
            "onset_mae_days": float(np.mean([abs(pair["onset_error_days"]) for pair in pairs])) if pairs else None,
            "offset_mae_days": float(np.mean([abs(pair["offset_error_days"]) for pair in pairs])) if pairs else None,
            "stage_macro_f1": aggregate["stage_macro_f1"], "requests": cost.get("requests", 0),
            "reported_tokens": cost.get("tokens", 0)})
        for family in sorted(set(families.values())):
            measured = aggregate_scores([per_case[cid] for cid in ids if families[cid] == family])
            groups.append({"method": method, "family": family, **measured["daily"]})
    internal = verify_metrics(predictions, results, run / "reference-subset.csv", run)
    for row in summaries:
        row["internal_stage_macro_f1"] = internal[row["method"]]["macro_f1"]
    indexed = {row["method"]: row for row in summaries}
    decisions, intervals = [], {}
    for rep in (0, 1):
        stem = f"adaptive_no_feedback_r{rep}"
        main, whole, direct, fixed = [indexed[name] for name in (
            stem + "__conflict_scope", stem + "__whole_target", f"direct_visual_r{rep}",
            f"fixed_image_review_r{rep}__whole_target")]
        by_control = changes[main["method"]]
        scope_change = by_control[whole["method"]]
        direct_change = by_control[direct["method"]]
        scope_gate = {
            "lost_tp_zero": sum(row["lost_tp"] for row in scope_change.values()) == 0,
            "weak_recall_nonloss": main["weak_recall"] >= whole["weak_recall"],
            "fp_reduced": main["fp"] < whole["fp"], "f1_nonloss": main["f1"] >= whole["f1"],
        }
        method_gate = {
            "f1_gain_over_direct": main["f1"] > direct["f1"], "f1_gain_over_fixed": main["f1"] > fixed["f1"],
            "weak_recall_nonloss_over_direct": main["weak_recall"] >= direct["weak_recall"],
            "positive_net_definite_correction": sum(row["net_correct"] for row in direct_change.values()) > 0,
            "multiple_records_helped": sum(row["net_correct"] > 0 for row in direct_change.values()) >= 2,
        }
        decisions.append({"repeat": rep, "scope_gate": scope_gate, "method_gate": method_gate,
            "scope_passed": all(scope_gate.values()), "method_passed": all(method_gate.values())})
        for control in (whole["method"], direct["method"], fixed["method"], "xgboost_84"):
            intervals[f"{main['method']}-minus-{control}"] = paired_record_interval(results, control, main["method"])
    receipt_paths = sorted((run / "requests").glob("*.response.json"))
    saved_receipts = [read_json(path) for path in receipt_paths]
    actual_costs = {
        "new_started": len(list((run / "requests").glob("*.started.json"))),
        "new_local_receipts": len(saved_receipts),
        "new_returned_json": sum(row["status"] == "returned_json" for row in saved_receipts),
        "new_reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in saved_receipts),
        "usage_unknown_receipts": sum("usage" not in row for row in saved_receipts),
        "new_records": 12, "inference_retries": 0,
        "failures": len(read_json(run / "failures.json")),
    }
    for name, value in (("results.json", results), ("internal-stages.json", internal), ("changes.json", changes),
                        ("paired-intervals.json", intervals), ("actual-costs.json", actual_costs),
                        ("decision.json", {"repeats": decisions,
                            "scope_confirmed": all(row["scope_passed"] for row in decisions),
                            "method_confirmed": all(row["method_passed"] for row in decisions),
                            "field_accuracy_confirmed": False, "new_tuning": False})):
        save_json(out / name, value)
    for name, rows in (("summary.csv", summaries), ("daily.csv", daily), ("events.csv", events), ("groups.csv", groups)):
        write_rows(out / name, rows)
    string_rows = [{key: str(value) if value is not None else "" for key, value in row.items()} for row in summaries]
    (out / "results.md").write_text("# 原范围方法跨种子冻结确认\n\n" + comparison_table(string_rows), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for rep in (0, 1):
        names = [f"direct_visual_r{rep}", f"fixed_image_review_r{rep}__whole_target",
                 f"adaptive_no_feedback_r{rep}__whole_target", f"adaptive_no_feedback_r{rep}__conflict_scope",
                 f"adaptive_no_feedback_r{rep}__matched_center_scope", "xgboost_84"]
        for axis, metric in zip(axes, ("f1", "weak_recall"), strict=True):
            axis.bar(np.arange(6) + (rep - .5) * .36, [indexed[name][metric] for name in names],
                     width=.36, label=f"Repeat {rep+1}")
            axis.set_xticks(range(6), ["Direct", "Fixed", "Whole", "Conflict", "Centre", "XGB84"])
            axis.set_ylabel(metric)
            axis.set_ylim(0, 1.02)
            axis.legend()
    fig.suptitle("New seed, unchanged method | controlled recovery, no field-accuracy claim")
    fig.tight_layout()
    fig.savefig(out / "cross-seed-comparison.png", dpi=150)
    plt.close(fig)
    save_json(out / "checks.json", {
        "data_scope_manifest_files": verified_files, "methods_independently_scored": len(internal),
        "request_payloads_verified": len(receipts), "numerical_records_recomputed": len(recomputed) * len(ids),
        "daily_rows": len(daily), "pre_generation_method_and_scorer_unchanged": True,
        "references_used_only_after_inference_frozen": True,
    })
    print({"decisions": decisions, "costs": actual_costs}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scoped", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run.resolve(), args.scoped.resolve(), args.out.resolve())
