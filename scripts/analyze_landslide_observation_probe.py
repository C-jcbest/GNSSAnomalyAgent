"""Score the frozen display factorial on targets and the common twelve records."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from finalize_landslide_adaptive_evidence import verify_metrics, verify_request_receipts
from probe_landslide_longer_motion_evidence import longer_conflict_scope
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_observation_probe import ARMS, verify
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_evaluation import (
    STAGES,
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    score_counts,
)
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_window_experiment import parse_window_activity


def evaluate_population(predictions, labels):
    results, summary, records = {}, [], []
    for method, cases in predictions.items():
        per_case, boundaries03, boundaries05 = {}, {}, {}
        for cid, value in cases.items():
            prediction = ActivityPrediction(np.array(value["activity"]), np.array(value["stage"]),
                                            status=value["status"], gated=value["gated"],
                                            stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            # This controlled schema marks weak activity separately from A/S/D.
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
            per_case[cid]["weak_days"] = int(weak.sum())
            per_case[cid]["weak_detected_days"] = int(np.count_nonzero(weak & (prediction.activity == 1)))
            boundaries03[cid] = boundary_errors(labels[cid], prediction.activity, .3)
            boundaries05[cid] = boundary_errors(labels[cid], prediction.activity, .5)
            records.append({"method": method, "case_id": cid, **per_case[cid]["daily"],
                            "weak_days": per_case[cid]["weak_days"],
                            "weak_detected_days": per_case[cid]["weak_detected_days"]})
        aggregate = aggregate_scores(list(per_case.values()))
        pairs = [pair for value in boundaries03.values() for pair in value["matched_pairs"]]
        if len(pairs) != aggregate["events"]["tp"]:
            raise ValueError("Event and boundary matching differ")
        event05 = score_counts(sum(len(row["matched_pairs"]) for row in boundaries05.values()),
                              sum(row["unmatched_prediction_events"] for row in boundaries05.values()),
                              sum(row["missed_events"] for row in boundaries05.values()))
        results[method] = {"aggregate": aggregate, "per_case": per_case,
                           "boundaries_iou03": boundaries03, "boundaries_iou05": boundaries05}
        summary.append({"method": method, **aggregate["daily"], "weak_recall": aggregate["weak_recall"],
                        "weak_days": aggregate["weak_days"], "event_f1_iou03": aggregate["events"]["f1"],
                        "event_f1_iou05": event05["f1"], "coverage": aggregate["coverage"],
                        "stage_macro_f1": aggregate["stage_macro_f1"],
                        "onset_mae_matched_days": float(np.mean([abs(p["onset_error_days"]) for p in pairs])) if pairs else None,
                        "offset_mae_matched_days": float(np.mean([abs(p["offset_error_days"]) for p in pairs])) if pairs else None})
    return results, summary, records


def csv_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_target_metrics(predictions, results, labels, records):
    """Keep absent stage classes visible in the deliberately selected population.

    The full-record verifier assumes all three reference stages occur. These five
    targets contain S/D but no A, so the repository macro may have two defined
    classes. Verify that convention and separately report the fixed ASD macro.
    """
    internal = {}
    for method, cases in predictions.items():
        actual_activity, predicted_activity, actual_stage, predicted_stage = [], [], [], []
        internal_actual, internal_predicted = [], []
        for cid, value in cases.items():
            activity, stage = np.array(value["activity"]), np.array(value["stage"])
            observed = records[cid][1].observed
            prediction = ActivityPrediction(activity, stage)
            prediction.validate(len(observed))
            if np.any(activity[~observed] != -1) or np.any(stage[~observed] != -1):
                raise ValueError("Target scoring invented a missing-day decision")
            for day, row in enumerate(labels[cid]):
                actual = int(row["activity_label"])
                if actual < 0:
                    continue
                expected_stage = STAGES[row["feature"]] if actual == 1 else 0
                actual_activity.append(actual)
                predicted_activity.append(int(activity[day] == 1))
                actual_stage.append(expected_stage)
                predicted_stage.append(int(stage[day]))
                if actual == 1:
                    internal_actual.append(expected_stage)
                    internal_predicted.append(int(stage[day]))
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_activity, predicted_activity, labels=[1], zero_division=0)
        aggregate = results[method]["aggregate"]
        for name, value in (("precision", precision[0]), ("recall", recall[0]), ("f1", f1[0])):
            if not np.isclose(value, aggregate["daily"][name]):
                raise ValueError(f"Independent target activity score differs: {method}")
        _, _, stage_f1, stage_support = precision_recall_fscore_support(
            actual_stage, predicted_stage, labels=[1, 2, 3], zero_division=0)
        defined = [code for code in (1, 2, 3) if code in actual_stage or code in predicted_stage]
        if not np.isclose(np.mean([stage_f1[code - 1] for code in defined]), aggregate["stage_macro_f1"]):
            raise ValueError(f"Independent defined-class target stage score differs: {method}")
        precision, recall, f1, support = precision_recall_fscore_support(
            internal_actual, internal_predicted, labels=[1, 2, 3], zero_division=0)
        internal[method] = {"scope": "True activity inside selected targets; missed activity and unknown retained",
            "precision_ASD": precision.tolist(), "recall_ASD": recall.tolist(), "f1_ASD": f1.tolist(),
            "support_ASD": support.tolist(), "macro_f1": float(np.mean(f1)),
            "end_to_end_fixed_ASD_macro_f1": float(np.mean(stage_f1)),
            "end_to_end_reference_support_ASD": stage_support.tolist(),
            "repository_defined_stage_classes": defined,
            "definite_stage_coverage": float(np.mean(np.array(internal_predicted) > 0))}
    return internal


def analyze(run, out):
    verify(run)
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Freeze all display predictions before scoring")
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError("Display predictions changed")
    config = read_json(run / "protocol.json")
    receipts = verify_request_receipts(run)
    if len(receipts) != config["maximum_new_requests"]:
        raise ValueError("Actual request population differs")
    target_map = {row["case_id"]: row for row in config["targets"]}
    records = {cid: load_case_features(run, cid) for cid in config["all_case_ids"]}
    full_labels = load_labels(run / "reference-subset.csv", {cid: row[0] for cid, row in records.items()})
    target_labels = load_labels(run / "target-reference-subset.csv", {cid: records[cid][0] for cid in target_map})
    prior = read_json(run / "prior-comparison-predictions.json")
    full_predictions = dict(prior)
    reviews = read_json(run / "target-reviews.json")
    cached = read_json(run / "cached-target-receipts.json")
    for rep in range(2):
        values = {}
        for cid, target in target_map.items():
            payload = json.loads(cached[f"{cid}-r{rep}"]["output"])
            values[cid] = {"activity": parse_window_activity(
                payload, records[cid][1].observed, target["start"], target["stop"]).tolist(),
                "status": "success", "start": target["start"], "stop": target["stop"]}
        reviews[f"cached_narrow_raw_r{rep}"] = values
    expected_targets = set(target_map)
    if len(reviews) != 10 or any(set(cases) != expected_targets for cases in reviews.values()):
        raise ValueError("Target comparison must include all arms, cached control and repeats")
    stage_head = XGBClassifier()
    stage_head.load_model(run / "stage-xgboost.json")
    stage_codes = {cid: stage_head.predict(feature.matrix).astype(int) + 1
                   for cid, (_, feature) in records.items()}
    source = Path(config["source_run"])
    initial = read_json(source / "original-predictions.json")
    scopes = {}
    policy = ReviewPolicy()
    for rep in range(2):
        for cid in target_map:
            case, feature = records[cid]
            _, _, scope, _ = longer_conflict_scope(np.array(initial[f"direct_visual_r{rep}"][cid]["activity"]),
                feature, raw_motion_evidence(case, feature), raw_motion_evidence(case, feature, 84), policy)
            target = target_map[cid]
            scope[:target["start"]] = False
            scope[target["stop"]:] = False
            scopes[f"{cid}-r{rep}"] = scope
    target_predictions, changes = {}, []
    for method, cases in reviews.items():
        rep = int(method[-1])
        target_predictions[method] = {}
        baseline = prior[f"extended84_evidence_scope_r{rep}"]
        full_values = dict(baseline)
        for cid, value in cases.items():
            feature = records[cid][1]
            activity = np.array(value["activity"], dtype=int)
            prediction = learned_stages(stage_codes[cid], activity, feature.supported)
            if value["status"] == "failed":
                prediction.status = "failed"
            target_predictions[method][cid] = prediction.to_dict()
            # All display arms inherit exactly the failed84 trial's revision rights.
            revised = np.array(baseline[cid]["activity"], dtype=int)
            scope = scopes[f"{cid}-r{rep}"]
            revised[scope] = activity[scope]
            full_values[cid] = learned_stages(stage_codes[cid], revised, feature.supported).to_dict()
            original = np.array(prior[f"adaptive_no_feedback_r{rep}__conflict_scope"][cid]["activity"])
            actual = np.array([int(row["activity_label"]) for row in full_labels[cid]])
            removed = (original == 1) & (revised != 1)
            gained = (original != 1) & (revised == 1)
            changes.append({"method": method, "case_id": cid,
                            "lost_tp_vs_base42": int(np.count_nonzero(removed & (actual == 1))),
                            "gained_tp_vs_base42": int(np.count_nonzero(gained & (actual == 1))),
                            "removed_fp_vs_base42": int(np.count_nonzero(removed & (actual == 0))),
                            "gained_fp_vs_base42": int(np.count_nonzero(gained & (actual == 0)))})
        if method.startswith("obs_"):
            full_predictions[method] = full_values
        else:
            for cid in full_values:
                if full_values[cid] != baseline[cid]:
                    raise ValueError("Cached replay must exactly recover the frozen base84 result")
    if len(full_predictions) != 47:
        raise ValueError("Unified comparison must contain prior39 plus eight display configurations")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "target-predictions.json", target_predictions)
    save_json(out / "predictions.json", full_predictions)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in (
        "target-predictions.json", "predictions.json")})
    target_results, target_summary, target_records = evaluate_population(target_predictions, target_labels)
    results, summary, full_records = evaluate_population(full_predictions, full_labels)
    target_internal = verify_target_metrics(target_predictions, target_results, target_labels, records)
    internal = verify_metrics(full_predictions, results, run / "reference-subset.csv", run)
    for row in target_summary:
        row["true_activity_stage_macro_f1"] = target_internal[row["method"]]["macro_f1"]
        row["end_to_end_fixed_ASD_macro_f1"] = target_internal[row["method"]]["end_to_end_fixed_ASD_macro_f1"]
        row["repository_defined_stage_classes"] = "/".join(
            str(code) for code in target_internal[row["method"]]["repository_defined_stage_classes"])
    for row in summary:
        row["true_activity_stage_macro_f1"] = internal[row["method"]]["macro_f1"]
    save_json(out / "target-results.json", target_results)
    save_json(out / "results.json", results)
    save_json(out / "true-activity-stage-scores.json", {"targets": target_internal, "full": internal})
    indexed = {row["method"]: row for row in target_summary}
    decisions, factorial = [], []
    for rep in range(2):
        candidate = f"obs_wide_blocks_r{rep}"
        control = f"obs_narrow_raw_r{rep}"
        a, b = indexed[candidate], indexed[control]
        step_ids = [row["case_id"] for row in config["targets"] if "step negative" in row["role"]]
        step_fp = {method: sum(target_results[method]["per_case"][cid]["daily"]["fp"] for cid in step_ids)
                   for method in (candidate, control)}
        tp_control = target_results[control]["per_case"]["case_0082"]["daily"]["tp"]
        tp_candidate = target_results[candidate]["per_case"]["case_0082"]["daily"]["tp"]
        decision = {"repeat": rep + 1, "f1_difference": a["f1"] - b["f1"],
                    "weak_recall_difference": a["weak_recall"] - b["weak_recall"],
                    "step_fp_difference": step_fp[candidate] - step_fp[control],
                    "case0082_tp_difference": tp_candidate - tp_control}
        decision["passes"] = (decision["f1_difference"] >= 0 and decision["weak_recall_difference"] > 0
                              and decision["step_fp_difference"] <= 0 and decision["case0082_tp_difference"] >= 0)
        decisions.append(decision)
        for treatment, comparison, factor in (
            ("wide_raw", "narrow_raw", "raw context"),
            ("narrow_blocks", "narrow_raw", "block display within narrow view"),
            ("wide_blocks", "wide_raw", "block display within wide view"),
            ("wide_blocks", "narrow_blocks", "context with block display")):
            x, y = indexed[f"obs_{treatment}_r{rep}"], indexed[f"obs_{comparison}_r{rep}"]
            factorial.append({"repeat": rep + 1, "factor": factor,
                              "f1_difference": x["f1"] - y["f1"],
                              "weak_recall_difference": x["weak_recall"] - y["weak_recall"],
                              "tp_difference": x["tp"] - y["tp"], "fp_difference": x["fp"] - y["fp"]})
    decision = {"pre_inference_stop": config["primary_stop"], "repetitions": decisions,
                "proceed_to_new_confirmation": all(row["passes"] for row in decisions),
                "scope": config["scope"], "factorial_differences": factorial}
    save_json(out / "candidate-decision.json", decision)
    csv_rows(out / "target-summary.csv", target_summary)
    csv_rows(out / "target-per-record.csv", target_records)
    csv_rows(out / "summary.csv", summary)
    csv_rows(out / "per-record.csv", full_records)
    csv_rows(out / "full-changes-vs-base42.csv", changes)
    arms = list(ARMS)
    fig, panels = plt.subplots(1, 3, figsize=(13, 4.5))
    for rep in range(2):
        for panel, metric, label in zip(panels, ("f1", "weak_recall", "fp"),
                                         ("Target activity F1", "Target weak recall", "Target false positive days")):
            panel.bar(np.arange(4) + (rep - .5) * .36,
                      [indexed[f"obs_{arm}_r{rep}"][metric] for arm in arms],
                      width=.34, label=f"repeat {rep + 1}")
            panel.set_xticks(range(4), ["narrow raw", "wide raw", "narrow blocks", "wide blocks"],
                             rotation=25, ha="right")
            panel.set_ylabel(label)
            panel.grid(axis="y", alpha=.2)
            panel.legend()
            if metric != "fp":
                panel.set_ylim(0, 1.02)
    fig.suptitle("Five viewed development targets: same prompt and target; context x block display")
    fig.tight_layout()
    fig.savefig(out / "observation-factorial.png", dpi=180)
    plt.close(fig)
    method_costs = {}
    for method in reviews:
        if method.startswith("obs_"):
            arm, repetition = method.removeprefix("obs_").rsplit("_r", 1)
            request_suffix = f"{arm}-r{repetition}"
            selected = [row for row in receipts if row["request_id"].endswith(request_suffix)]
            if len(selected) != 5:
                raise ValueError("Each display arm must have five actual target calls")
            rep = int(method[-1])
            prior_cost = read_json(source / "analysis-v3/costs.json")["per_extended_repetition"][str(rep)]
            cached_cost = [cached[f"{cid}-r{rep}"] for cid in target_map]
            method_costs[method] = {
                "new_target_requests": len(selected),
                "new_target_tokens": sum(row["usage"]["total_tokens"] for row in selected),
                "new_target_seconds": sum(row["seconds"] for row in selected),
                "full_method_requests": prior_cost["requests"],
                "full_method_tokens": prior_cost["tokens"] - sum(row["usage"]["total_tokens"] for row in cached_cost)
                                      + sum(row["usage"]["total_tokens"] for row in selected),
                "scope": "Five historical target receipts replaced by five new ones; shared initial "
                         "and other local calls never additive across hypothetical full methods"}
    save_json(out / "costs.json", {"new_unique_requests": len(receipts),
              "new_unique_tokens": sum(row["usage"]["total_tokens"] for row in receipts),
              "new_unique_seconds": sum(row["seconds"] for row in receipts), "per_method": method_costs})
    save_json(out / "verification.json", {"new_actual_payloads": len(receipts),
              "same_inputs_and_stage_head": 12, "target_groups_independently_scored": 10,
              "whole_record_groups_independently_scored": 47, "failed_targets": len(read_json(run / "failures.json")),
              "cached_replay_equals_prior84": True, "unseen_records": 0,
              "same_stage_head_sha256": sha(run / "stage-xgboost.json")})
    save_json(out / "manifest.json", {"source_bindings": {str(run / name): sha(run / name) for name in (
        "freeze.json", "prediction-freeze.json")},
        "files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Display mechanism decision:", decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.out)


if __name__ == "__main__":
    main()
