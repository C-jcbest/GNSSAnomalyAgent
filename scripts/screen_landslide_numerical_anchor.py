"""Test numerical-first supplementation on the fully cached first repeat, API-free."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from analyze_landslide_review_utility_v2 import verify_receipts
from analyze_landslide_utility_prototype import strong_additional_controls
from probe_landslide_revision_scope import target_scopes
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_background_benchmark import recompute_numerical
from summarize_landslide_missed_motion_repair import write_rows
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import raw_motion_evidence, review_prompt
from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    score_counts,
)
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_window_experiment import parse_window_activity


def screen(prepared, out):
    for name, expected in read_json(prepared / "bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Numerical anchor preflight changed")
    protocol = read_json(prepared / "protocol.json")
    complete, data = Path(protocol["complete"]), Path(protocol["data"])
    original = data.parent / "background-review-v1"
    ids = protocol["case_ids"]
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "protocol.json", {**protocol, "scored_repeat": 0, "new_api": 0,
        "early_gate": "Necessary first-repeat criteria from frozen preflight; if any fail, both-repeat "
        "continuation is impossible and three missing repeat1 calls are not collected. No repeat1 estimates filled"})
    records = {cid: load_case_features(complete, cid) for cid in ids}
    local = {**read_json(complete / "cached-locals.json"), **read_json(complete / "new-locals.json")}
    selections = read_json(prepared / "selections.json")
    head = XGBClassifier()
    head.load_model(data / "stage-xgboost.json")
    predictions = recompute_numerical(data, records)
    all_records = {cid: load_case_features(complete, cid) for cid in read_json(complete / "protocol.json")["case_ids"]}
    additional = strong_additional_controls(data, all_records, head)
    predictions.update({name: {cid: values[cid] for cid in ids} for name, values in additional.items()})
    prior = data.parent / "context-observation-development-v1"
    legacy = read_json(prior / "analysis-v1/predictions-with-controls.json")
    old_used = read_json(prior / "used-requests.json")
    used = {}
    for name in ("direct_visual_r0", "adaptive_local_r0", "fixed_local_r0"):
        predictions[name] = legacy[name]
        used[name] = old_used[name]
    permissions, actual_payload_keys = {}, set()
    for cid, (case, feature) in records.items():
        baseline = np.array(predictions["xgboost_84"][cid]["activity"])
        if baseline.tolist() != selections[cid]["numerical_initial"]:
            raise ValueError("Registered numerical initial changed")
        long = raw_motion_evidence(case, feature)
        neighbourhood = np.zeros(len(baseline), dtype=bool)
        neighbourhood[selections[cid]["neighbourhood"]] = True
        codes = head.predict(feature.matrix).astype(int) + 1
        short_motion = (feature.progressive_rate >= .03) & (feature.signal_to_noise >= 2)
        long_motion = (long["rate"] >= .03) & (long["signal_to_noise"] >= 2)
        raw_motion = feature.observed & (short_motion | long_motion)
        for arm in ("numerical_anchor_visual_supplement", "numerical_anchor_fixed", "numerical_anchor_raw_supplement"):
            targets = selections[cid]["fixed" if arm == "numerical_anchor_fixed" else "adaptive"]
            authority = target_scopes(neighbourhood, feature.observed, targets)[0] & (baseline != 1)
            activity = baseline.copy()
            method = arm + "_r0"
            keys = []
            if arm == "numerical_anchor_raw_supplement":
                activity[authority & raw_motion] = 1
            else:
                for target in targets:
                    key = f"{cid}-window-{target['index']:02d}-image-r0"
                    if key not in local:
                        raise ValueError("The first repeat is not fully cached; do not score partial coverage")
                    owner = original if (original / "requests" / f"{key}.response.json").exists() else complete
                    receipt = read_json(owner / "requests" / f"{key}.response.json")
                    prompt = review_prompt(feature, long, target["start"], target["stop"], False)
                    if receipt["prompt"] != prompt:
                        raise ValueError("Cached visual answer has a different prompt")
                    reviewed = np.full(len(activity), -1)
                    if local[key]["status"] == "success":
                        reviewed = parse_window_activity(json.loads(receipt["output"]), feature.observed, target["start"], target["stop"])
                    np.testing.assert_array_equal(reviewed, local[key]["activity"])
                    permission = target_scopes(neighbourhood, feature.observed, [target])[0] & (baseline != 1)
                    activity[permission] = reviewed[permission]
                    keys.append(key)
                    actual_payload_keys.add(key)
            if np.any(activity[baseline == 1] != 1) or np.any(activity[~authority] != baseline[~authority]):
                raise ValueError("Numerical anchor or revision authority was violated")
            predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
            permissions.setdefault(method, {})[cid] = np.flatnonzero(authority).tolist()
            used.setdefault(method, []).extend(keys)
    predictions.pop("ungated_rule", None)
    save_json(out / "predictions.json", predictions)
    save_json(out / "permissions.json", permissions)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in ("predictions.json", "permissions.json", "protocol.json", "source.py")})
    receipts = verify_receipts(prior / "cached")
    receipts.update(verify_receipts(complete))
    # References are read only after the complete first-repeat predictions freeze.
    with (data / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        reference = [row for row in csv.DictReader(handle) if row["case_id"] in ids]
    write_rows(out / "reference-subset.csv", reference)
    labels = load_labels(out / "reference-subset.csv", {cid: records[cid][0] for cid in ids})
    results, summary, changes, daily, events = {}, [], {}, [], []
    for method, values in predictions.items():
        per_case, boundaries = {}, []
        weak_total, weak_detected = 0, 0
        events05 = np.zeros(3, dtype=int)
        for cid, value in values.items():
            activity = np.array(value["activity"])
            prediction = ActivityPrediction(activity, np.array(value["stage"]))
            per_case[cid] = evaluate_case(labels[cid], prediction)
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
            weak_total += int(weak.sum())
            weak_detected += int((weak & (activity == 1)).sum())
            boundaries.extend(boundary_errors(labels[cid], activity, .3)["matched_pairs"])
            detail = boundary_errors(labels[cid], activity, .5)
            events05 += [len(detail["matched_pairs"]), detail["unmatched_prediction_events"], detail["missed_events"]]
            events.append({"method": method, "case_id": cid, "iou": .5, "matched": len(detail["matched_pairs"]),
                           "unmatched_predictions": detail["unmatched_prediction_events"], "missed": detail["missed_events"]})
            if method.startswith("numerical_anchor"):
                before = np.array(predictions["xgboost_84"][cid]["activity"])
                changes.setdefault(method, {})[cid] = {**correction_utility(actual, before, activity, actual >= 0),
                    "gained_tp": int(((actual == 1) & (before != 1) & (activity == 1)).sum()),
                    "lost_tp": int(((actual == 1) & (before == 1) & (activity != 1)).sum()),
                    "added_fp": int(((actual == 0) & (before != 1) & (activity == 1)).sum())}
            for day, row in enumerate(labels[cid]):
                daily.append({"method": method, "case_id": cid, "day_index": day, "actual_activity": row["activity_label"],
                              "actual_stage": row["feature"], "predicted_activity": value["activity"][day], "predicted_stage": value["stage"][day]})
        aggregate = aggregate_scores(list(per_case.values()))
        results[method] = {"aggregate": aggregate, "per_case": per_case}
        flow = [receipts[key] for key in used.get(method, [])]
        summary.append({"method": method, **aggregate["daily"], "weak_recall": weak_detected / weak_total,
            "weak_days": weak_total, "coverage": aggregate["coverage"], "event_f1_iou03": aggregate["events"]["f1"],
            "event_f1_iou05": score_counts(*events05)["f1"],
            "onset_mae_days": float(np.mean([abs(row["onset_error_days"]) for row in boundaries])) if boundaries else None,
            "offset_mae_days": float(np.mean([abs(row["offset_error_days"]) for row in boundaries])) if boundaries else None,
            "stage_macro_f1": aggregate["stage_macro_f1"], "requests": len(flow),
            "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in flow)})
    stages = verify_metrics(predictions, results, labels, records)
    for row in summary:
        row["internal_stage_macro_f1"] = stages[row["method"]]["macro_f1"]
    indexed = {row["method"]: row for row in summary}
    main, base, raw = [indexed[name] for name in ("numerical_anchor_visual_supplement_r0", "xgboost_84", "numerical_anchor_raw_supplement_r0")]
    record_changes = changes["numerical_anchor_visual_supplement_r0"]
    criteria = {"f1_gain_over_numeric_initial": main["f1"] > base["f1"], "f1_gain_over_raw_only": main["f1"] > raw["f1"],
        "weak_recall_gain": main["weak_recall"] > base["weak_recall"],
        "multiple_positive_correction_records": sum(row["net_correct"] > 0 for row in record_changes.values()) >= 2,
        "net_correct_positive": sum(row["net_correct"] for row in record_changes.values()) > 0}
    allows = all(criteria.values())
    for name, value in (("results.json", results), ("changes.json", changes), ("internal-stages.json", stages),
                        ("decision.json", {"allows_three_missing_repeat1_calls": allows, "criteria": criteria,
                         "repeat0_only": True, "repeat1_unscored": True, "new_api": 0,
                         "reason": "Both-repeat registered gate is impossible if necessary first-repeat criteria fail"} )):
        save_json(out / name, value)
    write_rows(out / "summary.csv", summary)
    write_rows(out / "daily.csv", daily)
    write_rows(out / "events.csv", events)
    save_json(out / "checks.json", {"methods_independently_scored": len(stages), "daily_rows": len(daily),
        "cached_new_flow_payloads": len(actual_payload_keys), "original_numeric_positive_retained": True,
        "outside_revision_scope_unchanged": True, "new_api": 0, "repeat1_predictions_not_created": True})
    print({"allows_three_missing_calls": allows, "criteria": criteria, "main_f1": main["f1"], "base_f1": base["f1"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    screen(args.prepared.resolve(), args.out.resolve())
