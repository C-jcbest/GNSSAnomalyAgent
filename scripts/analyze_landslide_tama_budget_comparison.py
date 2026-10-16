"""Replay the unchanged TAMA task adapter and append it to the frozen 37-arm comparison."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_observation_probe import evaluate_population
from analyze_landslide_review_utility_v2 import verify_receipts
from finalize_landslide_adaptive_evidence import verify_manifest
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_tama_adapter import REFERENCES, verify as verify_tama
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_missed_motion_repair import write_rows
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_visual import parse_activity


def replay_tama(tama, records, receipts):
    config = read_json(tama / "protocol.json")
    head = XGBClassifier()
    head.load_model(tama / "stage-xgboost.json")
    selections = read_json(tama / "selections.json")
    predictions, used = {}, {}
    for repeat in (0, 1):
        pattern_key = f"normal-pattern-r{repeat}"
        pattern = receipts[pattern_key]
        valid_pattern = False
        if pattern["status"] == "returned_json":
            value = json.loads(pattern["output"])
            valid_pattern = (isinstance(value, dict) and set(value) == {"normal_pattern"}
                and isinstance(value["normal_pattern"], str) and 1 <= len(value["normal_pattern"].strip()) <= 5000)
        for cid in config["case_ids"]:
            feature = records[cid][1]
            activity = np.full(len(feature.observed), -1)
            status = "failed"
            detect_key = f"{cid}-detect-r{repeat}"
            if valid_pattern:
                receipt = receipts[detect_key]
                try:
                    activity = parse_activity(json.loads(receipt["output"]), feature.observed)
                    status = "success"
                except (ValueError, KeyError, json.JSONDecodeError):
                    activity = np.full(len(feature.observed), -1)
            triggered = status == "success" and bool(np.any(activity == 1))
            if selections[f"{cid}-r{repeat}"]["positive_initial_trigger"] != triggered:
                raise ValueError("Positive-only adapter controller no longer replays")
            reviewed, reviewed_status = activity.copy(), status
            review_key = f"{cid}-review-r{repeat}"
            if triggered:
                receipt = receipts[review_key]
                try:
                    reviewed = parse_activity(json.loads(receipt["output"]), feature.observed)
                    reviewed_status = "success"
                except (ValueError, KeyError, json.JSONDecodeError):
                    reviewed = np.full(len(feature.observed), -1)
                    reviewed_status = "failed"
            codes = head.predict(feature.matrix).astype(int) + 1
            for stem, gate, state in (("tama_gnss_no_review", activity, status),
                                      ("tama_gnss_positive_review", reviewed, reviewed_status)):
                prediction = learned_stages(codes, gate, feature.supported)
                if state == "failed":
                    prediction = ActivityPrediction(gate, np.full(len(gate), -1), status="failed")
                method = f"{stem}_r{repeat}"
                predictions.setdefault(method, {})[cid] = prediction.to_dict()
                keys = [pattern_key]
                if valid_pattern:
                    keys.append(detect_key)
                if stem == "tama_gnss_positive_review" and triggered:
                    keys.append(review_key)
                used.setdefault(method, {})[cid] = keys
    if predictions != read_json(tama / "predictions.json"):
        raise ValueError("All four adapter predictions must reconstruct from actual answers")
    return predictions, used


def analyze(run, tama, registration, out):
    verify_freeze(run)
    verify_tama(tama)
    verify_manifest(registration)
    for name, expected in read_json(registration / "bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Before-inference comparator registration changed")
    for folder in (run, tama):
        if read_json(folder / "state.json")["status"] != "predictions_complete":
            raise ValueError("Complete both inference workflows before scoring")
        for name, expected in read_json(folder / "prediction-freeze.json").items():
            if sha(folder / name) != expected:
                raise ValueError("Frozen inference answer changed")
    original = run / "analysis-v1"
    verify_manifest(original)
    ids = read_json(run / "protocol.json")["case_ids"]
    if read_json(tama / "protocol.json")["case_ids"] != ids:
        raise ValueError("Nearest-neighbour comparison must use the identical population")
    if set(ids) & set(REFERENCES) or sha(run / "stage-xgboost.json") != sha(tama / "stage-xgboost.json"):
        raise ValueError("Training reference overlap or stage-head difference")
    records = {cid: load_case_features(run, cid) for cid in ids}
    for cid in ids:
        for name in ("input.json", "motion-31.json", "motion-61.json", "motion-91.json"):
            if sha(run / "inputs" / cid / name) != sha(tama / "inputs" / cid / name):
                raise ValueError("Adapter input differs from original comparison")
    receipts = verify_receipts(tama)
    extra, used = replay_tama(tama, records, receipts)
    predictions = {**read_json(original / "predictions.json"), **extra}
    if len(predictions) != 41:
        raise ValueError("The 37 original configurations must remain alongside four adapter arms")
    labels = load_labels(run / "reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    results, rows, per_case = evaluate_population(predictions, labels)
    for method, expected in read_json(original / "results.json").items():
        if results[method] != expected:
            raise ValueError("Appending adapter comparison altered a published original score")
    internal = verify_metrics(predictions, results, labels, records)
    with (original / "summary.csv").open(encoding="utf-8", newline="") as handle:
        original_rows = {row["method"]: row for row in csv.DictReader(handle)}
    daily, family_rows, correction_rows = [], [], []
    data = ROOT / read_json(run / "protocol.json")["source_bundle"]
    families = {cid: read_json(data / "mechanism-reference" / cid / "truth.json")["family"] for cid in ids}
    for row in rows:
        method = row["method"]
        row["internal_stage_macro_f1"] = internal[method]["macro_f1"]
        if method in used:
            keys = sorted({key for value in used[method].values() for key in value})
            row["deployment_requests"] = len(keys)
            row["reported_tokens"] = sum(receipts[key].get("usage", {}).get("total_tokens", 0) for key in keys)
        else:
            row["deployment_requests"] = int(original_rows[method]["deployment_requests"])
            row["reported_tokens"] = int(original_rows[method]["reported_tokens"])
        for family in sorted(set(families.values())):
            aggregate = aggregate_scores([results[method]["per_case"][cid] for cid in ids if families[cid] == family])
            family_rows.append({"method": method, "family": family, **aggregate["daily"],
                                "weak_recall": aggregate["weak_recall"], "coverage": aggregate["coverage"]})
        for cid, value in predictions[method].items():
            for day, actual in enumerate(labels[cid]):
                daily.append({"method": method, "case_id": cid, "day_index": day,
                    "actual_activity": actual["activity_label"], "actual_stage": actual["feature"],
                    "predicted_activity": value["activity"][day], "predicted_stage": value["stage"][day]})
    for repeat in (0, 1):
        comparisons = [(f"adaptive_no_feedback_r{repeat}__conflict_scope", f"tama_gnss_positive_review_r{repeat}"),
                       (f"adaptive_no_feedback_r{repeat}__conflict_scope", f"tama_gnss_no_review_r{repeat}"),
                       (f"tama_gnss_positive_review_r{repeat}", f"tama_gnss_no_review_r{repeat}")]
        for method, control in comparisons:
            for cid in ids:
                actual = np.array([int(item["activity_label"]) for item in labels[cid]])
                before = np.array(predictions[control][cid]["activity"])
                after = np.array(predictions[method][cid]["activity"])
                correction_rows.append({"method": method, "control": control, "case_id": cid, "family": families[cid],
                    **correction_utility(actual, before, after, actual >= 0),
                    "gained_tp": int(((actual == 1) & (before != 1) & (after == 1)).sum()),
                    "lost_tp": int(((actual == 1) & (before == 1) & (after != 1)).sum()),
                    "added_fp": int(((actual == 0) & (before != 1) & (after == 1)).sum()),
                    "removed_fp": int(((actual == 0) & (before == 1) & (after != 1)).sum())})
    costs = {"started": len(list((tama / "requests").glob("*.started.json"))), "receipts": len(receipts),
             "returned_json": sum(row["status"] == "returned_json" for row in receipts.values()),
             "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts.values()),
             "failures": read_json(tama / "failures.json"), "retries": 0,
             "inference_wall_seconds": read_json(tama / "state.json")["seconds"]}
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    for name, value in (("predictions.json", predictions), ("results.json", results), ("internal-stages.json", internal),
                        ("used-requests.json", used), ("actual-costs.json", costs),
                        ("checks.json", {"original_37_scores_unchanged": True, "adapter_case_replays": 48,
                                         "methods": 41, "daily_rows": len(daily), "family_rows": len(family_rows),
                                         "actual_payloads": len(receipts), "new_records": 0,
                                         "comparison_added_after_original_scores": True, "new_api_in_analysis": 0})):
        save_json(out / name, value)
    save_json(out / "prediction-freeze.json", {"predictions.json": sha(out / "predictions.json")})
    for name, values in (("summary.csv", rows), ("per-case.csv", per_case), ("daily.csv", daily),
                          ("family-results.csv", family_rows), ("corrections.csv", correction_rows)):
        write_rows(out / name, values)
    indexed = {row["method"]: row for row in rows}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    methods = ("direct_visual", "tama_gnss_no_review", "tama_gnss_positive_review", "adaptive_no_feedback")
    for repeat in (0, 1):
        names = [f"{stem}_r{repeat}" + ("__conflict_scope" if stem == "adaptive_no_feedback" else "") for stem in methods]
        for axis, metric in zip(axes, ("f1", "weak_recall"), strict=True):
            axis.bar(np.arange(4) + (repeat - .5) * .36, [indexed[name][metric] for name in names], width=.36,
                     label=f"Repeat {repeat}")
            axis.set_xticks(np.arange(4), ["Direct", "TAMA detect", "TAMA + review", "Raw conflict"], rotation=15)
            axis.set_ylim(0, 1)
            axis.set_ylabel(metric)
            axis.legend()
    figure.suptitle("Same twelve controlled records | unchanged GNSS task adaptation, not published TAMA scores")
    figure.tight_layout()
    figure.savefig(out / "comparison.png", dpi=150)
    plt.close(figure)
    save_json(out / "manifest.json", {"files": {str(path.relative_to(out)): sha(path)
                                               for path in out.rglob("*") if path.is_file()}})
    print({"costs": costs, "adapter": [row for row in rows if row["method"] in extra]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tama", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    analyze(arguments.run, arguments.tama, arguments.registration, arguments.out)
