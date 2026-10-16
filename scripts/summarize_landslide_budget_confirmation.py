"""Check exports and report every family, event and actual request in the frozen trial."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from analyze_landslide_review_utility_v2 import verify_receipts
from finalize_landslide_adaptive_evidence import verify_manifest
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_missed_motion_repair import write_rows

from gnss_sim.landslide_evaluation import aggregate_scores
from gnss_sim.landslide_review_utility import correction_utility


def summarize(run, out):
    verify_freeze(run)
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Summarize only the complete two-repeat inference")
    analysis = run / "analysis-v1"
    verify_manifest(analysis)
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError("Original frozen prediction changed")
    config = read_json(run / "protocol.json")
    data = ROOT / config["source_bundle"]
    families = {cid: read_json(data / "mechanism-reference" / cid / "truth.json")["family"]
                for cid in config["case_ids"]}
    population = dict(Counter(families.values()))
    if len(population) != 6 or set(population.values()) != {2}:
        raise ValueError("Actual population must include two records per family")
    receipts = verify_receipts(run)
    predictions = read_json(analysis / "predictions.json")
    results = read_json(analysis / "results.json")
    records = {cid: load_case_features(run, cid) for cid in config["case_ids"]}
    labels = load_labels(run / "reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    family_rows, event_rows = [], []
    for method, values in predictions.items():
        for cid, value in values.items():
            feature = records[cid][1]
            activity = np.asarray(value["activity"])
            stages = np.asarray(value["stage"])
            if len(activity) != 730 or len(stages) != 730:
                raise ValueError("Prediction length differs from full daily axis")
            if np.any(activity[~feature.observed] != -1):
                raise ValueError("Missing observation acquired a definite activity label")
            if np.any((stages > 0) & ((activity != 1) | ~feature.supported)):
                raise ValueError("A/S/D escapes confirmed supported activity")
        for family in sorted(population):
            selected = [results[method]["per_case"][cid] for cid in values if families[cid] == family]
            aggregate = aggregate_scores(selected)
            family_rows.append({"method": method, "family": family, **aggregate["daily"],
                                "weak_recall": aggregate["weak_recall"], "coverage": aggregate["coverage"]})
        for threshold, key in ((.3, "boundaries_iou03"), (.5, "boundaries_iou05")):
            for cid, detail in results[method][key].items():
                event_rows.append({"method": method, "case_id": cid, "iou": threshold,
                                   "matched": len(detail["matched_pairs"]),
                                   "unmatched_predictions": detail["unmatched_prediction_events"],
                                   "missed": detail["missed_events"]})
    daily_count = 0
    daily_seen = set()
    with (analysis / "daily.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method, cid, day = row["method"], row["case_id"], int(row["day_index"])
            key = (method, cid, day)
            if key in daily_seen:
                raise ValueError("Daily export duplicates a prediction")
            daily_seen.add(key)
            for field, array in (("predicted_activity", "activity"), ("predicted_stage", "stage")):
                if int(row[field]) != predictions[method][cid][array][day]:
                    raise ValueError("Daily export differs from frozen prediction")
            if (int(row["actual_activity"]) != int(labels[cid][day]["activity_label"])
                    or row["actual_stage"] != labels[cid][day]["feature"]):
                raise ValueError("Daily export differs from registered reference")
            daily_count += 1
    if daily_count != 37 * 12 * 730:
        raise ValueError("Daily export is incomplete")
    schedules = read_json(analysis / "schedules.json")
    for schedule in schedules.values():
        if not (len(schedule["adaptive"]) == len(schedule["fixed_early"])
                == len(schedule["fixed_late"]) == schedule["K"] <= 2):
            raise ValueError("Matched controls differ in per-record review count")
    correction_rows = []
    for repeat in (0, 1):
        for variant in ("whole_target", "conflict_scope"):
            method = f"adaptive_no_feedback_r{repeat}__{variant}"
            controls = [f"direct_visual_r{repeat}"] + [
                f"{stem}_r{repeat}__{variant}" for stem in
                ("fixed_image_review", "same_count_fixed_early", "same_count_fixed_late")]
            for control in controls:
                for cid in config["case_ids"]:
                    actual = np.array([int(item["activity_label"]) for item in labels[cid]])
                    before = np.array(predictions[control][cid]["activity"])
                    after = np.array(predictions[method][cid]["activity"])
                    correction_rows.append({"method": method, "control": control, "case_id": cid,
                        "family": families[cid], **correction_utility(actual, before, after, actual >= 0),
                        "gained_tp": int(((actual == 1) & (before != 1) & (after == 1)).sum()),
                        "lost_tp": int(((actual == 1) & (before == 1) & (after != 1)).sum()),
                        "added_fp": int(((actual == 0) & (before != 1) & (after == 1)).sum()),
                        "removed_fp": int(((actual == 0) & (before == 1) & (after != 1)).sum())})
    costs = {"started": len(list((run / "requests").glob("*.started.json"))),
             "receipts": len(receipts),
             "returned_json": sum(row["status"] == "returned_json" for row in receipts.values()),
             "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts.values()),
             "failures": read_json(run / "failures.json"), "retries": 0,
             "inference_wall_seconds": read_json(run / "state.json")["seconds"],
             "numeric_cpu_timing": "not separately measured", "billing": "reported usage, historical gaps retained"}
    if costs["started"] > config["maximum_requests"]:
        raise ValueError("Actual request budget exceeded registration")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    write_rows(out / "family-results.csv", family_rows)
    write_rows(out / "events.csv", event_rows)
    write_rows(out / "corrections.csv", correction_rows)
    save_json(out / "actual-costs.json", costs)
    save_json(out / "checks.json", {"population": population, "methods": len(predictions),
                                   "family_rows": len(family_rows), "event_rows": len(event_rows),
                                   "daily_rows_checked": daily_count, "payloads_checked": len(receipts),
                                   "record_repeat_counts_checked": len(schedules),
                                   "activity_stage_missing_masks_checked": True,
                                   "new_api": 0, "field_accuracy_confirmed": False})
    save_json(out / "manifest.json", {"files": {str(path.relative_to(out)): sha(path)
                                               for path in out.rglob("*") if path.is_file()}})
    print({"costs": costs, "family_rows": len(family_rows), "daily_rows": daily_count}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    summarize(arguments.run, arguments.out)
