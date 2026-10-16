"""Verify the new-record comparison without making any additional model calls."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_codex_observation_reference import scoring_inputs
from analyze_landslide_review_utility_v2 import verify_receipts
from prepare_landslide_ai_reference_holdout import verify_manifest
from run_landslide_ai_reference_holdout import replay, verify_bindings
from run_landslide_window_optimization import read_json
from sklearn.metrics import precision_recall_fscore_support

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import ActivityPrediction


def verify(experiment, out):
    if out.exists():
        raise FileExistsError("Keep prior verification artifacts")
    bindings = read_json(experiment / "source-bindings.json")
    data, training = Path(bindings["data"]), Path(bindings["training"])
    for directory, name, key in (
        (data, "data-freeze.json", "data_freeze_sha256"),
        (training, "training-freeze.json", "training_freeze_sha256"),
        (data / "codex-review", "delivery-manifest.json", "reference_delivery_sha256"),
    ):
        if sha(directory / name) != bindings[key]:
            raise ValueError("Original source binding changed")
    verify_bindings(training, "training-freeze.json")
    verify_bindings(data, "data-freeze.json")
    reference = data / "codex-review"
    verify_manifest(reference, "delivery-manifest.json")
    analysis = experiment / "analysis"
    verify_manifest(analysis)
    receipts = verify_receipts(experiment / "run")
    predictions, used, scopes = replay(experiment / "run", receipts)
    for name, value in (("predictions.json", predictions), ("used-requests.json", used), ("scopes.json", scopes)):
        if value != read_json(analysis / name):
            raise ValueError("Deterministic replay differs: " + name)
    labels, observed, _ = scoring_inputs(reference)
    results = read_json(analysis / "results.json")
    with (analysis / "summary.csv").open(encoding="utf-8", newline="") as stream:
        summaries = {row["method"]: row for row in csv.DictReader(stream)}
    codes = {"acceleration": 1, "steady_motion": 2, "deceleration": 3}
    inspected_positions = 0
    for method, records in predictions.items():
        activity_actual, activity_predicted, stage_actual, stage_predicted = [], [], [], []
        internal_actual, internal_predicted = [], []
        for cid, value in records.items():
            prediction = ActivityPrediction(np.array(value["activity"]), np.array(value["stage"]),
                status=value["status"], gated=value["gated"], stage_status=value["stage_status"])
            prediction.validate(len(labels[cid]))
            if np.any(prediction.activity[~observed[cid]] != -1) or np.any(prediction.stage[~observed[cid]] != -1):
                raise ValueError("Missing-day contract violated")
            actual = np.array([row["activity_label"] for row in labels[cid]])
            actual_stage = np.array([codes.get(row["feature"], -1) for row in labels[cid]])
            actual_stage[actual == 0] = 0
            known, stage_known, interior = actual >= 0, actual_stage >= 0, actual_stage > 0
            activity_actual.extend(actual[known])
            activity_predicted.extend(prediction.activity[known])
            stage_actual.extend(actual_stage[stage_known])
            stage_predicted.extend(prediction.stage[stage_known])
            internal_actual.extend(actual_stage[interior])
            internal_predicted.extend(prediction.stage[interior])
            inspected_positions += len(actual)
        p, r, f1, _ = precision_recall_fscore_support(activity_actual, activity_predicted, labels=[1], zero_division=0)
        for key, value in (("precision", p[0]), ("recall", r[0]), ("f1", f1[0])):
            if not np.isclose(value, results[method]["aggregate"]["daily"][key]):
                raise ValueError("Activity mismatch: " + method)
        for actual, predicted, expected in (
            (stage_actual, stage_predicted, results[method]["aggregate"]["stage_macro_f1"]),
            (internal_actual, internal_predicted, results[method]["internal_stages"]["macro_f1"]),
        ):
            values = precision_recall_fscore_support(actual, predicted, labels=[1, 2, 3], zero_division=0)[2]
            if not np.isclose(values.mean(), expected):
                raise ValueError("Stage mismatch: " + method)
    for rep in (0, 1):
        main = f"adaptive_no_feedback_r{rep}__conflict_scope"
        main_cost = int(summaries[main]["requests"])
        for stem in ("same_count_fixed_early", "same_count_fixed_late"):
            if int(summaries[f"{stem}_r{rep}__conflict_scope"]["requests"]) != main_cost:
                raise ValueError("Fixed-position ablation call counts differ")
        center = f"adaptive_no_feedback_r{rep}__matched_center_scope"
        for cid in labels:
            authority = scopes[main][cid]["editable_days"]
            if len(authority) != len(scopes[center][cid]["editable_days"]):
                raise ValueError("Matched-size scope differs")
            initial = np.array(predictions[f"direct_visual_r{rep}"][cid]["activity"])
            final = np.array(predictions[main][cid]["activity"])
            permission = np.zeros(len(initial), dtype=bool)
            permission[authority] = True
            if np.any(initial[~permission] != final[~permission]):
                raise ValueError("Method revised outside frozen authority")
    with (analysis / "daily-comparison.csv").open(encoding="utf-8", newline="") as stream:
        daily_count = 0
        for row in csv.DictReader(stream):
            value = predictions[row["method"]][row["case_id"]]
            day = int(row["day_index"])
            if (int(row["predicted_activity"]) != value["activity"][day]
                    or int(row["predicted_stage"]) != value["stage"][day]):
                raise ValueError("Daily export differs")
            daily_count += 1
    if daily_count != inspected_positions:
        raise ValueError("Daily export count differs")
    out.mkdir()
    shutil.copyfile(Path(__file__), out / "source.py")
    audit = {"status": "passed", "methods": len(predictions), "cases": len(labels),
             "daily_positions_checked": inspected_positions, "receipt_payloads": len(receipts),
             "prediction_schedule_scope_replay_equal": True, "source_hashes_unchanged": True,
             "sklearn_activity_end_to_end_and_internal_stage": True,
             "same_call_count_and_same_editable_count": True, "daily_export_equal": True,
             "new_remote_requests": 0, "independent_reviewer": False,
             "analysis_manifest_sha256": sha(analysis / "manifest.json")}
    write_json(out / "audit.json", audit)
    print(audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.experiment.resolve(), args.out.resolve())
