"""Verify supplemental predictions and export manuscript evidence without model calls."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import paired_record_interval
from run_landslide_comparison import STAGES, save_json, sha
from run_landslide_window_optimization import read_json
from sklearn.metrics import precision_recall_fscore_support


def verify_manifest(folder):
    manifest = read_json(folder / "manifest.json")
    for name, expected in manifest["files"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Frozen data changed: {folder / name}")
    return len(manifest["files"])


def verify_request_receipts(folder):
    receipts = []
    for response_path in sorted((folder / "requests").glob("*.response.json")):
        response = read_json(response_path)
        started = read_json(response_path.with_name(
            response_path.name.replace(".response.json", ".started.json")))
        content = [{"type": "text", "text": started["prompt"]}]
        for name, expected in started["image_sha256"].items():
            path = folder / "images" / name
            if sha(path) != expected:
                raise ValueError(f"Request image changed: {path}")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + encoded}})
        payload = {
            "model": started["model"], "messages": [{"role": "user", "content": content}],
            "temperature": .1, "top_p": .3, "max_tokens": 2048,
            "enable_thinking": False, "response_format": {"type": "json_object"},
        }
        payload_sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if payload_sha != started["request_sha256"] or payload_sha != response["request_sha256"]:
            raise ValueError(f"Actual request binding differs: {response_path}")
        if response["status"] != "returned_json" or response["http_status"] != 200:
            raise ValueError(f"Non-success response requires explicit accounting: {response_path}")
        receipts.append(response)
    return receipts


def verify_metrics(predictions, results, reference, source):
    rows = {cid: [] for cid in next(iter(predictions.values()))}
    with reference.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["case_id"]].append(row)
    internal_stage_results = {}
    for method, cases in predictions.items():
        actual_activity, predicted_activity, actual_stage, predicted_stage = [], [], [], []
        internal_actual, internal_predicted = [], []
        for cid, value in cases.items():
            observed_input = read_json(source / "inputs" / cid / "input.json")
            observed = np.array([item is not None for item in observed_input["displacement_mm"]])
            activity = np.array(value["activity"])
            stage = np.array(value["stage"])
            if activity.shape != observed.shape or stage.shape != observed.shape:
                raise ValueError("Prediction calendar differs from input")
            if not np.isin(activity, [-1, 0, 1]).all() or not np.isin(stage, [-1, 0, 1, 2, 3]).all():
                raise ValueError("Prediction has an invalid class")
            if np.any(activity[~observed] != -1) or np.any(stage[~observed] != -1):
                raise ValueError("Missing-day decision was invented")
            if np.any((stage > 0) & (activity != 1)):
                raise ValueError("Stage assigned outside confirmed activity")
            for day, row in enumerate(rows[cid]):
                actual = int(row["activity_label"])
                if actual < 0:
                    continue
                actual_activity.append(actual)
                predicted_activity.append(int(activity[day] == 1))
                expected_stage = STAGES[row["feature"]] if actual == 1 else 0
                actual_stage.append(expected_stage)
                predicted_stage.append(stage[day])
                if actual == 1:
                    internal_actual.append(expected_stage)
                    internal_predicted.append(stage[day])
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_activity, predicted_activity, labels=[1], zero_division=0)
        aggregate = results[method]["aggregate"]
        for name, measured in (("precision", precision[0]), ("recall", recall[0]), ("f1", f1[0])):
            if not np.isclose(measured, aggregate["daily"][name]):
                raise ValueError(f"Independent activity score differs: {method} {name}")
        _, _, stage_f1, _ = precision_recall_fscore_support(
            actual_stage, predicted_stage, labels=[1, 2, 3], zero_division=0)
        if not np.isclose(np.mean(stage_f1), aggregate["stage_macro_f1"]):
            raise ValueError(f"Independent stage score differs: {method}")
        precision, recall, f1, support = precision_recall_fscore_support(
            internal_actual, internal_predicted, labels=[1, 2, 3], zero_division=0)
        internal_stage_results[method] = {
            "scope": "Reference-active observed days only; missed activity and stage abstention retained",
            "precision_ASD": precision.tolist(), "recall_ASD": recall.tolist(),
            "f1_ASD": f1.tolist(), "support_ASD": support.tolist(),
            "macro_f1": float(np.mean(f1)),
            "definite_stage_coverage": float(np.mean(np.array(internal_predicted) > 0)),
        }
    return internal_stage_results


def finalize(root, out):
    out.mkdir(parents=True, exist_ok=False)
    frozen_files = sum(verify_manifest(root / name) for name in (
        "motion-probe-v2", "confirmation-data-v1"))
    raw = root / "raw-baselines-v1"
    if sha(raw / "predictions.json") != read_json(raw / "manifest.json")["prediction_sha256"]:
        raise ValueError("Additional mechanism baseline predictions changed")
    stage_evidence = {}
    for run_name, additional, analysis_name in (
        ("mechanism-v1", raw, "analysis-v2"),
        ("confirmation-v1", root / "confirmation-data-v1/additional-baselines", "analysis-v1"),
    ):
        run = root / run_name
        combined = read_json(run / analysis_name / "combined-results.json")
        predictions = read_json(run / "predictions.json")
        case_ids = read_json(run / "protocol.json")["case_ids"]
        for method, cases in read_json(run / "numeric-predictions.json").items():
            if method not in combined:
                continue
            predictions[method] = {cid: cases[cid] for cid in case_ids}
        predictions.update(read_json(additional / "predictions.json"))
        stage_evidence[run_name] = verify_metrics(
            predictions, combined, run / "reference-subset.csv", run)
    for name in ("native-stages-v1", "confirmation-native-stages-v1"):
        folder = root / name
        protocol = read_json(folder / "protocol.json")
        source = Path(protocol["activity_run"])
        if sha(source / "protocol.json") != protocol["activity_protocol_sha256"]:
            raise ValueError("Native stage source protocol changed")
        if sha(source / "predictions.json") != read_json(
                folder / "activity-prediction-binding.json")["sha256"]:
            raise ValueError("Native stage source activity changed")
        if sha(folder / "predictions.json") != read_json(folder / "prediction-freeze.json")["sha256"]:
            raise ValueError("Native stage predictions changed")
        predictions = read_json(folder / "predictions.json")
        activities = read_json(source / "predictions.json")
        for method, cases in predictions.items():
            for cid, value in cases.items():
                if value["activity"] != activities[method.replace("_native_stage", "")][cid]["activity"]:
                    raise ValueError("Native stage head revised frozen activity")
        stage_evidence[name] = verify_metrics(
            predictions, read_json(folder / "results.json"), folder / "reference-subset.csv", source)
    costs = {}
    for name in ("pilot-v3", "mechanism-v1", "confirmation-v1",
                 "native-stages-v1", "confirmation-native-stages-v1"):
        folder = root / name
        receipts = verify_request_receipts(folder)
        costs[name] = {
            "unique_actual_requests": len(receipts),
            "total_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts),
            "sum_response_seconds": sum(row["seconds"] for row in receipts),
            "failures": read_json(folder / "failures.json"),
        }
    confirmation = read_json(root / "confirmation-v1/analysis-v1/combined-results.json")
    intervals = {}
    for repetition in range(2):
        candidate = f"adaptive_no_feedback_r{repetition}"
        for control in (f"direct_visual_r{repetition}", f"fixed_image_review_r{repetition}",
                        "raw_multiscale_rule", "xgboost_raw_persistence"):
            intervals[f"{candidate}-minus-{control}"] = paired_record_interval(
                confirmation, control, candidate)
    save_json(out / "internal-stage-results.json", stage_evidence)
    save_json(out / "default-candidate-intervals.json", intervals)
    save_json(out / "cost-ledger.json", {
        "runs": costs,
        "unique_actual_requests": sum(row["unique_actual_requests"] for row in costs.values()),
        "total_tokens": sum(row["total_tokens"] for row in costs.values()),
        "scope": "Unique saved receipts, including native stages; shared flow costs never summed",
    })
    save_json(out / "verification.json", {
        "status": "verified", "data_manifest_files": frozen_files,
        "independent_metric_groups": sum(len(value) for value in stage_evidence.values()),
        "scope": "Engineering and provenance checks, not validation of observable reference",
        "source_sha256": sha(Path(__file__)),
        "inputs": {str(path.relative_to(root)): sha(path) for path in root.rglob("*.json")
                   if path.name in {"predictions.json", "combined-results.json", "results.json",
                                    "manifest.json", "protocol.json", "prediction-freeze.json"}},
    })
    print("Supplemental evidence verified; no new model calls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.root, args.out)


if __name__ == "__main__":
    main()
