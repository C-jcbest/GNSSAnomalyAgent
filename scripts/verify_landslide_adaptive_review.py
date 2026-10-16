"""Check persisted requests, frozen bytes, calendar masks and independent metrics."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from run_landslide_comparison import STAGES, save_json, sha
from run_landslide_window_optimization import read_json
from sklearn.metrics import precision_recall_fscore_support


def verify(run, analysis, out):
    if read_json(run / "state.json")["status"] != "complete":
        raise ValueError("Verify only a completed run")
    frozen = read_json(run / "freeze.json")
    for name, expected in frozen.items():
        if sha(run / name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError(f"Prediction binding changed: {name}")
    receipts = sorted((run / "requests").glob("*.response.json"))
    for receipt_path in receipts:
        receipt = read_json(receipt_path)
        started = read_json(receipt_path.with_name(
            receipt_path.name.replace(".response.json", ".started.json")))
        content = [{"type": "text", "text": started["prompt"]}]
        for name, expected in started["image_sha256"].items():
            path = run / "images" / name
            if sha(path) != expected:
                raise ValueError("Actual request image differs from frozen source")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + encoded}})
        payload = {"model": started["model"],
                   "messages": [{"role": "user", "content": content}],
                   "temperature": .1, "top_p": .3, "max_tokens": 2048,
                   "enable_thinking": False, "response_format": {"type": "json_object"}}
        request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if request_hash != started["request_sha256"]:
            raise ValueError("Actual request payload cannot be reconstructed")
        if receipt["request_sha256"] != started["request_sha256"]:
            raise ValueError("Request and response receipts are not bound")
    case_ids = read_json(run / "protocol.json")["case_ids"]
    labels = {cid: [] for cid in case_ids}
    with (run / "reference-subset.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["case_id"]].append(row)
    predictions = read_json(run / "predictions.json")
    numeric = read_json(run / "numeric-predictions.json")
    for method in ("robust_rule", "pelt_rule", "xgboost", "robust_xgboost_stage"):
        predictions[method] = {cid: numeric[method][cid] for cid in case_ids}
    scores = read_json(run / "results.json")
    for method, cases in predictions.items():
        activity_true, activity_predicted, stage_true, stage_predicted = [], [], [], []
        for cid, value in cases.items():
            observed_input = read_json(run / "inputs" / cid / "input.json")
            observed = np.array([row is not None for row in observed_input["displacement_mm"]])
            activity = np.array(value["activity"])
            stage = np.array(value["stage"])
            if np.any(activity[~observed] != -1) or np.any(stage[~observed] != -1):
                raise ValueError("Missing-day decisions were invented")
            if np.any((stage > 0) & (activity != 1)):
                raise ValueError("Stage was assigned outside confirmed activity")
            for index, row in enumerate(labels[cid]):
                reference_activity = int(row["activity_label"])
                if reference_activity >= 0:
                    activity_true.append(reference_activity)
                    activity_predicted.append(int(activity[index] == 1))
                reference_stage = 0 if reference_activity == 0 else STAGES.get(row["feature"], -1)
                if reference_stage >= 0:
                    stage_true.append(reference_stage)
                    stage_predicted.append(stage[index])
        precision, recall, f1, _ = precision_recall_fscore_support(
            activity_true, activity_predicted, labels=[1], average=None, zero_division=0)
        for name, value in (("precision", precision[0]), ("recall", recall[0]), ("f1", f1[0])):
            if not np.isclose(value, scores[method]["aggregate"]["daily"][name]):
                raise ValueError(f"Independent daily metric disagrees: {method} {name}")
        precision, recall, f1, _ = precision_recall_fscore_support(
            stage_true, stage_predicted, labels=[1, 2, 3], average=None, zero_division=0)
        if not np.isclose(np.mean(f1), scores[method]["aggregate"]["stage_macro_f1"]):
            raise ValueError("Independent end-to-end stage F1 disagrees")
    # Boundary matching uses a separate maximum-weight assignment implementation.
    boundary_analysis = read_json(analysis / "analysis.json")
    for method, row in boundary_analysis["metrics"].items():
        if method in scores and row["matched_event_count"] != scores[method]["aggregate"]["events"]["tp"]:
            raise ValueError("Independent event matching count disagrees")
    save_json(out, {"status": "verified", "frozen_files": len(frozen),
                    "actual_payloads_and_image_bindings": len(receipts),
                    "method_metric_groups": len(predictions),
                    "independent_checks": "sklearn daily/ASD P/R/F1, assignment event counts, "
                                          "frozen bytes, actual request payloads, masks and gates",
                    "scope": "Engineering/provenance verification; not expert reference validity",
                    "verification_source_sha256": sha(Path(__file__))})
    print("Verified", len(frozen), "frozen files,", len(receipts), "requests,",
          len(predictions), "metric groups")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run, args.analysis, args.out)


if __name__ == "__main__":
    main()
