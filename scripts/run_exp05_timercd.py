"""Run pinned author TimeRCD inference and explicitly separated GNSS adapters."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from time_rcd import TimeRCDDetector

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import aggregate, evaluate_record
from gnss_research.schema import Prediction, event_masks, observed_values

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_COMMIT = "372bb980426b2f67007311c6f3165ab789c79bef"


def interpolate_observations(values: np.ndarray) -> np.ndarray:
    """Fill only inference input, retain original dates and evaluation missing mask."""
    filled = values.copy()
    time_indices = np.arange(len(values))
    for axis in range(values.shape[1]):
        observed = np.isfinite(values[:, axis])
        filled[:, axis] = np.interp(time_indices, time_indices[observed], values[observed, axis])
    return filled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-raw", action="store_true")
    args = parser.parse_args()
    code = ROOT / "vendor/upstream/time-rcd"
    commit = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != AUTHOR_COMMIT:
        raise ValueError("author source differs from the declared commit")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(2026101401)
    registration = {
        "experiment_id": "GNSS-VLM-EXP05-TimeRCD", "split": args.split,
        "author_commit": commit, "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_source": read_json(ROOT / "papers/exp05-timercd/acquisition.json"),
        "script_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(ROOT / "docs/experiment-exp05-timercd.md"),
        "author_source_hashes": {str(p.relative_to(code)): sha256(p)
                                  for p in sorted((code / "time_rcd").rglob("*.py"))},
        "runtime": {"threads": 2, "device": "cpu", "packages": {
            name: version(name) for name in ("torch", "numpy", "scikit-learn", "time-rcd")}},
        "datasets": [{"path": str(p.resolve()), "manifest_sha256": sha256(p / "manifest.json")}
                     for p in args.datasets],
        "model_parameters": {"variant": "multi", "win_size": 5000, "batch_size": 1},
        "adjustments": ["missing values linearly interpolated for model input only",
                        "hard labels from anomaly-minus-normal logit > 0; equivalent to "
                        "posterior > .5; project decision adapter, not author default"],
        "check_raw_input": args.check_raw,
    }
    write_json(args.out / "registration.json", registration)
    detector = TimeRCDDetector.from_local(args.checkpoint, variant="multi", device="cpu")
    reports = []
    targets, probabilities = [], []
    failures = 0
    raw_checks = []
    receipts = []
    for dataset in args.datasets:
        manifest = read_json(dataset / "manifest.json")
        for record in manifest["records"]:
            if record["split"] != args.split:
                continue
            observation = read_json(dataset / record["observation"])
            values = observed_values(observation)
            observed = np.isfinite(values).all(axis=1)
            work = args.out / f"seed{manifest['seed']}" / record["station_id"]
            work.mkdir(parents=True)
            if args.check_raw:
                raw_scores = detector.predict(values)
                raw = {"station_id": record["station_id"],
                       "input": "original NEU with NaN, unchanged author API",
                       "finite_score_days": int(np.isfinite(raw_scores).sum()),
                       "total_days": len(values), "hard_decision": "not provided by author API"}
                write_json(work / "author-raw-input-check.json", raw)
                raw_checks.append(raw)
            started = time.perf_counter()
            result = {"prediction": None, "generic_mask": [False] * len(values),
                      "generic_score": None, "status": "failed"}
            try:
                scores, logits = detector.predict(interpolate_observations(values),
                                                   return_logits=True)
                if not np.isfinite(scores).all() or not np.isfinite(logits).all():
                    raise ValueError("nonfinite author scores after input adaptation")
                result.update(status="success", generic_mask=(logits > 0).tolist(),
                              generic_score=scores.tolist(), logit_difference=logits.tolist())
            except (ValueError, OSError, RuntimeError) as error:
                result["error_type"] = type(error).__name__
                failures += 1
            result["elapsed_seconds"] = time.perf_counter() - started
            write_json(work / "result.json", result)
            truth = read_json(dataset / record["truth"])
            report = evaluate_record(truth, result, observed, observation["background_end"])
            reports.append(report)
            eligible = observed & (np.arange(len(values)) > observation["background_end"])
            if result["status"] == "success":
                truth_masks = event_masks(Prediction.model_validate({"events": truth["events"]}),
                                          observed)
                targets.append(np.any(np.column_stack(list(truth_masks.values())), axis=1)[eligible])
                probabilities.append(np.asarray(result["generic_score"])[eligible])
            receipts.append({"dataset_seed": manifest["seed"], "station_id": record["station_id"],
                             "status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                             "observation_sha256": record["observation_sha256"],
                             "result_sha256": sha256(work / "result.json")})
            print(f"seed{manifest['seed']} {record['station_id']}: {result['status']}", flush=True)
    summary = {"experiment_id": "GNSS-VLM-EXP05-TimeRCD", "split": args.split,
               "method": "TimeRCD author multi + GNSS interpolation + posterior argmax adapter",
               "metrics": aggregate(reports), "failed_records": failures,
               "generic_pooled_average_precision": float(average_precision_score(
                   np.concatenate(targets), np.concatenate(probabilities))) if not failures else None,
               "raw_input_checks": raw_checks, "receipts": receipts,
               "types": "not available; author model outputs generic scores only",
               "comparison_scope": "F1 requires declared input and decision adapters; "
               "no validation/test threshold tuning; AP unavailable if any adapted inference fails"}
    write_json(args.out / "per-station.json", reports)
    write_json(args.out / "summary.json", summary)
    print(json.dumps({"stations": len(reports), "failed": failures,
                      "generic_f1": summary["metrics"]["generic_daily_micro"]["f1"],
                      "ap": summary["generic_pooled_average_precision"]}))


if __name__ == "__main__":
    main()
