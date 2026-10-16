"""Apply the nine frozen numerical baselines to the complete 120-record extension.

Only existing observation inputs, selected parameters and fitted models are read.
Replay checks use the original prediction function and never fit or select a model.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_evidence_display_confirmation import numerical_predictions
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_evaluation import ActivityPrediction

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
DEFAULT_DATA = BASE / "persistence-scale-data-v1"
DEFAULT_OUT = BASE / "persistence-scale-baselines-v1"
METHODS = (
    "robust_rule",
    "pelt_rule",
    "xgboost",
    "robust_xgboost_stage",
    "raw_multiscale_rule",
    "xgboost_raw_persistence",
    "raw_rule_84",
    "xgboost_42_matched_support",
    "xgboost_84",
)
MODEL_FILES = (
    "stage-xgboost.json",
    "activity-xgboost.json",
    "original-selection.json",
    "raw-selection.json",
    "long-selection.json",
    "matched-selection.json",
    "activity-xgboost-raw-persistence.json",
    "activity-xgboost-84.json",
    "activity-xgboost-42-matched-support.json",
)


def verify_mapping(directory: Path, bindings: dict) -> None:
    for name, expected in bindings.items():
        if sha(directory / name) != expected:
            raise ValueError(f"Frozen artifact changed: {directory / name}")


def verify_data(data: Path) -> dict:
    verify_mapping(data, read_json(data / "pre-generation-freeze.json"))
    verify_mapping(data, read_json(data / "data-freeze.json"))
    training_bindings = read_json(data / "training-bindings.json")
    verify_mapping(ROOT, training_bindings)
    protocol = read_json(data / "protocol.json")
    if protocol["refitting"] or protocol["training_data_expanded"]:
        raise ValueError("This runner only accepts frozen inference without refitting")
    expected_ids = [f"case_{index:04d}" for index in range(217, 337)]
    if protocol["case_ids"] != expected_ids:
        raise ValueError("Registered population must contain all case_0217--case_0336")
    if protocol["generation"]["days"] != 730:
        raise ValueError("Unexpected observation calendar")
    frozen_digests = set(training_bindings.values())
    for name in MODEL_FILES:
        if sha(data / name) not in frozen_digests:
            raise ValueError(f"Model or selection is not bound to training: {name}")
    return protocol


def validate_predictions(predictions: dict, records: dict) -> None:
    if set(predictions) != set(METHODS):
        raise ValueError("The original numerical runner did not return all nine methods")
    for method, cases in predictions.items():
        if set(cases) != set(records):
            raise ValueError(f"Incomplete record coverage: {method}")
        for cid, payload in cases.items():
            case, features = records[cid]
            prediction = ActivityPrediction(
                np.asarray(payload["activity"], dtype=int),
                np.asarray(payload["stage"], dtype=int),
                status=payload["status"],
                gated=payload["gated"],
                stage_status=payload["stage_status"],
            )
            prediction.validate(len(case.dates))
            if np.any(prediction.activity[~features.observed] != -1):
                raise ValueError(f"Missing observations received activity labels: {method}/{cid}")
            if np.any(prediction.stage[~features.observed] != -1):
                raise ValueError(f"Missing observations received stage labels: {method}/{cid}")


def predict_records(data: Path, case_ids: list[str], out: Path) -> tuple[dict, list]:
    predictions = {method: {} for method in METHODS}
    costs = []
    for number, cid in enumerate(case_ids, start=1):
        started = time.perf_counter()
        record = load_case_features(data, cid)
        feature_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result = numerical_predictions(data, {cid: record})
        prediction_seconds = time.perf_counter() - started
        validate_predictions(result, {cid: record})
        for method in METHODS:
            predictions[method][cid] = result[method][cid]
        costs.append({
            "case_id": cid,
            "feature_seconds": feature_seconds,
            "all_nine_prediction_seconds": prediction_seconds,
            "observed_days": int(record[1].observed.sum()),
            "missing_days": int((~record[1].observed).sum()),
        })
        # Checkpoints retain completed records if any later input or method fails.
        save_json(out / "predictions.json", predictions)
        save_json(out / "per-record-cost.json", costs)
        print(f"Predicted {cid}: {number}/{len(case_ids)}; all nine {prediction_seconds:.3f}s",
              flush=True)
    return predictions, costs


def run(data: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    save_json(out / "state.json", {"status": "verifying_frozen_inputs"})
    try:
        protocol = verify_data(data)
        source_files = {Path(__file__).resolve()}
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if path.suffix == ".py" and path.is_relative_to(ROOT / "scripts"):
                    source_files.add(path)
                elif path.suffix == ".py" and path.is_relative_to(ROOT / "src/gnss_sim"):
                    source_files.add(path)
        source_bindings = {path.relative_to(ROOT).as_posix(): sha(path)
                           for path in sorted(source_files)}
        for name in source_bindings:
            destination = out / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        save_json(out / "source-bindings.json", source_bindings)
        save_json(out / "input-bindings.json", {
            "data_directory": str(data),
            "protocol_sha256": sha(data / "protocol.json"),
            "data_freeze_sha256": sha(data / "data-freeze.json"),
            "training_bindings_sha256": sha(data / "training-bindings.json"),
            "observation_files": {
                name: digest for name, digest in read_json(data / "data-freeze.json").items()
                if name.startswith("inputs/")
            },
            "model_files": {name: sha(data / name) for name in MODEL_FILES},
        })
        save_json(out / "protocol.json", {
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "data_protocol_sha256": sha(data / "protocol.json"),
            "case_ids": protocol["case_ids"],
            "methods": list(METHODS),
            "prediction_entry": "run_landslide_evidence_display_confirmation.numerical_predictions",
            "refitting": False,
            "new_parameter_selection": False,
            "new_reference_contents_opened": False,
            "new_api_calls": 0,
            "missing_observations": "Activity and stage both remain unknown (-1)",
            "cost_scope": "All nine methods per record; includes loading frozen models every record. Feature loading reported separately. No per-method attribution.",
            "failures": "Preserve partial predictions and traceback; never retry or overwrite an existing experiment directory",
            "environment": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "numpy": version("numpy"),
                "xgboost": version("xgboost"),
                "ruptures": version("ruptures"),
            },
        })
        save_json(out / "pre-inference-freeze.json", {
            path.relative_to(out).as_posix(): sha(path)
            for path in sorted(out.rglob("*")) if path.is_file() and path.name != "state.json"
        })
        save_json(out / "state.json", {"status": "predicting"})
        predictions, costs = predict_records(data, protocol["case_ids"], out)
        if any(len(cases) != 120 for cases in predictions.values()):
            raise ValueError("Each numerical method must cover every registered record")
        save_json(out / "cost.json", {
            "all_nine_prediction_seconds": sum(row["all_nine_prediction_seconds"] for row in costs),
            "feature_seconds": sum(row["feature_seconds"] for row in costs),
            "wall_seconds": time.perf_counter() - started,
            "cases": len(costs), "methods": len(METHODS),
        })
        save_json(out / "validation.json", {
            "methods": 9, "cases_per_method": 120, "method_record_predictions": 1080,
            "calendar_days_per_record": 730,
            "valid_activity_stage_codes": True,
            "definite_stages_require_confirmed_activity": True,
            "missing_observations_remain_unknown": True,
            "original_entry_called_directly": True,
        })
        verify_data(data)
        verify_mapping(ROOT, source_bindings)
        save_json(out / "state.json", {"status": "predictions_complete"})
        save_json(out / "manifest.json", {
            path.relative_to(out).as_posix(): sha(path)
            for path in sorted(out.rglob("*")) if path.is_file()
        })
    except Exception as error:
        (out / "failure-traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        save_json(out / "state.json", {"status": "failed", "error": str(error)})
        raise


def replay(data: Path, out: Path) -> None:
    protocol = verify_data(data)
    verify_mapping(out, read_json(out / "manifest.json"))
    verify_mapping(ROOT, read_json(out / "source-bindings.json"))
    original = read_json(out / "predictions.json")
    started = time.perf_counter()
    for cid in protocol["case_ids"]:
        record = load_case_features(data, cid)
        result = numerical_predictions(data, {cid: record})
        validate_predictions(result, {cid: record})
        for method in METHODS:
            if result[method][cid] != original[method][cid]:
                raise ValueError(f"Original frozen numerical prediction differs: {method}/{cid}")
    print(f"Replay identical: 9 methods x 120 records; {time.perf_counter() - started:.3f}s",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay(args.data.resolve(), args.out.resolve())
    else:
        run(args.data.resolve(), args.out.resolve())
