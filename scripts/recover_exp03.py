"""Finish offline derivation in a new directory after a terminated inference process.

No original file is changed and no provider request is made. Incomplete author calls
remain a failed station; recorded attempts are a lower bound, not six fabricated calls.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from gnss_research.experiment import evaluate, verify_dataset
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.fusion import (
    DERIVED_METHODS,
    anchored_review,
    boundary_review,
    ranked_boundary_review,
)
from gnss_research.plots import export_labels, render_observation
from gnss_research.schema import Prediction

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "runs/exp03-main-v2"
    target = ROOT / "runs/exp03-main-v2-recovered"
    config = read_json(source / "registration.json")
    if (source / "execution-complete.json").exists():
        raise ValueError("original completed; recovery not needed")
    if config["source_hashes"] != source_hashes():
        raise ValueError("recovery must use the frozen implementation")
    dataset = Path(config["dataset"])
    manifest = verify_dataset(dataset)
    if sha256(dataset / "manifest.json") != config["manifest_sha256"]:
        raise ValueError("dataset changed")
    shutil.copytree(source, target)
    recovered_files = []
    unresolved_attempts = 0
    for record in manifest["records"]:
        station = record["station_id"]
        if station not in config["station_ids"]:
            continue
        observation = read_json(dataset / record["observation"])
        for method in config["methods"]:
            work = target / "predictions" / method / station
            if method in DERIVED_METHODS:
                parent = target / "predictions" / DERIVED_METHODS[method] / station
                visual = read_json(parent / "result.json")
                candidates = read_json(parent / "candidates.json")
                if method.startswith("ranked_"):
                    result = ranked_boundary_review(observation, visual, candidates)
                elif method.startswith("boundary_"):
                    result = boundary_review(observation, visual, candidates,
                                             numerical_points=method == "boundary_numeric_points")
                else:
                    result = anchored_review(observation, visual, candidates)
                result.update(method=method, source_visual_method=DERIVED_METHODS[method],
                              source_result_sha256=sha256(parent / "result.json"))
                write_json(work / "result.json", result)
            elif not (work / "result.json").exists():
                if method != "visualtime_gnss":
                    raise ValueError(f"missing non-author result: {method}/{station}")
                receipts = []
                missing = []
                for axis in "NEU":
                    for task in ("point", "range"):
                        path = work / f"{axis}-{task}/receipt.json"
                        if path.exists():
                            receipts.append(read_json(path))
                        else:
                            missing.append(f"{axis}-{task}")
                # Calls within an author station are sequential. Only the first receipt-less
                # call could have been in flight; subsequent calls were not reached.
                upper_bound = int(bool(missing))
                unresolved_attempts += upper_bound
                result = {"status": "failed", "error_type": "InterruptedExecution",
                          "prediction": None, "generic_mask": [False] * len(observation["dates"]),
                          "generic_score": None, "request_count": sum(
                              part["request_count"] for part in receipts),
                          "unresolved_attempt_upper_bound": upper_bound,
                          "receiptless_subrequests": missing, "subrequests": receipts}
                write_json(work / "result.json", result)
                recovered_files.append({"station": station, "method": method,
                                        "missing_receipts": missing, "status": "failed"})
            else:
                result = read_json(work / "result.json")
            if result.get("prediction") is not None:
                prediction = Prediction.model_validate(result["prediction"])
                if not (work / "labels.csv").exists():
                    export_labels(observation, prediction, work / "labels.csv")
                if not (work / "prediction.png").exists():
                    render_observation(observation, work / "prediction.png", prediction=prediction)
    write_json(target / "recovery.json", {
        "original_run": str(source), "original_registration_sha256": sha256(source / "registration.json"),
        "new_requests": 0, "source_hashes": source_hashes(), "recovered_at": time.time(),
        "incomplete_stations": recovered_files, "unresolved_attempt_upper_bound": unresolved_attempts,
        "scope": "offline completion of original frozen inference, not independent experiment"})
    write_json(target / "execution-complete.json", {"completed": time.time(), "offline_recovery": True})
    print(evaluate(target)["report"])


if __name__ == "__main__":
    main()
