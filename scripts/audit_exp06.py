"""Check frozen source/data and label masks once after completed EXP06 execution."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from gnss_research.experiment import verify_dataset
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.schema import EVENT_TYPES, Prediction, event_masks, observed_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipts = []
    for run in args.runs:
        config = read_json(run / "registration.json")
        if not (run / "execution-complete.json").exists():
            raise ValueError("incomplete run")
        dataset = Path(config["dataset"])
        manifest = verify_dataset(dataset)
        if sha256(dataset / "manifest.json") != config["manifest_sha256"]:
            raise ValueError("manifest differs from registration")
        for name, digest in config["source_hashes"].items():
            if sha256(run / "source" / name) != digest:
                raise ValueError("frozen source mismatch")
        if sha256(run / "source/method-protocol.md") != config["method_protocol_sha256"]:
            raise ValueError("method protocol mismatch")
        if config["split"] == "test" and config["source_hashes"] != source_hashes():
            raise ValueError("current source differs from formal frozen source")
        requests = failures = skipped = label_files = 0
        for record in manifest["records"]:
            if record["station_id"] not in config["station_ids"]:
                continue
            obs = read_json(dataset / record["observation"])
            observed = np.isfinite(observed_values(obs)).all(axis=1)
            for method in config["methods"]:
                work = run / "predictions" / method / record["station_id"]
                result = read_json(work / "result.json")
                requests += result.get("request_count", 0)
                if result.get("request_count", 0):
                    if result["retry_count"] != 0:
                        raise ValueError("model request replay detected")
                    failures += result["status"] != "success"
                skipped += result.get("review_mode") == "not_requested_no_ambiguity"
                if result.get("prediction") is None:
                    continue
                masks = event_masks(Prediction.model_validate(result["prediction"]), observed)
                with (work / "labels.csv").open(encoding="utf8", newline="") as stream:
                    labels = list(csv.DictReader(stream))
                if len(labels) != len(observed):
                    raise ValueError("daily index count changed")
                for day, row in enumerate(labels):
                    expected = {kind: str(int(masks[kind][day])) if observed[day] else ""
                                for kind in EVENT_TYPES}
                    if any(row[kind] != expected[kind] for kind in EVENT_TYPES):
                        raise ValueError("export labels differ from final missing-aware event mask")
                label_files += 1
        receipts.append({"run": run.name, "registration_sha256": sha256(run / "registration.json"),
                         "frozen_source_verified": True, "data_verified": True,
                         "model_requests": requests, "model_failures": failures,
                         "unrequested_parent_jobs": skipped, "label_files_verified": label_files})
    write_json(args.out, {"experiment_id": "GNSS-VLM-EXP06", "receipts": receipts,
                         "total_requests": sum(row["model_requests"] for row in receipts),
                         "model_retries": 0})
    print({"runs_verified": len(receipts), "total_requests": sum(row["model_requests"] for row in receipts)})


if __name__ == "__main__":
    main()
