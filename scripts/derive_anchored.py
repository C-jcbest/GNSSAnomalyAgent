"""Declared offline development analysis of frozen visual responses; no paid calls."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from gnss_research.files import read_json, sha256, write_json
from gnss_research.fusion import DERIVED_METHODS, anchored_review
from gnss_research.metrics import aggregate, evaluate_record
from gnss_research.schema import observed_values


def derive(run: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    config = read_json(run / "registration.json")
    dataset = Path(config["dataset"])
    reports = {method: [] for method in DERIVED_METHODS}
    failures = {method: 0 for method in DERIVED_METHODS}
    for record in read_json(dataset / "manifest.json")["records"]:
        if record["station_id"] not in config["station_ids"]:
            continue
        observation = read_json(dataset / record["observation"])
        truth = read_json(dataset / record["truth"])
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        for method, parent in DERIVED_METHODS.items():
            work = run / "predictions" / parent / record["station_id"]
            result = anchored_review(observation, read_json(work / "result.json"),
                                     read_json(work / "candidates.json"))
            result["source_result_sha256"] = sha256(work / "result.json")
            write_json(out / "predictions" / method / f"{record['station_id']}.json", result)
            if result["status"] != "success":
                failures[method] += 1
                result = {"prediction": {"events": []}, "generic_mask": [False] * len(observed),
                          "generic_score": None}
            reports[method].append(evaluate_record(truth, result, observed,
                                                   observation["background_end"]))
    write_json(out / "summary.json", {
        "methods": {method: {**aggregate(rows), "failed_records": failures[method],
                             "not_run_records": 0} for method, rows in reports.items()},
        "source_run": str(run.resolve()), "new_model_requests": 0,
        "analysis_source_sha256": sha256(Path(__file__)),
        "fusion_source_sha256": sha256(Path(__file__).resolve().parents[1]
                                      / "src/gnss_research/fusion.py"),
        "note": "Post-v2 validation-side method development; original predictions unchanged"})
    write_json(out / "per-station.json", reports)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "src/gnss_research/fusion.py",
                    out / "fusion-source.py")
    print(out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    derive(args.run, args.out)
