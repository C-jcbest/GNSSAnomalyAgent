"""Explicit six-station development derivation; saved responses, zero new requests."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from report_exp02 import activity_errors

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.metrics import aggregate, evaluate_record
from gnss_research.schema import observed_values
from gnss_research.segmented_review import repair_return_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.run / "registration.json")
    if config["split"] != "validation" or "segment_pair_repair" in config["methods"]:
        raise ValueError("only pre-repair validation pilot responses are eligible")
    args.out.mkdir(parents=True, exist_ok=False)
    dataset = Path(config["dataset"])
    rows = []
    receipts = []
    for record in read_json(dataset / "manifest.json")["records"]:
        if record["station_id"] not in config["station_ids"]:
            continue
        station = record["station_id"]
        work = args.run / "predictions/segment_guided" / station
        obs = read_json(dataset / record["observation"])
        visual = read_json(work / "result.json")
        result = repair_return_pairs(obs, visual, read_json(work / "candidates.json"))
        receipt = {"station_id": station, "source_result_sha256": sha256(work / "result.json"),
                   "source_candidates_sha256": sha256(work / "candidates.json"),
                   "visual_dependency_status": visual["status"]}
        receipts.append(receipt)
        write_json(args.out / "predictions/segment_pair_repair" / station / "result.json",
                   {**result, **receipt})
        truth = read_json(dataset / record["truth"])
        observed = np.isfinite(observed_values(obs)).all(axis=1)
        rows.append(evaluate_record(truth, result, observed, obs["background_end"]))
    write_json(args.out / "per-station.json", {"segment_pair_repair": rows})
    write_json(args.out / "summary.json", {"methods": {"segment_pair_repair": aggregate(rows)},
               "scope": "six seen validation stations; post-pilot offline method development",
               "new_requests": 0, "source_run": str(args.run.resolve()),
               "source_registration_sha256": sha256(args.run / "registration.json"),
               "source_hashes": source_hashes(), "receipts": receipts})
    write_json(args.out / "activity-errors.json", {"segment_pair_repair": activity_errors(
        args.out, dataset, "segment_pair_repair", config["station_ids"])})
    shutil.copyfile(Path(__file__).resolve().parents[1] / "src/gnss_research/segmented_review.py",
                    args.out / "segmented-review-source.py")
    print(aggregate(rows)["types_daily_macro_f1"])


if __name__ == "__main__":
    main()
