"""Save original-displacement block evidence for post-score visual error review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import runs


def inspect(data, run, out):
    out.mkdir(parents=True, exist_ok=False)
    predictions = read_json(run / "predictions.json")
    blocks = {
        "case_0390": [
            (25, 40),
            (60, 75),
            (135, 150),
            (170, 185),
            (360, 375),
            (415, 430),
            (475, 490),
            (520, 535),
        ],
        "case_0385": [
            (115, 130),
            (135, 150),
            (160, 175),
            (220, 235),
            (245, 260),
            (280, 295),
            (315, 330),
            (500, 515),
        ],
    }
    labels = {cid: [] for cid in blocks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(int(row["activity_label"]))
    result = {}
    for cid, ranges in blocks.items():
        case, _ = load_case_features(data, cid)
        values = np.array(
            [row if row is not None else [np.nan] * 3 for row in case.displacement_mm]
        )
        evidence = []
        for start, stop in ranges:
            sample = values[start:stop]
            observed = np.isfinite(sample).all(axis=1)
            evidence.append(
                {
                    "start": start,
                    "stop_exclusive": stop,
                    "observed_days": int(observed.sum()),
                    "raw_neu_median_mm": np.median(sample[observed], axis=0).tolist()
                    if observed.any()
                    else None,
                }
            )
        result[cid] = {
            "original_block_evidence": evidence,
            "process_active_runs": runs(np.array(labels[cid]) == 1),
            "predicted_active_runs": {
                name: runs(np.array(rows[cid]["activity"]) == 1)
                for name, rows in predictions.items()
            },
        }
    write_json(out / "original-evidence.json", result)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    inspect(args.data.resolve(), args.run.resolve(), args.out.resolve())
