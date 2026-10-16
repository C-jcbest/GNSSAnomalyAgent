"""Prepare original numerical evidence and fixed training-pair visual QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from probe_landslide_endpoint_fit import export
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha
from gnss_sim.landslide_paired_persistence import render_encoder_chart


def prepare_review(run, out):
    out.mkdir(exist_ok=False)
    rows = read_json(run / "selected-training-pairs.json")
    first_weak = next(row for row in rows if row["weak_center"])
    chosen = rows[:4]
    if first_weak not in chosen:
        chosen.append(first_weak)
    evidence = []
    for row in chosen:
        with np.load(run / "pairs" / (row["pair_id"] + ".npz")) as data:
            endpoints = row["endpoints"]
            positive, negative = data["original_observations"], data["negative_observations"]
            observed = np.isfinite(positive).all(axis=1)
            if not np.array_equal(observed, np.isfinite(negative).all(axis=1)):
                raise ValueError("Counterpart changed missing dates")
            np.testing.assert_array_equal(positive[endpoints], negative[endpoints])
            error = positive - data["original_latent"]
            counterpart_error = negative - data["negative_latent"]
            maximum_error_difference = float(np.max(np.abs((error - counterpart_error)[observed])))
            blocks = []
            for center in range(row["day"] - 84, row["day"] + 85, 14):
                values = []
                for observations in (positive, negative):
                    block = observations[center - 7:center + 7]
                    complete = np.isfinite(block).all(axis=1)
                    values.append(np.median(block[complete], axis=0).tolist() if complete.sum() >= 8 else None)
                blocks.append({"original_days": [center - 7, center + 7], "positive": values[0], "negative": values[1]})
            evidence.append({**row, "equal_observed_endpoint_values": positive[endpoints].tolist(),
                             "maximum_error_difference_mm": maximum_error_difference, "blocks": blocks,
                             "positive_noise_proxy": data["positive_noise"].tolist(),
                             "negative_noise_proxy": data["negative_noise"].tolist()})
            for side in ("positive", "negative"):
                render_encoder_chart(data[side + "_image"]).save(out / (row["pair_id"] + "-" + side + ".png"))
    export(out / "numeric-evidence.json", evidence)
    export(out / "scope.json", {"selection": "First four training pairs and first weak training pair, before calibration results",
                                "pair_count": len(rows), "weak_pair_count": sum(row["weak_center"] for row in rows),
                                "independent_expert": False, "historical_label_exposure": True, "reference_changed": False,
                                "preparation_source_sha256": sha(Path(__file__))})
    print("Prepared", [row["pair_id"] for row in chosen], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    prepare_review(arguments.run.resolve(), arguments.out.resolve())
