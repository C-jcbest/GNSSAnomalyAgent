"""Describe paired-record development variation without a significance claim."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json


def bootstrap(run, out):
    verify_manifest(run, "freeze.json")
    ids = read_json(run / "protocol.json")["split"]["validation"]
    results = read_json(run / "results.json")
    draws = np.random.default_rng(2026101254).integers(0, len(ids), (2000, len(ids)))
    rows = []
    for control in (
        "fixed_blend",
        "scores_only_review",
        "generic_raw_stacker",
        "visual_numeric",
        "xgb_raw84",
    ):
        for metric in ("daily", "events"):
            values = {}
            valid = np.ones(2000, dtype=bool)
            for name in ("raw_evidence_review", control):
                counts = np.array(
                    [
                        [results[name]["per_case"][cid][metric][key] for key in ("tp", "fp", "fn")]
                        for cid in ids
                    ]
                )
                summed = counts[draws].sum(axis=1)
                denominator = 2 * summed[:, 0] + summed[:, 1] + summed[:, 2]
                valid &= denominator > 0
                values[name] = np.divide(
                    2 * summed[:, 0],
                    denominator,
                    out=np.zeros(2000, dtype=float),
                    where=denominator > 0,
                )
            difference = (values["raw_evidence_review"] - values[control])[valid]
            rows.append(
                dict(
                    control=control,
                    metric=metric,
                    defined_draws=int(valid.sum()),
                    low=float(np.quantile(difference, 0.025)),
                    high=float(np.quantile(difference, 0.975)),
                )
            )
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "paired-record-bootstrap.json",
        dict(
            seed=2026101254,
            draws=2000,
            unit="paired complete original record",
            scope="exposed development; no selection-adjusted significance",
            rows=rows,
        ),
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(rows[:4], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    bootstrap(args.run.resolve(), args.out.resolve())
