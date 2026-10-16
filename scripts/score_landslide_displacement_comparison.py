"""Replay the frozen display experiment with JSON-safe scalar serialization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import run_landslide_displacement_comparison as experiment

from gnss_sim.artifacts import sha, write_json


def native_scalars(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: native_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native_scalars(item) for item in value]
    return value


def score(run, analysis):
    # Preserve the prerequest source and every decision expression. Only its
    # JSON output receives native Python scalars; no new inference is performed.
    def export(path, value):
        write_json(path, native_scalars(value))

    experiment.write_json = export
    experiment.score(run, analysis)
    write_json(
        analysis / "serialization-repair.json",
        {
            "scope": "NumPy scalar to native JSON scalar only",
            "frozen_runner_sha256": sha(run / "source/run_landslide_displacement_comparison.py"),
            "new_remote_requests": 0,
            "prediction_and_gate_expressions_changed": False,
            "source_sha256": sha(Path(__file__)),
        },
    )
    (analysis / "serialization-source.py").write_bytes(Path(__file__).read_bytes())
    # The first manifest remains an exact record of the original scoring pass.
    write_json(
        analysis / "repair-manifest.json",
        {
            "files": {
                str(path.relative_to(analysis)): sha(path)
                for path in analysis.rglob("*")
                if path.is_file()
            }
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    score(args.run.resolve(), args.analysis.resolve())
