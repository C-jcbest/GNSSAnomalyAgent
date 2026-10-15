"""Explicit technical replication of identical frozen requests into a fresh run."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from experiment_exp26 import check_frozen

from gnss_research.files import read_json, sha256, write_json


def prepare(original: Path, output: Path, phase: str):
    registration = read_json(original / f"{phase}-registration.json")
    check_frozen(registration)
    if phase == "pilot":
        output.mkdir(parents=True, exist_ok=False)
        shutil.copytree(original / "frozen-source", output / "frozen-source")
    elif not (output / "pilot-completed.json").exists():
        raise ValueError("complete technical pilot before remaining requests")
    jobs = {(job["mode"], job["station"]): job for job in registration["jobs"]}
    modes = [p.name for p in (original / "predictions").iterdir()]
    for record in registration["records"]:
        station = record["station_id"]
        shutil.copytree(original / "inputs" / station, output / "inputs" / station)
        for mode in modes:
            source = original / "predictions" / mode / station
            work = output / "predictions" / mode / station
            work.mkdir(parents=True)
            shutil.copyfile(source / "candidates.json", work / "candidates.json")
            if (mode, station) in jobs:
                shutil.copyfile(source / "request.json", work / "request.json")
                if sha256(work / "request.json") != jobs[(mode, station)]["request_sha256"]:
                    raise ValueError("technical replication changed request bytes")
            else:
                shutil.copyfile(source / "result.json", work / "result.json")
                final = output / "final" / mode / station
                final.mkdir(parents=True)
                shutil.copyfile(original / "final" / mode / station / "result.json", final / "result.json")
        final = output / "final/vector_nearest" / station
        final.mkdir(parents=True)
        shutil.copyfile(original / "final/vector_nearest" / station / "result.json", final / "result.json")
    registration.update(technical_attempt=2, original_visual=str(original),
                        original_registration_sha256=sha256(original / f"{phase}-registration.json"))
    registration["frozen_files"][str(Path(__file__))] = sha256(Path(__file__))
    registration["frozen_files"]["docs/exp28-transport-replication.md"] = sha256(Path("docs/exp28-transport-replication.md"))
    write_json(output / f"{phase}-registration.json", registration)
    print("registered exact-request technical replication", phase, len(registration["jobs"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "remaining"), required=True)
    args = parser.parse_args()
    prepare(args.original, args.output, args.phase)
