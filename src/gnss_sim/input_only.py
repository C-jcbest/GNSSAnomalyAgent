"""Frozen numerical inference from an input-only package, without truth or calibration."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from gnss_sim.numerical import score_to_points, score_to_ranges, spectral_residual, theilsen_score
from gnss_sim.schemas import CaseInput, PointResult, RangeResult

AXES = ("N", "E", "U")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(row.model_dump_json() + "\n")


def read_inputs(directory: Path) -> list[tuple[dict, CaseInput]]:
    manifest = json.loads((directory / "manifest.json").read_bytes())
    if set(manifest) != {"protocol", "cases"} or manifest["protocol"] != "input-only-v1":
        raise ValueError("expected input-only manifest")
    rows, seen = [], set()
    for entry in manifest["cases"]:
        if set(entry) != {"case_id", "input_sha256", "image_sha256"}:
            raise ValueError("non-input metadata in inference manifest")
        case_id = entry["case_id"]
        if not re.fullmatch(r"case_[0-9]{4}", case_id) or case_id in seen:
            raise ValueError("invalid or duplicate case identity")
        seen.add(case_id)
        path = directory / "cases" / case_id / "input.json"
        if sha(path) != entry["input_sha256"]:
            raise ValueError("input digest mismatch")
        case = CaseInput.model_validate_json(path.read_bytes())
        if case.case_id != case_id or len(case.dates) != 365:
            raise ValueError("input identity or length mismatch")
        if sha(directory / "images" / f"{case_id}.png") != entry["image_sha256"]:
            raise ValueError("image digest mismatch")
        rows.append((entry, case))
    if not rows:
        raise ValueError("empty input package")
    return rows


def predict_frozen(case: CaseInput, task: str, frozen: dict):
    if task not in ("point", "range"):
        raise ValueError("unknown task")
    method = "sr" if task == "point" else "theilsen"
    values = np.asarray(case.displacement_mm, dtype=float)
    if values.shape != (365, 3) or not np.isfinite(values).all():
        raise ValueError("expected finite 365-day N/E/U input")
    scorer = spectral_residual if task == "point" else theilsen_score
    convert = score_to_points if task == "point" else score_to_ranges
    thresholds = frozen["methods"][method]["calibration"]["threshold"]
    model = PointResult if task == "point" else RangeResult
    return model(case_id=case.case_id, method=method, status="success", predictions={
        axis: convert(scorer(values[:, index]), thresholds[axis])
        for index, axis in enumerate(AXES)})


def run_numerical(inputs: Path, frozen_path: Path, out: Path) -> dict:
    frozen = json.loads(frozen_path.read_bytes())
    source = Path(__file__).with_name("numerical.py").read_text(encoding="utf-8")
    if hashlib.sha256(source.encode()).hexdigest() != frozen["source_sha256"]["numerical.py"]:
        raise ValueError("frozen numerical implementation changed")
    cases = read_inputs(inputs)
    out.mkdir(parents=True, exist_ok=False)
    timings = []
    for task in ("point", "range"):
        rows = []
        for _, case in cases:
            started = time.perf_counter()
            rows.append(predict_frozen(case, task, frozen))
            timings.append({"task": task, "case_id": case.case_id,
                            "seconds": time.perf_counter() - started})
        write_rows(out / task / "predictions.jsonl", rows)
    report = {"input_manifest_sha256": sha(inputs / "manifest.json"),
              "parameters_sha256": sha(frozen_path), "calibration_calls": 0,
              "timings": timings, "note": "First sample includes cold initialization."}
    write_json(out / "run.json", report)
    return report
