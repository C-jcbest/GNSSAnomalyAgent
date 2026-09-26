"""Run and freeze the four P5 baselines against the unchanged P4 evaluator."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from gnss_sim.evaluation import evaluate_pilot
from gnss_sim.numerical import (
    DAYS,
    WINDOW,
    matrix_profile,
    pelt_points,
    score_to_points,
    score_to_ranges,
    spectral_residual,
    theilsen_score,
)
from gnss_sim.pilot import verify_pilot
from gnss_sim.schemas import CaseInput, PointResult, RangeResult

AXES = ("N", "E", "U")
METHOD_TASK = {"sr": "point", "pelt": "point",
               "matrix-profile": "range", "theilsen": "range"}
SCORE_METHODS = {"sr": spectral_residual, "matrix-profile": matrix_profile,
                 "theilsen": theilsen_score}
BETAS = (1, 2, 4, 8, 16, 32)
METHOD_ORDER = tuple(METHOD_TASK)


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n").encode("utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {name: hashlib.sha256((directory / name).read_text(encoding="utf-8")
                                 .replace("\r\n", "\n").encode()).hexdigest()
            for name in ("numerical.py", "numerical_runner.py", "evaluation.py")}


def _revision() -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
        check=False, capture_output=True, text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def _input(directory: Path, case_id: str) -> CaseInput:
    case = CaseInput.model_validate_json(
        (directory / "cases" / case_id / "input.json").read_bytes())
    if case.case_id != case_id or len(case.displacement_mm) != DAYS:
        raise ValueError("invalid pilot input identity or day count")
    return case


def _axes(case: CaseInput) -> dict[str, np.ndarray]:
    displacement = np.asarray(case.displacement_mm, dtype=np.float64)
    if displacement.shape != (DAYS, 3) or not np.all(np.isfinite(displacement)):
        raise ValueError("invalid signed N/E/U displacement")
    return {axis: displacement[:, index] for index, axis in enumerate(AXES)}


def calibrate_normal(method: str, normal_inputs: list[CaseInput]) -> tuple[dict, dict]:
    """Calibrate exclusively from designated Normal CaseInput objects."""
    if method not in METHOD_TASK or len(normal_inputs) != 30:
        raise ValueError("P5 calibration requires a method and exactly 30 Normal inputs")
    series = [_axes(case) for case in normal_inputs]
    if method == "pelt":
        betas = {}
        alarms = {}
        for axis in AXES:
            counts = {}
            for beta in BETAS:
                count = 0
                for case in series:
                    count += bool(pelt_points(case[axis], beta))
                    if count > 1:
                        break
                counts[beta] = count
                if count <= 1:
                    break
            chosen = next((beta for beta in BETAS if counts.get(beta, 2) <= 1), BETAS[-1])
            betas[axis] = chosen
            alarms[axis] = sum(bool(pelt_points(case[axis], chosen)) for case in series)
        return ({"beta": betas, "penalty": {axis: float(betas[axis] * np.log(DAYS))
                                          for axis in AXES}}, alarms)

    scorer = SCORE_METHODS[method]
    maxima = {axis: [float(np.max(scorer(case[axis]))) for case in series]
              for axis in AXES}
    thresholds = {axis: float(np.quantile(maxima[axis], 0.95, method="linear"))
                  for axis in AXES}
    alarms = {axis: sum(value > thresholds[axis] for value in maxima[axis]) for axis in AXES}
    return {"threshold": thresholds}, alarms


def _predict(method: str, case: CaseInput, calibration: dict) -> PointResult | RangeResult:
    axes = _axes(case)
    if method == "pelt":
        predictions = {axis: pelt_points(axes[axis], calibration["beta"][axis])
                       for axis in AXES}
    else:
        scorer = SCORE_METHODS[method]
        convert = score_to_points if METHOD_TASK[method] == "point" else score_to_ranges
        predictions = {axis: convert(scorer(axes[axis]), calibration["threshold"][axis])
                       for axis in AXES}
    model = PointResult if METHOD_TASK[method] == "point" else RangeResult
    return model(case_id=case.case_id, method=method, status="success",
                 predictions=predictions)


def _failed(case_id: str, method: str) -> PointResult | RangeResult:
    model = PointResult if METHOD_TASK[method] == "point" else RangeResult
    return model(case_id=case_id, method=method, status="failed",
                 predictions={axis: [] for axis in AXES})


def _parameters(method: str, calibration: dict) -> dict:
    if method == "sr":
        fixed = {"log_amplitude_window": 3, "epsilon": 1e-12, "threshold_quantile": 0.95}
    elif method == "pelt":
        fixed = {"model": "l2", "min_size": 3, "jump": 1, "beta_candidates": list(BETAS),
                 "normal_max_false_axes_per_axis": 1}
    elif method == "matrix-profile":
        fixed = {"window_days": WINDOW, "normalize": True, "center_offset": WINDOW // 2,
                 "threshold_quantile": 0.95}
    else:
        fixed = {"window_days": WINDOW, "center_offset": WINDOW // 2,
                 "threshold_quantile": 0.95}
    return {"fixed": fixed, "calibration": calibration}


def _table(root: Path, task: str, methods: tuple[str, str]) -> str:
    columns = (("Precision", "Recall", "F1", "FAR", "Success", "Time (s)")
               if task == "point" else
               ("Affiliation P", "Affiliation R", "Affiliation F1", "FAR", "Success", "Time (s)"))
    fields = (("precision", "recall", "f1") if task == "point" else
              ("affiliation_precision", "affiliation_recall", "affiliation_f1"))
    rows = ["| Method | " + " | ".join(columns) + " |",
            "| --- | " + " | ".join("---:" for _ in columns) + " |"]
    for method in methods:
        report = json.loads((root / method / "report.json").read_text(encoding="utf-8"))
        run = json.loads((root / method / "run.json").read_text(encoding="utf-8"))
        values = [report[field] for field in fields] + [report["far"],
                 report["execution_success_rate"], run["runtime_seconds"]]
        cells = ["N/A" if value is None else f"{value:.4f}" for value in values]
        rows.append("| " + method + " | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def freeze_selection(root: Path) -> dict | None:
    """Create two task tables and the frozen winners once all valid runs exist."""
    if not all((root / method / name).is_file() for method in METHOD_ORDER
               for name in ("report.json", "run.json", "predictions.jsonl")):
        return None
    runs = {method: json.loads((root / method / "run.json").read_text(encoding="utf-8"))
            for method in METHOD_ORDER}
    reports = {method: json.loads((root / method / "report.json").read_text(encoding="utf-8"))
               for method in METHOD_ORDER}
    reference = runs[METHOD_ORDER[0]]
    for method in METHOD_ORDER:
        run = runs[method]
        if (run["pilot_sha256"] != reference["pilot_sha256"] or
                run["source_sha256"] != reference["source_sha256"] or
                run["failed_cases"] or reports[method]["execution_success_rate"] != 1.0 or
                run["task"] != METHOD_TASK[method] or reports[method]["task"] != METHOD_TASK[method]):
            raise ValueError("P5 runs are incomplete or use different pilot/source versions")
    winners = {}
    for task, methods, metric in (("point", ("sr", "pelt"), "f1"),
                                  ("range", ("matrix-profile", "theilsen"),
                                   "affiliation_f1")):
        winner = min(methods, key=lambda method: (
            -reports[method][metric],
            reports[method]["far"] if reports[method]["far"] is not None else float("inf"),
            methods.index(method),
        ))
        winners[task] = {"method": winner, "parameters": runs[winner]["parameters"]}
        (root / f"{task}-table.md").write_text(_table(root, task, methods), encoding="utf-8")
    frozen = {"pilot_sha256": reference["pilot_sha256"],
              "source_sha256": reference["source_sha256"], "selected": winners}
    (root / "frozen-numerical.json").write_bytes(_json_bytes(frozen))
    return frozen


def run_numerical(method: str, pilot_dir: Path, output_root: Path) -> dict:
    if method not in METHOD_TASK:
        raise ValueError(f"unknown P5 method: {method}")
    pilot_dir = pilot_dir.resolve()
    output_root = output_root.resolve()
    if output_root == pilot_dir or pilot_dir in output_root.parents:
        raise ValueError("P5 outputs must not be inside the frozen pilot")
    manifest = verify_pilot(pilot_dir)
    pilot_sha256 = {name: _hash(pilot_dir / f"{name}.json")
                    for name in ("manifest", "summary")}
    started = time.perf_counter()
    normal = [_input(pilot_dir, entry["case_id"]) for entry in manifest["cases"]
              if entry["case_type"] == "normal"]
    calibration, normal_alarms = calibrate_normal(method, normal)
    results: list[PointResult] | list[RangeResult] = []
    errors = []
    for entry in manifest["cases"]:
        case_id = entry["case_id"]
        try:
            results.append(_predict(method, _input(pilot_dir, case_id), calibration))
        except Exception as exc:
            results.append(_failed(case_id, method))
            errors.append({"case_id": case_id, "error_type": type(exc).__name__})
    directory = output_root / method
    directory.mkdir(parents=True, exist_ok=True)
    lines = [result.model_dump_json() for result in results]
    (directory / "predictions.jsonl").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    runtime = time.perf_counter() - started

    verify_pilot(pilot_dir)
    if pilot_sha256 != {name: _hash(pilot_dir / f"{name}.json")
                        for name in ("manifest", "summary")}:
        raise ValueError("pilot manifest or summary changed during P5")
    report = evaluate_pilot(pilot_dir, results, method, METHOD_TASK[method])
    (directory / "report.json").write_bytes(_json_bytes(report))
    run = {"method": method, "task": METHOD_TASK[method],
           "parameters": _parameters(method, calibration),
           "normal_calibration_alarm_count": normal_alarms,
           "pilot_sha256": pilot_sha256,
           "source_sha256": _source_hashes(), "source_revision": _revision(),
           "versions": {name: version(name) for name in
                        ("numpy", "scipy", "ruptures", "stumpy")},
           "python_version": platform.python_version(),
           "runtime_seconds": round(runtime, 6), "failed_cases": len(errors),
           "errors": errors}
    (directory / "run.json").write_bytes(_json_bytes(run))
    freeze_selection(output_root)
    return report
