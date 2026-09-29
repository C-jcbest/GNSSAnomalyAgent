"""Project orchestration/unit conversion; anomaly calculations run in original upstream code."""
from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from gnss_sim.schemas import CASE_TYPES, CaseInput, CaseTruth, Event, GenerationRequest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR_VERSION = "synthetic-v1"
SIGMA_MM = np.array([0.5, 0.5, 1.0])


def allocate_case_types(request: GenerationRequest):
    if request.case_type == "all":
        return list(CASE_TYPES) * (request.count // 4)
    return [request.case_type] * request.count


def make_plans(request: GenerationRequest):
    plans = []
    for index, kind in enumerate(allocate_case_types(request)):
        group = index // 4 if request.case_type == "all" else index
        states = np.random.SeedSequence([request.seed, group, 0x555053]).generate_state(3)
        plans.append({"case_id": f"case_{index + 1:04d}", "case_type": kind,
                      "background_group": f"group_{group + 1:04d}",
                      "noise_seed": int(states[0]), "position_seed": int(states[1]),
                      "tods_seed": int(states[2]), "axis": (group // 2) % 3,
                      "sign": -1 if group % 2 == 0 else 1})
    return plans


def convert_native(result):
    plan = result["plan"]
    kind, axis = plan["case_type"], plan["axis"]
    observed = np.asarray(result["observed"]) * SIGMA_MM
    noise = np.asarray(result["noise"]) * SIGMA_MM
    delta = observed - noise
    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(365)]
    events = []
    if kind != "normal":
        start, end = result["start"], result["end"]
        events = [Event(type=kind, task="point" if kind == "global_extremum" else "range",
                        axis=("N", "E", "U")[axis], start_index=start, end_index=end,
                        start_date=dates[start], end_date=dates[end], persistent=kind == "trend",
                        operation="native_extremum" if kind == "global_extremum" else "add",
                        target_offset_mm=float(delta[end, axis]),
                        observed_end_mm=float(observed[end, axis]), sigma_mm=float(SIGMA_MM[axis]),
                        source=result["source"], native_parameters=result["native_parameters"])]
    case = CaseInput(case_id=plan["case_id"], dates=dates, reference_coordinate_mm=(0, 0, 0),
                     observed_coordinate_mm=observed.tolist(), displacement_mm=observed.tolist(),
                     horizontal_offset_mm=np.linalg.norm(observed[:, :2], axis=1).tolist(),
                     spatial_offset_mm=np.linalg.norm(observed, axis=1).tolist())
    truth = CaseTruth(case_id=plan["case_id"], background_group=plan["background_group"],
                      measurement_noise_mm=noise.tolist(), anomaly_delta_mm=delta.tolist(),
                      noise_seed=plan["noise_seed"], position_seed=plan["position_seed"],
                      tods_seed=plan["tods_seed"], native_labels=result["labels"],
                      axis_labels=result["axis_labels"], events=events,
                      source_lock_sha256=result["source_lock_sha256"])
    return case, truth


def generate_cases(request: GenerationRequest):
    environment = ROOT / ".venv-generator"
    python = environment / "Scripts/python.exe"
    if not python.is_file():
        python = environment / "bin/python"
    if not python.is_file():
        raise RuntimeError("请先运行 uv run python scripts/setup_generator.py")
    plans = make_plans(request)
    # A separate process owns TODS' global NumPy RNG; stderr cannot fill a pipe and deadlock.
    with tempfile.TemporaryFile(mode="w+b") as errors:
        process = subprocess.Popen([str(python), str(ROOT / "scripts/generate_worker.py")],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                   text=True, encoding="utf-8", cwd=ROOT)
        try:
            process.stdin.write(json.dumps(plans) + "\n")
            process.stdin.close()
            received = 0
            for line in process.stdout:
                result = json.loads(line)
                if received >= len(plans) or result["plan"] != plans[received]:
                    raise RuntimeError("Unexpected upstream result order")
                received += 1
                case, truth = convert_native(result)
                yield case, truth, result["plan"]
            code = process.wait(timeout=30)
            if code or received != len(plans):
                errors.seek(0)
                detail = errors.read().decode("utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"Upstream worker failed ({code}, {received}/{len(plans)}): {detail}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
