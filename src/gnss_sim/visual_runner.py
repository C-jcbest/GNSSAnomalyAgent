"""P6 image-only inference; truth is opened only by the separate evaluation command."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from importlib.metadata import version
from pathlib import Path

import httpx
import numpy as np

from gnss_sim.schemas import CaseInput, PointResult, RangeResult
from gnss_sim.visual import AXES, DAYS, failed_result, parse_result, render_case

TASKS = ("point", "range")
MODEL_ID = "qwen3.8-flash"
METHOD = "visual-v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: _digest((root / name).read_text(encoding="utf-8").replace("\r\n", "\n"))
            for name in ("visual.py", "visual_runner.py")}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    model = config["model"]
    if (config["protocol"] != "visual-v1" or config["renderer"] != "renderer-v1" or
            config["method"] != METHOD or model["request_id"] != MODEL_ID or
            model["temperature"] != 0 or model["enable_thinking"] is not False or
            model["attempts_per_task_case"] != 1 or
            config["image"]["panels"] != list(AXES) or
            (config["image"]["width_px"], config["image"]["height_px"]) != (1800, 1200)):
        raise ValueError("P6 frozen configuration mismatch")
    return config


def _credentials(env_file: Path | None) -> tuple[str, str]:
    values = {}
    if env_file is not None:
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            if key.strip() in ("QWEN_BASE_URL", "QWEN_API_KEY"):
                values[key.strip()] = value.strip().strip("\"'")
    base = os.getenv("QWEN_BASE_URL") or values.get("QWEN_BASE_URL")
    key = os.getenv("QWEN_API_KEY") or values.get("QWEN_API_KEY")
    if not base or not key:
        raise ValueError("QWEN_BASE_URL and QWEN_API_KEY are required")
    if not base.startswith(("https://", "http://")):
        raise ValueError("invalid QWEN_BASE_URL")
    return base.rstrip("/"), key


def _pilot_inputs(pilot_dir: Path, config: dict) -> list[tuple[str, Path]]:
    manifest_path = pilot_dir / "manifest.json"
    summary_path = pilot_dir / "summary.json"
    if (_sha(manifest_path) != config["pilot_manifest_sha256"] or
            _sha(summary_path) != config["pilot_summary_sha256"]):
        raise ValueError("pilot-v1 manifest or summary hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["pilot_id"] != config["pilot_id"] or len(manifest["cases"]) != 300:
        raise ValueError("pilot-v1 identity or case count changed")
    inputs = []
    for index, item in enumerate(manifest["cases"], 1):
        case_id = item["case_id"]
        if case_id != f"case_{index:04d}":
            raise ValueError("pilot-v1 case order changed")
        path = pilot_dir / "cases" / case_id / "input.json"
        if _sha(path) != item["input_sha256"]:
            raise ValueError(f"pilot-v1 input hash changed: {case_id}")
        inputs.append((case_id, path))
    return inputs


def _engineering_cases() -> list[CaseInput]:
    """Six hand-built transport/coordinate checks outside the frozen Pilot."""
    times = np.arange(DAYS, dtype=float)
    dates = [date(2025, 1, 1) + timedelta(days=day) for day in range(DAYS)]
    cases = []
    for index in range(6):
        values = np.column_stack((0.4 * np.sin(times / 24),
                                  0.4 * np.cos(times / 28),
                                  0.7 * np.sin(times / 37)))
        if index == 1:
            values[60, 0] += 8
        elif index == 2:
            values[120:, 1] += 7
        elif index == 3:
            values[100:190, 2] += np.linspace(0, 9, 90)
            values[190:, 2] += 9
        elif index == 4:
            values[220:234, 0] += 6
        elif index == 5:
            values[80, 1] -= 8
            values[180:270, 2] += np.linspace(0, -7, 90)
            values[270:, 2] -= 7
        vectors = [tuple(map(float, row)) for row in values]
        cases.append(CaseInput(
            case_id=f"engineering_{index + 1:02d}", dates=dates,
            reference_coordinate_mm=(0.0, 0.0, 0.0),
            observed_coordinate_mm=vectors, displacement_mm=vectors,
            horizontal_offset_mm=[float(np.hypot(row[0], row[1])) for row in values],
            spatial_offset_mm=[float(np.linalg.norm(row)) for row in values],
        ))
    return cases


def _request(client: httpx.Client, base: str, key: str, image: bytes,
             task: str, case_id: str, config: dict) -> tuple[PointResult | RangeResult, dict]:
    model = config["model"]
    content = [{"type": "text", "text": config["prompts"][task]},
               {"type": "image_url", "image_url": {
                   "url": "data:image/png;base64," + base64.b64encode(image).decode("ascii"),
               }, "max_pixels": model["max_pixels"]}]
    payload = {"model": MODEL_ID, "temperature": 0,
               "max_tokens": model["max_tokens"],
               "enable_thinking": False,
               "vl_high_resolution_images": model["vl_high_resolution_images"],
               "response_format": {"type": "json_object"},
               "messages": [{"role": "user", "content": content}]}
    if model["top_p"] is not None:
        payload["top_p"] = model["top_p"]
    record = {"case_id": case_id, "task": task, "request_model": MODEL_ID,
              "image_sha256": hashlib.sha256(image).hexdigest(),
              "status": "failed", "response": None, "response_model": None,
              "response_id": None, "usage": None, "error_type": None}
    started = time.perf_counter()
    try:
        response = client.post(f"{base}/chat/completions", json=payload,
                               headers={"Authorization": f"Bearer {key}"})
        record["http_status"] = response.status_code
        response.raise_for_status()
        body = response.json()
        record["response_model"] = body.get("model")
        record["response_id"] = body.get("id")
        record["usage"] = body.get("usage")
        choice = body["choices"][0]
        record["finish_reason"] = choice.get("finish_reason")
        record["reasoning_present"] = bool(choice["message"].get("reasoning_content"))
        record["response"] = choice["message"]["content"]
        if record["finish_reason"] not in (None, "stop"):
            raise ValueError("model did not finish normally")
        if record["reasoning_present"]:
            raise ValueError("thinking response despite enable_thinking=false")
        if record["response_model"] != MODEL_ID:
            raise ValueError("response model ID differs from frozen request ID")
        result = parse_result(record["response"], case_id, task)
        record["status"] = "success"
        return result, record
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError,
            json.JSONDecodeError) as exc:
        record["error_type"] = type(exc).__name__
        return failed_result(case_id, task), record
    finally:
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)


def run_preflight(config_path: Path, out_dir: Path, env_file: Path | None) -> dict:
    config = load_config(config_path)
    base, key = _credentials(env_file)
    directory = out_dir / MODEL_ID / "preflight"
    if directory.exists():
        raise FileExistsError("P6 preflight exists; preserve its raw responses")
    directory.mkdir(parents=True)
    records = []
    with httpx.Client(timeout=config["model"]["timeout_seconds"],
                      follow_redirects=False) as client:
        for case in _engineering_cases():
            image_path = directory / "images" / f"{case.case_id}.png"
            render_case(case, image_path)
            image = image_path.read_bytes()
            for task in TASKS:
                _, record = _request(client, base, key, image, task, case.case_id, config)
                records.append(record)
    with (directory / "raw.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    report = {"protocol": config["protocol"], "model": MODEL_ID,
              "config_sha256": _sha(config_path), "source_sha256": _source_hashes(),
              "calls": len(records),
              "valid_responses": sum(item["status"] == "success" for item in records),
              "response_models": sorted({item["response_model"] for item in records
                                         if item["response_model"]}),
              "errors": [{"case_id": item["case_id"], "task": item["task"],
                          "error_type": item["error_type"]} for item in records
                         if item["status"] != "success"]}
    _write_json(directory / "report.json", report)
    return report


def _load_existing(directory: Path, task: str) -> dict[str, PointResult | RangeResult]:
    path = directory / task / "predictions.jsonl"
    if not path.exists():
        return {}
    result_type = PointResult if task == "point" else RangeResult
    results = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        result = result_type.model_validate_json(line)
        if result.case_id in results:
            raise ValueError("duplicate P6 result in existing run")
        results[result.case_id] = result
    return results


def run_visual(config_path: Path, pilot_dir: Path, out_dir: Path,
               env_file: Path | None) -> dict:
    config = load_config(config_path)
    inputs = _pilot_inputs(pilot_dir, config)
    directory = out_dir / MODEL_ID
    preflight = json.loads((directory / "preflight" / "report.json").read_text(
        encoding="utf-8"))
    if (preflight["config_sha256"] != _sha(config_path) or
            preflight["source_sha256"] != _source_hashes() or
            preflight["valid_responses"] != preflight["calls"] or
            preflight["calls"] != 12 or
            preflight["response_models"] != [MODEL_ID]):
        raise ValueError("P6 preflight did not pass for this frozen source/config")
    base, key = _credentials(env_file)
    identity = {"config_sha256": _sha(config_path), "source_sha256": _source_hashes(),
                "pilot_manifest_sha256": _sha(pilot_dir / "manifest.json"),
                "pilot_summary_sha256": _sha(pilot_dir / "summary.json")}
    identity_path = directory / "run-identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("existing P6 run has different source, config, or pilot")
    else:
        _write_json(identity_path, identity)
    images = {}
    for case_id, path in inputs:
        case = CaseInput.model_validate_json(path.read_bytes())
        if case.case_id != case_id:
            raise ValueError("case input identity mismatch")
        image_path = directory / "images" / f"{case_id}.png"
        previous_hash = _sha(image_path) if image_path.exists() else None
        render_case(case, image_path)
        if previous_hash is not None and _sha(image_path) != previous_hash:
            raise ValueError("rendered image changed during resumed P6 run")
        images[case_id] = image_path
    existing = {task: _load_existing(directory, task) for task in TASKS}
    expected = {case_id for case_id, _ in inputs}
    if any(set(existing[task]) - expected for task in TASKS):
        raise ValueError("existing P6 run contains unknown case")
    todo = [(case_id, task) for case_id, _ in inputs for task in TASKS
            if case_id not in existing[task]]
    started = time.perf_counter()
    with httpx.Client(timeout=config["model"]["timeout_seconds"],
                      follow_redirects=False) as client:
        with ThreadPoolExecutor(max_workers=config["model"]["workers"]) as pool:
            futures = {pool.submit(_request, client, base, key, images[case_id].read_bytes(),
                                   task, case_id, config): (case_id, task)
                       for case_id, task in todo}
            for completed, future in enumerate(as_completed(futures), 1):
                case_id, task = futures[future]
                result, record = future.result()
                task_dir = directory / task
                task_dir.mkdir(parents=True, exist_ok=True)
                with (task_dir / "raw.jsonl").open("a", encoding="utf-8") as raw:
                    raw.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                with (task_dir / "predictions.jsonl").open("a", encoding="utf-8") as output:
                    output.write(result.model_dump_json() + "\n")
                existing[task][case_id] = result
                if completed % 25 == 0 or completed == len(todo):
                    print(f"P6 requests {completed}/{len(todo)}; latest {case_id}/{task}",
                          flush=True)
    _pilot_inputs(pilot_dir, config)
    if any(set(existing[task]) != expected for task in TASKS):
        raise ValueError("P6 did not produce 300 results for each task")
    run = {**identity, "protocol": config["protocol"], "method": METHOD,
           "request_model": MODEL_ID, "provider_endpoint": base,
           "settings": config["model"], "prompt_sha256": {
               task: _digest(config["prompts"][task]) for task in TASKS},
           "versions": {name: version(name) for name in ("matplotlib", "httpx", "numpy")},
           "python_version": platform.python_version(),
           "task_counts": {task: {"results": len(existing[task]),
                                  "success": sum(result.status == "success"
                                                 for result in existing[task].values())}
                           for task in TASKS},
           "new_requests": len(todo), "runtime_seconds": round(time.perf_counter() - started, 3),
           "cost": None}
    _write_json(directory / "run.json", run)
    return run


def evaluate_visual(config_path: Path, pilot_dir: Path, out_dir: Path) -> dict:
    """Separate scoring process: this is the only P6 path importing the truth reader."""
    from gnss_sim.evaluation import evaluate_pilot

    config = load_config(config_path)
    inputs = _pilot_inputs(pilot_dir, config)
    directory = out_dir / MODEL_ID
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if (run["config_sha256"] != _sha(config_path) or
            run["source_sha256"] != _source_hashes() or
            run["pilot_manifest_sha256"] != _sha(pilot_dir / "manifest.json")):
        raise ValueError("P6 run identity changed before evaluation")
    expected = {case_id for case_id, _ in inputs}
    reports = {}
    for task in TASKS:
        results = _load_existing(directory, task)
        if set(results) != expected:
            raise ValueError(f"P6 {task} predictions are incomplete")
        reports[task] = evaluate_pilot(pilot_dir, list(results.values()), METHOD, task)
        _write_json(directory / task / "report.json", reports[task])
    return reports
