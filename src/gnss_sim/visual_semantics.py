"""Versioned prompt-only experiment; reuse frozen P6 transport, parser and images."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import httpx

from gnss_sim.schemas import PointResult, RangeResult
from gnss_sim.visual import failed_result, render_case
from gnss_sim.visual_runner import (
    TASKS,
    _credentials,
    _engineering_cases,
    _pilot_inputs,
    _request,
    _sha,
)

METHOD = "visual-semantics-v2"
ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes() -> dict:
    return {name: hashlib.sha256((Path(__file__).parent / name).read_text(
        encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
        for name in ("visual_semantics.py", "visual.py", "visual_runner.py", "schemas.py")}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_bytes())
    original = json.loads((ROOT / "configs/p6-visual.json").read_bytes())
    frozen = json.loads((ROOT / "configs/p6-frozen.json").read_bytes())
    if _sha(ROOT / "configs/p6-visual.json") != frozen["config_sha256"]:
        raise ValueError("Parent P6 config changed")
    expected = dict(original, protocol=METHOD, method=METHOD, prompts=config.get("prompts"))
    if config != expected or set(config["prompts"]) != set(TASKS):
        raise ValueError("Only protocol/method/prompts may differ from frozen P6")
    if any(not isinstance(prompt, str) or not prompt.strip()
           for prompt in config["prompts"].values()):
        raise ValueError("Two nonempty prompts required")
    for name, digest in frozen["source_sha256"].items():
        if source_hashes()[name] != digest:
            raise ValueError("Frozen P6 inference source changed")
    return config


def request_once(client, base, key, image, task, case_id, config):
    result, record = _request(client, base, key, image, task, case_id, config)
    result = result.model_copy(update={"method": METHOD})
    record.update(method=METHOD, completed_at=datetime.now(timezone.utc).isoformat())
    return result, record


def run_preflight(config_path: Path, out: Path, env_file: Path | None) -> dict:
    config = load_config(config_path)
    base, key = _credentials(env_file)
    directory = out / "preflight"
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    with httpx.Client(timeout=config["model"]["timeout_seconds"],
                      follow_redirects=False) as client:
        for case in _engineering_cases():
            path = directory / "images" / f"{case.case_id}.png"
            render_case(case, path)
            for task in TASKS:
                _, record = request_once(client, base, key, path.read_bytes(), task,
                                         case.case_id, config)
                records.append(record)
                with (directory / "raw.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {"config_sha256": _sha(config_path), "source_sha256": source_hashes(),
              "calls": len(records),
              "valid_responses": sum(row["status"] == "success" for row in records)}
    write_json(directory / "report.json", report)
    return report


def frozen_images(pilot: Path, config: dict, original_run: Path) -> dict[str, Path]:
    inputs = _pilot_inputs(pilot, config)  # Hash only input files, never truth.
    paths = {}
    expected = {case_id for case_id, _ in inputs}
    per_task = []
    for task in TASKS:
        rows = [json.loads(line) for line in (original_run / task / "raw.jsonl").read_text(
            encoding="utf-8").splitlines()]
        if len(rows) != 300 or {row["case_id"] for row in rows} != expected:
            raise ValueError("Original image provenance is incomplete")
        per_task.append({row["case_id"]: row["image_sha256"] for row in rows})
    for case_id, _ in inputs:
        path = original_run / "images" / f"{case_id}.png"
        if not (_sha(path) == per_task[0][case_id] == per_task[1][case_id]):
            raise ValueError("Cached image differs from original P6 requests")
        paths[case_id] = path
    return paths


def run_visual(config_path: Path, pilot: Path, out: Path, original_run: Path,
               env_file: Path | None) -> dict:
    config = load_config(config_path)
    images = frozen_images(pilot, config, original_run)
    identity = {"config_sha256": _sha(config_path), "source_sha256": source_hashes(),
                "pilot_manifest_sha256": _sha(pilot / "manifest.json"),
                "pilot_summary_sha256": _sha(pilot / "summary.json"),
                "images_sha256": {key: _sha(path) for key, path in images.items()}}
    preflight = json.loads((out / "preflight/report.json").read_bytes())
    if (preflight["config_sha256"] != identity["config_sha256"] or
            preflight["source_sha256"] != identity["source_sha256"] or
            preflight["calls"] != 12 or preflight["valid_responses"] != 12):
        raise ValueError("Preflight must pass before Pilot calls")
    lock = out / ".runner.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        identity_path = out / "run-identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_bytes()) != identity:
                raise ValueError("Run identity changed; preserve existing version")
        else:
            write_json(identity_path, identity)
        if (out / "run.json").exists():
            return json.loads((out / "run.json").read_bytes())
        base, key = _credentials(env_file)

        def invoke(case_id, task, client):
            path = out / "calls" / task / f"{case_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                return json.loads(path.read_bytes()), False
            claim = path.with_suffix(".started")
            try:
                with claim.open("x", encoding="utf-8") as marker:
                    marker.write(datetime.now(timezone.utc).isoformat())
            except FileExistsError:
                # An interrupted request has unknown remote outcome: never submit it again.
                result = failed_result(case_id, task, METHOD)
                record = {"case_id": case_id, "task": task, "method": METHOD,
                          "status": "failed", "error_type": "InterruptedRequest",
                          "usage": None, "response": None,
                          "image_sha256": identity["images_sha256"][case_id]}
                submitted = False
            else:
                result, record = request_once(client, base, key, images[case_id].read_bytes(),
                                              task, case_id, config)
                submitted = True
            packet = {"result": result.model_dump(mode="json"), "raw": record}
            write_json(path, packet)
            return packet, submitted

        packets = {task: {} for task in TASKS}
        submitted = 0
        with httpx.Client(timeout=config["model"]["timeout_seconds"],
                          follow_redirects=False) as client:
            with ThreadPoolExecutor(max_workers=config["model"]["workers"]) as pool:
                jobs = {pool.submit(invoke, key, task, client): (key, task)
                        for key in images for task in TASKS}
                for count, future in enumerate(as_completed(jobs), 1):
                    case_id, task = jobs[future]
                    packet, new_call = future.result()
                    model = PointResult if task == "point" else RangeResult
                    result = model.model_validate(packet["result"])
                    if (result.case_id != case_id or result.method != METHOD or
                            packet["raw"]["image_sha256"] != identity["images_sha256"][case_id]):
                        raise ValueError("Stored call identity mismatch")
                    packets[task][case_id] = packet
                    submitted += new_call
                    if count % 50 == 0:
                        print(f"Semantic experiment {count}/600 completed", flush=True)
        run = {**identity, "method": METHOD, "settings": config["model"],
               "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
               "runtime_seconds_this_invocation": round(time.perf_counter() - started, 3),
               "new_requests": submitted, "cost": None,
               "python_version": platform.python_version(),
               "versions": {name: version(name) for name in ("httpx", "numpy", "matplotlib")},
               "prompt_sha256": {task: hashlib.sha256(config["prompts"][task].encode()).hexdigest()
                                 for task in TASKS}, "tasks": {}}
        for task, rows in packets.items():
            directory = out / task
            directory.mkdir(exist_ok=True)
            for filename, field in (("predictions.jsonl", "result"), ("raw.jsonl", "raw")):
                temporary = directory / (filename + ".tmp")
                with temporary.open("w", encoding="utf-8") as stream:
                    for key in sorted(rows):
                        stream.write(json.dumps(rows[key][field], ensure_ascii=False) + "\n")
                temporary.replace(directory / filename)
            run["tasks"][task] = {"cases": len(rows),
                "success": sum(row["result"]["status"] == "success" for row in rows.values()),
                "predictions_sha256": _sha(directory / "predictions.jsonl"),
                "usage": {name: sum((row["raw"].get("usage") or {}).get(name, 0)
                                    for row in rows.values())
                          for name in ("prompt_tokens", "completion_tokens", "total_tokens")}}
        _pilot_inputs(pilot, config)
        write_json(out / "run.json", run)
        return run
    finally:
        lock.unlink()


def evaluate(config_path: Path, pilot: Path, out: Path) -> dict:
    from gnss_sim.evaluation import evaluate_pilot, load_results_jsonl

    load_config(config_path)
    run = json.loads((out / "run.json").read_bytes())
    if run["config_sha256"] != _sha(config_path) or run["source_sha256"] != source_hashes():
        raise ValueError("Run source or config differs")
    reports = {}
    for task in TASKS:
        path = out / task / "predictions.jsonl"
        if _sha(path) != run["tasks"][task]["predictions_sha256"]:
            raise ValueError("Predictions changed after run")
        rows, errors = load_results_jsonl(path, task)
        if errors or len(rows) != 300:
            raise ValueError("Invalid or incomplete prediction artifact")
        reports[task] = evaluate_pilot(pilot, rows, METHOD, task)
        write_json(out / task / "report.json", reports[task])
    return reports
