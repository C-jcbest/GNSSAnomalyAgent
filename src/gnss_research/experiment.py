"""Freeze inputs before inference and report failures without counting them as success."""
from __future__ import annotations

import csv
import platform
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np

from gnss_research.baselines import NUMERICAL_METHODS, fit_baselines, predict_numerical
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.metrics import aggregate, evaluate_record, paired_bootstrap
from gnss_research.plots import export_labels, render_observation
from gnss_research.schema import Prediction, observed_values
from gnss_research.upstream import (
    UPSTREAM_METHODS,
    call_visualtime,
    prepare_visualtime,
    setup_sources,
)
from gnss_research.vision import VISUAL_METHODS, call_visual, credentials, prepare_request


def verify_dataset(dataset: Path) -> dict:
    manifest = read_json(dataset / "manifest.json")
    station_ids = [record["station_id"] for record in manifest["records"]]
    if len(set(station_ids)) != len(station_ids):
        raise ValueError("station overlap across splits")
    for record in manifest["records"]:
        for kind in ("observation", "truth"):
            path = (dataset / record[kind]).resolve()
            if not path.is_relative_to(dataset.resolve()):
                raise ValueError("dataset path escapes root")
            if sha256(path) != record[kind + "_sha256"]:
                raise ValueError(f"dataset changed: {record['station_id']} {kind}")
    return manifest


def register(dataset: Path, out: Path, methods: list[str], split: str, seed: int,
             model: str | None, max_requests: int, workers: int) -> dict:
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("nonempty distinct methods required")
    if set(methods) - set(NUMERICAL_METHODS + VISUAL_METHODS + UPSTREAM_METHODS):
        raise ValueError("unknown method")
    if split not in ("validation", "test") or not 1 <= workers <= 4 or seed < 0:
        raise ValueError("invalid split, seed or concurrency")
    manifest = verify_dataset(dataset)
    records = [record for record in manifest["records"] if record["split"] == split]
    expected = len(records) * sum(6 if method in UPSTREAM_METHODS else
                                  int(method in VISUAL_METHODS) for method in methods)
    if expected and (not model or max_requests < expected):
        raise ValueError(f"model and budget >= {expected} required; all selected stations included")
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source"
    snapshot.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(source, snapshot / source.name)
    project_root = Path(__file__).resolve().parents[2]
    shutil.copyfile(project_root / "docs/experiment-protocol.md", snapshot / "protocol.md")
    shutil.copyfile(project_root / "configs/upstream.json", snapshot / "upstream.json")
    frozen = {"schema": "gnss-research-run-v1", "dataset": str(dataset.resolve()),
              "manifest_sha256": sha256(dataset / "manifest.json"), "split": split,
              "station_ids": [record["station_id"] for record in records],
              "methods": methods, "seed": seed, "model": model, "max_requests": max_requests,
              "expected_requests": expected, "workers": workers, "source_hashes": source_hashes(),
              "protocol_sha256": sha256(Path(__file__).resolve().parents[2]
                                        / "docs/experiment-protocol.md"),
              "upstream": setup_sources() if set(methods) & set(UPSTREAM_METHODS) else {},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": {name: version(name) for name in
                                           ("numpy", "scipy", "scikit-learn", "matplotlib",
                                            "pydantic", "httpx")}},
              "rules": {"threshold_tuning": "none; numerical library default decisions",
                        "primary": "deformation daily F1 and event F1 at IoU 0.5",
                        "generic_comparison": "union of morphological anomaly masks",
                        "point_adjustment": False, "missing": "excluded and coverage reported",
                        "failures": "zero detections for unconditional score; separately counted",
                        "online": False}}
    write_json(out / "registration.json", frozen)
    return {"run": str(out.resolve()), "expected_requests": expected, "stations": len(records)}


def execute(run: Path, env_file: Path | None = None) -> dict:
    config = read_json(run / "registration.json")
    if config["source_hashes"] != source_hashes():
        raise ValueError("source changed after registration; register a new run")
    if config["upstream"] and config["upstream"] != setup_sources():
        raise ValueError("upstream source differs from registration")
    if (run / "execution-started.json").exists():
        raise ValueError("run already started; no implicit replays or paid retries")
    dataset = Path(config["dataset"])
    if config["manifest_sha256"] != sha256(dataset / "manifest.json"):
        raise ValueError("dataset manifest changed")
    manifest = verify_dataset(dataset)
    settings = None
    if config["expected_requests"]:
        settings = credentials(env_file)
        if settings[2] != config["model"]:
            raise ValueError("model differs from registration")
    write_json(run / "execution-started.json", {"started": time.time()})
    models, training_receipt = fit_baselines(dataset, manifest["records"], config["seed"])
    write_json(run / "training.json", training_receipt)
    records = [record for record in manifest["records"] if record["split"] == config["split"]]
    pending = []
    completed = []
    for record in records:
        observation = read_json(dataset / record["observation"])
        for method in config["methods"]:
            work = run / "predictions" / method / record["station_id"]
            if method in VISUAL_METHODS:
                prepare_request(observation, method, work, settings[2])
                pending.append((record, method, observation, work))
            elif method in UPSTREAM_METHODS:
                prepare_visualtime(observation, work, settings[2])
                pending.append((record, method, observation, work))
            else:
                work.mkdir(parents=True, exist_ok=False)
                start = time.perf_counter()
                result = predict_numerical(models[method], method, observation)
                result.update({"status": "success", "elapsed_seconds": time.perf_counter() - start,
                               "request_count": 0, "method": method})
                write_json(work / "result.json", result)
                completed.append((record, method, observation, work, result))

    def infer(job):
        record, method, observation, work = job
        if method in UPSTREAM_METHODS:
            result = call_visualtime(observation, work, settings)
        else:
            result = call_visual(observation, method, work, settings)
        print(f"{record['station_id']} {method}: {result['status']}", flush=True)
        return record, method, observation, work, result

    # Prepare all Matplotlib artifacts before concurrent network requests. Figure rendering
    # remains serial, while independent station requests share a bounded provider budget.
    with ThreadPoolExecutor(max_workers=config["workers"]) as executor:
        completed.extend(executor.map(infer, pending))
    for record, method, observation, work, result in completed:
        if result.get("prediction") is not None:
            prediction = Prediction.model_validate(result["prediction"])
            export_labels(observation, prediction, work / "labels.csv")
            render_observation(observation, work / "prediction.png", prediction=prediction)
    write_json(run / "execution-complete.json", {"completed": time.time(), "jobs": len(completed)})
    return evaluate(run)


def evaluate(run: Path, out: Path | None = None, dataset_override: Path | None = None) -> dict:
    config = read_json(run / "registration.json")
    dataset = dataset_override or Path(config["dataset"])
    manifest = verify_dataset(dataset)
    if config["manifest_sha256"] != sha256(dataset / "manifest.json"):
        raise ValueError("dataset manifest changed")
    records = [record for record in manifest["records"] if record["split"] == config["split"]]
    reports = {method: [] for method in config["methods"]}
    strata = {method: {} for method in config["methods"]}
    failures = {method: 0 for method in config["methods"]}
    missing_results = {method: 0 for method in config["methods"]}
    csv_rows = []
    for record in records:
        observation = read_json(dataset / record["observation"])
        truth = read_json(dataset / record["truth"])
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        for method in config["methods"]:
            path = run / "predictions" / method / record["station_id"] / "result.json"
            result = read_json(path) if path.exists() else {"status": "not_run"}
            if result["status"] != "success":
                failures[method] += result["status"] != "not_run"
                missing_results[method] += result["status"] == "not_run"
                typed_method = method in VISUAL_METHODS or (
                    method in NUMERICAL_METHODS and method != "isolation_forest")
                result = {"prediction": {"events": []} if typed_method else None,
                          "generic_mask": [False] * len(observed), "generic_score": None}
            report = evaluate_record(truth, result, observed, observation["background_end"])
            reports[method].append(report)
            strata[method].setdefault(truth["scenario"], []).append(report)
            csv_rows.append({"station_id": record["station_id"], "method": method,
                             "scenario": truth["scenario"], "result_status":
                             read_json(path)["status"] if path.exists() else "not_run",
                             **report["generic_daily"],
                             "false_alarm_runs": report["false_alarm_runs"]})
    summary = {method: {**aggregate(rows), "failed_records": failures[method],
                        "not_run_records": missing_results[method]}
               for method, rows in reports.items()}
    pairs = {}
    if "fusion" in reports:
        for method in reports:
            if method != "fusion":
                pairs[f"fusion_minus_{method}"] = paired_bootstrap(
                    reports["fusion"], reports[method], config["seed"])
                if method not in ("isolation_forest", "visualtime_gnss"):
                    pairs[f"fusion_minus_{method}_deformation"] = paired_bootstrap(
                        reports["fusion"], reports[method], config["seed"], endpoint="deformation")
    output = out or run / "evaluation"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "per-station.json", reports)
    write_json(output / "summary.json", {"methods": summary, "paired_comparisons": pairs,
               "scenario_strata": {method: {name: aggregate(rows) for name, rows in groups.items()}
                                   for method, groups in strata.items()},
               "evaluation_source_hashes": source_hashes(),
               "dataset_used": str(dataset.resolve()),
               "endpoints": {"internal_phases": "not evaluated by this activity-only protocol",
                             "generic_only_methods": ["isolation_forest", *UPSTREAM_METHODS]},
               "interpretation": "descriptive; synthetic station sample, no field validity claim"})
    fields = sorted({key for row in csv_rows for key in row})
    with (output / "per-station.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    return {"report": str((output / "summary.json").resolve()), "methods": summary}
