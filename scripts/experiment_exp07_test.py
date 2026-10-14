"""Frozen NVR versus original SDR-VLM on fresh, paired test stations."""
from __future__ import annotations

import argparse
import copy
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from experiment_exp07 import COEXISTENCE_REVIEW, coexist_repair
from summarize_exp06 import pooled_bootstrap

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.metrics import aggregate, evaluate_record
from gnss_research.schema import observed_values
from gnss_research.vision import call_visual, credentials


def candidate_run(run: Path, out: Path, env: Path):
    original = read_json(run / "registration.json")
    if original["split"] != "test" or original["source_hashes"] != source_hashes():
        raise ValueError("frozen test baseline with unchanged source required")
    if not (run / "evaluation/summary.json").exists():
        raise ValueError("baseline must complete before candidate evaluation")
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source"
    shutil.copytree(run / "source", snapshot)
    scripts = [Path(__file__), Path(__file__).with_name("experiment_exp07.py"),
               Path(__file__).with_name("summarize_exp06.py")]
    for file in scripts:
        shutil.copyfile(file, snapshot / file.name)
    dataset = Path(original["dataset"])
    manifest = read_json(dataset / "manifest.json")
    records = [record for record in manifest["records"]
               if record["station_id"] in original["station_ids"]]
    settings = credentials(env)
    if settings[2] != original["model"]:
        raise ValueError("model changed from paired baseline")
    pending = []
    for record in records:
        station = record["station_id"]
        source = run / "predictions/pair_focus" / station
        work = out / "predictions/coexist_visual" / station
        payload_path = source / "request.json"
        if not payload_path.exists():
            shutil.copytree(source, work)
            continue
        shutil.copytree(source, work, ignore=shutil.ignore_patterns(
            "request.json", "input-receipt.json", "result.json", "response.json"))
        payload = copy.deepcopy(read_json(payload_path))
        payload["messages"][0]["content"] += COEXISTENCE_REVIEW
        write_json(work / "request.json", payload)
        write_json(work / "input-receipt.json", {
            "original_request_sha256": sha256(payload_path),
            "request_sha256": sha256(work / "request.json"),
            "original_candidates_sha256": sha256(source / "candidates.json"),
            "change": "same images and numerical candidates; nonexclusive prompt appended"})
        pending.append((record, work))
    frozen = {"experiment": read_json(Path("configs/experiment-exp07-test.json")),
              "baseline_registration_sha256": sha256(run / "registration.json"),
              "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
              "source_hashes": source_hashes(),
              "script_sha256": {file.name: sha256(file) for file in scripts},
              "method_protocol_sha256": sha256(Path("docs/experiment-exp07-test.md")),
              "model": settings[2], "requests": len(pending), "workers": 3,
              "station_ids": original["station_ids"],
              "request_sha256": {record["station_id"]: sha256(work / "request.json")
                                 for record, work in pending}}
    write_json(out / "registration.json", frozen)
    write_json(out / "execution-started.json", {"requests": len(pending), "retry_count": 0})

    def infer(job):
        record, work = job
        if sha256(work / "request.json") != frozen["request_sha256"][record["station_id"]]:
            raise ValueError("request changed")
        observation = read_json(dataset / record["observation"])
        result = call_visual(observation, "coexist_visual", work, settings)
        print(record["station_id"], "coexist_visual", result["status"], flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(infer, pending))
    rows = {method: [] for method in ("coexist_repair", "pair_focus_repair", "segment_numeric",
                                     "rf_split_temporal", "rf_balanced")}
    requests = failures = retained_steps = 0
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        truth = read_json(dataset / record["truth"])
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        work = out / "predictions/coexist_visual" / station
        visual = read_json(work / "result.json")
        result = coexist_repair(observation, visual, read_json(work / "candidates.json"))
        write_json(out / "predictions/coexist_repair" / station / "result.json", result)
        requests += visual.get("request_count", 0)
        failures += visual["status"] != "success"
        retained_steps += len(result.get("coexisting_step_indices", []))
        for method in rows:
            prediction = result if method == "coexist_repair" else read_json(
                run / "predictions" / method / station / "result.json")
            rows[method].append(evaluate_record(truth, prediction, observed,
                                               observation["background_end"]))
    summary = {method: aggregate(values) for method, values in rows.items()}
    write_json(out / "summary.json", {"methods": summary, "dataset_seed": manifest["seed"],
               "baseline_run": str(run.resolve()), "requests": requests, "failures": failures,
               "coexisting_steps_retained": retained_steps,
               "paired_bootstrap": pooled_bootstrap(rows["coexist_repair"], rows["pair_focus_repair"], manifest["seed"])})
    write_json(out / "per-station.json", rows)
    write_json(out / "execution-completed.json", {"requests": requests, "failures": failures})
    print({method: value["types_daily_macro_f1"] for method, value in summary.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    candidate_run(args.run, args.out, args.env_file)


if __name__ == "__main__":
    main()
