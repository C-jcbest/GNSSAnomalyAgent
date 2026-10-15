"""Frozen, staged test inference and explicit whole-pipeline image ablations."""
from __future__ import annotations

import argparse
import base64
import copy
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from exp14_fingerprint import PairDecision, apply, call
from experiment_exp08 import data_records
from experiment_exp23 import PROMPT, numeric_decision, recovery_graph, render_local
from experiment_exp24 import gap_guard

from gnss_research.files import read_json, sha256, write_json
from gnss_research.schema import observed_values
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.upstream import call_visualtime, prepare_visualtime
from gnss_research.vision import SYSTEM, call_visual, credentials, prepare_request

PROTOCOL = Path("docs/experiment-exp25.md")
BASE_MODES = {"sdr_flash": "qwen3.8-flash", "sdr_flash_noimage": "qwen3.8-flash",
              "sdr_vl": "qwen3-vl-plus", "sdr_vl_noimage": "qwen3-vl-plus"}
RECOVERY_MODES = {"recovery_flash": "qwen3.8-flash", "recovery_flash_noimage": "qwen3.8-flash",
                  "recovery_vl": "qwen3-vl-plus", "recovery_vl_noimage": "qwen3-vl-plus"}
DEPENDENCIES = [Path(__file__), Path("scripts/experiment_exp23.py"), Path("scripts/experiment_exp24.py"),
                Path("scripts/exp14_fingerprint.py"), PROTOCOL]


def settings(model):
    base, key, _ = credentials(Path(".env"))
    return base, key, model


def remove_images(payload):
    result = copy.deepcopy(payload)
    for message in result["messages"]:
        if isinstance(message["content"], list):
            message["content"] = [item for item in message["content"] if item.get("type") != "image_url"]
    return result


def register_stage(output: Path, name: str, value: dict):
    value["experiment_id"] = "GNSS-VLM-EXP25"
    value["frozen_files"] = {str(path): sha256(path) for path in DEPENDENCIES}
    write_json(output / (name + "-registration.json"), value)
    frozen = output / "frozen-source"
    frozen.mkdir(exist_ok=True)
    for path in DEPENDENCIES:
        target = frozen / path.name
        if not target.exists():
            shutil.copyfile(path, target)


def check_frozen(registration):
    if any(sha256(Path(path)) != value for path, value in registration["frozen_files"].items()):
        raise ValueError("frozen experiment sources changed")


def workdir(output: Path, mode: str, station: str):
    return output / "predictions" / mode / station


def prepare(baseline: Path, output: Path, phase: str):
    config, dataset, manifest, records = data_records(baseline)
    if config["split"] != "test" or not (baseline / "execution-complete.json").exists():
        raise ValueError("completed held-out test baselines required")
    if phase == "pilot":
        output.mkdir(parents=True, exist_ok=False)
    else:
        if not (output / "pilot-derived.json").exists():
            raise ValueError("complete pilot before remaining stations")
        check_frozen(read_json(output / "pilot-base-registration.json"))
    records = sorted(records, key=lambda record: record["station_id"])
    records = records[:6] if phase == "pilot" else records[6:]
    jobs = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        rf = read_json(baseline / "predictions/rf_split_temporal" / station / "result.json")
        focus = workdir(output, "sdr_flash", station)
        payload = prepare_pair_focus(observation, rf, focus, "qwen3.8-flash", SYSTEM)
        for mode, model in BASE_MODES.items():
            work = workdir(output, mode, station)
            if mode != "sdr_flash":
                work.mkdir(parents=True)
                shutil.copyfile(focus / "candidates.json", work / "candidates.json")
            if payload is None:
                if mode != "sdr_flash":
                    shutil.copyfile(focus / "result.json", work / "result.json")
                continue
            request = remove_images(payload) if mode.endswith("noimage") else copy.deepcopy(payload)
            request["model"] = model
            if mode != "sdr_flash":
                write_json(work / "request.json", request)
            jobs.append({"station": station, "mode": mode, "model": model, "kind": "visual",
                         "request_sha256": sha256(work / "request.json")})
        raw = workdir(output, "sdr_raw", station)
        request = prepare_pair_focus(observation, rf, raw, "qwen3.8-flash", SYSTEM, "historical_residual")
        if request is not None:
            jobs.append({"station": station, "mode": "sdr_raw", "model": "qwen3.8-flash", "kind": "visual",
                         "request_sha256": sha256(raw / "request.json")})
        visual = workdir(output, "visual_only", station)
        prepare_request(observation, "visual_only", visual, "qwen3.8-flash")
        jobs.append({"station": station, "mode": "visual_only", "model": "qwen3.8-flash", "kind": "visual",
                     "request_sha256": sha256(visual / "request.json")})
        author = workdir(output, "visualtime_gnss", station)
        prepare_visualtime(observation, author, "qwen3.8-flash")
        jobs.append({"station": station, "mode": "visualtime_gnss", "model": "qwen3.8-flash", "kind": "upstream",
                     "requests": {str(p.relative_to(author)): sha256(p) for p in author.glob("*/request.json")}})
        print("prepared", phase, station, flush=True)
    register_stage(output, phase + "-base", {"baseline": str(baseline), "dataset": str(dataset),
        "seed": manifest["seed"], "records": records, "phase": phase, "jobs": jobs,
        "baseline_registration_sha256": sha256(baseline / "registration.json")})
    print("base requests", sum(6 if job["kind"] == "upstream" else 1 for job in jobs), flush=True)


def infer(output: Path, phase: str, stage: str):
    registration = read_json(output / f"{phase}-{stage}-registration.json")
    check_frozen(registration)
    sentinel = output / f"{phase}-{stage}-started.json"
    if sentinel.exists():
        raise ValueError("stage already started; no implicit retries")
    jobs = registration["jobs"]
    write_json(sentinel, {"jobs": len(jobs), "retry_count": 0})
    records = {record["station_id"]: record for record in registration["records"]}
    dataset = Path(registration["dataset"])

    def process(job):
        work = workdir(output, job["mode"], job["station"])
        observation = read_json(dataset / records[job["station"]]["observation"])
        if job["kind"] == "upstream":
            if any(sha256(work / path) != value for path, value in job["requests"].items()):
                raise ValueError("author request changed")
            result = call_visualtime(observation, work, settings(job["model"]))
        else:
            if sha256(work / "request.json") != job["request_sha256"]:
                raise ValueError("request changed")
            if job["kind"] == "recovery":
                result = call(observation, work, settings(job["model"]))
            else:
                result = call_visual(observation, job["mode"], work, settings(job["model"]))
        print(phase, job["station"], job["mode"], result["status"], flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(process, jobs))
    write_json(output / f"{phase}-{stage}-completed.json", {"jobs": len(jobs)})


def base_result(output: Path, mode: str, station: str, observation: dict):
    work = workdir(output, mode, station)
    return repair_return_pairs(observation, read_json(work / "result.json"), read_json(work / "candidates.json"))


def prepare_recovery_request(observation, base, graph, output, station, mode, model):
    work = workdir(output, mode, station)
    work.mkdir(parents=True)
    write_json(work / "candidates.json", {"graph": graph, "numerical_prediction": base})
    missing = np.flatnonzero(~np.isfinite(observed_values(observation)).all(axis=1)).tolist()
    content = [{"type": "text", "text": json.dumps({"graph": graph, "missing_indices": missing})}]
    if not mode.endswith("noimage"):
        image = work / "local.png"
        render_local(observation, graph, image)
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                        + base64.b64encode(image.read_bytes()).decode("ascii")}})
    request = {"model": model, "temperature": 0, "enable_thinking": False,
        "response_format": {"type": "json_schema", "json_schema": {"name": "gnss_recovery",
            "strict": True, "schema": PairDecision.model_json_schema()}},
        "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}]}
    write_json(work / "request.json", request)
    return {"station": station, "mode": mode, "model": model, "kind": "recovery",
            "request_sha256": sha256(work / "request.json")}


def prepare_recovery(output: Path, phase: str):
    if not (output / f"{phase}-base-completed.json").exists():
        raise ValueError("base calls not complete")
    registered = read_json(output / f"{phase}-base-registration.json")
    check_frozen(registered)
    dataset, baseline = Path(registered["dataset"]), Path(registered["baseline"])
    jobs, graphs = [], {}
    for record in registered["records"]:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numeric = read_json(baseline / "predictions/segment_numeric" / station / "result.json")
        base = base_result(output, "sdr_flash", station, observation)
        noimage = base_result(output, "sdr_flash_noimage", station, observation)
        graph = recovery_graph(observation, numeric, base)
        noimage_graph = recovery_graph(observation, numeric, noimage)
        graphs[station] = {"image_base": graph, "noimage_base": noimage_graph,
                           "shared_noimage_response": graph == noimage_graph}
        if graph["pairs"]:
            for mode, model in RECOVERY_MODES.items():
                jobs.append(prepare_recovery_request(observation, base, graph, output, station, mode, model))
        if noimage_graph["pairs"] and noimage_graph != graph:
            jobs.append(prepare_recovery_request(observation, noimage, noimage_graph, output, station,
                                                 "recovery_pipeline_noimage", "qwen3.8-flash"))
        print("recovery candidates", station, len(graph["pairs"]), len(noimage_graph["pairs"]), flush=True)
    register_stage(output, phase + "-recovery", {**{key: registered[key] for key in ("baseline", "dataset", "seed", "records", "phase")},
        "jobs": jobs, "graphs": graphs})
    print("recovery requests", len(jobs), flush=True)


def automatic_response(observation, numeric, graph, select_all=False):
    ids = [pair["id"] for pair in graph["pairs"]] if select_all else numeric_decision(observation, numeric, graph)
    used, compatible = set(), []
    for pair in graph["pairs"]:
        edges = {pair["start"], pair["end"] + 1}
        if pair["id"] in ids and not edges & used:
            compatible.append(pair["id"])
            used.update(edges)
    return {"status": "success", "request_count": 0, "pair_decision": {
        "pair_ids": compatible, "evidence": "Same-candidate automatic control"}}


def recover(observation, base, graph, response, guarded):
    if guarded:
        response = gap_guard(observation, graph, response)
    result = apply(observation, base, graph, response)
    result["gap_guard_rejected_ids"] = response.get("gap_guard_rejected_ids", [])
    return result


def derive(output: Path, phase: str):
    if not (output / f"{phase}-recovery-completed.json").exists():
        raise ValueError("recovery calls not complete")
    registered = read_json(output / f"{phase}-recovery-registration.json")
    check_frozen(registered)
    dataset, baseline = Path(registered["dataset"]), Path(registered["baseline"])
    for record in registered["records"]:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numeric = read_json(baseline / "predictions/segment_numeric" / station / "result.json")
        bases = {mode: base_result(output, mode, station, observation) for mode in (*BASE_MODES, "sdr_raw")}
        dependencies = {mode: [str(workdir(output, mode, station))] for mode in bases}
        results = dict(bases)
        for mode in ("visual_only", "visualtime_gnss"):
            results[mode] = read_json(workdir(output, mode, station) / "result.json")
            dependencies[mode] = [str(workdir(output, mode, station))]
        graph_info = registered["graphs"][station]
        graph = graph_info["image_base"]
        base = bases["sdr_flash"]
        mapping = {"sdr_gr_flash": ("recovery_flash", True), "sdr_gr_vl": ("recovery_vl", True),
                   "sdr_cr_flash": ("recovery_flash", False), "sdr_cr_vl": ("recovery_vl", False),
                   "sdr_gr_recovery_noimage": ("recovery_flash_noimage", True),
                   "sdr_gr_vl_recovery_noimage": ("recovery_vl_noimage", True),
                   "sdr_cr_recovery_noimage": ("recovery_flash_noimage", False),
                   "sdr_cr_vl_recovery_noimage": ("recovery_vl_noimage", False)}
        for name, (mode, guarded) in mapping.items():
            results[name] = base
            dependencies[name] = list(dependencies["sdr_flash"])
            if graph["pairs"]:
                work = workdir(output, mode, station)
                results[name] = recover(observation, base, graph, read_json(work / "result.json"), guarded)
                dependencies[name].append(str(work))
        for name, select_all in (("sdr_gr_numeric_recovery", False), ("sdr_gr_candidate_all", True)):
            response = automatic_response(observation, numeric, graph, select_all)
            results[name] = recover(observation, base, graph, response, True)
            dependencies[name] = list(dependencies["sdr_flash"])
        noimage_graph = graph_info["noimage_base"]
        noimage_base = bases["sdr_flash_noimage"]
        results["sdr_gr_pipeline_noimage"] = noimage_base
        dependencies["sdr_gr_pipeline_noimage"] = list(dependencies["sdr_flash_noimage"])
        if noimage_graph["pairs"]:
            mode = "recovery_flash_noimage" if graph_info["shared_noimage_response"] else "recovery_pipeline_noimage"
            work = workdir(output, mode, station)
            results["sdr_gr_pipeline_noimage"] = recover(observation, noimage_base, noimage_graph,
                                                         read_json(work / "result.json"), True)
            dependencies["sdr_gr_pipeline_noimage"].append(str(work))
        numeric_graph = recovery_graph(observation, numeric, numeric)
        results["numeric_gr"] = recover(observation, numeric, numeric_graph,
                                         automatic_response(observation, numeric, numeric_graph), True)
        dependencies["numeric_gr"] = []
        for name, result in results.items():
            final = {**result, "evaluation_dependency_workdirs": dependencies[name]}
            write_json(output / "final" / name / station / "result.json", final)
    write_json(output / f"{phase}-derived.json", {"stations": len(registered["records"]),
        "note": "Frozen configuration; no performance-based method selection"})
    print("derived", phase, len(registered["records"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "infer", "prepare-recovery", "infer-recovery", "derive"))
    parser.add_argument("--baseline", type=Path, default=Path("runs/exp25-test-1516-baseline-v1"))
    parser.add_argument("--visual", type=Path, default=Path("runs/exp25-test-1516-visual-v1"))
    parser.add_argument("--phase", choices=("pilot", "remaining"), default="pilot")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.baseline, args.visual, args.phase)
    elif args.stage == "prepare-recovery":
        prepare_recovery(args.visual, args.phase)
    elif args.stage == "derive":
        derive(args.visual, args.phase)
    else:
        infer(args.visual, args.phase, "recovery" if args.stage == "infer-recovery" else "base")


if __name__ == "__main__":
    main()
