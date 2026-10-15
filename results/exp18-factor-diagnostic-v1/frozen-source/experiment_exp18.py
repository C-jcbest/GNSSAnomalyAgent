"""Registered contrastive-distribution study with unchanged numerical controls."""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import exp18_distribution as study
from experiment_exp07 import score
from experiment_exp08 import data_records
from experiment_exp17 import settings
from summarize_exp06 import pooled_bootstrap

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.vision import SYSTEM, call_visual

FILES = [Path(__file__), Path(study.__file__)] + [Path("scripts") / name for name in (
    "experiment_exp17.py", "exp17_resolution.py", "exp16_overlap.py", "exp14_fingerprint.py",
    "exp15_vector.py", "exp10_visual_graph.py", "experiment_exp07.py", "experiment_exp08.py",
    "summarize_exp06.py")]


def prepare(run: Path, visual: Path, saved_control: Path | None):
    registered, dataset, manifest, records = data_records(run)
    reference = study.training_reference(dataset, manifest["records"])
    visual.mkdir(parents=True, exist_ok=False)
    write_json(visual / "training-reference.json", reference)
    pending = []
    reused = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numeric = read_json(run / "predictions/segment_numeric" / station / "result.json")
        for mode in (*study.MODES, "original"):
            work = visual / "predictions" / mode / station
            if mode == "original" and saved_control is not None:
                saved = saved_control / "predictions/original" / station
                shutil.copytree(saved, work)
                reused.append({"station": station, "source": str(saved.resolve()),
                               "result_sha256": sha256(saved / "result.json")})
                continue
            if mode == "original":
                rf = read_json(run / "predictions/rf_split_temporal" / station / "result.json")
                request = prepare_pair_focus(observation, rf, work, settings()[2], SYSTEM)
            else:
                request = study.prepare(observation, numeric, reference, work, mode, settings()[2])
            if request is not None:
                pending.append({"station": station, "mode": mode,
                                "request_sha256": sha256(work / "request.json")})
    source = visual / "frozen-source"
    source.mkdir()
    for path in FILES:
        shutil.copyfile(path, source / path.name)
    protocol = Path("docs/experiment-exp18.md")
    write_json(visual / "registration.json", {"experiment_id": "GNSS-VLM-EXP18",
        "baseline_run": str(run.resolve()), "baseline_registration_sha256": sha256(run / "registration.json"),
        "dataset": str(dataset.resolve()), "seed": manifest["seed"], "split": registered["split"],
        "station_ids": registered["station_ids"], "model": settings()[2], "pending": pending,
        "pilot_ids": sorted({job["station"] for job in pending})[:1], "reused": reused,
        "training_reference_sha256": sha256(visual / "training-reference.json"),
        "source_hashes": source_hashes(), "experiment_files": {str(p): sha256(p) for p in FILES},
        "protocol_path": str(protocol), "protocol_sha256": sha256(protocol)})
    print({"requests": len(pending), "reference_station": reference["station_id"],
           "triggered_ids": sorted({job["station"] for job in pending}),
           "pilot_ids": sorted({job["station"] for job in pending})[:1]}, flush=True)


def infer(visual: Path, phase: str):
    frozen = read_json(visual / "registration.json")
    if source_hashes() != frozen["source_hashes"]:
        raise ValueError("production source changed")
    for path, digest in frozen["experiment_files"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("experiment changed during frozen run")
    if sha256(Path(frozen["protocol_path"])) != frozen["protocol_sha256"]:
        raise ValueError("protocol changed")
    if sha256(visual / "training-reference.json") != frozen["training_reference_sha256"]:
        raise ValueError("reference changed")
    dataset = Path(frozen["dataset"])
    records = {row["station_id"]: row for row in read_json(dataset / "manifest.json")["records"]}
    jobs = [job for job in frozen["pending"]
            if (job["station"] in frozen["pilot_ids"]) == (phase == "pilot")]
    marker = visual / f"{phase}-started.json"
    if marker.exists():
        raise ValueError("phase already started; no retry")
    write_json(marker, {"requests": len(jobs), "retry_count": 0})

    def request(job):
        work = visual / "predictions" / job["mode"] / job["station"]
        if sha256(work / "request.json") != job["request_sha256"]:
            raise ValueError("request changed")
        observation = read_json(dataset / records[job["station"]]["observation"])
        if job["mode"] == "original":
            result = call_visual(observation, "pair_focus", work, settings())
        else:
            result = study.call(observation, work, settings())
        print(job["station"], job["mode"], result["status"],
              result.get("pair_decision", {}).get("pair_ids"), flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(request, jobs))
    write_json(visual / f"{phase}-completed.json", {"requests": len(jobs)})


def gates(values: dict, count: int) -> dict:
    result = {}
    for mode in ("distribution", "reference_distribution"):
        main = values[mode]
        score_value = main["types_daily_macro_f1"]
        differences = {other: score_value - values[other]["types_daily_macro_f1"]
                       for other in ("original", "segment_numeric", "vector_nearest", "distribution_text")}
        worst = min(main["types"][kind]["daily_micro"]["f1"]
                    - values["original"]["types"][kind]["daily_micro"]["f1"]
                    for kind in main["types"])
        controls = [differences["vector_nearest"], differences["distribution_text"]]
        result[mode] = {"differences": differences, "worst_type_difference": worst,
            "passes": count >= 6 and differences["original"] >= .01
            and differences["segment_numeric"] >= .01 and min(controls) >= -.005
            and max(controls) >= .005 and worst >= -.03}
    return result


def summarize(run: Path, visual: Path, output: Path):
    registered, dataset, manifest, records = data_records(run)
    frozen = read_json(visual / "registration.json")
    if sha256(run / "registration.json") != frozen["baseline_registration_sha256"]:
        raise ValueError("baseline changed")
    for phase in ("pilot", "remaining"):
        if not (visual / f"{phase}-completed.json").exists():
            raise ValueError("incomplete registered experiment")
    output.mkdir(parents=True, exist_ok=False)
    methods = (*study.MODES, "original", "vector_nearest", "segment_numeric", "rf_split_temporal", "rf_balanced")
    predictions = {method: {} for method in methods}
    costs = {mode: {"new_requests": 0, "reused_requests": 0, "failures": 0}
             for mode in (*study.MODES, "original")}
    reused = {row["station"] for row in frozen["reused"]}
    multi_ids = []
    decisions = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numeric = read_json(run / "predictions/segment_numeric" / station / "result.json")
        for mode in (*study.MODES, "original"):
            work = visual / "predictions" / mode / station
            result = read_json(work / "result.json")
            candidates = read_json(work / "candidates.json")
            if mode == "original":
                prediction = repair_return_pairs(observation, result, candidates)
            else:
                prediction = study.apply(observation, numeric, candidates["graph"], result)
            predictions[mode][station] = prediction
            key = "reused_requests" if mode == "original" and station in reused else "new_requests"
            costs[mode][key] += result.get("request_count", 0)
            costs[mode]["failures"] += result["status"] != "success"
            if result.get("request_count", 0):
                decisions.append({"station": station, "mode": mode, "status": result["status"],
                                  "decision": result.get("pair_decision")})
            for name in ("response.json", "result.json", "input-receipt.json"):
                path = work / name
                if path.exists():
                    target = output / "responses" / mode / station / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
        candidates = read_json(visual / "predictions/distribution" / station / "candidates.json")
        if len(candidates["graph"]["jumps"]) >= 3:
            multi_ids.append(station)
        decision = study.nearest(candidates["graph"], candidates["local_vectors"])
        predictions["vector_nearest"][station] = study.apply(observation, numeric, candidates["graph"], decision)
        for method in ("segment_numeric", "rf_split_temporal", "rf_balanced"):
            predictions[method][station] = read_json(run / "predictions" / method / station / "result.json")
    summaries, rows = score(dataset, records, predictions)
    subset = {}
    if multi_ids:
        subset_records = [record for record in records if record["station_id"] in multi_ids]
        subset_predictions = {method: {station: results[station] for station in multi_ids}
                              for method, results in predictions.items()}
        subset, _ = score(dataset, subset_records, subset_predictions)
    summary = {"experiment_id": "GNSS-VLM-EXP18", "seed": manifest["seed"], "split": registered["split"],
        "station_ids": registered["station_ids"], "methods": summaries,
        "multi_jump_ids": multi_ids, "multi_jump_methods": subset, "costs": costs,
        "development_gates": {"all": gates(summaries, len(records)),
                              "multi_jump": gates(subset, len(multi_ids)) if subset else {}},
        "visual_increment_bootstrap": pooled_bootstrap(rows["reference_distribution"],
                                                       rows["segment_numeric"], manifest["seed"]),
        "baseline_run": str(run.resolve()), "visual_run": str(visual.resolve())}
    write_json(output / "summary.json", summary)
    write_json(output / "per-station.json", rows)
    write_json(output / "decisions.json", decisions)
    shutil.copytree(visual / "frozen-source", output / "frozen-source")
    for name in ("registration.json", "training-reference.json"):
        shutil.copyfile(visual / name, output / name)
    lines = ["# EXP18：局部分布与对照示例视觉核验（DCV-VLM）", "",
             f"种子{manifest['seed']}，{registered['split']}，{len(records)}站，qwen3-vl-plus。", "",
             "| 方法 | 四类日F1 | 四类事件F1≥.5 | 通用日F1 | 新请求 | 复用请求 | 失败 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, values in summaries.items():
        cost = costs.get(method, {"new_requests": 0, "reused_requests": 0, "failures": 0})
        lines.append(f"| {method} | {values['types_daily_macro_f1']:.4f} | "
                     f"{values['types_event_macro_f1']['iou_0.5']:.4f} | "
                     f"{values['generic_daily_micro']['f1']:.4f} | {cost['new_requests']} | "
                     f"{cost['reused_requests']} | {cost['failures']} |")
    lines.extend(["", f"预定义多跳变子集{len(multi_ids)}站：{multi_ids}。",
                  "完整类别指标/子集/门槛见summary.json；训练参考来源见training-reference.json。", ""])
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "remaining", "summarize"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--saved-control", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run, args.visual, args.saved_control)
    elif args.stage == "summarize":
        if args.out is None:
            parser.error("--out required")
        summarize(args.run, args.visual, args.out)
    else:
        infer(args.visual, args.stage)


if __name__ == "__main__":
    main()
