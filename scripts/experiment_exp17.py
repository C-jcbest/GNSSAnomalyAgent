"""Run all registered representation factors before selecting a new-data candidate."""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import exp16_overlap
import exp17_resolution as study
from experiment_exp07 import score
from experiment_exp08 import data_records
from summarize_exp06 import pooled_bootstrap

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.vision import SYSTEM, call_visual, credentials

ROOT = Path(__file__).resolve().parents[1]
FILES = [Path(__file__), Path(study.__file__), ROOT / "scripts/exp16_overlap.py",
         ROOT / "scripts/exp14_fingerprint.py", ROOT / "scripts/exp15_vector.py",
         ROOT / "scripts/exp10_visual_graph.py"]


def settings() -> tuple:
    base, key, _ = credentials(Path(".env"))
    return base, key, "qwen3-vl-plus"


def prepare(run: Path, visual: Path, control: Path | None, modes: list[str], protocol: Path):
    registered, dataset, manifest, records = data_records(run)
    visual.mkdir(parents=True, exist_ok=False)
    pending = []
    reused = []
    modes = list(dict.fromkeys([*modes, "bar_text", "local_text", "no_image", "original"]))
    if control is not None:
        modes.append("bar_low_saved")
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numeric = read_json(run / "predictions/segment_numeric" / station / "result.json")
        for mode in modes:
            work = visual / "predictions" / mode / station
            if control is not None and mode in ("original", "bar_text", "bar_low_saved"):
                saved_mode = {"original": "original", "bar_text": "pair_text",
                              "bar_low_saved": "pair_overlap"}[mode]
                saved = control / "predictions" / saved_mode / station
                shutil.copytree(saved, work)
                reused.append({"mode": mode, "station": station, "source": str(saved.resolve()),
                               "result_sha256": sha256(saved / "result.json")})
                continue
            if mode == "original":
                rf = read_json(run / "predictions/rf_split_temporal" / station / "result.json")
                request = prepare_pair_focus(observation, rf, work, settings()[2], SYSTEM)
            elif mode == "bar_text":
                request = exp16_overlap.prepare(observation, numeric, work, settings()[2], "pair_text")
            else:
                request = study.prepare(observation, numeric, work, settings()[2], mode)
            if request is not None:
                pending.append({"mode": mode, "station": station,
                                "request_sha256": sha256(work / "request.json")})
    source = visual / "frozen-source"
    source.mkdir()
    for path in FILES:
        shutil.copyfile(path, source / path.name)
    write_json(visual / "registration.json", {"experiment_id": "GNSS-VLM-EXP17",
        "baseline_run": str(run.resolve()), "baseline_registration_sha256": sha256(run / "registration.json"),
        "dataset": str(dataset.resolve()), "seed": manifest["seed"], "split": registered["split"],
        "station_ids": registered["station_ids"], "model": settings()[2], "modes": modes,
        "pending": pending, "reused": reused,
        "pilot_ids": sorted({job["station"] for job in pending})[:1],
        "source_hashes": source_hashes(), "experiment_files": {str(p): sha256(p) for p in FILES},
        "protocol_path": str(protocol), "protocol_sha256": sha256(protocol)})
    maximum = sum(2 if job["mode"] == "staged_readout" else 1 for job in pending)
    print({"jobs": len(pending), "maximum_requests": maximum,
           "triggered_ids": sorted({job["station"] for job in pending}),
           "pilot_ids": sorted({job["station"] for job in pending})[:1]}, flush=True)


def infer(visual: Path, phase: str):
    frozen = read_json(visual / "registration.json")
    if source_hashes() != frozen["source_hashes"]:
        raise ValueError("production sources changed")
    for path, digest in frozen["experiment_files"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("experiment source changed during frozen run")
    if sha256(Path(frozen["protocol_path"])) != frozen["protocol_sha256"]:
        raise ValueError("protocol changed during frozen run")
    dataset = Path(frozen["dataset"])
    records = {row["station_id"]: row for row in read_json(dataset / "manifest.json")["records"]}
    jobs = [job for job in frozen["pending"]
            if (job["station"] in frozen["pilot_ids"]) == (phase == "pilot")]
    start = visual / f"{phase}-started.json"
    if start.exists():
        raise ValueError("phase already started; no automatic retry")
    write_json(start, {"jobs": len(jobs), "retry_count": 0})

    def request(job):
        work = visual / "predictions" / job["mode"] / job["station"]
        if sha256(work / "request.json") != job["request_sha256"]:
            raise ValueError("request changed")
        observation = read_json(dataset / records[job["station"]]["observation"])
        if job["mode"] == "original":
            result = call_visual(observation, "pair_focus", work, settings())
        elif job["mode"] == "staged_readout":
            result = study.call_staged(observation, work, settings())
        else:
            result = study.call(observation, work, settings())
        print(job["station"], job["mode"], result["status"],
              result.get("pair_decision", {}).get("pair_ids"), flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(request, jobs))
    write_json(visual / f"{phase}-completed.json", {"jobs": len(jobs)})


def summarize(run: Path, visual: Path, output: Path):
    registered, dataset, manifest, records = data_records(run)
    frozen = read_json(visual / "registration.json")
    if sha256(run / "registration.json") != frozen["baseline_registration_sha256"]:
        raise ValueError("baseline changed")
    for phase in ("pilot", "remaining"):
        if not (visual / f"{phase}-completed.json").exists():
            raise ValueError("incomplete frozen run")
    output.mkdir(parents=True, exist_ok=False)
    methods = [*frozen["modes"], "vector_nearest", "segment_numeric", "rf_split_temporal", "rf_balanced"]
    predictions = {method: {} for method in methods}
    costs = {mode: {"new_requests": 0, "reused_requests": 0, "failures": 0}
             for mode in frozen["modes"]}
    reused = {(row["mode"], row["station"]) for row in frozen["reused"]}
    multi_jump_ids = []
    decisions = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numeric = read_json(run / "predictions/segment_numeric" / station / "result.json")
        for mode in frozen["modes"]:
            work = visual / "predictions" / mode / station
            result = read_json(work / "result.json")
            candidates = read_json(work / "candidates.json")
            if mode == "original":
                prediction = repair_return_pairs(observation, result, candidates)
            else:
                prediction = study.apply(observation, numeric, candidates["graph"], result)
            predictions[mode][station] = prediction
            cost_key = "reused_requests" if (mode, station) in reused else "new_requests"
            costs[mode][cost_key] += result.get("request_count", 0)
            costs[mode]["failures"] += result["status"] != "success"
            if result.get("request_count", 0):
                decisions.append({"station": station, "mode": mode,
                                  "status": result["status"], "decision": result.get("pair_decision")})
            for name in ("response.json", "result.json", "readout-result.json", "input-receipt.json",
                         "decision/response.json", "decision/result.json", "decision/request.json"):
                path = work / name
                if path.exists():
                    target = output / "responses" / mode / station / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
        candidates = read_json(visual / "predictions/bar_text" / station / "candidates.json")
        if len(candidates["graph"]["jumps"]) >= 3:
            multi_jump_ids.append(station)
        nearest = study.nearest(candidates["graph"], candidates["local_vectors"])
        predictions["vector_nearest"][station] = study.apply(observation, numeric, candidates["graph"], nearest)
        for method in ("segment_numeric", "rf_split_temporal", "rf_balanced"):
            predictions[method][station] = read_json(run / "predictions" / method / station / "result.json")
    summaries, rows = score(dataset, records, predictions)
    subset = {}
    if multi_jump_ids:
        subset_predictions = {method: {station: pred[station] for station in multi_jump_ids}
                              for method, pred in predictions.items()}
        subset_records = [record for record in records if record["station_id"] in multi_jump_ids]
        subset, _ = score(dataset, subset_records, subset_predictions)
    ranking = sorted((mode for mode in frozen["modes"] if mode in study.MODES
                      and mode not in ("local_text", "no_image")),
                     key=lambda mode: (-summaries[mode]["types_daily_macro_f1"], study.MODES.index(mode)))
    eligible = {}
    for scope, values in (("all", summaries), ("multi_jump", subset)):
        if not values:
            continue
        for mode in ranking:
            text = "local_text" if mode.startswith("local_") else "bar_text"
            main = values[mode]
            value = main["types_daily_macro_f1"]
            differences = {other: value - values[other]["types_daily_macro_f1"]
                           for other in ("original", "segment_numeric", "vector_nearest", text)}
            worst = min(main["types"][kind]["daily_micro"]["f1"]
                        - values["original"]["types"][kind]["daily_micro"]["f1"]
                        for kind in main["types"])
            representational = [differences["vector_nearest"], differences[text]]
            passes = (differences["original"] >= .01 and differences["segment_numeric"] >= .01
                      and max(representational) >= .005 and min(representational) >= -.005
                      and worst >= -.03 and (scope == "all" or len(multi_jump_ids) >= 6))
            eligible[f"{scope}/{mode}"] = {"passes": passes, "differences": differences,
                                           "worst_type_difference": worst}
    summary = {"experiment_id": "GNSS-VLM-EXP17", "seed": manifest["seed"],
        "split": registered["split"], "station_ids": registered["station_ids"],
        "methods": summaries, "multi_jump_ids": multi_jump_ids, "multi_jump_methods": subset,
        "costs": costs, "visual_ranking": ranking, "development_gates": eligible,
        "bootstrap_vs_numeric": {mode: pooled_bootstrap(rows[mode], rows["segment_numeric"], manifest["seed"])
                                 for mode in ranking},
        "baseline_run": str(run.resolve()), "visual_run": str(visual.resolve())}
    write_json(output / "summary.json", summary)
    write_json(output / "per-station.json", rows)
    write_json(output / "decisions.json", decisions)
    shutil.copyfile(visual / "registration.json", output / "registration.json")
    shutil.copytree(visual / "frozen-source", output / "frozen-source")
    lines = ["# EXP17：分辨率、坐标粒度与视觉读出流程（RGW-VLM）", "",
             f"种子{manifest['seed']}，{registered['split']}，{len(records)}站，qwen3-vl-plus。", "",
             "| 方法 | 四类日F1 | 四类事件F1≥.5 | 通用日F1 | 新请求 | 复用请求 | 失败 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, values in summaries.items():
        cost = costs.get(method, {"new_requests": 0, "reused_requests": 0, "failures": 0})
        lines.append(f"| {method} | {values['types_daily_macro_f1']:.4f} | "
                     f"{values['types_event_macro_f1']['iou_0.5']:.4f} | "
                     f"{values['generic_daily_micro']['f1']:.4f} | {cost['new_requests']} | "
                     f"{cost['reused_requests']} | {cost['failures']} |")
    lines.extend(["", f"多跳变候选子集{len(multi_jump_ids)}站：{multi_jump_ids}。",
                  "完整类别/子集指标、门槛与按站描述性bootstrap见summary.json。",
                  "所有模式为本项目适配，不是相关论文的作者默认方法复现。", ""])
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "remaining", "summarize"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--saved-control", type=Path)
    parser.add_argument("--modes", nargs="+", choices=study.MODES, default=list(study.MODES))
    parser.add_argument("--protocol", type=Path, default=Path("docs/experiment-exp17.md"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run, args.visual, args.saved_control, args.modes, args.protocol)
    elif args.stage == "summarize":
        if args.out is None:
            parser.error("--out required for summarize")
        summarize(args.run, args.visual, args.out)
    else:
        infer(args.visual, args.stage)


if __name__ == "__main__":
    main()
