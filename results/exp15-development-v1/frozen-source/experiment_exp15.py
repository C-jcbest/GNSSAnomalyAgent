"""Isolated aligned-local-image study, then optional independently registered test."""
from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import exp15_vector as fingerprint
from experiment_exp07 import score
from experiment_exp08 import data_records
from summarize_exp06 import pooled_bootstrap

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.vision import SYSTEM, call_visual, credentials

MODES = ("vector_visual", "vector_text", "vector_no_evidence")


def prepare(run: Path, visual: Path, control: Path | None):
    registered, dataset, manifest, records = data_records(run)
    settings = credentials(Path(".env"))
    visual.mkdir(parents=True, exist_ok=False)
    pending = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numeric = read_json(run / "predictions/segment_numeric" / station / "result.json")
        for mode in (*MODES, "original"):
            work = visual / "predictions" / mode / station
            if mode == "original" and control is not None:
                shutil.copytree(control / "predictions/original" / station, work)
                continue
            if mode == "original":
                rf = read_json(run / "predictions/rf_split_temporal" / station / "result.json")
                payload = prepare_pair_focus(observation, rf, work, settings[2], SYSTEM)
            else:
                payload = fingerprint.prepare(observation, numeric, work, settings[2], mode)
            if payload is not None:
                pending.append({"mode": mode, "station": station, "request_sha256": sha256(work / "request.json")})
    scripts = [Path(__file__), Path(fingerprint.__file__), Path("scripts/exp14_fingerprint.py"), Path("scripts/exp10_visual_graph.py")]
    for path in scripts:
        shutil.copyfile(path, visual / path.name)
    protocol = Path("docs/experiment-exp15.md")
    write_json(visual / "registration.json", {"experiment_id": "GNSS-VLM-EXP15",
        "baseline_run": str(run.resolve()), "baseline_registration_sha256": sha256(run / "registration.json"),
        "dataset": str(dataset.resolve()), "seed": manifest["seed"], "split": registered["split"],
        "station_ids": registered["station_ids"], "model": settings[2], "pending": pending,
        "pilot_ids": sorted({p["station"] for p in pending})[:2], "source_hashes": source_hashes(),
        "experiment_files": {str(p): sha256(p) for p in scripts},
        "protocol_path": str(protocol), "protocol_sha256": sha256(protocol),
        "saved_control": str(control.resolve()) if control is not None else None})
    print({"requests": len(pending), "pilot_ids": sorted({p["station"] for p in pending})[:2]}, flush=True)


def infer(visual: Path, phase: str):
    frozen = read_json(visual / "registration.json")
    assert source_hashes() == frozen["source_hashes"]
    for path, digest in frozen["experiment_files"].items():
        assert sha256(Path(path)) == digest
    assert sha256(Path(frozen["protocol_path"])) == frozen["protocol_sha256"]
    settings = credentials(Path(".env"))
    assert settings[2] == frozen["model"]
    dataset = Path(frozen["dataset"])
    records = {r["station_id"]: r for r in read_json(dataset / "manifest.json")["records"]}
    jobs = [j for j in frozen["pending"] if (j["station"] in frozen["pilot_ids"]) == (phase == "pilot")]
    write_json(visual / f"{phase}-started.json", {"requests": len(jobs), "retry_count": 0})

    def request(job):
        work = visual / "predictions" / job["mode"] / job["station"]
        assert sha256(work / "request.json") == job["request_sha256"]
        observation = read_json(dataset / records[job["station"]]["observation"])
        if job["mode"] == "original":
            result = call_visual(observation, "pair_focus", work, settings)
        else:
            result = fingerprint.call(observation, work, settings)
        print(job["station"], job["mode"], result["status"], flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(request, jobs))
    write_json(visual / f"{phase}-completed.json", {"requests": len(jobs)})


def summarize(run: Path, visual: Path, output: Path):
    registered, dataset, manifest, records = data_records(run)
    frozen = read_json(visual / "registration.json")
    assert sha256(run / "registration.json") == frozen["baseline_registration_sha256"]
    for phase in ("pilot", "remaining"):
        assert (visual / f"{phase}-completed.json").exists()
    output.mkdir(parents=True, exist_ok=False)
    methods = (*MODES, "vector_nearest", "pair_focus_repair", "segment_numeric", "rf_split_temporal", "rf_balanced")
    predictions = {method: {} for method in methods}
    costs = {mode: {"requests": 0, "failures": 0} for mode in (*MODES, "original")}
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        for mode in (*MODES, "original"):
            work = visual / "predictions" / mode / station
            response, candidates = read_json(work / "result.json"), read_json(work / "candidates.json")
            costs[mode]["requests"] += response.get("request_count", 0)
            costs[mode]["failures"] += response["status"] != "success"
            if mode == "original":
                method = "pair_focus_repair"
                result = repair_return_pairs(observation, response, candidates)
            else:
                method = mode
                result = fingerprint.apply(observation, candidates["numerical_prediction"], candidates["graph"], response)
            predictions[method][station] = result
            write_json(output / "predictions" / method / station / "result.json", result)
        for method in ("segment_numeric", "rf_split_temporal", "rf_balanced"):
            predictions[method][station] = read_json(run / "predictions" / method / station / "result.json")
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        candidates = read_json(visual / "predictions/vector_visual" / station / "candidates.json")
        decision = fingerprint.nearest(candidates["graph"], candidates["local_vectors"])
        result = fingerprint.apply(observation, candidates["numerical_prediction"], candidates["graph"], decision)
        predictions["vector_nearest"][station] = result
        write_json(output / "predictions/vector_nearest" / station / "result.json", result)
    summaries, rows = score(dataset, records, predictions)
    main, original, numeric = [summaries[m] for m in ("vector_visual", "pair_focus_repair", "segment_numeric")]
    difference = main["types_daily_macro_f1"] - original["types_daily_macro_f1"]
    gain = main["types_daily_macro_f1"] - numeric["types_daily_macro_f1"]
    worst = min(main["types"][k]["daily_micro"]["f1"] - original["types"][k]["daily_micro"]["f1"] for k in main["types"])
    representation_gain = max(main["types_daily_macro_f1"] - summaries[m]["types_daily_macro_f1"]
                              for m in ("vector_text", "vector_nearest"))
    summary = {"experiment_id": "GNSS-VLM-EXP15", "seed": manifest["seed"], "split": registered["split"],
        "station_ids": registered["station_ids"], "methods": summaries, "costs": costs,
        "decision": {"eligible_for_new_test": difference >= .015 and gain >= .015 and worst >= -.03 and representation_gain > 0,
                     "difference_vs_original": difference, "visual_gain_vs_numeric": gain, "worst_type_difference": worst, "max_gain_vs_text_or_nearest": representation_gain},
        "visual_increment_bootstrap": pooled_bootstrap(rows["vector_visual"], rows["segment_numeric"], manifest["seed"]),
        "baseline_run": str(run.resolve()), "visual_run": str(visual.resolve()), "saved_control": frozen["saved_control"]}
    write_json(output / "summary.json", summary)
    write_json(output / "per-station.json", rows)
    shutil.copyfile(visual / "registration.json", output / "registration.json")
    lines = ["# EXP15：局部跳变向量可视化核验（VEC-VLM）", "",
             f"种子{manifest['seed']}，{registered['split']}，24同组站，模型{frozen['model']}。", "",
             "| 方法 | 四类日F1 | 四类事件F1≥.5 | 通用日F1 |", "|---|---:|---:|---:|"]
    for method, values in summaries.items():
        lines.append(f"| {method} | {values['types_daily_macro_f1']:.4f} | {values['types_event_macro_f1']['iou_0.5']:.4f} | {values['generic_daily_micro']['f1']:.4f} |")
    lines.extend(["", f"主候选比同模型原SDR{difference:+.4f}，比数值{gain:+.4f}；开发条件{summary['decision']['eligible_for_new_test']}。",
                  "旧控制请求与本轮新增请求分列于registration，不将复用控制算作新调用。", ""])
    (output / "results.md").write_text("\n".join(lines), encoding="utf8")
    print(summary["decision"])
    print({method: values["types_daily_macro_f1"] for method, values in summaries.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "remaining", "summarize"))
    parser.add_argument("--run", type=Path, default=Path("runs/exp10-development-baseline-v1"))
    parser.add_argument("--visual", type=Path, default=Path("runs/exp15-development-visual-v1"))
    parser.add_argument("--out", type=Path, default=Path("results/exp15-development-v1"))
    parser.add_argument("--saved-control", type=Path)
    args = parser.parse_args()
    os.environ["GNSS_VLM_MODEL"] = "qwen3-vl-plus"
    if args.stage == "prepare":
        prepare(args.run, args.visual, args.saved_control)
    elif args.stage == "summarize":
        summarize(args.run, args.visual, args.out)
    else:
        infer(args.visual, args.stage)


if __name__ == "__main__":
    main()
