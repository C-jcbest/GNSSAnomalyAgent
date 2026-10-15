"""Activity-only visual intervention over the same numerical GNSS event dates."""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from exp11_activity_review import (
    apply_activity,
    call_activity,
    prepare_activity,
    render_activity_observation,
    render_references,
    select_references,
)
from experiment_exp07 import score
from experiment_exp08 import data_records
from summarize_exp06 import pooled_bootstrap

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.metrics import aggregate
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.vision import SYSTEM, call_visual, credentials

ACTIVITY_MODES = ("activity_reference", "activity_zero", "activity_text")


def prepare(run: Path, visual: Path, env: Path):
    registered, dataset, manifest, records = data_records(run)
    settings = credentials(env)
    visual.mkdir(parents=True, exist_ok=False)
    references = select_references(dataset, manifest["records"])
    reference_image = visual / "training-references.png"
    render_references(dataset, references, reference_image)
    write_json(visual / "training-references.json", references)
    pending = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        source = read_json(run / "predictions/rf_split_temporal" / station / "result.json")
        original_work = visual / "predictions/original" / station
        payload = prepare_pair_focus(observation, source, original_work, settings[2], SYSTEM)
        if payload is not None:
            pending.append({"station": station, "mode": "original", "request_sha256": sha256(original_work / "request.json")})
        target_image = visual / "inputs" / (station + ".png")
        target_image.parent.mkdir(exist_ok=True)
        render_activity_observation(observation, target_image)
        for mode in ACTIVITY_MODES:
            work = visual / "predictions" / mode / station
            prepare_activity(observation, work, settings[2], mode, references, reference_image, target_image)
            pending.append({"station": station, "mode": mode, "request_sha256": sha256(work / "request.json")})
    sources = [Path(__file__), Path(__file__).with_name("exp11_activity_review.py")]
    for path in sources:
        shutil.copyfile(path, visual / path.name)
    identity = registered["experiment"]
    write_json(visual / "registration.json", {"baseline_run": str(run.resolve()),
        "dataset": str(dataset.resolve()), "seed": manifest["seed"], "split": registered["split"],
        "station_ids": registered["station_ids"], "model": settings[2], "source_hashes": source_hashes(),
        "experiment_files": {str(path): sha256(path) for path in sources}, "identity": identity,
        "protocol_path": identity["protocol_path"], "protocol_sha256": sha256(Path(identity["protocol_path"])),
        "references": references, "pending": pending, "pilot_ids": registered["station_ids"][:6]})
    print({"request_upper_bound": len(pending), "pilot_ids": registered["station_ids"][:6]}, flush=True)


def infer(visual: Path, env: Path, phase: str):
    frozen = read_json(visual / "registration.json")
    if frozen["source_hashes"] != source_hashes():
        raise ValueError("registered production source changed")
    for path, digest in frozen["experiment_files"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("registered experiment source changed")
    if sha256(Path(frozen["protocol_path"])) != frozen["protocol_sha256"]:
        raise ValueError("registered protocol changed")
    sentinel = visual / f"{phase}-started.json"
    if sentinel.exists():
        raise ValueError("phase already started; no implicit retry")
    settings = credentials(env)
    if settings[2] != frozen["model"]:
        raise ValueError("registered model changed")
    dataset = Path(frozen["dataset"])
    records = {record["station_id"]: record for record in read_json(dataset / "manifest.json")["records"]}
    jobs = [job for job in frozen["pending"]
            if (job["station"] in frozen["pilot_ids"]) == (phase == "pilot")]
    write_json(sentinel, {"requests": len(jobs), "retry_count": 0})

    def request(job):
        work = visual / "predictions" / job["mode"] / job["station"]
        if sha256(work / "request.json") != job["request_sha256"]:
            raise ValueError("registered request changed")
        if job["mode"] == "original":
            observation = read_json(dataset / records[job["station"]]["observation"])
            result = call_visual(observation, "pair_focus", work, settings)
        else:
            result = call_activity(work, settings)
        print(job["station"], job["mode"], result["status"], flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(request, jobs))
    write_json(visual / f"{phase}-completed.json", {"requests": len(jobs)})


def summarize(run: Path, visual: Path, output: Path):
    registered, dataset, manifest, records = data_records(run)
    frozen = read_json(visual / "registration.json")
    if frozen["baseline_run"] != str(run.resolve()):
        raise ValueError("baseline and visual observations differ")
    for phase in ("pilot", "remaining"):
        if not (visual / f"{phase}-completed.json").exists():
            raise ValueError("all planned phases must finish")
    output.mkdir(parents=True, exist_ok=False)
    methods = (*ACTIVITY_MODES, "activity_on_numeric", "pair_focus_repair", "segment_numeric",
               "rf_split_temporal", "rf_balanced")
    predictions = {method: {} for method in methods}
    costs = {mode: {"requests": 0, "failures": 0, "total_tokens": 0} for mode in ("original", *ACTIVITY_MODES)}
    activity_decisions = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        original_work = visual / "predictions/original" / station
        original = repair_return_pairs(observation, read_json(original_work / "result.json"),
                                       read_json(original_work / "candidates.json"))
        numeric = read_json(run / "predictions/segment_numeric" / station / "result.json")
        predictions["pair_focus_repair"][station] = original
        for mode in ("original", *ACTIVITY_MODES):
            response = read_json(visual / "predictions" / mode / station / "result.json")
            costs[mode]["requests"] += response.get("request_count", 0)
            costs[mode]["failures"] += response["status"] != "success"
            costs[mode]["total_tokens"] += (response.get("usage") or {}).get("total_tokens", 0)
            if mode == "original":
                continue
            predictions[mode][station] = apply_activity(observation, original, numeric, response)
            if mode == "activity_reference":
                predictions["activity_on_numeric"][station] = apply_activity(observation, numeric, numeric, response)
                activity_decisions.append({"station": station, "numeric_selected": numeric["activity_fit"]["selected"],
                                           "status": response["status"], "decision": response.get("activity_decision")})
        for method in ("segment_numeric", "rf_split_temporal", "rf_balanced"):
            predictions[method][station] = read_json(run / "predictions" / method / station / "result.json")
        for method in (*ACTIVITY_MODES, "activity_on_numeric", "pair_focus_repair"):
            write_json(output / "predictions" / method / station / "result.json", predictions[method][station])
    summaries, rows = score(dataset, records, predictions)
    selected, original, numeric = [summaries[method] for method in ("activity_reference", "pair_focus_repair", "segment_numeric")]
    difference = selected["types_daily_macro_f1"] - original["types_daily_macro_f1"]
    visual_gain = selected["types_daily_macro_f1"] - numeric["types_daily_macro_f1"]
    activity_gain = selected["types"]["deformation"]["daily_micro"]["f1"] - original["types"]["deformation"]["daily_micro"]["f1"]
    identity = registered["experiment"]
    accepted = (difference >= identity["minimum_vs_original"] and visual_gain >= identity["minimum_visual_vs_numeric"]
                and activity_gain >= identity["minimum_activity_f1_gain"])
    strata = {}
    for method, reports in rows.items():
        groups = {}
        for record, row in zip(records, reports):
            scenario = read_json(dataset / record["truth"])["scenario"]
            groups.setdefault(scenario, []).append(row)
        strata[method] = {name: aggregate(group) for name, group in groups.items()}
    summary = {"experiment_id": "GNSS-VLM-EXP11", "seed": manifest["seed"], "split": registered["split"],
               "station_ids": registered["station_ids"], "methods": summaries, "scenario_strata": strata,
               "costs": costs, "activity_decisions": activity_decisions,
               "decision": {"eligible_for_new_test": accepted, "difference_vs_original": difference,
                            "visual_gain_vs_numeric": visual_gain, "activity_f1_gain": activity_gain},
               "visual_increment_bootstrap": pooled_bootstrap(rows["activity_reference"], rows["segment_numeric"], registered["seed"]),
               "activity_increment_bootstrap": pooled_bootstrap(rows["activity_reference"], rows["pair_focus_repair"], registered["seed"]),
               "baseline_run": str(run.resolve()), "visual_run": str(visual.resolve())}
    write_json(output / "summary.json", summary)
    write_json(output / "per-station.json", rows)
    shutil.copyfile(visual / "registration.json", output / "registration.json")
    for name in ("experiment_exp11.py", "exp11_activity_review.py"):
        shutil.copyfile(visual / name, output / name)
    lines = ["# EXP11：视觉持续活动判别与固定数值边界融合", "",
             f"种子{manifest['seed']}，{registered['split']}，{len(records)}同组站；四类等权日F1，无point adjustment。", "",
             "| 方法 | 类型日F1 | 类型事件F1≥.5 | 通用日F1 | 活动日F1 |", "|---|---:|---:|---:|---:|"]
    for method, values in summaries.items():
        numbers = [values["types_daily_macro_f1"], values["types_event_macro_f1"]["iou_0.5"],
                   values["generic_daily_micro"]["f1"], values["types"]["deformation"]["daily_micro"]["f1"]]
        lines.append("| " + method + " | " + " | ".join(f"{value:.4f}" for value in numbers) + " |")
    lines.extend(["", f"相对原SDR-VLM{difference:+.4f}，视觉相对数值{visual_gain:+.4f}，活动F1变化{activity_gain:+.4f}。",
                  f"开发条件通过：{accepted}。尖峰/阶跃/暂态相同，活动日期候选未改。",
                  "activity_zero去活动参考，activity_text去活动图像；原水平视觉响应共享。", ""])
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary["decision"])
    print({method: values["types_daily_macro_f1"] for method, values in summaries.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "remaining", "summarize"))
    parser.add_argument("--run", type=Path, default=Path("runs/exp11-development-baseline-v1"))
    parser.add_argument("--visual", type=Path, default=Path("runs/exp11-development-visual-v1"))
    parser.add_argument("--out", type=Path, default=Path("results/exp11-development-v1"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run, args.visual, args.env_file)
    elif args.stage == "summarize":
        summarize(args.run, args.visual, args.out)
    else:
        infer(args.visual, args.env_file, args.stage)


if __name__ == "__main__":
    main()
