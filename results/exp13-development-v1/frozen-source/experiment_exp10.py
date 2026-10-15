"""Frozen visual-contribution study with numerical, original, and image ablations."""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from exp10_visual_graph import (
    apply_graph,
    call_graph,
    prepare_graph,
    render_references,
    select_training_references,
)
from experiment_exp07 import score
from experiment_exp08 import data_records
from summarize_exp06 import pooled_bootstrap

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.metrics import aggregate
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.vision import SYSTEM, call_visual, credentials

MODES = ("original", "reference_graph", "zero_graph", "text_graph")


def prepare(run: Path, visual: Path, env: Path):
    registered, dataset, manifest, records = data_records(run)
    settings = credentials(env)
    visual.mkdir(parents=True, exist_ok=False)
    references = select_training_references(dataset, manifest["records"])
    reference_image = visual / "training-references.png"
    render_references(dataset, references, reference_image)
    write_json(visual / "training-references.json", references)
    pending = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numerical = read_json(run / "predictions/segment_numeric" / station / "result.json")
        source = read_json(run / "predictions/rf_split_temporal" / station / "result.json")
        for mode in MODES:
            work = visual / "predictions" / mode / station
            if mode == "original":
                payload = prepare_pair_focus(observation, source, work, settings[2], SYSTEM)
            else:
                payload = prepare_graph(observation, numerical, work, settings[2], mode, references, reference_image)
            if payload is not None:
                pending.append({"station": station, "mode": mode, "request_sha256": sha256(work / "request.json")})
    sources = [Path(__file__), Path(__file__).with_name("exp10_visual_graph.py")]
    for path in sources:
        shutil.copyfile(path, visual / path.name)
    identity = registered["experiment"]
    write_json(visual / "registration.json", {"baseline_run": str(run.resolve()),
        "dataset": str(dataset.resolve()), "dataset_seed": manifest["seed"], "split": registered["split"],
        "station_ids": registered["station_ids"], "model": settings[2],
        "source_hashes": source_hashes(), "experiment_files": {str(path): sha256(path) for path in sources},
        "protocol_path": identity["protocol_path"], "protocol_sha256": sha256(Path(identity["protocol_path"])),
        "identity": identity, "references": references, "reference_image_sha256": sha256(reference_image),
        "pending": pending, "pilot_ids": sorted({job["station"] for job in pending})[:6]})
    print({"request_upper_bound": len(pending), "pilot_ids": sorted({job["station"] for job in pending})[:6]}, flush=True)


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
        observation = read_json(dataset / records[job["station"]]["observation"])
        if job["mode"] == "original":
            result = call_visual(observation, "pair_focus", work, settings)
        else:
            numerical = read_json(work / "candidates.json")["numerical_prediction"]
            result = call_graph(observation, numerical, work, settings)
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
            raise ValueError("every planned phase must finish")
    output.mkdir(parents=True, exist_ok=False)
    predictions = {method: {} for method in ("pair_focus_repair", "reference_graph", "zero_graph", "text_graph",
                                             "segment_numeric", "rf_split_temporal", "rf_balanced")}
    costs = {mode: {"requests": 0, "failures": 0, "total_tokens": 0} for mode in MODES}
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        for mode in MODES:
            work = visual / "predictions" / mode / station
            response = read_json(work / "result.json")
            candidates = read_json(work / "candidates.json")
            if mode == "original":
                method = "pair_focus_repair"
                result = repair_return_pairs(observation, response, candidates)
            else:
                method = mode
                result = apply_graph(observation, candidates["numerical_prediction"], candidates["graph"], response)
            predictions[method][station] = result
            write_json(output / "predictions" / method / station / "result.json", result)
            costs[mode]["requests"] += response.get("request_count", 0)
            costs[mode]["failures"] += response["status"] != "success"
            costs[mode]["total_tokens"] += (response.get("usage") or {}).get("total_tokens", 0)
        for method in ("segment_numeric", "rf_split_temporal", "rf_balanced"):
            predictions[method][station] = read_json(run / "predictions" / method / station / "result.json")
    summaries, rows = score(dataset, records, predictions)
    selected = summaries["reference_graph"]
    original = summaries["pair_focus_repair"]
    numeric = summaries["segment_numeric"]
    identity = registered["experiment"]
    difference = selected["types_daily_macro_f1"] - original["types_daily_macro_f1"]
    visual_gain = selected["types_daily_macro_f1"] - numeric["types_daily_macro_f1"]
    worst = min(selected["types"][kind]["daily_micro"]["f1"] - original["types"][kind]["daily_micro"]["f1"]
                for kind in original["types"])
    accepted = (difference >= identity["minimum_vs_original"]
                and visual_gain >= identity["minimum_visual_vs_numeric"] and worst >= -.05)
    strata = {}
    for method, reports in rows.items():
        groups = {}
        for record, row in zip(records, reports):
            truth = read_json(dataset / record["truth"])
            groups.setdefault(truth["scenario"], []).append(row)
        strata[method] = {scenario: aggregate(reports) for scenario, reports in groups.items()}
    summary = {"experiment_id": "GNSS-VLM-EXP10", "seed": manifest["seed"], "split": registered["split"],
               "station_ids": registered["station_ids"], "methods": summaries, "scenario_strata": strata,
               "costs": costs, "decision": {"eligible_for_new_test": accepted,
                   "difference_vs_original": difference, "visual_gain_vs_numeric": visual_gain,
                   "worst_type_difference": worst},
               "visual_increment_bootstrap": pooled_bootstrap(rows["reference_graph"], rows["segment_numeric"], registered["seed"]),
               "vs_original_bootstrap": pooled_bootstrap(rows["reference_graph"], rows["pair_focus_repair"], registered["seed"]),
               "baseline_run": str(run.resolve()), "visual_run": str(visual.resolve())}
    write_json(output / "summary.json", summary)
    write_json(output / "per-station.json", rows)
    shutil.copyfile(visual / "registration.json", output / "registration.json")
    for name in ("experiment_exp10.py", "exp10_visual_graph.py"):
        shutil.copyfile(visual / name, output / name)
    lines = ["# EXP10：竞争跳变联合视觉判别与训练形态参考", "",
             f"种子{manifest['seed']}，{registered['split']}，{len(records)}同组站。四类等权日F1，不做point adjustment。", "",
             "| 方法 | 类型日F1 | 类型事件F1≥.5 | 通用日F1 | 阶跃 | 暂态 |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, values in summaries.items():
        numbers = [values["types_daily_macro_f1"], values["types_event_macro_f1"]["iou_0.5"],
                   values["generic_daily_micro"]["f1"], values["types"]["step"]["daily_micro"]["f1"],
                   values["types"]["transient"]["daily_micro"]["f1"]]
        lines.append("| " + method + " | " + " | ".join(f"{value:.4f}" for value in numbers) + " |")
    lines.extend(["", f"主候选相对原视觉{difference:+.4f}，相对同一无视觉数值{visual_gain:+.4f}。",
                  f"开发条件是否通过：{accepted}；最差类型变化{worst:+.4f}。", "",
                  "原数值训练/尖峰/活动完全相同，只有视觉重分配水平事件。",
                  "text_graph删除全部图像；zero_graph删除训练形态参考；不从消融事后挑选主候选。", ""])
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary["decision"])
    print({method: values["types_daily_macro_f1"] for method, values in summaries.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "pilot", "remaining", "summarize"))
    parser.add_argument("--run", type=Path, default=Path("runs/exp10-development-baseline-v1"))
    parser.add_argument("--visual", type=Path, default=Path("runs/exp10-development-visual-v1"))
    parser.add_argument("--out", type=Path, default=Path("results/exp10-development-v1"))
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
