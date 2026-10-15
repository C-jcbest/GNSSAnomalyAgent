"""Fresh paired image removal on the historical SDR range-repair task."""
from __future__ import annotations

import argparse
import copy
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from exp20_range_metrics import aggregate, evaluate

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import aggregate as aggregate_original
from gnss_research.metrics import evaluate_record
from gnss_research.schema import observed_values
from gnss_research.segmented_review import repair_return_pairs
from gnss_research.vision import call_visual, credentials

ROOT = Path(__file__).resolve().parents[1]
RUNS = [Path("runs/exp06-seed2026101405-v1"), Path("runs/exp06-seed2026101406-v1")]
MODES = {"flash_image": "qwen3.8-flash", "flash_noimage": "qwen3.8-flash",
         "vl_image": "qwen3-vl-plus", "vl_noimage": "qwen3-vl-plus"}
PROTOCOL = Path("docs/experiment-exp22.md")


def cases():
    stations = []
    triggered = []
    for run in RUNS:
        registration = read_json(run / "registration.json")
        dataset = Path(registration["dataset"])
        manifest = read_json(dataset / "manifest.json")
        for record in sorted(manifest["records"], key=lambda r: r["station_id"]):
            if record["station_id"] not in registration["station_ids"]:
                continue
            case = {"run": str(run), "dataset": str(dataset), "seed": manifest["seed"], "record": record,
                    "id": f"{manifest['seed']}-{record['station_id']}"}
            stations.append(case)
            old_work = run / "predictions/pair_focus" / record["station_id"]
            if read_json(old_work / "result.json").get("request_count", 0):
                triggered.append(case)
    return stations, triggered


def prepare(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    all_cases, triggered = cases()
    pending = []
    for case in triggered:
        source = Path(case["run"]) / "predictions/pair_focus" / case["record"]["station_id"]
        for mode, model in MODES.items():
            work = output / "predictions" / case["id"] / mode
            work.mkdir(parents=True)
            request = copy.deepcopy(read_json(source / "request.json"))
            request["model"] = model
            if mode.endswith("noimage"):
                for message in request["messages"]:
                    if isinstance(message["content"], list):
                        message["content"] = [item for item in message["content"] if item.get("type") != "image_url"]
            write_json(work / "request.json", request)
            image_count = sum(item.get("type") == "image_url" for message in request["messages"]
                              if isinstance(message["content"], list) for item in message["content"])
            write_json(work / "input-receipt.json", {"original_request_sha256": sha256(source / "request.json"),
                "request_sha256": sha256(work / "request.json"), "image_count": image_count,
                "modification": "model name; image items removed only for noimage"})
            pending.append({"case_id": case["id"], "mode": mode, "model": model,
                            "request_sha256": sha256(work / "request.json")})
    write_json(output / "registration.json", {"experiment_id": "GNSS-VLM-EXP22", "cases": all_cases,
        "triggered": triggered, "pilot_ids": [case["id"] for case in triggered[:2]], "pending": pending,
        "protocol_sha256": sha256(PROTOCOL), "script_sha256": sha256(Path(__file__))})
    shutil.copyfile(__file__, output / "experiment-source.py")
    print({"triggered_stations": len(triggered), "pilot_requests": 8, "maximum_requests": len(pending)}, flush=True)


def infer(output: Path, phase: str):
    frozen = read_json(output / "registration.json")
    if sha256(PROTOCOL) != frozen["protocol_sha256"] or sha256(Path(__file__)) != frozen["script_sha256"]:
        raise ValueError("registered source/protocol changed")
    if phase == "remaining" and not read_json(output / "pilot-summary.json")["expand"]:
        raise ValueError("pilot did not justify expansion")
    sentinel = output / f"{phase}-started.json"
    if sentinel.exists():
        raise ValueError("phase already started; no retries")
    selected = [job for job in frozen["pending"] if (job["case_id"] in frozen["pilot_ids"]) == (phase == "pilot")]
    write_json(sentinel, {"requests": len(selected), "retry_count": 0})
    lookup = {case["id"]: case for case in frozen["triggered"]}
    base, key, _ = credentials(Path(".env"))

    def request(job):
        case = lookup[job["case_id"]]
        observation = read_json(Path(case["dataset"]) / case["record"]["observation"])
        work = output / "predictions" / job["case_id"] / job["mode"]
        if sha256(work / "request.json") != job["request_sha256"]:
            raise ValueError("registered request changed")
        result = call_visual(observation, "pair_focus", work, (base, key, job["model"]))
        print(job["case_id"], job["mode"], result["status"],
              [(e["kind"], e["start"], e["end"]) for e in result.get("prediction", {}).get("events", [])], flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(request, selected))
    write_json(output / f"{phase}-completed.json", {"requests": len(selected)})


def summarize(visual: Path, output: Path, phase: str):
    frozen = read_json(visual / "registration.json")
    if not (visual / f"{phase}-completed.json").exists():
        raise ValueError("all planned requests must complete")
    selected = frozen["cases"] if phase == "remaining" else [
        case for case in frozen["triggered"] if case["id"] in frozen["pilot_ids"]]
    output.mkdir(parents=True, exist_ok=False)
    methods = (*MODES, "historical_sdr", "segment_numeric")
    rows = {method: [] for method in methods}
    old_rows = {method: [] for method in methods}
    costs = {mode: {"requests": 0, "failed": 0} for mode in MODES}
    for case in selected:
        dataset, run = Path(case["dataset"]), Path(case["run"])
        record = case["record"]
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        truth = read_json(dataset / record["truth"])
        old_work = run / "predictions/pair_focus" / station
        candidates = read_json(old_work / "candidates.json")
        triggered = bool(read_json(old_work / "result.json").get("request_count", 0))
        for method in methods:
            if method in MODES:
                if triggered:
                    work = visual / "predictions" / case["id"] / method
                    response = read_json(work / "result.json")
                    costs[method]["requests"] += response["request_count"]
                    costs[method]["failed"] += response["status"] != "success"
                    result = repair_return_pairs(observation, response, candidates)
                    for name in ("result.json", "response.json", "input-receipt.json"):
                        target = output / "responses" / case["id"] / method / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if (work / name).exists():
                            shutil.copyfile(work / name, target)
                else:
                    result = read_json(run / "predictions/segment_numeric" / station / "result.json")
            else:
                name = "pair_focus_repair" if method == "historical_sdr" else method
                result = read_json(run / "predictions" / name / station / "result.json")
            write_json(output / "predictions" / method / case["id"] / "result.json", result)
            rows[method].append({"seed": case["seed"], "station": station, "metrics": evaluate(observation, truth, result)})
            observed = np.isfinite(observed_values(observation)).all(axis=1)
            old_rows[method].append(evaluate_record(truth, result, observed, observation["background_end"]))
    metrics = {method: aggregate(reports) for method, reports in rows.items()}
    original = {method: aggregate_original(reports) for method, reports in old_rows.items()}
    comparisons = {}
    for model in ("flash", "vl"):
        image, noimage = [metrics[model + suffix]["transient"] for suffix in ("_image", "_noimage")]
        comparisons[model] = {
            "affiliation_f1_difference": image["affiliation"]["station_macro_f1"] - noimage["affiliation"]["station_macro_f1"],
            "event_f1_difference": image["events_micro"]["iou_0.5"]["f1"] - noimage["events_micro"]["iou_0.5"]["f1"],
            "matched_event_difference": image["events_micro"]["iou_0.5"]["tp"] - noimage["events_micro"]["iou_0.5"]["tp"]}
    expand = any(any(value > 0 for value in comparison.values()) for comparison in comparisons.values())
    summary = {"experiment_id": "GNSS-VLM-EXP22", "phase": phase, "stations": len(selected),
        "methods": metrics, "original_metrics": original, "costs": costs, "paired_image_comparison": comparisons,
        "expand": expand, "note": "Post-hoc diagnosis on already observed historical data; fresh paired requests"}
    write_json(output / "summary.json", summary)
    write_json(output / "per-station.json", rows)
    write_json(visual / f"{phase}-summary.json", summary)
    shutil.copyfile(visual / "registration.json", output / "registration.json")
    shutil.copyfile(visual / "experiment-source.py", output / "experiment-source.py")
    lines = ["# EXP22：SDR暂态修复的同轮有图/去图复核", "", f"{phase}，{len(selected)}站；已见数据事后诊断。", "",
             "| 方法 | 暂态Aff-F1 | 暂态事件F1 .5 | 暂态日F1 | 暂态负站FP日 | 四类日F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, values in metrics.items():
        r = values["transient"]
        lines.append(f"| {method} | {r['affiliation']['station_macro_f1']:.4f} | {r['events_micro']['iou_0.5']['f1']:.4f}"
                     + f" | {r['daily_micro']['f1']:.4f} | {r['negative_station_fp_days']}"
                     + f" | {original[method]['types_daily_macro_f1']:.4f} |")
    lines += ["", f"配对比较：{comparisons}；扩量条件：{expand}。", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "infer", "summarize"))
    parser.add_argument("--visual", type=Path, default=Path("runs/exp22-image-ablation-v1"))
    parser.add_argument("--out", type=Path, default=Path("results/exp22-image-ablation-pilot-v1"))
    parser.add_argument("--phase", choices=("pilot", "remaining"), default="pilot")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.visual)
    elif args.stage == "infer":
        infer(args.visual, args.phase)
    else:
        summarize(args.visual, args.out, args.phase)


if __name__ == "__main__":
    main()
