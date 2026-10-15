"""Registered small-batch multiscale range localization experiment."""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import exp21_range_localization as study
from exp20_range_metrics import aggregate, evaluate
from experiment_exp07 import score
from experiment_exp08 import data_records

from gnss_research.files import read_json, sha256, write_json
from gnss_research.vision import credentials

ROOT = Path(__file__).resolve().parents[1]
FILES = [Path(__file__), Path(study.__file__), ROOT / "scripts/exp20_range_metrics.py"]
MODES = ("range_visual", "range_text", "range_no_image")


def settings():
    base, key, _ = credentials(Path(".env"))
    return base, key, "qwen3-vl-plus"


def prepare(run: Path, visual: Path, phase: str):
    registered, dataset, manifest, records = data_records(run)
    if not (run / "execution-complete.json").exists():
        raise ValueError("numerical predictions must be complete before visual registration")
    if phase == "pilot":
        visual.mkdir(parents=True, exist_ok=False)
        selected = registered["station_ids"][:6]
    else:
        pilot = read_json(visual / "pilot-summary.json")
        if not pilot["expand"]:
            raise ValueError("small-batch range experiment did not justify expansion")
        selected = registered["station_ids"][6:]
    pending = []
    for record in records:
        station = record["station_id"]
        if station not in selected:
            continue
        observation = read_json(dataset / record["observation"])
        overview = visual / "inputs" / (station + ".png")
        study.render(observation, overview)
        for mode in MODES:
            work = visual / "predictions" / station / mode
            study.prepare(observation, work, settings()[2], mode, overview)
            pending.append({"station": station, "mode": mode, "request_sha256": sha256(work / "request.json")})
    frozen = visual / f"{phase}-registration.json"
    write_json(frozen, {"experiment_id": "GNSS-VLM-EXP21", "baseline_run": str(run.resolve()),
        "dataset": str(dataset.resolve()), "seed": manifest["seed"], "split": registered["split"],
        "station_ids": selected, "model": settings()[2], "pending": pending,
        "baseline_registration_sha256": sha256(run / "registration.json"),
        "protocol_sha256": sha256(Path("docs/experiment-exp21.md")),
        "source_hashes": registered["source_hashes"],
        "experiment_files": {str(p): sha256(p) for p in FILES}})
    source = visual / "frozen-source"
    source.mkdir(exist_ok=True)
    for path in FILES:
        if not (source / path.name).exists():
            shutil.copyfile(path, source / path.name)
    print({"phase": phase, "stations": selected, "first_pass_requests": len(pending),
           "zoom_request_upper_bound": len(selected)}, flush=True)


def infer(visual: Path, phase: str):
    frozen = read_json(visual / f"{phase}-registration.json")
    if sha256(Path("docs/experiment-exp21.md")) != frozen["protocol_sha256"]:
        raise ValueError("registered protocol changed")
    for path, digest in frozen["experiment_files"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("registered experiment source changed")
    sentinel = visual / f"{phase}-started.json"
    if sentinel.exists():
        raise ValueError("already started; no implicit retries")
    dataset = Path(frozen["dataset"])
    records = {r["station_id"]: r for r in read_json(dataset / "manifest.json")["records"]}
    write_json(sentinel, {"requests_upper_bound": len(frozen["pending"]) + len(frozen["station_ids"]), "retry_count": 0})

    def request(job):
        station, mode = job["station"], job["mode"]
        observation = read_json(dataset / records[station]["observation"])
        work = visual / "predictions" / station / mode
        if sha256(work / "request.json") != job["request_sha256"]:
            raise ValueError("registered input changed")
        response = study.call(observation, work, settings())
        print(station, mode, response["status"], response.get("range_decision"), flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(request, frozen["pending"]))
    for station in frozen["station_ids"]:
        observation = read_json(dataset / records[station]["observation"])
        work = visual / "predictions" / station
        coarse = read_json(work / "range_visual/result.json")
        zoom_work = work / "range_zoom"
        if coarse["status"] != "success" or not coarse["range_decision"]["events"]:
            zoom_work.mkdir()
            write_json(zoom_work / "result.json", {**coarse, "request_count": 0,
                "zoom_skipped": "No usable coarse interval; same prediction, no new call"})
            continue
        study.prepare(observation, zoom_work, settings()[2], "range_zoom", visual / "inputs" / (station + ".png"), coarse)
        response = study.call(observation, zoom_work, settings())
        print(station, "range_zoom", response["status"], response.get("range_decision"), flush=True)
    write_json(visual / f"{phase}-completed.json", {"stations": frozen["station_ids"], "retry_count": 0})


def summarize(run: Path, visual: Path, output: Path, phase: str):
    registered, dataset, manifest, records = data_records(run)
    if not (visual / f"{phase}-completed.json").exists():
        raise ValueError("incomplete run")
    frozen = read_json(visual / f"{phase}-registration.json")
    selected = frozen["station_ids"]
    if phase == "remaining":
        selected = registered["station_ids"]
    records = [r for r in records if r["station_id"] in selected]
    output.mkdir(parents=True, exist_ok=False)
    methods = (*MODES, "range_zoom", "segment_numeric", "rf_split_temporal")
    predictions = {method: {} for method in methods}
    rows = {method: [] for method in methods}
    costs = {mode: {"requests": 0, "failed": 0} for mode in (*MODES, "range_zoom")}
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        truth = read_json(dataset / record["truth"])
        numeric = read_json(run / "predictions/segment_numeric" / station / "result.json")
        for method in methods:
            if method in ("segment_numeric", "rf_split_temporal"):
                result = read_json(run / "predictions" / method / station / "result.json")
            else:
                work = visual / "predictions" / station / method
                response = read_json(work / "result.json")
                costs[method]["requests"] += response.get("request_count", 0)
                costs[method]["failed"] += response["status"] != "success"
                result = study.apply(observation, numeric, response)
                for name in ("response.json", "result.json", "input-receipt.json"):
                    if (work / name).exists():
                        target = output / "responses" / station / method / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(work / name, target)
            predictions[method][station] = result
            write_json(output / "predictions" / method / station / "result.json", result)
            rows[method].append({"seed": manifest["seed"], "station": station, "scenario": truth["scenario"],
                                "metrics": evaluate(observation, truth, result)})
    metrics = {method: aggregate(reports) for method, reports in rows.items()}
    daily, _ = score(dataset, records, predictions)
    decisions = {}
    for method in ("range_visual", "range_zoom"):
        m, n, t = [metrics[name] for name in (method, "segment_numeric", "range_text")]
        f1_gain = m["range_types_affiliation_macro_f1"] - n["range_types_affiliation_macro_f1"]
        event_gain = np_mean_event(m) - np_mean_event(n)
        visual_gain = m["range_types_affiliation_macro_f1"] - t["range_types_affiliation_macro_f1"]
        event_text_gain = np_mean_event(m) - np_mean_event(t)
        normal_fp = sum(m[k]["negative_station_fp_days"] for k in ("transient", "deformation"))
        numeric_fp = sum(n[k]["negative_station_fp_days"] for k in ("transient", "deformation"))
        passes = max(f1_gain, event_gain) >= .02 and max(visual_gain, event_text_gain) > 0 and normal_fp <= numeric_fp
        decisions[method] = {"affiliation_gain_vs_numeric": f1_gain, "event_f1_gain_vs_numeric": event_gain,
                             "affiliation_gain_vs_text": visual_gain, "event_f1_gain_vs_text": event_text_gain,
                             "negative_station_fp_days": normal_fp, "passes": passes}
    summary = {"experiment_id": "GNSS-VLM-EXP21", "method_name": "MRL-VLM", "phase": phase,
        "seed": manifest["seed"], "split": registered["split"], "station_ids": selected,
        "methods": metrics, "original_metrics": daily, "costs": costs, "decisions": decisions,
        "expand": any(d["passes"] for d in decisions.values()),
        "note": "Development evidence only; zoom requests reuse coarse visual first pass; no retries"}
    write_json(output / "summary.json", summary)
    write_json(output / "per-station.json", rows)
    write_json(visual / f"{phase}-summary.json", summary)
    shutil.copytree(visual / "frozen-source", output / "frozen-source")
    shutil.copyfile(visual / f"{phase}-registration.json", output / "registration.json")
    lines = ["# EXP21：MRL-VLM多尺度视觉区间定位", "", f"{phase}，{len(selected)}验证站；开发证据。", "",
        "| 方法 | 暂态Aff-F1 | 形变Aff-F1 | 两类Aff均值 | 暂态事件F1 .5 | 形变事件F1 .5 | 两类负站FP日 | 四类日F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, m in metrics.items():
        cells = [m[k]["affiliation"]["station_macro_f1"] for k in ("transient", "deformation")]
        cells += [m["range_types_affiliation_macro_f1"]]
        cells += [m[k]["events_micro"]["iou_0.5"]["f1"] for k in ("transient", "deformation")]
        lines.append(f"| {method} | " + " | ".join(f"{v:.4f}" for v in cells)
                     + f" | {sum(m[k]['negative_station_fp_days'] for k in ('transient', 'deformation'))}"
                     + f" | {daily[method]['types_daily_macro_f1']:.4f} |")
    lines += ["", f"小批扩量条件：{summary['expand']}。完整P/R、覆盖、边界、IoU和敏感性见summary.json。", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def np_mean_event(metrics):
    return sum(metrics[k]["events_micro"]["iou_0.5"]["f1"] for k in ("transient", "deformation")) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "infer", "summarize"))
    parser.add_argument("--run", type=Path, default=Path("runs/exp21-development-1515-baseline-v1"))
    parser.add_argument("--visual", type=Path, default=Path("runs/exp21-development-1515-visual-v1"))
    parser.add_argument("--out", type=Path, default=Path("results/exp21-development-1515-pilot-v1"))
    parser.add_argument("--phase", choices=("pilot", "remaining"), default="pilot")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run, args.visual, args.phase)
    elif args.stage == "infer":
        infer(args.visual, args.phase)
    else:
        summarize(args.run, args.visual, args.out, args.phase)


if __name__ == "__main__":
    main()
