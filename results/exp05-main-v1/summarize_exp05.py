"""Pool saved EXP05 station counts and retain request costs and source receipts."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from report_exp02 import activity_errors

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import aggregate


def summarize(runs: list[Path], out: Path) -> dict:
    pooled = {}
    usage = {}
    receipts = []
    activity = {}
    seen = set()
    splits = set()
    for run in runs:
        registration = read_json(run / "registration.json")
        if not (run / "execution-complete.json").exists():
            raise ValueError("only completed runs can enter a report")
        dataset = Path(registration["dataset"])
        dataset_seed = read_json(dataset / "manifest.json")["seed"]
        splits.add(registration["split"])
        for station in registration["station_ids"]:
            key = (dataset_seed, registration["split"], station)
            if key in seen:
                raise ValueError("duplicate station in pooled report")
            seen.add(key)
        reports = read_json(run / "evaluation/per-station.json")
        receipts.append({"run": run.name, "registration_sha256": sha256(run / "registration.json"),
                         "evaluation_sha256": sha256(run / "evaluation/summary.json"),
                         "station_ids": registration["station_ids"],
                         "dataset_manifest_sha256": registration["manifest_sha256"],
                         "dataset_seed": dataset_seed, "split": registration["split"]})
        for method, rows in reports.items():
            pooled.setdefault(method, []).extend(rows)
            if rows[0]["types"] is not None:
                errors = activity_errors(run, dataset, method)
                totals = activity.setdefault(method, {key: 0 for key in errors})
                for key, value in errors.items():
                    totals[key] += value
            costs = usage.setdefault(method, {"requests": 0, "failed": 0, "prompt_tokens": 0,
                                               "completion_tokens": 0, "seconds": [],
                                               "image_counts": [], "missing_token_receipts": 0})
            for station_id in registration["station_ids"]:
                work = run / "predictions" / method / station_id
                result = read_json(work / "result.json")
                costs["requests"] += result.get("request_count", 0)
                costs["failed"] += int(result["status"] != "success")
                if result.get("request_count", 0):
                    costs["seconds"].append(result["elapsed_seconds"])
                    tokens = result.get("usage") or {}
                    costs["missing_token_receipts"] += int(not tokens)
                    costs["prompt_tokens"] += tokens.get("prompt_tokens", 0)
                    costs["completion_tokens"] += tokens.get("completion_tokens", 0)
                    receipt = read_json(work / "input-receipt.json")
                    image_count = 1 if method == "visual_only" else len(receipt["images"])
                    costs["image_counts"].append(image_count)
    for costs in usage.values():
        seconds = costs.pop("seconds")
        costs["median_request_seconds"] = float(np.median(seconds)) if seconds else None
    summary = {"experiment_id": "GNSS-VLM-EXP05", "sources": receipts,
               "splits": sorted(splits), "unique_stations": len(seen),
               "methods": {method: {**aggregate(rows), "failed_records": usage[method]["failed"]}
                           for method, rows in pooled.items()}, "costs": usage,
               "interpretation": "Validation is development evidence; fresh frozen tests are "
               "reported separately. Counts pooled within type before equally averaging four types."}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "summary.json", summary)
    write_json(out / "per-station.json", pooled)
    write_json(out / "activity-errors.json", activity)
    shutil.copyfile(Path(__file__), out / "summarize_exp05.py")
    for run in runs:
        target = out / "receipts" / run.name
        target.mkdir(parents=True)
        for filename in ("registration.json", "training.json", "execution-started.json",
                         "execution-complete.json"):
            shutil.copyfile(run / filename, target / filename)
    lines = ["# GNSS-VLM-EXP05：GNSS视觉形态确认与数值精确定位实验", "",
             "从所列运行的全站计数汇总；失败按零检测保留，无point adjustment。", "",
             "| 方法 | 类型日F1 | 类型事件F1≥.5 | 通用日F1 | 阶跃严格F1 | 失败 |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, values in summary["methods"].items():
        typed = values.get("types", {})
        macro = values.get("types_daily_macro_f1")
        event = values.get("types_event_macro_f1", {}).get("iou_0.5")
        step = typed.get("step", {}).get("daily_micro", {}).get("f1")
        cells = [f"{value:.4f}" if value is not None else "—" for value in
                 (macro, event, values["generic_daily_micro"]["f1"], step)]
        lines.append(f"| {method} | " + " | ".join(cells) + f" | {values['failed_records']} |")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary": str(out / "summary.json"), "methods": len(pooled)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.runs, args.out))


if __name__ == "__main__":
    main()
