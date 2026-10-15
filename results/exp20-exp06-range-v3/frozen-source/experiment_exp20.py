"""Re-evaluate existing complete predictions; no API calls or prediction tuning."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import exp17_resolution
import exp18_distribution
from exp20_range_metrics import RANGE_TYPES, aggregate, evaluate

from gnss_research.files import read_json, sha256, write_json
from gnss_research.segmented_review import repair_return_pairs


def predictions_for(run, visual, family, record):
    station = record["station_id"]
    registered = read_json(run / "registration.json")
    dataset = Path(registered["dataset"])
    observation = read_json(dataset / record["observation"])
    if family == "production":
        methods = [name for name in registered["methods"] if name not in ("pair_focus", "pair_raw")]
        return {method: read_json(run / "predictions" / method / station / "result.json")
                for method in methods}
    numeric = read_json(run / "predictions/segment_numeric" / station / "result.json")
    frozen = read_json(visual / "registration.json")
    outputs = {}
    module = exp17_resolution if family == "exp17" else exp18_distribution
    modes = frozen["modes"] if family == "exp17" else (*module.MODES, "original")
    for method in modes:
        work = visual / "predictions" / method / station
        result = read_json(work / "result.json")
        candidates = read_json(work / "candidates.json")
        if method == "original":
            outputs[method] = repair_return_pairs(observation, result, candidates)
        else:
            outputs[method] = module.apply(observation, numeric, candidates["graph"], result)
    for method in ("segment_numeric", "rf_split_temporal", "rf_balanced"):
        outputs[method] = read_json(run / "predictions" / method / station / "result.json")
    candidates = read_json(visual / "predictions/bar_text" / station / "candidates.json") \
        if family == "exp17" else read_json(visual / "predictions/distribution_text" / station / "candidates.json")
    nearest = module.nearest(candidates["graph"], candidates["local_vectors"])
    outputs["vector_nearest"] = module.apply(observation, numeric, candidates["graph"], nearest)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--visual", type=Path)
    parser.add_argument("--family", choices=("production", "exp17", "exp18"), default="production")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    pooled = {}
    sources = []
    common_methods = set.intersection(*[set(read_json(run / "registration.json")["methods"])
                                        for run in args.runs])
    for run in args.runs:
        registered = read_json(run / "registration.json")
        dataset = Path(registered["dataset"])
        manifest = read_json(dataset / "manifest.json")
        records = [record for record in manifest["records"] if record["station_id"] in registered["station_ids"]]
        for record in records:
            observation = read_json(dataset / record["observation"])
            truth = read_json(dataset / record["truth"])
            predictions = predictions_for(run, args.visual, args.family, record)
            for method, result in predictions.items():
                if args.family == "production" and method not in common_methods:
                    continue
                # Registered visual failures/not-run records remain as zero detections.
                if result.get("status") != "success":
                    result = {**result, "prediction": {"events": []}}
                metrics = evaluate(observation, truth, result)
                if metrics is None:
                    continue
                pooled.setdefault(method, []).append({"seed": manifest["seed"], "station": record["station_id"],
                    "scenario": truth["scenario"], "status": result["status"], "metrics": metrics})
        sources.append({"run": str(run), "seed": manifest["seed"], "split": registered["split"],
                        "station_ids": registered["station_ids"], "registration_sha256": sha256(run / "registration.json"),
                        "manifest_sha256": sha256(dataset / "manifest.json")})
    expected = sum(len(source["station_ids"]) for source in sources)
    if any(len(rows) != expected for rows in pooled.values()):
        raise ValueError("all compared methods must retain the same stations including failures")
    methods = {method: aggregate(rows) for method, rows in pooled.items()}
    write_json(args.out / "summary.json", {"experiment_id": "GNSS-VLM-EXP20", "sources": sources,
        "family": args.family, "visual": str(args.visual) if args.visual else None,
        "interpretation": "Post-hoc range analysis of already observed data; no new requests; no point adjustment",
        "protocol_sha256": sha256(Path("docs/experiment-exp20.md")), "methods": methods})
    write_json(args.out / "per-station.json", pooled)
    source = args.out / "frozen-source"
    source.mkdir()
    for name in ("experiment_exp20.py", "exp20_range_metrics.py"):
        shutil.copyfile(Path(__file__).with_name(name), source / name)
    lines = ["# EXP20：异常区间识别多指标复核", "", "既有预测事后评价，未产生新请求。Affiliation按正样本站汇总；负站误报另列。", ""]
    for kind in RANGE_TYPES:
        lines += [f"## {kind}", "",
            "| 方法 | Aff-P | Aff-R | 站均Aff-F1 | 日F1 | 事件F1 .3 | 事件F1 .5 | 最佳IoU | 覆盖 | 起/止MAE(.5) | 匹配/真值 | 负站FP日 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
        for method, report in methods.items():
            r = report[kind]
            a = r["affiliation"]
            e = r["events_micro"]["iou_0.5"]
            values = [a["precision"], a["recall"], a["station_macro_f1"], r["daily_micro"]["f1"],
                      r["events_micro"]["iou_0.3"]["f1"], e["f1"], r["truth_event_mean_best_iou"],
                      r["truth_event_mean_coverage"]]
            cells = [f"{v:.4f}" if v is not None else "—" for v in values]
            boundary = "/".join(f"{e[key]:.2f}" if e[key] is not None else "—"
                                for key in ("matched_start_mae_days", "matched_end_mae_days"))
            lines.append(f"| {method} | " + " | ".join(cells)
                         + f" | {boundary} | {e['tp']}/{r['truth_events']} | {r['negative_station_fp_days']} |")
        lines.append("")
    (args.out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
