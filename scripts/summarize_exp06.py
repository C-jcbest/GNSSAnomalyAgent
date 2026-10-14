"""EXP06 pooled endpoints and paired uncertainty on the same station set."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from summarize_exp05 import summarize

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import aggregate
from gnss_research.schema import EVENT_TYPES


def pooled_bootstrap(left: list[dict], right: list[dict], seed: int = 2026101404) -> dict:
    def counts(rows):
        return np.array([[[row["types"][kind]["daily"][key] for key in ("tp", "fp", "fn")]
                          for kind in EVENT_TYPES] for row in rows])

    def score(total):
        numerator = 2 * total[..., 0]
        denominator = numerator + total[..., 1] + total[..., 2]
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float),
                         where=denominator > 0).mean(axis=-1)

    a, b = counts(left), counts(right)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), (5000, len(a)))
    differences = score(a[indices].sum(axis=1)) - score(b[indices].sum(axis=1))
    return {"endpoint": "four-type pooled daily macro F1 difference",
            "difference": float(score(a.sum(axis=0)) - score(b.sum(axis=0))),
            "percentile_ci_95": np.quantile(differences, [.025, .975]).tolist(),
            "resamples": 5000, "seed": seed,
            "note": "Descriptive paired station bootstrap; development selection not corrected. "
                    "Resampled stations share fitted training models."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot-repair", type=Path,
                        help="explicit offline repair derivation for the original six pilot stations")
    parser.add_argument("--primary-method", choices=("segment_pair_repair", "pair_focus_repair"),
                        default="segment_pair_repair")
    parser.add_argument("--controls", type=Path, help="previous development report on exactly the same stations")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    base = args.out / "base-summary"
    summarize(args.runs, base, experiment_id="GNSS-VLM-EXP06",
              experiment_name="GNSS位移结构分解与视觉语义核验实验")
    rows = read_json(base / "per-station.json")
    summary = read_json(base / "summary.json")
    activity = read_json(base / "activity-errors.json")
    if args.controls is not None:
        control_summary = read_json(args.controls / "summary.json")
        control_rows = read_json(args.controls / "per-station.json")
        control_activity = read_json(args.controls / "activity-errors.json")
        order = [(source["dataset_seed"], station) for source in summary["sources"]
                 for station in source["station_ids"]]
        previous_order = [(source["dataset_seed"], station) for source in control_summary["sources"]
                          for station in source["station_ids"]]
        if set(order) != set(previous_order) or summary["splits"] != ["validation"]:
            raise ValueError("saved controls must match all the same development validation stations")
        lookup = {station: position for position, station in enumerate(previous_order)}
        for method, previous_rows in control_rows.items():
            if method in rows:
                continue
            rows[method] = [previous_rows[lookup[station]] for station in order]
            summary["methods"][method] = control_summary["methods"][method]
            summary["costs"][method] = control_summary["costs"][method]
            activity[method] = control_activity[method]
        summary["saved_development_controls"] = {"path": str(args.controls),
                                                  "summary_sha256": sha256(args.controls / "summary.json"),
                                                  "new_requests": 0}
    if args.pilot_repair is not None:
        extra = read_json(args.pilot_repair / "per-station.json")["segment_pair_repair"]
        provenance = read_json(args.pilot_repair / "summary.json")
        if provenance["source_registration_sha256"] != sha256(args.runs[0] / "registration.json"):
            raise ValueError("pilot derivation must correspond to the first listed run")
        rows["segment_pair_repair"] = extra + rows.get("segment_pair_repair", [])
        extra_errors = read_json(args.pilot_repair / "activity-errors.json")["segment_pair_repair"]
        errors = activity.setdefault("segment_pair_repair", {key: 0 for key in extra_errors})
        for key, value in extra_errors.items():
            errors[key] += value
        if "segment_pair_repair" not in summary["costs"]:
            summary["costs"]["segment_pair_repair"] = {
                "requests": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "image_counts": [], "missing_token_receipts": 0, "median_request_seconds": None}
        summary["offline_pilot_derivation"] = {"path": str(args.pilot_repair),
                                               "summary_sha256": sha256(args.pilot_repair / "summary.json"),
                                               "new_model_requests": 0}
    if any(len(method_rows) != summary["unique_stations"] for method_rows in rows.values()):
        raise ValueError("every comparison method must have all the same stations")
    primary = args.primary_method
    parent = "pair_focus" if primary == "pair_focus_repair" else "segment_guided"
    summary["methods"][primary] = {
        **aggregate(rows[primary]), "failed_records": 0,
        "visual_dependency_failures": summary["costs"][parent]["failed"]}
    summary["primary_method"] = primary
    summary["interpretation"] = (
        "SDR-VLM retains independent numerical decisions when visual repair fails; "
        "parent visual failures separately disclosed. Visual-veto/visual-only failures use zero detections. "
        "Development stations were used for method selection; no point adjustment.")
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "per-station.json", rows)
    write_json(args.out / "activity-errors.json", activity)
    comparisons = ["localized_fusion", "morphology_fusion", "segment_numeric", "rf_split_temporal"]
    if "segment_visual_bounds" in rows:
        comparisons.append("segment_visual_bounds")
    if "pair_raw_repair" in rows:
        comparisons.append("pair_raw_repair")
    pairs = {name: pooled_bootstrap(rows[primary], rows[name])
             for name in comparisons}
    write_json(args.out / "paired-bootstrap.json", pairs)
    lines = ["# GNSS-VLM-EXP06：GNSS位移结构分解与视觉语义核验实验", "",
             summary["interpretation"], "",
             f"主方法SDR-VLM=`{primary}`。派生方法无新增请求，视觉成本按{parent}父方法理解。", "",
             "| 方法 | 类型日F1 | 类型事件F1≥.5 | 通用日F1 | 尖峰 | 阶跃严格 | 暂态 | 活动 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, metrics in summary["methods"].items():
        types = metrics.get("types", {})
        values = [metrics.get("types_daily_macro_f1"),
                  metrics.get("types_event_macro_f1", {}).get("iou_0.5"),
                  metrics["generic_daily_micro"]["f1"],
                  *[types.get(kind, {}).get("daily_micro", {}).get("f1") for kind in EVENT_TYPES]]
        label = method
        if method in ("pair_focus", "pair_raw"):
            label += "（配对子任务输出）"
            values[:4] = [None] * 4
            values[-1] = None
        lines.append(f"| {label} | " + " | ".join(
            f"{value:.4f}" if value is not None else "—" for value in values) + " |")
    lines.extend(["", f"父视觉失败：当前主方法输入{summary['costs'][parent]['failed']}次，"
                  f"旧完整提示{summary['costs']['guided_fusion']['failed']}次，"
                  f"纯视觉{summary['costs']['visual_only']['failed']}次；全部保留，无重试。"])
    (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copyfile(Path(__file__), args.out / "summarize-exp06-source.py")
    print({"summary": str(args.out / "summary.json"), "stations": summary["unique_stations"]})
    print({name: metrics.get("types_daily_macro_f1")
           for name, metrics in summary["methods"].items()})


if __name__ == "__main__":
    main()
