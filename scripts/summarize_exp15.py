"""Same-station pooled endpoints for the two prospectively specified test seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

from summarize_exp06 import pooled_bootstrap

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs=2, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summaries = [read_json(p / "summary.json") for p in args.results]
    assert {s["seed"] for s in summaries} == {2026101503, 2026101504}
    assert all(s["split"] == "test" and len(s["station_ids"]) == 24 for s in summaries)
    reports = [read_json(p / "per-station.json") for p in args.results]
    rows = {method: [row for report in reports for row in report[method]] for method in reports[0]}
    methods = {method: aggregate(values) for method, values in rows.items()}
    primary = methods["vector_visual"]
    differences = {method: primary["types_daily_macro_f1"] - values["types_daily_macro_f1"]
                   for method, values in methods.items() if method != "vector_visual"}
    per_seed = {str(s["seed"]): s["methods"]["vector_visual"]["types_daily_macro_f1"]
                - s["methods"]["pair_focus_repair"]["types_daily_macro_f1"] for s in summaries}
    worst = min(primary["types"][kind]["daily_micro"]["f1"]
                - methods["pair_focus_repair"]["types"][kind]["daily_micro"]["f1"]
                for kind in primary["types"])
    accepted = (differences["pair_focus_repair"] >= .015 and differences["segment_numeric"] >= .015
                and all(value > 0 for value in per_seed.values()) and worst >= -.03
                and max(differences["vector_text"], differences["vector_nearest"]) > 0)
    summary = {"experiment_id": "GNSS-VLM-EXP15", "name": "GNSS局部跳变向量可视化核验正式复核",
        "primary_method": "vector_visual", "model": "qwen3-vl-plus", "station_count": 48,
        "methods": methods, "differences_vs_primary": differences,
        "per_seed_gain_vs_original": per_seed, "worst_type_difference_vs_original": worst,
        "eligible_for_adoption": accepted,
        "bootstrap": {method: pooled_bootstrap(rows["vector_visual"], rows[method], 2026101503)
                      for method in ("pair_focus_repair", "segment_numeric", "vector_text", "vector_nearest")},
        "sources": [{"path": str(p), "summary_sha256": sha256(p / "summary.json"),
                     "seed": s["seed"], "station_ids": s["station_ids"], "costs": s["costs"]}
                    for p, s in zip(args.results, summaries)]}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "per-station.json", rows)
    lines = ["# EXP15：VEC-VLM两个新种子48站正式复核", "",
             "主方法vector_visual，四类等权日F1；无point adjustment。", "",
             "| 方法 | 四类日F1 | 四类事件F1≥.5 | 通用日F1 | 尖峰 | 阶跃 | 暂态 | 活动 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, values in methods.items():
        numbers = [values["types_daily_macro_f1"], values["types_event_macro_f1"]["iou_0.5"],
                   values["generic_daily_micro"]["f1"],
                   *[values["types"][kind]["daily_micro"]["f1"] for kind in ("spike", "step", "transient", "deformation")]]
        lines.append("| " + method + " | " + " | ".join(f"{n:.4f}" for n in numbers) + " |")
    lines.extend(["", f"预设采纳条件通过：{accepted}；两种子对原SDR差值：{per_seed}。", "",
                  "| 主方法减对照 | 四类日F1差值 | 描述性95%区间 |", "|---|---:|---|"])
    for method, bootstrap in summary["bootstrap"].items():
        low, high = bootstrap["percentile_ci_95"]
        lines.append(f"| {method} | {bootstrap['difference']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
    lines.extend(["", "vector_text提供相同局部统计的数字文字；vector_nearest为公开未调参数值规则。",
                  "图像版还包含原始曲线，因此与文字版的差异不能只归因条形图。",
                  "重采样按站、共享训练模型，区间未纠正开发多次试验；未将其当强显著性证据。", ""])
    (args.out / "results.md").write_text("\n".join(lines), encoding="utf8")
    print({"eligible_for_adoption": accepted, "differences": differences, "per_seed": per_seed})


if __name__ == "__main__":
    main()
