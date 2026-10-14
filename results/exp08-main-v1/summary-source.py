"""Apply the frozen ISP-VLM decision to both new test seeds together."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from summarize_exp06 import pooled_bootstrap

from gnss_research.files import read_json, write_json
from gnss_research.metrics import aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs=2, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/exp08-main-v1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = read_json(Path("configs/experiment-exp08-test.json"))
    candidate = protocol["primary_method"]
    methods = [candidate, "pair_focus_repair", "binary_rf_default_numeric", "segment_numeric",
               "rf_split_temporal", "rf_balanced"]
    pooled = {method: [] for method in methods}
    sources = []
    costs = {method: {"requests": 0, "failures": 0} for method in (candidate, "pair_focus_repair")}
    for report in args.reports:
        summary = read_json(report / "summary.json")
        rows = read_json(report / "per-station.json")
        if summary["split"] != "test":
            raise ValueError("formal test reports required")
        for method in methods:
            pooled[method].extend(rows[method])
        source = {"seed": summary["dataset_seed"], "report": str(report.resolve()),
                  "station_ids": summary["station_ids"],
                  "candidate_f1": summary["methods"][candidate]["types_daily_macro_f1"],
                  "stable_f1": summary["methods"]["pair_focus_repair"]["types_daily_macro_f1"]}
        source["difference"] = source["candidate_f1"] - source["stable_f1"]
        sources.append(source)
        for method in costs:
            for key in costs[method]:
                costs[method][key] += summary["costs"][method][key]
    if [source["seed"] for source in sources] != protocol["test_seeds"]:
        raise ValueError("every frozen seed must appear exactly once in registered order")
    summary = {method: aggregate(rows) for method, rows in pooled.items()}
    stable, selected = summary["pair_focus_repair"], summary[candidate]
    difference = selected["types_daily_macro_f1"] - stable["types_daily_macro_f1"]
    worst = min(selected["types"][kind]["daily_micro"]["f1"]
                - stable["types"][kind]["daily_micro"]["f1"] for kind in stable["types"])
    accepted = (difference >= .02 and worst >= -.05
                and all(source["difference"] > 0 for source in sources))
    bootstrap = pooled_bootstrap(pooled[candidate], pooled["pair_focus_repair"], 2026101411)
    result = {"experiment_id": "GNSS-VLM-EXP08", "name": protocol["name"], "sources": sources,
              "unique_stations": 48, "methods": summary, "costs": costs,
              "upgrade_decision": {"accepted": accepted, "difference": difference,
                                   "worst_type_difference": worst,
                                   "seed_differences": [source["difference"] for source in sources]},
              "paired_bootstrap": bootstrap}
    write_json(args.out / "summary.json", result)
    write_json(args.out / "per-station.json", pooled)
    shutil.copyfile(__file__, args.out / "summary-source.py")
    names = {candidate: "ISP-VLM（独立默认RF尖峰+原视觉）", "pair_focus_repair": "原SDR-VLM",
             "binary_rf_default_numeric": "独立默认RF尖峰+分解，无视觉",
             "segment_numeric": "原位移分解，无视觉", "rf_split_temporal": "RF点型/时序分工",
             "rf_balanced": "平衡RF"}
    lines = ["# EXP08：GNSS独立尖峰先验与视觉位移分解融合正式复核", "",
             "两个新种子、48同组测试站；四类等权日F1为主，不做point adjustment。", "",
             "| 方法 | 类型日F1 | 类型事件F1≥0.5 | 通用日F1 | 尖峰 | 阶跃 | 暂态 | 活动 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in methods:
        values = summary[method]
        numbers = [values["types_daily_macro_f1"], values["types_event_macro_f1"]["iou_0.5"],
                   values["generic_daily_micro"]["f1"],
                   *[values["types"][kind]["daily_micro"]["f1"]
                     for kind in ("spike", "step", "transient", "deformation")]]
        lines.append("| " + names[method] + " | " + " | ".join(f"{value:.4f}" for value in numbers) + " |")
    lines.extend(["", "| 新种子 | 原SDR-VLM | ISP-VLM | 差值 |", "|---|---:|---:|---:|"])
    for source in sources:
        lines.append(f"| {source['seed']} | {source['stable_f1']:.4f} | {source['candidate_f1']:.4f} | {source['difference']:+.4f} |")
    lines.extend(["", f"合并主指标增量{difference:+.4f}，最差类型差值{worst:+.4f}。",
                  f"配对站点bootstrap 95%区间{bootstrap['percentile_ci_95']}；不把点估计自动表述成显著。",
                  "", f"决定：**{'达到预设升级条件' if accepted else '未达预设条件，取消集成并保留EXP06'}**。",
                  "条件：合并至少+0.02、两种子方向均为正、任何类型不下降超过0.05。", "",
                  f"实际视觉请求：原版{costs['pair_focus_repair']['requests']}、候选{costs[candidate]['requests']}；",
                  f"失败：原版{costs['pair_focus_repair']['failures']}、候选{costs[candidate]['failures']}，无重试。",
                  "", "不同种子均重新训练；默认RF只指定random_state，不调整类别/分数阈值。",
                  "无视觉消融区分数值新增收益与原视觉收益。近期完整VLM4TS/R-FPOP未运行，不填成绩。", ""])
    (args.out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    order = ["rf_balanced", "rf_split_temporal", "segment_numeric", "binary_rf_default_numeric",
             "pair_focus_repair", candidate]
    labels = ["Balanced RF", "RF split + temporal", "Original structure numeric",
              "Independent spike + structure", "SDR-VLM original", "ISP-VLM candidate"]
    values = [summary[method]["types_daily_macro_f1"] for method in order]
    figure, axis = plt.subplots(figsize=(9, 4.8), dpi=180)
    bars = axis.barh(labels, values, color=["#a2afbd"] * 4 + ["#245c70", "#a66835"])
    for bar, value in zip(bars, values):
        axis.text(value + .008, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    axis.set_xlim(0, 1.03)
    axis.set_xlabel("Four-type pooled daily macro F1 (no point adjustment)")
    axis.set_title("EXP08: two frozen test seeds, 48 paired GNSS stations")
    axis.grid(axis="x", alpha=.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(args.out / "comparison.png")
    figure.savefig(args.out / "comparison.svg")
    plt.close(figure)
    print(result["upgrade_decision"])


if __name__ == "__main__":
    main()
