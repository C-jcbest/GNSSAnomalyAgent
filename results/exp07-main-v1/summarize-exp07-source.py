"""Pool the two frozen NVR tests and apply the predeclared upgrade decision."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from summarize_exp06 import pooled_bootstrap

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import aggregate


def summarize(runs: list[Path], out: Path):
    out.mkdir(parents=True, exist_ok=False)
    protocol = read_json(Path("configs/experiment-exp07-test.json"))
    pooled = {}
    sources = []
    seed_differences = []
    requests = failures = retained_steps = baseline_requests = baseline_failures = 0
    for run in runs:
        if not (run / "execution-completed.json").exists():
            raise ValueError("candidate test must complete before pooling")
        summary = read_json(run / "summary.json")
        rows = read_json(run / "per-station.json")
        for method, values in rows.items():
            pooled.setdefault(method, []).extend(values)
        source = {"seed": summary["dataset_seed"], "stations": len(rows["coexist_repair"]),
                  "run": str(run.resolve()), "summary_sha256": sha256(run / "summary.json"),
                  "registration_sha256": sha256(run / "registration.json"),
                  "candidate_f1": summary["methods"]["coexist_repair"]["types_daily_macro_f1"],
                  "stable_f1": summary["methods"]["pair_focus_repair"]["types_daily_macro_f1"]}
        source["difference"] = source["candidate_f1"] - source["stable_f1"]
        sources.append(source)
        seed_differences.append(source["difference"])
        requests += summary["requests"]
        failures += summary["failures"]
        retained_steps += summary["coexisting_steps_retained"]
        baseline = Path(summary["baseline_run"])
        frozen = read_json(baseline / "registration.json")
        for station in frozen["station_ids"]:
            receipt = read_json(baseline / "predictions/pair_focus" / station / "result.json")
            baseline_requests += receipt.get("request_count", 0)
            baseline_failures += receipt["status"] != "success"
    if [source["seed"] for source in sources] != protocol["test_seeds"]:
        raise ValueError("pool every frozen test seed once, in registered order")
    methods = {method: aggregate(rows) for method, rows in pooled.items()}
    candidate, stable = methods["coexist_repair"], methods["pair_focus_repair"]
    difference = candidate["types_daily_macro_f1"] - stable["types_daily_macro_f1"]
    worst = min(candidate["types"][kind]["daily_micro"]["f1"]
                - stable["types"][kind]["daily_micro"]["f1"] for kind in stable["types"])
    accepted = (difference >= protocol["minimum_absolute_improvement"]
                and all(value > 0 for value in seed_differences)
                and worst >= -protocol["maximum_per_type_degradation"])
    bootstrap = pooled_bootstrap(pooled["coexist_repair"], pooled["pair_focus_repair"], 2026101407)
    result = {"experiment_id": "GNSS-VLM-EXP07", "name": protocol["name"],
              "methods": methods, "sources": sources, "split": "test", "unique_stations": 48,
              "upgrade_decision": {"accepted": accepted, "difference": difference,
                                   "worst_type_difference": worst, "seed_differences": seed_differences,
                                   "minimum_absolute_improvement": .02,
                                   "decision": "upgrade" if accepted else "cancel integration; retain EXP06"},
              "requests": {"candidate": requests, "stable": baseline_requests},
              "failures": {"candidate": failures, "stable": baseline_failures},
              "coexisting_steps_retained": retained_steps, "paired_bootstrap": bootstrap}
    write_json(out / "summary.json", result)
    write_json(out / "per-station.json", pooled)
    shutil.copyfile(__file__, out / "summarize-exp07-source.py")
    lines = ["# EXP07：GNSS非互斥视觉提示与配对修复正式复核", "",
             "2026-10-14；两个冻结新种子、48个同组测试站，四类等权日F1为主，不做point adjustment。", "",
             "| 方法 | 类型日F1 | 类型事件F1≥0.5 | 通用日F1 | 尖峰 | 阶跃 | 暂态 | 活动 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"coexist_repair": "候选SDR-VLM-NVR", "pair_focus_repair": "原SDR-VLM（EXP06原实现）",
             "segment_numeric": "位移结构数值（同原分解）", "rf_split_temporal": "RF点型/时序分工",
             "rf_balanced": "平衡RF"}
    for method, values in methods.items():
        numbers = [values["types_daily_macro_f1"], values["types_event_macro_f1"]["iou_0.5"],
                   values["generic_daily_micro"]["f1"],
                   *[values["types"][kind]["daily_micro"]["f1"]
                     for kind in ("spike", "step", "transient", "deformation")]]
        lines.append("| " + names[method] + " | " + " | ".join(f"{value:.4f}" for value in numbers) + " |")
    lines.extend(["", f"主指标增量：{difference:+.4f}；配对站点bootstrap 95%区间：{bootstrap['percentile_ci_95']}。",
                  "", "| 新种子 | 原SDR-VLM | NVR候选 | 差值 |", "|---|---:|---:|---:|"])
    for source in sources:
        lines.append(f"| {source['seed']} | {source['stable_f1']:.4f} | {source['candidate_f1']:.4f} | {source['difference']:+.4f} |")
    lines.extend(["", f"决定：**{'达到冻结收益条件' if accepted else '取消新方案集成，保留EXP06稳定实现'}**。",
                  "接受条件为合并增量至少0.02、两种子方向均为正、任何类型不下降超过0.05；不是显著性标准。",
                  "", f"实际视觉请求：原版{baseline_requests}、候选{requests}；失败分别{baseline_failures}/{failures}，无自动重试。",
                  f"共存阶跃保留规则实际触发{retained_steps}次。无歧义站明确不请求，独立数值结果保留。",
                  "", "开发负结果见`../exp07-numeric-v1/summary.json`，修正版数值见`../exp07-numeric-v2/summary.json`；",
                  "24站视觉开发见`../exp07-development-v1/summary.json`。不同种子成绩不得当作同站提升。",
                  "本轮没有运行完整VLM4TS或R-FPOP，不据本表宣称击败它们。"])
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_order = ["rf_balanced", "rf_split_temporal", "segment_numeric", "pair_focus_repair", "coexist_repair"]
    figure, axis = plt.subplots(figsize=(9, 4.4), dpi=180)
    values = [methods[method]["types_daily_macro_f1"] for method in plot_order]
    labels = ["Balanced RF", "RF split+temporal", "Structure numeric", "SDR-VLM original", "NVR candidate"]
    bars = axis.barh(labels, values, color=["#98a7b8"] * 3 + ["#23576c", "#ad7142"])
    for bar, value in zip(bars, values):
        axis.text(value + .009, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Four-type pooled daily macro F1 (no point adjustment)")
    axis.set_title("EXP07: two frozen test seeds, 48 paired GNSS stations")
    axis.grid(axis="x", alpha=.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(out / "comparison.png")
    figure.savefig(out / "comparison.svg")
    plt.close(figure)
    print(result["upgrade_decision"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs=2, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/exp07-main-v1"))
    args = parser.parse_args()
    summarize(args.runs, args.out)


if __name__ == "__main__":
    main()
