"""Independent-seed replication summaries; repeated inference is excluded."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import aggregate


def stratified_difference(groups: list[dict], primary: str, baseline: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    primary_counts = np.zeros((2000, 3))
    baseline_counts = np.zeros((2000, 3))
    for group in groups:
        samples = rng.integers(0, len(group[primary]), (2000, len(group[primary])))
        for method, total in ((primary, primary_counts), (baseline, baseline_counts)):
            counts = np.array([[row["types"]["deformation"]["daily"][key]
                                for key in ("tp", "fp", "fn")] for row in group[method]])
            total += counts[samples].sum(axis=1)

    def f1(counts):
        tp, fp, fn = counts.T
        divisor = 2 * tp + fp + fn
        return np.divide(2 * tp, divisor, out=np.zeros(len(tp)), where=divisor > 0)

    difference = f1(primary_counts) - f1(baseline_counts)
    return {"endpoint": "deformation daily micro-F1 difference", "repeats": 2000,
            "strata": "dataset seeds, stations resampled within each seed",
            "percentile_ci_95": np.quantile(difference, [.025, .975]).tolist(),
            "seed": seed, "scope": "conditional on three fitted models, not all training variability"}


def report(runs: list[Path], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    registrations = [read_json(run / "registration.json") for run in runs]
    summaries = [read_json(run / "evaluation/summary.json") for run in runs]
    groups = [read_json(run / "evaluation/per-station.json") for run in runs]
    methods = [method for method in registrations[0]["methods"]
               if all(method in group for group in groups)]
    # Same implementation and training random seed; data seeds and training sets differ.
    if any(config["source_hashes"] != registrations[0]["source_hashes"]
           for config in registrations):
        raise ValueError("replications use different implementations")
    seeds = [config["experiment"]["dataset_seed"] for config in registrations]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed repetition cannot count as independent stations")
    rows = []
    combined = {}
    for method in methods:
        combined[method] = aggregate([row for group in groups for row in group[method]])
        combined[method]["failed_records"] = sum(s["methods"][method]["failed_records"]
                                                 for s in summaries)
        for label, values in [*[(str(seed), summary["methods"][method])
                                for seed, summary in zip(seeds, summaries)],
                              ("pooled_72", combined[method])]:
            row = {"dataset_seed": label, "method": method, "stations": values["stations"],
                   "generic_daily_f1": values["generic_daily_micro"]["f1"],
                   "deformation_daily_f1": values["types"]["deformation"]["daily_micro"]["f1"],
                   "deformation_event_f1_iou_05": values["types"]["deformation"]["events_micro"]["iou_0.5"]["f1"],
                   "spike_daily_f1": values["types"]["spike"]["daily_micro"]["f1"],
                   "step_daily_f1": values["types"]["step"]["daily_micro"]["f1"],
                   "transient_daily_f1": values["types"]["transient"]["daily_micro"]["f1"],
                   "normal_false_alarm_runs_per_station_year": values[
                       "false_alarm_runs_per_normal_station_year_controls"],
                   "failed_records": values["failed_records"]}
            rows.append(row)
    primary = registrations[0]["experiment"]["primary_method"]
    differences = {method: stratified_difference(groups, primary, method, 2026101203)
                   for method in methods if method != primary}
    for method, result in differences.items():
        result["difference"] = (combined[primary]["types"]["deformation"]["daily_micro"]["f1"]
                                - combined[method]["types"]["deformation"]["daily_micro"]["f1"])
    write_json(out / "summary.json", {"dataset_seeds": seeds, "methods": combined,
                                       "paired_stratified_bootstrap": differences})
    with (out / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# GNSS-VLM-EXP02-v3：三个独立种子复核", "",
             "训练/验证/测试各24站，每种子测试24站，合计72个独立模拟测试站；同站推理复测不计入。",
             "主方法anchored_fusion，数值改进rf_split_temporal，自由视觉改写对照guided_fusion，EXP01原融合fusion。",
             "数据种子分别2026101203、2026101204、2026101205；分类器种子固定2026101203。", "",
             "| 种子 | 方法 | 通用日F1 | 形变日F1 | 形变事件F1≥.5 | 尖峰日F1 | 阶跃日F1 | 失败站 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset_seed']} | {row['method']} | {row['generic_daily_f1']:.4f} | "
                     f"{row['deformation_daily_f1']:.4f} | {row['deformation_event_f1_iou_05']:.4f} | "
                     f"{row['spike_daily_f1']:.4f} | {row['step_daily_f1']:.4f} | "
                     f"{row['failed_records']}/{row['stations']} |")
    lines.extend(["", "## 分层成对bootstrap", "",
                  "形变日micro-F1差值，种子内站点重采样2000次；条件于这三组已拟合模型，不估计全部训练模型变异。", ""])
    for method, values in differences.items():
        low, high = values["percentile_ci_95"]
        lines.append(f"- anchored_fusion − {method}：{values['difference']:.4f}，95%区间[{low:.4f}, {high:.4f}]。")
    lines.extend(["", "各信息组消融和作者源码适配仅在主种子完成，见主报告；不能称所有三种子均执行这些对照。",
                  "四类型全部指标、暂态及正常误报见CSV；无point adjustment、无测试阈值调参。失败零检测，派生融合与其视觉对照共用响应，不是独立推理。",
                  "均为同一简单仿真分布，新种子复核不等于跨分布、现场或真实滑坡验证。"])
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "receipt.json", {
        "runs": [str(run.resolve()) for run in runs],
        "registration_sha256": [sha256(run / "registration.json") for run in runs],
        "report_script_sha256": sha256(Path(__file__))})
    print(out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.runs, args.out)
