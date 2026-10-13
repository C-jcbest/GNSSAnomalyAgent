"""Report all EXP04 endpoints with identical scoring and paired station comparisons."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_exp02 import activity_errors

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import aggregate
from gnss_research.schema import EVENT_TYPES

PRIMARY = "morphology_fusion"
ORDER = ["random_forest", "rf_balanced", "extra_trees", "isolation_forest",
         "rf_split_temporal", "visual_only", "guided_fusion", "boundary_numeric_points",
         "ranked_boundary_fusion", "morphology_fixed_activity", PRIMARY,
         "guided_no_local", "guided_no_scores", "morphology_no_local", "morphology_no_scores"]


def metric_row(method, result):
    row = {"method": method, "stations": result["stations"],
           "failures": result["failed_records"],
           "generic_daily_f1": result["generic_daily_micro"]["f1"],
           "type_macro_daily_f1": result.get("types_daily_macro_f1"),
           "type_macro_event_f1_iou_05": result.get("types_event_macro_f1", {}).get("iou_0.5"),
           "type_macro_event_f1_iou_03": result.get("types_event_macro_f1", {}).get("iou_0.3"),
           "normal_false_runs_per_station_year": result[
               "false_alarm_runs_per_normal_station_year_controls"]}
    for kind, values in result.get("types", {}).items():
        for metric in ("f1", "precision", "recall", "tp", "fp", "fn"):
            row[f"{kind}_daily_{metric}"] = values["daily_micro"][metric]
        for threshold, metrics in values["events_micro"].items():
            row[f"{kind}_event_{threshold}_f1"] = metrics["f1"]
    return row


def paired_macro(groups, left, right, event=False):
    rng = np.random.default_rng(2026101301)
    sampled = {name: np.zeros((2000, 4, 3)) for name in (left, right)}
    totals = {name: np.zeros((1, 4, 3)) for name in (left, right)}
    for group in groups:
        count = len(group[left])
        indices = rng.integers(0, count, (2000, count))
        for name in (left, right):
            rows = []
            for station in group[name]:
                values = [station["types"][kind]["events"]["iou_0.5"] if event else
                          station["types"][kind]["daily"] for kind in EVENT_TYPES]
                rows.append([[value[key] for key in ("tp", "fp", "fn")] for value in values])
            matrix = np.asarray(rows)
            sampled[name] += matrix[indices].sum(axis=1)
            totals[name] += matrix.sum(axis=0)

    def macro(counts):
        tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
        denominator = 2 * tp + fp + fn
        return np.divide(2 * tp, denominator, out=np.zeros_like(tp),
                         where=denominator > 0).mean(axis=1)

    delta = macro(sampled[left]) - macro(sampled[right])
    return {"difference": float(macro(totals[left])[0] - macro(totals[right])[0]),
            "percentile_ci_95": np.quantile(delta, [.025, .975]).tolist(),
            "repeats": 2000, "seed": 2026101301,
            "scope": "paired stations resampled within run; conditional on fitted models"}


def table(rows):
    lines = ["| 方法 | 四类等权日F1 | 四类等权事件F1≥.5 | 通用日F1 | 尖峰日F1 | 阶跃日F1 | 暂态日F1 | 形变日F1 | 失败站 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    keys = ("type_macro_daily_f1", "type_macro_event_f1_iou_05", "generic_daily_f1",
            "spike_daily_f1", "step_daily_f1", "transient_daily_f1", "deformation_daily_f1")
    for row in rows:
        values = ["—" if row.get(key) is None else f"{row[key]:.4f}" for key in keys]
        lines.append(f"| {row['method']} | " + " | ".join(values)
                     + f" | {row['failures']}/{row['stations']} |")
    return lines


def plot(rows, out, stations):
    selected = [row for row in rows if row["method"] in (
        "rf_balanced", "rf_split_temporal", "visual_only", "ranked_boundary_fusion", PRIMARY)]
    labels = {"rf_balanced": "Balanced RF", "rf_split_temporal": "Temporal RF",
              "visual_only": "Visual only", "ranked_boundary_fusion": "EXP03 ranked fusion",
              PRIMARY: "EXP04 morphology fusion"}
    colors = ["#7b8b9d", "#3971a5", "#cb9a36", "#a66ca1", "#19846c"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    keys = [("type_macro_daily_f1", "Daily macro"),
            ("type_macro_event_f1_iou_05", "Event macro\nIoU >= 0.5")]
    width = .15
    for index, row in enumerate(selected):
        offset = (index - (len(selected) - 1) / 2) * width
        axes[0].bar(np.arange(2) + offset, [row[key] for key, _ in keys], width,
                    color=colors[index], label=labels[row["method"]])
        axes[1].bar(np.arange(4) + offset,
                    [row[f"{kind}_daily_f1"] for kind in EVENT_TYPES], width,
                    color=colors[index])
    axes[0].set_xticks(np.arange(2), [label for _, label in keys])
    axes[1].set_xticks(np.arange(4), [kind.capitalize() for kind in EVENT_TYPES])
    for ax in axes:
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 (no point adjustment)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", fontsize=7)
    axes[0].set_title("Equal weight across four morphologies")
    axes[1].set_title("Daily F1 by morphology")
    fig.suptitle(f"GNSS-VLM-EXP04 | {stations} simulated stations", fontsize=13)
    fig.savefig(out / "comparison.png", dpi=200)
    fig.savefig(out / "comparison.svg")
    plt.close(fig)


def report(runs, out, scope):
    configs = [read_json(run / "registration.json") for run in runs]
    groups = [read_json(run / "evaluation/per-station.json") for run in runs]
    summaries = [read_json(run / "evaluation/summary.json") for run in runs]
    out.mkdir(parents=True, exist_ok=False)
    receipts, per_run, errors, seen = [], [], [], set()
    for run, config, summary in zip(runs, configs, summaries):
        if not (run / "execution-complete.json").exists():
            raise ValueError("only completed runs can enter this report")
        dataset = Path(config["dataset"])
        manifest = read_json(dataset / "manifest.json")
        for station in config["station_ids"]:
            key = (manifest["seed"], config["split"], station)
            if key in seen:
                raise ValueError("duplicate station in combined report")
            seen.add(key)
        requests = sum(read_json(path).get("request_count", 0)
                       for path in (run / "predictions").glob("*/*/result.json"))
        receipts.append({"run": str(run.resolve()), "split": config["split"],
                         "dataset_seed": manifest["seed"], "request_attempts": requests,
                         "registration_sha256": sha256(run / "registration.json"),
                         "evaluation_sha256": sha256(run / "evaluation/summary.json"),
                         "same_source": config["source_hashes"] == configs[0]["source_hashes"]})
        for name in ORDER:
            if name not in summary["methods"]:
                continue
            values = summary["methods"][name]
            per_run.append({"run": run.name, **metric_row(name, values)})
            if "types" in values:
                errors.append({"run": run.name, "method": name,
                               **activity_errors(run, dataset, name)})
    common = set.intersection(*(set(group) for group in groups))
    combined = {}
    strata = {}
    for method in ORDER:
        if method not in common:
            continue
        combined[method] = aggregate([row for group in groups for row in group[method]])
        combined[method]["failed_records"] = sum(
            summary["methods"][method]["failed_records"] for summary in summaries)
        strata[method] = {}
        by_scenario = {}
        for config, group in zip(configs, groups):
            dataset = Path(config["dataset"])
            records = {r["station_id"]: r for r in read_json(dataset / "manifest.json")["records"]}
            for station_id, row in zip(config["station_ids"], group[method]):
                scenario = read_json(dataset / records[station_id]["truth"])["scenario"]
                by_scenario.setdefault(scenario, []).append(row)
        strata[method] = {name: aggregate(rows) for name, rows in by_scenario.items()}
    rows = [metric_row(method, values) for method, values in combined.items()]
    paired = {method: {"daily_macro": paired_macro(groups, PRIMARY, method),
                       "event_macro_iou_05": paired_macro(groups, PRIMARY, method, event=True)}
              for method, values in combined.items() if "types" in values and method != PRIMARY}
    write_json(out / "summary.json", {"experiment": configs[0]["experiment"], "scope": scope,
        "receipts": receipts, "combined": combined, "paired": paired, "scenario_strata": strata})
    write_json(out / "activity-errors.json", errors)
    fields = sorted({key for row in rows for key in row})
    with (out / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "per-run-metrics.json", per_run)
    lines = ["# GNSS-VLM-EXP04：GNSS形态分工视觉数值融合异常识别实验", "",
             f"范围：{scope}。主方法对应 `{PRIMARY}` 行。", "",
             "先汇总每类全站TP/FP/FN得到F1，再对四类等权平均；日F1与事件F1分列。",
             "失败保留且计零检测，无point adjustment。完整precision/recall、事件IoU≥.3、",
             "尖峰/阶跃±3日容差见comparison.csv及summary.json；所有方法使用同一标准。", "",
             *table(rows), "", "## 相同站点方法差值", "",
             "四类等权日F1；种子/批次内站点成对bootstrap 2000次，条件于已拟合模型。", ""]
    for name, values in paired.items():
        delta = values["daily_macro"]
        low, high = delta["percentile_ci_95"]
        lines.append(f"- 主方法 − {name}：{delta['difference']:+.4f}，95%CI [{low:.4f}, {high:.4f}]。")
    lines.extend(["", "## 分批结果与消融", "", *table(per_run), "",
                  "分批身份按per-run-metrics.json的run字段读取。合并表只纳入各批共同方法；",
                  "主批信息消融不假充完整双种子样本。no_scores仍使用数值选窗。", "",
                  "## 实现身份与解释", "",
                  "RF/ExtraTrees/IsolationForest是sklearn库实现；balanced、特征与时间解码是已披露的",
                  "GNSS适配。visual_only是本项目同模型纯视觉对照；guided与两代融合是本项目方法。",
                  "上述方法不冒称VisualTimeAnomaly、VLM4TS或TAMA的完整源码复现。",
                  "作者原提示/API/原解析与GNSS适配的已执行对照见EXP03报告，未混入此批新站数值。", "",
                  "![四类指标对照](comparison.png)", ""])
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    plot(rows, out, len(seen))
    source = out / "analysis-source"
    source.mkdir()
    for path in (Path(__file__), Path(__file__).with_name("report_exp02.py")):
        shutil.copyfile(path, source / path.name)
    print({"report": str(out), "stations": len(seen), "requests": sum(
        item["request_attempts"] for item in receipts)})
    for row in rows:
        print({key: row.get(key) for key in ("method", "type_macro_daily_f1",
              "type_macro_event_f1_iou_05", "spike_daily_f1", "step_daily_f1", "failures")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    report(args.runs, args.out, args.scope)
