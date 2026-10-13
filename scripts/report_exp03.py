"""Report frozen EXP03 runs and paired training-scale controls without changing outputs."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from report_exp02 import activity_errors

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import aggregate, counts_metrics


def difference(groups, left, right, kind="deformation"):
    rng = np.random.default_rng(2026101206)
    draws = {name: np.zeros((2000, 3)) for name in (left, right)}
    totals = {name: np.zeros(3) for name in (left, right)}
    for group in groups:
        samples = rng.integers(0, len(group[left]), (2000, len(group[left])))
        for name in (left, right):
            counts = np.array([[row["types"][kind]["daily"][key]
                                for key in ("tp", "fp", "fn")] for row in group[name]])
            draws[name] += counts[samples].sum(axis=1)
            totals[name] += counts.sum(axis=0)

    def f1(counts):
        tp, fp, fn = counts.T
        divisor = 2 * tp + fp + fn
        return np.divide(2 * tp, divisor, out=np.zeros_like(tp, dtype=float), where=divisor > 0)

    return {"endpoint": f"{kind} daily micro-F1 difference", "repeats": 2000,
            "difference": float(f1(totals[left][None])[0] - f1(totals[right][None])[0]),
            "percentile_ci_95": np.quantile(f1(draws[left]) - f1(draws[right]), [.025, .975]).tolist(),
            "scope": "stations resampled within dataset seed; conditional on fitted models"}


def report(runs, controls, out, scope):
    out.mkdir(parents=True, exist_ok=False)
    configs = [read_json(run / "registration.json") for run in runs]
    groups = [read_json(run / "evaluation/per-station.json") for run in runs]
    summaries = [read_json(run / "evaluation/summary.json")["methods"] for run in runs]
    identity = configs[0].get("experiment", {})
    primary = identity.get("primary_method", "boundary_fusion")
    rows = []
    errors = []
    receipts = []
    for run, config, group, summary in zip(runs, configs, groups, summaries):
        dataset = Path(config["dataset"])
        manifest = read_json(dataset / "manifest.json")
        training = read_json(run / "training.json")["training_stations"]
        for method, values in summary.items():
            row = {"run": run.name, "dataset_seed": manifest["seed"],
                   "training_stations": len(training), "method": method,
                   "stations": values["stations"], "failures": values["failed_records"],
                   "generic_f1": values["generic_daily_micro"]["f1"],
                   "normal_false_runs_per_station_year": values[
                       "false_alarm_runs_per_normal_station_year_controls"]}
            for kind, metrics in values.get("types", {}).items():
                row[kind + "_f1"] = metrics["daily_micro"]["f1"]
                row[kind + "_event_f1"] = metrics["events_micro"]["iou_0.5"]["f1"]
                if kind in ("spike", "step"):
                    counts = [sum(station["types"][kind]["events"][
                        "tolerance_3_days_secondary"][key] for station in group[method])
                              for key in ("tp", "fp", "fn")]
                    row[kind + "_tolerance3_f1"] = counts_metrics(*counts)["f1"]
            rows.append(row)
            if "types" in values:
                errors.append({"run": run.name, "method": method,
                               **activity_errors(run, dataset, method)})
        receipts.append({"run": str(run.resolve()), "registration_sha256": sha256(
            run / "registration.json"), "summary_sha256": sha256(run / "evaluation/summary.json"),
            "source_hashes_match": config["source_hashes"] == configs[0]["source_hashes"],
            "dataset_seed": manifest["seed"], "training_stations": len(training)})
    common = set.intersection(*(set(group) for group in groups))
    combined = {method: aggregate([row for group in groups for row in group[method]])
                for method in sorted(common)}
    paired = {}
    if primary in common:
        for baseline in ("anchored_fusion", "boundary_fusion", "guided_fusion", "rf_split_temporal",
                         "random_forest", "rf_balanced", "fusion", "ranked_no_local",
                         "ranked_no_scores", "boundary_numeric_points"):
            if baseline in common and baseline != primary:
                paired[baseline] = difference(groups, primary, baseline)
    scaling = {}
    if controls:
        if len(controls) != len(runs):
            raise ValueError("one training-scale control per run required")
        scaled_groups = []
        fusion_groups = []
        for run, control, group, config in zip(runs, controls, groups, configs):
            small_config = read_json(control / "registration.json")
            small_group = read_json(control / "evaluation/per-station.json")
            if config["station_ids"] != small_config["station_ids"]:
                raise ValueError("training controls must use identical ordered stations")
            records = read_json(Path(config["dataset"]) / "manifest.json")["records"]
            small_records = read_json(Path(small_config["dataset"]) / "manifest.json")["records"]
            selected = set(config["station_ids"])
            if {r["station_id"]: r["observation_sha256"] for r in records
                if r["station_id"] in selected} != {
                    r["station_id"]: r["observation_sha256"] for r in small_records
                    if r["station_id"] in selected}:
                raise ValueError("training controls have different observations")
            scaled_groups.append({"train96": group["rf_split_temporal"],
                                  "train24": small_group["rf_split_temporal"]})
            if primary in group and primary in small_group:
                fusion_groups.append({"train96": group[primary], "train24": small_group[primary]})
        scaling = {"paired_deformation": difference(scaled_groups, "train96", "train24"),
                   "controls": [str(p.resolve()) for p in controls]}
        for label in ("train24", "train96"):
            scaling[label] = aggregate([row for group in scaled_groups for row in group[label]])
        if len(fusion_groups) == len(runs):
            scaling["primary_fusion"] = {"paired_deformation": difference(
                fusion_groups, "train96", "train24"), **{label: aggregate(
                    [row for group in fusion_groups for row in group[label]])
                    for label in ("train24", "train96")},
                "limitation": "training changes candidate inputs; visual calls independently sampled"}
    write_json(out / "summary.json", {"experiment": configs[0].get("experiment"),
        "scope": scope, "receipts": receipts, "combined": combined,
        "paired_deformation": paired, "training_scale": scaling})
    write_json(out / "activity-errors.json", errors)
    fields = sorted({key for row in rows for key in row})
    with (out / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {identity.get('experiment_id')}-{identity.get('protocol_version')}：{identity.get('name')}", "",
             f"范围：{scope}。主方法{primary}，旧主方法anchored_fusion；相同视觉响应派生，非独立调用。",
             "", "| 批次 | 训练站 | 方法 | 通用日F1 | 形变日F1 | 形变事件F1≥.5 | 尖峰日F1 | 阶跃日F1 | 失败站 |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        metrics = [f"{row[key]:.4f}" if key in row else "—" for key in
                   ("generic_f1", "deformation_f1", "deformation_event_f1", "spike_f1", "step_f1")]
        lines.append(f"| {row['run']} | {row['training_stations']} | {row['method']} | "
                     + " | ".join(metrics) + f" | {row['failures']}/{row['stations']} |")
    lines.extend(["", "## 成对差值", "", "形变日micro-F1，种子内站点bootstrap 2000次；条件于已拟合模型。", ""])
    for baseline, values in paired.items():
        low, high = values["percentile_ci_95"]
        lines.append(f"- {primary} − {baseline}：{values['difference']:+.4f}，95%CI [{low:.4f}, {high:.4f}]。")
    if len(runs) > 1:
        lines.extend(["", "## 两种子合并（共同方法）", "",
                      "| 方法 | 形变日F1 | 形变事件F1≥.5 | 尖峰日F1 | 阶跃日F1 | 暂态日F1 |",
                      "|---|---:|---:|---:|---:|---:|"])
        for method, values in combined.items():
            if "types" not in values:
                continue
            types = values["types"]
            numbers = [types["deformation"]["daily_micro"]["f1"],
                       types["deformation"]["events_micro"]["iou_0.5"]["f1"],
                       *[types[kind]["daily_micro"]["f1"] for kind in ("spike", "step", "transient")]]
            lines.append(f"| {method} | " + " | ".join(f"{v:.4f}" for v in numbers) + " |")
    if scaling:
        lines.extend(["", "## 同站训练规模对照", ""])
        for label in ("train24", "train96"):
            values = scaling[label]
            lines.append(f"- {label} rf_split_temporal：形变日F1 "
                         f"{values['types']['deformation']['daily_micro']['f1']:.4f}；阶跃日F1 "
                         f"{values['types']['step']['daily_micro']['f1']:.4f}。")
        if "primary_fusion" in scaling:
            for label in ("train24", "train96"):
                values = scaling["primary_fusion"][label]
                lines.append(f"- {label} {primary}：形变日F1 "
                             f"{values['types']['deformation']['daily_micro']['f1']:.4f}；阶跃日F1 "
                             f"{values['types']['step']['daily_micro']['f1']:.4f}。")
            lines.append("训练规模改变候选输入并重新调用视觉模型，差值也包含响应随机性；同响应融合消融另列。")
    lines.extend(["", "全部四类型、±3日辅助及正常误报见CSV；缺测排除，无point adjustment。",
                  "开发验证不能充当未见测试；同分布仿真不能证明现场效果。完整论文复现或未运行方法不填成绩。"])
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    source = out / "analysis-source"
    source.mkdir()
    for path in (Path(__file__), Path(__file__).with_name("report_exp02.py")):
        shutil.copyfile(path, source / path.name)
    print(out / "results.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--train24-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    report(args.runs, args.train24_runs, args.out, args.scope)
