"""Build transparent tables and a standalone figure from saved experiments only."""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_research.files import read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def costs(run: Path) -> dict:
    calls = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    seconds = 0.0
    errors = Counter()
    for path in run.glob("predictions/*/*/result.json"):
        receipt = read_json(path)
        calls += receipt.get("request_count", 0)
        subrequests = receipt.get("subrequests", [receipt])
        for request in subrequests:
            seconds += request.get("elapsed_seconds", 0)
            for key in usage:
                usage[key] += request.get("usage", {}).get(key, 0)
            if request.get("status") != "success" and request.get("request_count", 0):
                errors[request.get("error_type", "unknown")] += 1
    # The one-off capability probe was intentionally outside a benchmark run.
    if (run / "receipt.json").exists():
        probe = read_json(run / "receipt.json")
        calls += probe.get("request_count", 0)
        seconds += probe.get("elapsed_seconds", 0)
        for key in usage:
            usage[key] += probe.get("usage", {}).get(key, 0)
    return {"requests": calls, **usage, "summed_inference_seconds": seconds,
            "failed_request_error_types": dict(errors),
            "timing_note": "sum of request/record times, not concurrent wall time or billing"}


def table_rows(summary: dict) -> list[dict]:
    rows = []
    for method, values in summary["methods"].items():
        activity = None
        if method != "isolation_forest" and not method.startswith("visualtime_gnss"):
            activity = values.get("types", {}).get("deformation")
        rows.append({"method": method,
                     "generic_daily_f1": values["generic_daily_micro"]["f1"],
                     "deformation_daily_f1": activity["daily_micro"]["f1"] if activity else None,
                     "deformation_event_f1_iou_05": activity["events_micro"]["iou_0.5"]["f1"]
                     if activity else None,
                     "normal_false_alarm_runs_per_station_year": values[
                         "false_alarm_runs_per_normal_station_year_controls"],
                     "failed_records": values["failed_records"],
                     "not_run_records": values["not_run_records"]})
    return rows


def format_number(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def report(run: Path, repeat: Path | None, out: Path, supplement: bool,
           author_analysis: Path | None = None, evaluation: str = "evaluation",
           author_repeat_analysis: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=False)
    summary = read_json(run / evaluation / "summary.json")
    rows = table_rows(summary)
    if author_analysis is not None:
        author = read_json(author_analysis / "summary.json")
        rows.extend(table_rows(author))
        summary["paired_comparisons"].update(author.get("paired_comparisons", {}))
    if author_repeat_analysis is not None:
        repeated_author = read_json(author_repeat_analysis / "summary.json")
        repeated_author_config = read_json(author_repeat_analysis.parent / "registration.json")
        original_config = read_json(run / "registration.json")
        for key in ("station_ids", "manifest_sha256", "model"):
            if original_config[key] != repeated_author_config[key]:
                raise ValueError(f"author repeat differs: {key}")
        for row in table_rows(repeated_author):
            row["method"] += "_repeat"
            rows.append(row)
    registration = read_json(run / "registration.json")
    receipt = {"main": costs(run), "run": str(run.resolve()),
               "registration_sha256": sha256(run / "registration.json"),
               "summary_sha256": sha256(run / evaluation / "summary.json"),
               "evaluation_directory": evaluation,
               "report_script_sha256": sha256(Path(__file__))}
    if author_repeat_analysis is not None:
        receipt["author_repeat"] = costs(author_repeat_analysis.parent)
        receipt["author_repeat_analysis_sha256"] = sha256(author_repeat_analysis / "summary.json")
    receipt["all_recorded_runs"] = {path.name: costs(path)
                                    for path in sorted((ROOT / "runs").iterdir()) if path.is_dir()}
    receipt["total_recorded_requests"] = sum(
        item["requests"] for item in receipt["all_recorded_runs"].values())
    lines = ["# 新分支实际实验结果", "", "## Material Passport", "",
             "- Origin Skill: academic-research-suite / experiment-agent",
             "- Origin Mode: run（用户授权）", "- Origin Date: 2026-10-12",
             "- Verification Status: 保存响应的确定性离线评分；不是现场验证或完整论文复现。",
             "- Version Label: gnss-vlm-research-v4", "", "## 范围", "",
             f"模型：`{registration['model']}`。{len(registration['station_ids'])}独立测试站；"
             "训练与验证站分别独立。无统一物理阈值、无测试阈值扫描、无point adjustment。",
             "所有表格由脚本读取保存结果生成。作者源码只提供通用异常，形变列留空；"
             "rf_balanced是特别标注的class_weight调整，默认RF单列。", "",
             "| 方法 | 通用异常日F1 | 持续形变日F1 | 形变事件F1 IoU≥0.5 | 正常站误报段/站年 | 失败/未执行 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["method"], format_number(row["generic_daily_f1"]),
                  format_number(row["deformation_daily_f1"]),
                  format_number(row["deformation_event_f1_iou_05"]),
                  format_number(row["normal_false_alarm_runs_per_station_year"]),
                  f"{row['failed_records']}/{row['not_run_records']}"]
        lines.append("| " + " | ".join(values) + " |")
    if author_analysis is not None:
        lines.extend(["", "`visualtime_gnss`为原提示＋GNSS输入＋本项目严格解析；"
                      "`visualtime_gnss_author_parser`为同批保存响应调用作者原解析和向量转换。"
                      "后者是主实验登记后的单列离线分析，无新增请求、提示修改或阈值调优。"
                      "原代码的列表截取、截断补括号、索引跳过/裁剪行为均保留；"
                      "两列都要求六个子请求全部成功，否则整站记失败和零检测。"
                      "具体源文件哈希与每个子响应解析结果保存在author-parser-analysis中。"])
    if author_repeat_analysis is not None:
        lines.extend(["", "作者源码复测列来自同24站的另一批144次真实调用，其请求与首批逐一核对。"
                      "主批有43次连接错误及3次读取错误，因此复测用于区分接口可靠性与检测表现。"
                      "原批、失败与复测同时保留，没有以高分批次替换首批，也没有增加独立站点数。"
                      "新批训练过程和源码快照独立保存；提示、默认参数和输入未改变。"])
    lines.extend(["", "## 分类型表现", "",
                  "逐日P/R/F1包含失败站的漏检；事件F1为一对一匹配、IoU≥0.5。"
                  "尖峰/阶跃真值是发生日期，永久阶跃后的平台不计为每日异常。", "",
                  "| 方法 | 类型 | 日P | 日R | 日F1 | 事件F1 |",
                  "|---|---|---:|---:|---:|---:|"])
    for method, values in summary["methods"].items():
        if method in ("isolation_forest", "visualtime_gnss"):
            continue
        for kind, metrics in values.get("types", {}).items():
            daily = metrics["daily_micro"]
            event = metrics["events_micro"]["iou_0.5"]
            lines.append("| " + " | ".join([method, kind,
                format_number(daily["precision"]), format_number(daily["recall"]),
                format_number(daily["f1"]), format_number(event["f1"])]) + " |")
    lines.extend(["", "## 配对不确定性", "",
                  "以下为整站macro-F1差值的2000次成对bootstrap，和上表micro-F1不是同一统计量。"
                  "区间只描述该仿真样本，不证明现场有效性或首次创新。", "",
                  "| 比较 | 差值 | 95%百分位区间 |", "|---|---:|---:|"])
    for name, values in summary["paired_comparisons"].items():
        low, high = values["percentile_ci_95"]
        lines.append(f"| {name} | {values['difference']:.4f} | [{low:.4f}, {high:.4f}] |")
    if repeat is not None:
        repeated = read_json(repeat / evaluation / "summary.json")
        repeated_config = read_json(repeat / "registration.json")
        if registration["station_ids"] != repeated_config["station_ids"]:
            raise ValueError("repeat stations differ")
        if registration["manifest_sha256"] != repeated_config["manifest_sha256"]:
            raise ValueError("repeat dataset differs")
        receipt["repeat"] = costs(repeat)
        lines.extend(["", "## 同测试集独立响应复测", "",
                      "仅重复推理，不增加独立样本。RF确定性重放，视觉再次真实调用；不选择高分一遍。", "",
                      "| 方法 | 首遍通用F1 | 第二遍通用F1 | 首遍形变F1 | 第二遍形变F1 |",
                      "|---|---:|---:|---:|---:|"])
        repeated_rows = table_rows(repeated)
        for row in repeated_rows:
            first = next(item for item in rows if item["method"] == row["method"])
            lines.append("| " + " | ".join([row["method"],
                format_number(first["generic_daily_f1"]), format_number(row["generic_daily_f1"]),
                format_number(first["deformation_daily_f1"]),
                format_number(row["deformation_daily_f1"])]) + " |")
        lines.extend(["", "第二遍失败站数：" + "；".join(
            f"{name}={value['failed_records']}/{value['stations']}"
            for name, value in repeated["methods"].items()) + "。"])
    lines.extend(["", "## 代价与限制", "", "```json", json.dumps(receipt, indent=2), "```", "",
                  "背景正常前缀是仿真实验提供的条件，真实部署需要独立确认；这是重要适用性限制。"
                  "只有4个纯正常测试站与8个有形变测试站，事件统计较粗，不能以大量日点数冒充独立样本。"
                  "质量尖峰/阶跃标签很稀少，默认RF与平衡RF的局限须与视觉一起报告。",
                  "缺测排除评分并保留图和导出掩码；缺测会切开数值预测的连续段，事件F1可能因此降低，"
                  "本轮没有为了提高事件分数而填补或桥接。", "",
                  "本轮扁平视觉契约不输出内部阶段；早期嵌套阶段接口失败开发批次保留。"
                  "本报告不是可直接提交的论文，方法优势和新颖性不能由单批合成结果保证。"])
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "costs.json", receipt)
    with (out / "comparison.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, panels = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    colors = ["#bb6953" if row["method"] == "fusion" else "#42778a" for row in rows]
    positions = np.arange(len(rows))
    panels[0].barh(positions, [row["generic_daily_f1"] for row in rows], color=colors)
    panels[0].set_yticks(positions, [row["method"] for row in rows])
    panels[0].set_title("Generic anomaly: daily micro F1")
    typed = [row for row in rows if row["deformation_daily_f1"] is not None]
    panels[1].barh(np.arange(len(typed)), [row["deformation_daily_f1"] for row in typed],
                   color=["#bb6953" if row["method"] == "fusion" else "#42778a" for row in typed])
    panels[1].set_yticks(np.arange(len(typed)), [row["method"] for row in typed])
    panels[1].set_title("Sustained deformation: daily micro F1")
    for panel, entries in zip(panels, (rows, typed)):
        panel.set_xlim(0, 1)
        panel.invert_yaxis()
        panel.grid(axis="x", alpha=0.2)
        panel.tick_params(axis="y", labelsize=9)
        endpoint = "generic_daily_f1" if panel is panels[0] else "deformation_daily_f1"
        for index, row in enumerate(entries):
            value = row[endpoint]
            panel.text(value + 0.01, index,
                       f"{value:.3f} | {row['failed_records']}/24 failed",
                       va="center", fontsize=8)
    figure.text(0.5, 0.025,
                "24 synthetic test stations; failed records contribute zero detections. "
                "Repeat uses the same stations.\n"
                "Generic author-source methods have no deformation endpoint. "
                "No field-validation or superiority claim.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(out / "comparison.png")
    figure.savefig(out / "comparison.svg")
    plt.close(figure)
    if supplement:
        roots = [ROOT / "runs", ROOT / "data", ROOT / "configs", ROOT / "papers/evidence",
                 ROOT / "src/gnss_research", ROOT / "docs", ROOT / "scripts", ROOT / "tests"]
        if (ROOT / "results").exists():
            roots.append(ROOT / "results")
        # Include all development failures and locked author files, never credentials or .git.
        roots.append(ROOT / "vendor/upstream")
        if (ROOT / "artifacts/label-review").exists():
            roots.append(ROOT / "artifacts/label-review")
        with zipfile.ZipFile(out / "supplement.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for root in roots:
                for path in sorted(root.rglob("*")):
                    if path.is_file() and not {"__pycache__", ".git"} & set(path.parts):
                        archive.write(path, str(path.relative_to(ROOT)))
            for name in ("README.md", "AGENTS.md", "pyproject.toml", "uv.lock", ".env.example"):
                archive.write(ROOT / name, name)
            for name in ("results.md", "comparison.csv", "comparison.png", "comparison.svg",
                         "costs.json"):
                archive.write(out / name, "final-report/" + name)
    print(out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supplement", action="store_true")
    parser.add_argument("--author-analysis", type=Path)
    parser.add_argument("--author-repeat-analysis", type=Path)
    parser.add_argument("--evaluation", default="evaluation")
    arguments = parser.parse_args()
    report(arguments.run, arguments.repeat, arguments.out, arguments.supplement,
           arguments.author_analysis, arguments.evaluation, arguments.author_repeat_analysis)
