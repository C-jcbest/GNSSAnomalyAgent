"""Report EXP02's saved predictions; never generate or alter inference outputs."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import counts_metrics
from gnss_research.schema import EVENT_TYPES, Prediction, event_masks, observed_values

ROOT = Path(__file__).resolve().parents[1]


def paired_micro(left: list[dict], right: list[dict], seed: int, kind: str) -> dict:
    def count_matrix(rows):
        return np.array([[row["types"][kind]["daily"][key] for key in ("tp", "fp", "fn")]
                         for row in rows])

    a, b = count_matrix(left), count_matrix(right)
    samples = np.random.default_rng(seed).integers(0, len(left), (2000, len(left)))

    def f1(counts):
        tp, fp, fn = counts.T
        denominator = 2 * tp + fp + fn
        return np.divide(2 * tp, denominator, out=np.zeros(len(tp)), where=denominator > 0)

    differences = f1(a[samples].sum(axis=1)) - f1(b[samples].sum(axis=1))
    return {"endpoint": f"{kind} daily micro-F1 difference", "repeats": 2000, "seed": seed,
            "difference": float(f1(a.sum(axis=0)[None])[0] - f1(b.sum(axis=0)[None])[0]),
            "percentile_ci_95": np.quantile(differences, [.025, .975]).tolist()}


def activity_errors(run: Path, dataset: Path, method: str) -> dict:
    config = read_json(run / "registration.json")
    totals = {"observed_activity_days": 0, "missed_activity_days": 0,
              "false_activity_days_before_onset": 0, "false_activity_days_after_arrest": 0,
              "activity_stations": 0}
    for record in read_json(dataset / "manifest.json")["records"]:
        if record["station_id"] not in config["station_ids"]:
            continue
        observation = read_json(dataset / record["observation"])
        truth = read_json(dataset / record["truth"])
        events = [event for event in truth["events"] if event["kind"] == "deformation"]
        if not events:
            continue
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        result = read_json(run / "predictions" / method / record["station_id"] / "result.json")
        prediction = Prediction.model_validate(result.get("prediction") or {"events": []})
        predicted = event_masks(prediction, observed)["deformation"]
        target = event_masks(Prediction.model_validate({"events": events}), observed)["deformation"]
        first, last = min(e["start"] for e in events), max(e["end"] for e in events)
        totals["activity_stations"] += 1
        totals["observed_activity_days"] += int(target.sum())
        totals["missed_activity_days"] += int((target & ~predicted).sum())
        totals["false_activity_days_before_onset"] += int(
            predicted[observation["background_end"] + 1:first].sum())
        totals["false_activity_days_after_arrest"] += int(predicted[last + 1:].sum())
    return totals


def report(run: Path, out: Path, author: Path | None, repeat: Path | None) -> None:
    out.mkdir(parents=True, exist_ok=False)
    config = read_json(run / "registration.json")
    identity = config["experiment"]
    summary = read_json(run / "evaluation/summary.json")
    methods = dict(summary["methods"])
    if author is not None:
        methods.update(read_json(author / "summary.json")["methods"])
    rows = []
    for name, result in methods.items():
        types = result.get("types", {})
        quality_counts = [sum(types[k]["events_micro"]["iou_0.5"][key]
                              for k in ("spike", "step", "transient"))
                          for key in ("tp", "fp", "fn")] if types else None
        row = {"method": name, "generic_daily_f1": result["generic_daily_micro"]["f1"],
               "normal_false_alarm_runs_per_station_year": result[
                   "false_alarm_runs_per_normal_station_year_controls"],
               "failed_records": result["failed_records"],
               "quality_event_micro_f1_iou_05": counts_metrics(*quality_counts)["f1"]
               if quality_counts else None}
        for kind in EVENT_TYPES:
            row[f"{kind}_daily_f1"] = types[kind]["daily_micro"]["f1"] if types else None
            row[f"{kind}_event_f1_iou_05"] = types[kind]["events_micro"]["iou_0.5"]["f1"] \
                if types else None
        rows.append(row)
    with (out / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_station = read_json(run / "evaluation/per-station.json")
    primary = identity["primary_method"]
    intervals = {}
    if primary in per_station:
        for comparator in ("rf_temporal", "rf_split_temporal", "fusion", "visual_only",
                           "guided_fusion", "anchored_no_local", "anchored_no_scores"):
            if comparator in per_station:
                intervals[comparator] = paired_micro(per_station[primary], per_station[comparator],
                                                     config["seed"], "deformation")
    diagnosis = {method: activity_errors(run, Path(config["dataset"]), method)
                for method in config["methods"] if method != "isolation_forest"
                and not method.startswith("visualtime")}
    write_json(out / "activity-errors.json", diagnosis)
    write_json(out / "paired-micro.json", intervals)
    shutil.copyfile(run / "evaluation/summary.json", out / "summary.json")
    shutil.copyfile(run / "evaluation/per-station.json", out / "per-station.json")
    shutil.copyfile(run / "registration.json", out / "registration.json")
    attempts = 0
    tokens = 0
    costs = {}
    for method in config["methods"]:
        receipts = [read_json(path) for path in (run / "predictions" / method).glob("*/result.json")]
        calls = sum(r.get("request_count", 0) for r in receipts)
        usage = sum((part.get("usage") or {}).get("total_tokens", 0)
                    for r in receipts for part in r.get("subrequests", [r]))
        costs[method] = {"request_attempts": calls, "reported_total_tokens": usage}
        attempts += calls
        tokens += usage
    write_json(out / "costs.json", {"methods": costs, "request_attempts": attempts,
                                   "reported_total_tokens": tokens,
                                   "note": "usage is provider-reported; failed calls can lack usage"})
    repeated = read_json(repeat / "evaluation/summary.json") if repeat else None
    if repeated:
        shutil.copyfile(repeat / "evaluation/summary.json", out / "repeat-summary.json")
    def numeric(value):
        return "—" if value is None else f"{value:.4f}"

    lines = [f"# {identity['experiment_id']}：{identity['name']}", "",
             f"冻结协议：{identity['protocol_version']}；数据种子：{identity['dataset_seed']}；"
             f"{config['split']}集{len(config['station_ids'])}个站点；模型：{config['model']}。", "",
             f"主方法对应 `{primary}`；数值改进对应 `{identity['numerical_contribution']}`。"
             " `fusion` 是EXP01原方案在本轮新数据上的重新执行。", "",
             "| 方法列 | 通用日F1 | 形变日F1 | 形变事件F1≥.5 | 质量事件F1≥.5 | 正常误报段/站年 | 失败站 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {numeric(row['generic_daily_f1'])} | "
                     f"{numeric(row['deformation_daily_f1'])} | "
                     f"{numeric(row['deformation_event_f1_iou_05'])} | "
                     f"{numeric(row['quality_event_micro_f1_iou_05'])} | "
                     f"{numeric(row['normal_false_alarm_runs_per_station_year'])} | "
                     f"{row['failed_records']}/{len(config['station_ids'])} |")
    lines.extend(["", "四类型日/事件精确指标见comparison.csv，点事件±3日辅助指标见per-station.json。",
                  "没有point adjustment、没有测试集阈值搜索；失败按零检测计入。正常对照站仅4个，不能稳定估计现场误报率。",
                  "", "## 形变日micro-F1成对站点bootstrap（2000次）", ""])
    for comparator, estimate in intervals.items():
        low, high = estimate["percentile_ci_95"]
        lines.append(f"- {primary} − {comparator}：{estimate['difference']:.4f}；"
                     f"95%区间[{low:.4f}, {high:.4f}]。")
    lines.extend(["", "## 活动起止错误", "",
                  "以下为所有机制形变站的观测日计数，包含漏检站；停后误标数指真实活动结束后仍标形变的日期，不能解释为现场边界准确性。", "",
                  "| 方法 | 活动漏标日 | 停止后误标活动日 |",
                  "|---|---:|---:|"])
    for method in ("rf_balanced", "rf_temporal", "rf_split_temporal", "fusion", primary):
        if method in diagnosis:
            d = diagnosis[method]
            lines.append(f"| {method} | {d['missed_activity_days']} | "
                         f"{d['false_activity_days_after_arrest']} |")
    if repeated:
        lines.extend(["", "## 同测试集独立调用复测", "",
                      "仅检验服务响应波动，不能算作新的独立测试站。"])
        for method, values in repeated["methods"].items():
            lines.append(f"- {method}：通用日F1={values['generic_daily_micro']['f1']:.4f}，"
                         f"形变日F1={values['types']['deformation']['daily_micro']['f1']:.4f}，"
                         f"形变事件F1={values['types']['deformation']['events_micro']['iou_0.5']['f1']:.4f}，"
                         f"失败{values['failed_records']}站。")
    lines.extend(["", "## 方法来源与适配", "",
                  "独立RF、时间解码和局部复核是本项目实现。rf_balanced/独立RF均明确调整了类别权重，独立RF还调整了多任务训练方式；时间先验加1，候选窗口/预算在冻结协议披露。",
                  "VisualTime列实际复用锁定作者提示/请求函数；原始解析与本项目严格解析分别报告。GNSS图、NEU联合、模型更换、六路全部成功政策是适配，不称完整论文复现。VLM4TS完整模型未执行，不填成绩。",
                  "多尺度候选复核已有VLM4TS/TAMA近邻，本实验不以组合本身宣称首次。监督RF与零样本视觉输入条件不同，需结合相同RF的对照和消融判断增量。",
                  "", f"主批请求尝试：{attempts}；服务报告total_tokens：{tokens}，失败可能不返回用量。这些不是有效响应或计费确认。",
                  "仅简单仿真；固定一年正常历史、低信噪比机制边界、少量独立站和服务失败限制结论。未作现场验证，也没有投稿保证。"])
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    typed = [row for row in rows if row["deformation_daily_f1"] is not None]
    figure, panels = plt.subplots(1, 2, figsize=(15, max(5.5, len(typed) * .4)), dpi=140)
    positions = np.arange(len(typed))
    for panel, endpoint, title in zip(panels,
                                    ("deformation_daily_f1", "deformation_event_f1_iou_05"),
                                    ("Deformation daily F1", "Deformation event F1 (IoU >= 0.5)")):
        colors = ["#267f75" if row["method"] == primary else "#899fac" for row in typed]
        panel.barh(positions, [row[endpoint] for row in typed], color=colors)
        panel.set_yticks(positions, [row["method"] for row in typed])
        panel.invert_yaxis()
        panel.set_xlim(0, 1)
        panel.set_title(title)
        panel.grid(axis="x", alpha=.2)
    figure.suptitle(f"{identity['experiment_id']} — {config['split']}, {len(config['station_ids'])} stations")
    figure.tight_layout()
    figure.savefig(out / "comparison.png")
    figure.savefig(out / "comparison.svg")
    plt.close(figure)
    write_json(out / "report-receipt.json", {
        "experiment": identity, "run": str(run.resolve()),
        "registration_sha256": sha256(run / "registration.json"),
        "summary_sha256": sha256(run / "evaluation/summary.json"),
        "script_sha256": sha256(Path(__file__))})
    print(out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--author-analysis", type=Path)
    parser.add_argument("--repeat", type=Path)
    args = parser.parse_args()
    report(args.run, args.out, args.author_analysis, args.repeat)
