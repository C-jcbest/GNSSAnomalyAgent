"""Score all registered scale records only after every frozen forecast is complete."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_codex_observation_reference import scoring_inputs
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import BASE, ROOT, export
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import load_labels, save_json
from run_landslide_persistence_scale_baselines import METHODS, verify_data, verify_mapping
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from score_landslide_paired_persistence_confirmation import corrections, decision, replay_heads

from gnss_sim.artifacts import sha
from gnss_sim.landslide_evaluation import STAGES, ActivityPrediction, evaluate_case

PRIMARY = "visual_numeric_pair"
FOCUS_CONTROLS = (
    "visual_numeric_plain",
    "visual_numeric_augmentation",
    "visual_numeric_shuffled_pairs",
    "numeric_pair",
    "xgboost_84",
)
METHOD_DESCRIPTIONS = {
    "visual_numeric_pair": "视觉与数值输入，同增广数据，加配对约束；预先指定的主方法",
    "visual_numeric_plain": "视觉与数值输入，普通监督训练，无增广、无配对约束",
    "visual_numeric_augmentation": "视觉与数值输入，与主方法相同增广，无配对约束",
    "visual_numeric_shuffled_pairs": "视觉与数值输入，相同增广和配对约束，但负配对身份打乱",
    "visual_pair_no_numeric": "只使用图像编码，加增广和配对约束",
    "numeric_pair": "只使用数值输入，加增广和配对约束",
    "numeric_augmentation": "只使用数值输入，加相同增广，无配对约束",
    "xgb_raw18": "对相同原始数值输入训练的树模型；名称沿用已有模型编号",
    "xgb_expert24": "原始数值输入加既有人工设计特征的树模型；名称沿用已有模型编号",
    "xgb_chart384": "原始数值输入加图表数值描述的树模型；不使用图像编码器",
    "robust_rule": "稳健速度及信噪比规则确认活动，规则划分阶段",
    "pelt_rule": "变化点分段速度确认活动，规则划分阶段",
    "xgboost": "原始诊断特征树模型确认活动，树模型划分阶段",
    "robust_xgboost_stage": "稳健规则确认活动，树模型划分阶段",
    "raw_multiscale_rule": "加入 42 天原始位移持续性证据的活动规则，树模型划分阶段",
    "xgboost_raw_persistence": "加入 42 天原始位移持续性特征的活动树模型",
    "raw_rule_84": "加入 42 天和 84 天原始位移证据的活动规则",
    "xgboost_42_matched_support": "42 天特征活动树模型，使用与 84 天方法一致的可判断日期范围",
    "xgboost_84": "同时使用 42 天与 84 天持续性证据的活动树模型",
}


def verify_complete_forecasts(data: Path, heads: Path, baselines: Path) -> tuple[dict, dict]:
    """Reject partial methods before any process-reference loader can be called."""
    protocol = read_json(data / "protocol.json")
    expected_ids = [f"case_{index:04d}" for index in range(217, 337)]
    if protocol["case_ids"] != expected_ids or len(protocol["selection"]) != 10:
        raise ValueError("Require the entire registered 120-record, ten-head population")
    for run, freeze in ((heads, "result-freeze.json"), (baselines, "manifest.json")):
        if read_json(run / "state.json")["status"] != "predictions_complete":
            raise ValueError("Freeze all 19 complete forecasts before opening process references")
        verify_bindings(run, freeze)
    learned = read_json(heads / "new-predictions.json")
    numeric = read_json(baselines / "predictions.json")
    if set(learned) != set(protocol["selection"]) or set(numeric) != set(METHODS):
        raise ValueError("Every registered method is required; no method selection at scoring")
    if set(learned) & set(numeric):
        raise ValueError("Learning heads and numerical baseline names must be distinct")
    predictions = {**learned, **numeric}
    for name, cases in predictions.items():
        if set(cases) != set(expected_ids):
            raise ValueError(f"Every method must predict all 120 records: {name}")
        for payload in cases.values():
            prediction = ActivityPrediction(
                np.asarray(payload["activity"]), np.asarray(payload["stage"]),
                status=payload["status"], gated=payload["gated"],
                stage_status=payload["stage_status"],
            )
            prediction.validate(protocol["generation"]["days"])
    return protocol, predictions


def paired_count_interval(results: dict, control: str, treatment: str,
                          seed: int = 20261009, metric: str = "daily") -> dict:
    """Resample whole records, preserving the historical random draws and F1 rule."""
    cases = sorted(results[control]["per_case"])
    if set(cases) != set(results[treatment]["per_case"]):
        raise ValueError("Paired bootstrap requires exactly the same records")
    draws = np.random.default_rng(seed).integers(0, len(cases), (2000, len(cases)))
    samples = {}
    for name in (control, treatment):
        counts = np.asarray([
            [results[name]["per_case"][cid][metric][key] for key in ("tp", "fp", "fn")]
            for cid in cases
        ], dtype=np.int64)
        totals = counts[draws].sum(axis=1)
        denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        values = np.full(len(draws), np.nan)
        np.divide(2 * totals[:, 0], denominator, out=values, where=denominator != 0)
        samples[name] = values
    differences = samples[treatment] - samples[control]
    differences = differences[np.isfinite(differences)]
    before = results[control]["aggregate"][metric]["f1"]
    after = results[treatment]["aggregate"][metric]["f1"]
    return {
        "record_count": len(cases), "draws_with_defined_f1": len(differences),
        "difference": after - before if after is not None and before is not None else None,
        "exploratory_95_percentile_interval": np.quantile(differences, [.025, .975]).tolist()
        if len(differences) else None,
        "scope": "Paired whole-record descriptive bootstrap; no confirmatory significance, "
                 "no pooling of repetitions into independent samples",
    }


def enrich_scores(labels: dict, observed: dict, predictions: dict, results: dict,
                  summary: list[dict], records: list[dict]) -> None:
    """Expose counts and denominators alongside rates rather than hiding unknown days."""
    for row in summary:
        result = results[row["method"]]
        aggregate = result["aggregate"]
        if sum(result["internal_stages"]["support_ASD"]) == 0:
            # Empty stage interiors have no coverage denominator, including quiet families.
            result["internal_stages"]["coverage"] = None
            row["stage_internal_coverage"] = None
        row.update({key: value for key, value in aggregate.items()
                    if isinstance(value, int) and key not in ("weak_days", "weak_detected_days")})
        row["negative_alarm_events_per_decided_station_year"] = (
            aggregate["negative_alarm_events_per_decided_station_year"]
        )
        row["negative_false_positive_rate_all_known"] = (
            row["fp"] / aggregate["negative_days"] if aggregate["negative_days"] else None
        )
        row["negative_false_positive_rate_conditional_on_decision"] = (
            row["fp"] / aggregate["decided_negative_days"]
            if aggregate["decided_negative_days"] else None
        )
        row["activity_positive_days"] = row["tp"] + row["fn"]
        for prefix, events in (("events_iou03", aggregate["events"]),
                               ("events_iou05", result["events_iou05"])):
            row.update({f"{prefix}_{key}": value for key, value in events.items()})
        for name, code in STAGES.items():
            index = code - 1
            row[f"stage_internal_{name}_support"] = result["internal_stages"]["support_ASD"][index]
            row[f"stage_internal_{name}_f1"] = result["internal_stages"]["f1_ASD"][index]
            for key, value in aggregate["stages"][name].items():
                row[f"stage_{name}_{key}"] = value
        result["auxiliary_weak_signal"] = {
            "days": row["weak_days"], "detected_days": row["weak_detected_days"],
            "recall": row["weak_recall"],
            "definition": "Explicit target weak mask; process target uses frozen weak_signal. "
                          "Observation target has no pre-specified mask and is not estimated.",
        }
    for row in records:
        method, cid = row["method"], row["case_id"]
        case_score = results[method]["per_case"][cid]
        row.update({key: value for key, value in case_score.items()
                    if isinstance(value, int) and key not in ("weak_days", "weak_detected_days")})
        row["coverage"] = (
            1 - row["abstained_known_days"] / row["known_days"] if row["known_days"] else None
        )
        row["weak_recall"] = (
            row["weak_detected_days"] / row["weak_days"] if row["weak_days"] else None
        )
        row["negative_false_positive_rate_all_known"] = (
            row["fp"] / row["negative_days"] if row["negative_days"] else None
        )
        row["negative_false_positive_rate_conditional_on_decision"] = (
            row["fp"] / row["decided_negative_days"] if row["decided_negative_days"] else None
        )
        row["negative_alarm_events_per_decided_station_year"] = (
            row["negative_exposure_alarm_events"] / (row["decided_negative_days"] / 365.25)
            if row["decided_negative_days"] else None
        )
        value = predictions[method][cid]
        prediction = ActivityPrediction(
            np.asarray(value["activity"]), np.asarray(value["stage"]),
            status=value["status"], gated=value["gated"], stage_status=value["stage_status"],
        )
        events05 = evaluate_case(labels[cid], prediction, .5)["events"]
        for prefix, events in (("events_iou03", case_score["events"]), ("events_iou05", events05)):
            row.update({f"{prefix}_{key}": value for key, value in events.items()})
        actual = np.asarray([int(item["activity_label"]) for item in labels[cid]])
        unknown = observed[cid] & (actual == -1)
        row["missing_observation_days"] = int((~observed[cid]).sum())
        for code, name in ((1, "positive"), (0, "negative"), (-1, "abstained")):
            row[f"unknown_reference_predicted_{name}"] = int((unknown & (prediction.activity == code)).sum())
        for name in STAGES:
            row[f"stage_{name}_support"] = case_score["stages"][name]["tp"] + case_score["stages"][name]["fn"]
            row[f"stage_{name}_f1"] = case_score["stages"][name]["f1"]


def correction_summary(rows: list[dict], summary: list[dict], intervals: dict) -> list[dict]:
    table = {row["method"]: row for row in summary}
    totals = []
    for control in table:
        if control == PRIMARY:
            continue
        selected = [row for row in rows if row["control"] == control]
        primary, before = table[PRIMARY], table[control]
        value = {
            "control": control, "treatment": PRIMARY,
            "gained_correct_days": sum(row["gained"] for row in selected),
            "lost_correct_days": sum(row["lost"] for row in selected),
            "net_correct_days": sum(row["net"] for row in selected),
            "benefited_records": sum(row["net"] > 0 for row in selected),
            "harmed_records": sum(row["net"] < 0 for row in selected),
            "tied_records": sum(row["net"] == 0 for row in selected),
            "daily_f1_difference": intervals[control]["daily"]["difference"],
            "daily_f1_interval_lower": intervals[control]["daily"]["exploratory_95_percentile_interval"][0],
            "daily_f1_interval_upper": intervals[control]["daily"]["exploratory_95_percentile_interval"][1],
        }
        for key in ("precision", "recall", "weak_recall", "event_f1_iou03", "event_f1_iou05",
                    "onset_core_mae_days", "offset_core_mae_days"):
            value[f"{key}_difference"] = (
                primary[key] - before[key]
                if primary[key] is not None and before[key] is not None else None
            )
        totals.append(value)
    return totals


def display(value) -> str:
    if value is None:
        return "未定义"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(title for _, title in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(display(row[key]) for key, _ in columns) + " |")
    return "\n".join(lines)


def target_report(target: str, summary: list[dict], correction_totals: list[dict],
                  heading_level: int = 1) -> str:
    primary = target == "primary_process"
    heading = "完整 120 条过程参考评测" if primary else "预先选定 24 条观测参考评测"
    heading_prefix = "#" * heading_level
    subheading_prefix = "#" * (heading_level + 1)
    lines = [heading_prefix + " " + heading, "",
             "同一批已冻结预测分别对过程参考和观测参考评分；两种参考回答不同问题，分开报告。", "",
             subheading_prefix + " 全部 19 种方法", "",
             markdown_table(summary, [
                 ("method", "方法"), ("precision", "日精确率"), ("recall", "日召回率"),
                 ("f1", "日 F1"), ("weak_recall", "弱活动召回"),
                 ("event_f1_iou03", "事件 F1，IoU≥0.3"),
                 ("event_f1_iou05", "事件 F1，IoU≥0.5"), ("coverage", "活动判断覆盖率"),
             ]), "", subheading_prefix + " 事件边界、阶段和稳定时段误报", "",
             markdown_table(summary, [
                 ("method", "方法"), ("onset_core_mae_days", "起点平均绝对误差/天"),
                 ("offset_core_mae_days", "终点平均绝对误差/天"),
                 ("stage_internal_macro_f1", "活动内部阶段宏 F1"),
                 ("stage_end_to_end_macro_f1", "全过程阶段宏 F1"),
                 ("negative_false_positive_rate_conditional_on_decision", "稳定已判断日误报率"),
                 ("negative_alarm_events_per_decided_station_year", "每稳定已判断站年误报事件"),
                 ("failed_cases", "活动失败记录"), ("stage_failed_cases", "阶段失败记录"),
             ]), "",
             "边界误差只统计已经匹配的事件，因此同时查看事件 TP/FP/FN；不能将较少匹配后的较小误差理解为普遍改善。", "",
             "稳定已判断日误报率 = FP / 已作出判断的参考稳定日数；每站年事件误报指标沿用共同评分器，分母 = 这些稳定日数 / 365.25。完整 CSV 另保留所有参考稳定日分母的误报率及未判断天数。", "",
             "每站年指标的分子是触及参考稳定日的预测事件数：同一事件即使匹配到真实活动，只要延伸进稳定日仍计一次。因此它与未匹配事件 FP 回答不同问题，两项都保留。", "",
             subheading_prefix + " 主方法相对所有对照", "",
             "以下均为配对一致性训练视觉数值方法减去对照。受益/受损按整条记录净纠错天数的正负计；纠错是对已知活动参考的正确判断变化，不能等同于 F1 改善。", "",
             markdown_table(correction_totals, [
                 ("control", "对照"), ("daily_f1_difference", "日 F1 差"),
                 ("daily_f1_interval_lower", "描述性区间下限"),
                 ("daily_f1_interval_upper", "描述性区间上限"),
                 ("event_f1_iou03_difference", "事件 F1 差"),
                 ("gained_correct_days", "纠正天数"), ("lost_correct_days", "新增错误天数"),
                 ("net_correct_days", "净纠错天数"),
                 ("benefited_records", "受益记录"), ("harmed_records", "受损记录"),
                 ("tied_records", "持平记录"),
             ]), "",
             "整条记录配对 bootstrap 2,000 次，固定种子 20261009，同一次重采样同时取主方法与对照，重算汇总 F1，以 2.5%/97.5% 分位为描述性区间。保留所有 18 个对照，未做多重比较校正，不能表述为总体显著或现场准确率。", ""]
    if not primary:
        lines += ["观测参考由同一项目中的 Codex 原始位移优先复核制定，既有设计与历史结果已知；并非独立专家真值。此次没有预测前指定的观测弱活动子集，因此弱活动召回留空；没有借用过程标签或事后挑选区间。", ""]
    return "\n".join(lines)


def comparison_plot(folder: Path, summary: list[dict], target: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 9))
    names = [row["method"] for row in summary]
    colors = ["#c45a22" if name == PRIMARY else "#4d788c" for name in names]
    for axis, metric in zip(axes, ("f1", "event_f1_iou03"), strict=True):
        values = [np.nan if row[metric] is None else row[metric] for row in summary]
        axis.barh(names, values, color=colors)
        axis.set_xlim(0, 1)
        axis.invert_yaxis()
        axis.set_title(metric)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle(target + " | All 19 frozen methods, all registered records")
    figure.tight_layout()
    figure.savefig(folder / "comparison.png", dpi=150)
    plt.close(figure)


def score_reference(folder: Path, target: str, labels: dict, observed: dict, weak: dict,
                    predictions: dict) -> tuple[dict, list[dict], list[dict]]:
    folder.mkdir()
    results, summary, records = score_target(labels, observed, weak, predictions, {}, {})
    enrich_scores(labels, observed, predictions, results, summary, records)
    correction_rows = corrections(labels, predictions)
    intervals = {
        name: {"daily": paired_count_interval(results, name, PRIMARY),
               "events_iou03": paired_count_interval(results, name, PRIMARY, metric="events")}
        for name in predictions if name != PRIMARY
    }
    totals = correction_summary(correction_rows, summary, intervals)
    for name, value in (("results.json", results), ("summary.json", summary),
                        ("per-record.json", records), ("reference-labels.json", labels),
                        ("predictions.json", predictions), ("corrections.json", correction_rows),
                        ("correction-summary.json", totals),
                        ("descriptive-bootstrap.json", intervals)):
        export(folder / name, value)
    write_csv(folder / "summary.csv", summary)
    write_csv(folder / "per-record.csv", records)
    write_csv(folder / "correction-summary.csv", totals)
    (folder / "report.md").write_text(target_report(target, summary, totals), encoding="utf-8")
    comparison_plot(folder, summary, target)
    return results, summary, totals


def full_report(protocol: dict, primary: list[dict], secondary: list[dict],
                primary_totals: list[dict], secondary_totals: list[dict],
                family_rows: list[dict], historical_checks: dict, audit: dict) -> str:
    table = {row["method"]: row for row in primary}
    main = table[PRIMARY]
    lines = ["# 冻结模型在全部 120 条新记录上的对比评测", "",
             "本次执行预先登记的 120 条新记录评测：每条 730 天、6 类过程各 20 条，10 个冻结学习模型与 9 个既有数值基线全部参加，共 2,280 份完整记录预测。模型权重、阈值、数据生成种子和复核参考均沿用冻结设置，没有再训练、调参或新增远程 API 调用。", "",
             f"主方法过程参考日 F1 为 {main['f1']:.6f}，事件 F1（IoU≥0.3）为 {main['event_f1_iou03']:.6f}，弱活动召回为 {main['weak_recall']:.6f}。下面保留改善与代价，不按结果筛除方法或记录。", "",
             "## 如何理解这次比较", "",
             "过程参考来自生成时已经写定的真实过程区间，用于检验已知受控位移恢复；观测参考取预先选定的前 24 个不透明记录编号，由 Codex 在看到本批预测之前先看原始位移、再看区间内部阶段，用于检验模型能否符合这一可观察任务定义。二者分别评分，同一预测不作任务适配。", "",
             "这次扩大的是同一生成器的测试数量，训练数据没有扩大，也没有新的站点、现场观测或独立专家验证。早期 24 条确认试验单独保留；本次不合并其样本，也不把早期负面结果替换掉。", "",
             "## 参加比较的方法", "",
             markdown_table([{"method": row["method"], "description": METHOD_DESCRIPTIONS[row["method"]]}
                             for row in primary], [("method", "结果文件中的名称"),
                                                   ("description", "实际比较的做法")]), "",
             "10 个学习模型以及 5 个持续性数值基线都沿用冻结的阶段树模型；稳健规则和变化点规则采用既有阶段规则，其他基线的阶段设置也保持原实现。此次主要学习与比较活动确认，阶段输出先受各方法活动范围约束。", "",
             "## 过程参考：关键对照的改善与代价", ""]
    focus = [row for row in primary_totals if row["control"] in FOCUS_CONTROLS]
    lines.append(markdown_table(focus, [
        ("control", "对照"), ("daily_f1_difference", "日 F1 差"),
        ("event_f1_iou03_difference", "事件 F1 差"), ("weak_recall_difference", "弱召回差"),
        ("net_correct_days", "净纠错天数"), ("benefited_records", "受益记录"),
        ("harmed_records", "受损记录"),
        ("onset_core_mae_days_difference", "起点误差差/天"),
        ("offset_core_mae_days_difference", "终点误差差/天"),
    ]))
    lines += ["", "误差差为正表示边界误差变大；F1 和召回差为正表示相应分数提高。视觉数值普通训练、同增广训练和打乱配对训练分别帮助判断配对约束是否超出普通训练/增广以及正确配对的作用；纯数值配对方法用于观察视觉输入的增量价值；84 天数值树方法作为既有较强数值对照。上述解释限于这些固定模型和生成条件，不能据此作普遍因果结论。", "",
              target_report("primary_process", primary, primary_totals, heading_level=2),
              "## 每类过程的完整结果", "",
              "每类 20 条、每类全部 19 方法，未挑选成功家族。仅活动参考含有正例的家族才有日召回的定义；稳定类 F1/召回出现空值时保留，不填为满分或零分。", "",
              "六类分别为：`quiet`（平稳）、`instrument_step`（仪器阶跃）、`weak_constant`（持续弱活动）、`asd`（加速—匀速—减速）、`reactivation`（再活动）、`weak_after_strong`（强活动后的弱活动）。", "",
              markdown_table(family_rows, [
                  ("family", "过程类型"), ("method", "方法"), ("records", "记录数"),
                  ("precision", "精确率"), ("recall", "召回率"), ("f1", "日 F1"),
                  ("event_f1_iou03", "事件 F1"), ("weak_recall", "弱活动召回"),
              ]), "",
              target_report("secondary_observation", secondary, secondary_totals, heading_level=2),
              "## 观测参考中的关键差异", ""]
    for row in secondary_totals:
        if row["control"] in FOCUS_CONTROLS:
            sign = "提高" if row["daily_f1_difference"] > 0 else "下降" if row["daily_f1_difference"] < 0 else "持平"
            lines.append(
                f"- 相对 `{row['control']}`：日 F1 {sign}（差 {row['daily_f1_difference']:+.6f}），"
                f"事件 F1 差 {row['event_f1_iou03_difference']:+.6f}，"
                f"净纠错 {row['net_correct_days']:+d} 天，受益 {row['benefited_records']} 条、受损 {row['harmed_records']} 条。"
            )
    lines += ["", "这些观测目标结果全部保留；过程恢复改善并不保证观测参考下改善。阶段头沿用旧模型，活动范围改变会影响其可输出范围，故阶段变化不能直接解释为阶段头独立进步。", "",
              "## 沿用历史 10 条条件的描述性核对", "",
              "下面只是旧 24 条确认试验的条件在此次新批次上的表现，未将其事后升级为本次新登记的成功门槛。", "",
              f"旧条件通过 {historical_checks['passed']}/{historical_checks['total']}；旧函数返回 `{historical_checks['decision']}`。", "",
              markdown_table([{"check": name, "passed": passed}
                              for name, passed in historical_checks["checks"].items()],
                             [("check", "旧条件"), ("passed", "本批是否满足")]), "",
              "## 完整性、重放和运行成本", "",
              f"所有 19 种方法均覆盖 120 条，观测参考覆盖预先指定的 24 条；过程逐记录评分 {audit['primary_method_record_scores']} 行，观测逐记录评分 {audit['secondary_method_record_scores']} 行，家族汇总 114 行。10 个学习模型的全部概率、活动/阶段输出逐值重放一致；9 个数值基线通过原始预测入口全量重放一致。", "",
              f"九数值基线首次运行墙钟 {audit['numerical_cost']['wall_seconds']:.3f} 秒，特征读取合计 {audit['numerical_cost']['feature_seconds']:.3f} 秒，九方法计算合计 {audit['numerical_cost']['all_nine_prediction_seconds']:.3f} 秒。", "",
              f"学习模型运行墙钟 {audit['heads_cost']['wall_seconds']:.3f} 秒（含旧记录预检 {audit['heads_cost']['preflight_seconds']:.3f} 秒）；采用 {audit['heads_cost']['workers']} 个独立记录进程，每进程保持原 2 线程与 32 张图像批量。并行记录时间相加不是实际墙钟；成本是本机本次实测，不能作硬件无关速度结论。", "",
              "首次学习推断准备目录 `persistence-scale-heads-v1` 在写状态文件时发生 FileExistsError，当时旧记录预检已通过、0 条新记录完成；其失败证据保留。修复仅涉及运行状态写入，正式预测使用新目录 `persistence-scale-heads-v2`，不能把准备失败计为模型预测失败或把失败目录当成功结果。", "",
              "评分保留失败、未知判断、未匹配事件及阶段支持数。日 F1 以全部已知参考日的 TP/FP/FN 汇总；未知预测若落在参考活动日会算漏检，参考未知日不算正确稳定日。缺测处活动与阶段始终为 -1。阶段内部宏 F1 固定对加速/匀速/减速三类平均，无支持类按共同评分器记零；全过程阶段宏 F1 使用共同评分器对已定义类别平均，故应同时查看三类支持和单类分数。", "",
              "## 统计解释复查（建议性质，未改变门槛）", "",
              "1. 完整采样：固定种子一次生成 120 条，未补样、重生成或筛选成功记录。",
              "2. 家族覆盖：6 类各 20 条，家族平衡是实验人为设计，不等于现场频率。",
              "3. 基率：完整 CSV 记录活动/稳定/未知天数；精确率依赖这些比例。",
              "4. 记录相关：重采样单位为完整记录，不把 87,600 个日点视作独立样本。",
              "5. 多重比较：同时保留 18 个主方法对照，区间未校正，不宣布总体显著。",
              "6. 未知与失败：参考未知、输出未知、缺测与模型失败分开保留；未知不算真阴性。",
              "7. 指标定义：日、事件、边界、阶段和弱活动分别报告，不用单一 F1 替代全部任务。",
              "8. 边界条件：误差仅基于匹配事件，须同时看漏检、误报和匹配数量。",
              "9. 历史选择：模型及设计经过早期结果选择；当前新记录独立于权重阈值选择，但同生成器和既有设计知识仍限制外推。",
              "10. 参考独立性：观测复核者是 Codex 且知道既有设计，并非独立专家；弱活动辅助指标没有事后补标签。",
              "11. 因果与外推：固定模型对照用于定位可能机制，不证明所有架构下的配对因果作用、现场准确率或独立专家一致性。", "",
              "## 文件与复现", "",
              "完整机器结果在 `artifacts/landslide-adaptive-review-2026-10-09/persistence-scale-analysis-v1/`：`primary_process/` 和 `secondary_observation/` 各包含全部汇总、逐记录、事件与边界结果、参考、预测、净纠错和描述性区间；`primary-family-summary.csv` 覆盖 6×19 行。`source-bindings.json`、`pre-score-prediction-freeze.json`、`audit.json` 与 `manifest.json` 绑定本次代码和证据。", "",
              "```powershell",
              "$env:PYTHONPATH = 'artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-runtime-v1/site-packages;src;scripts'",
              "& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/score_landslide_persistence_scale.py `",
              "  --data artifacts/landslide-adaptive-review-2026-10-09/persistence-scale-data-v1 `",
              "  --heads artifacts/landslide-adaptive-review-2026-10-09/persistence-scale-heads-v2 `",
              "  --baselines artifacts/landslide-adaptive-review-2026-10-09/persistence-scale-baselines-v1 `",
              "  --numeric-replay-evidence <已保存的九方法全量重放证据.json> `",
              "  --out <新的评分目录>",
              "```", "",
              "使用新的输出目录重放评分，不覆盖任何历史结果。"]
    return "\n".join(lines) + "\n"


def score(data: Path, heads: Path, baselines: Path, numeric_replay_evidence: Path,
          out: Path) -> None:
    started = time.perf_counter()
    protocol, predictions = verify_complete_forecasts(data, heads, baselines)
    verify_data(data)
    verify_mapping(ROOT, read_json(baselines / "source-bindings.json"))
    replay_evidence = read_json(numeric_replay_evidence)
    if (
        replay_evidence.get("status") != "passed"
        or replay_evidence.get("methods") != 9
        or replay_evidence.get("records") != 120
        or not replay_evidence.get("all_fields_exact")
        or replay_evidence.get("failed_comparisons") != 0
        or replay_evidence.get("source_manifest_sha256") != sha(baselines / "manifest.json")
        or replay_evidence.get("source_predictions_sha256") != sha(baselines / "predictions.json")
    ):
        raise ValueError("Nine numerical methods must have passed complete replay")
    verify_mapping(numeric_replay_evidence.parent, replay_evidence["logs"])
    reference = data / "codex-review/compiled/reference-freeze.json"
    verify_mapping(reference.parent, read_json(reference)["files"])
    if sha(reference) != read_json(heads / "protocol.json")["review_binding"]:
        raise ValueError("Observation reference binding differs from frozen inference")
    replayed = replay_heads(heads)
    if any(replayed[name] != predictions[name] for name in replayed):
        raise ValueError("Full probability/activity/stage replay differs")
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "state.json", {"status": "verified_all_forecasts_before_references"})
    try:
        source_files = {Path(__file__).resolve()}
        for module in tuple(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if path.suffix == ".py" and (
                    path.is_relative_to(ROOT / "scripts") or path.is_relative_to(ROOT / "src/gnss_sim")
                ):
                    source_files.add(path)
        source_bindings = {path.relative_to(ROOT).as_posix(): sha(path)
                           for path in sorted(source_files)}
        for name in source_bindings:
            destination = out / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        export(out / "source-bindings.json", source_bindings)
        export(out / "pre-score-prediction-freeze.json", {
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "heads_result_freeze_sha256": sha(heads / "result-freeze.json"),
            "heads_predictions_sha256": sha(heads / "new-predictions.json"),
            "numerical_manifest_sha256": sha(baselines / "manifest.json"),
            "numerical_predictions_sha256": sha(baselines / "predictions.json"),
            "numerical_replay_evidence_sha256": sha(numeric_replay_evidence),
            "data_freeze_sha256": sha(data / "data-freeze.json"),
            "observation_reference_freeze_sha256": sha(reference),
            "source_sha256": sha(Path(__file__)),
            "all_methods_complete_before_process_reference_opened": True,
        })
        shutil.copyfile(numeric_replay_evidence, out / "numerical-replay-evidence.json")
        export(out / "predictions.json", predictions)
        features = {cid: load_case_features(data, cid) for cid in protocol["case_ids"]}
        primary = load_labels(data / "mechanism-sealed/reference-labels.csv",
                              {cid: record[0] for cid, record in features.items()})
        observed = {cid: record[1].observed for cid, record in features.items()}
        weak = {cid: np.asarray([int(row["activity_label"]) == 1 and int(row["weak_signal"]) == 1
                                 for row in rows]) for cid, rows in primary.items()}
        ai, ai_observed, _ = scoring_inputs(data / "codex-review")
        if set(ai) != set(protocol["secondary_review_ids"]):
            raise ValueError("Observation subset differs from its pre-generation registration")
        ai_weak = {cid: np.zeros(len(rows), dtype=bool) for cid, rows in ai.items()}
        secondary = {name: {cid: cases[cid] for cid in ai} for name, cases in predictions.items()}
        _, primary_summary, primary_totals = score_reference(
            out / "primary_process", "primary_process", primary, observed, weak, predictions
        )
        _, secondary_summary, secondary_totals = score_reference(
            out / "secondary_observation", "secondary_observation", ai, ai_observed, ai_weak, secondary
        )
        historical_checks = decision(primary_summary, corrections(primary, predictions))
        export(out / "historical-conditions-descriptive.json", {
            "scope": "Old ten conditions described on the scale batch; not a new registered gate",
            **historical_checks,
        })
        generation = read_json(data / "mechanism-sealed/generation-ledger.json")
        family_rows = []
        for family in protocol["generation"]["families"]:
            ids = [row["case_id"] for row in generation if row["truth"]["family"] == family]
            if len(ids) != protocol["generation"]["records_per_family"]:
                raise ValueError("Registered family counts changed")
            family_labels = {cid: primary[cid] for cid in ids}
            family_observed = {cid: observed[cid] for cid in ids}
            family_predictions = {name: {cid: cases[cid] for cid in ids}
                                  for name, cases in predictions.items()}
            results, summaries, records = score_target(
                family_labels, family_observed, {cid: weak[cid] for cid in ids},
                family_predictions, {}, {}
            )
            enrich_scores(family_labels, family_observed, family_predictions, results, summaries, records)
            family_rows.extend(dict(row, family=family, records=len(ids)) for row in summaries)
        export(out / "primary-family-summary.json", family_rows)
        write_csv(out / "primary-family-summary.csv", family_rows)
        audit = {
            "full_primary_records": len(primary), "secondary_records": len(ai),
            "primary_methods": len(predictions), "secondary_methods": len(secondary),
            "primary_method_record_scores": len(primary) * len(predictions),
            "secondary_method_record_scores": len(ai) * len(secondary),
            "registered_families": len(protocol["generation"]["families"]),
            "records_per_family": protocol["generation"]["records_per_family"],
            "learning_probability_activity_stage_replay": "exact: all 10 methods x 120 records",
            "numerical_replay": replay_evidence,
            "new_api_calls": 0, "refitting": False, "new_parameter_selection": False,
            "secondary_weak_mask": "Absent before inference; auxiliary weak recall remains undefined",
            "secondary_observable_target_adaptation_added": False,
            "field_accuracy_verified": False, "independent_expert_reference": False,
            "heads_cost": read_json(heads / "inference-cost.json"),
            "numerical_cost": read_json(baselines / "cost.json"),
            "bootstrap": {"record_unit": "whole record", "draws": 2000, "seed": 20261009,
                          "quantiles": [.025, .975], "confirmatory_significance": False},
            "total_analysis_seconds_including_replay": time.perf_counter() - started,
        }
        report = full_report(protocol, primary_summary, secondary_summary, primary_totals,
                             secondary_totals, family_rows, historical_checks, audit)
        (out / "report.md").write_text(report, encoding="utf-8")
        export(out / "audit.json", audit)
        verify_mapping(ROOT, source_bindings)
        save_json(out / "state.json", {
            "status": "complete", "records": len(primary), "methods": len(predictions),
            "primary_method_record_scores": len(primary) * len(predictions),
            "seconds": time.perf_counter() - started,
        })
        export(out / "manifest.json", {"files": {
            path.relative_to(out).as_posix(): sha(path)
            for path in sorted(out.rglob("*")) if path.is_file()
        }})
    except Exception as error:
        (out / "failure-traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        save_json(out / "state.json", {"status": "failed", "error": str(error)})
        raise
    print(f"Complete: {len(predictions)} methods x {len(primary)} primary records; "
          f"secondary {len(ai)} records; {time.perf_counter() - started:.3f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=BASE / "persistence-scale-data-v1")
    parser.add_argument("--heads", type=Path, default=BASE / "persistence-scale-heads-v2")
    parser.add_argument("--baselines", type=Path, default=BASE / "persistence-scale-baselines-v1")
    parser.add_argument("--numeric-replay-evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    score(args.data.resolve(), args.heads.resolve(), args.baselines.resolve(),
          args.numeric_replay_evidence.resolve(), args.out.resolve())
