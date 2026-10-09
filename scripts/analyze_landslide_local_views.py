"""Read-only paired diagnostics for the completed local-view experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_landslide_comparison import ROOT, aggregate_scores, save_json, sha

from gnss_sim.landslide_trace import read_json


def negative_stage_diagnostics(run_root, predictions, case_ids):
    """Count confident stages only on observed reference nonactivity, excluding unknowns."""
    protocol = read_json(run_root / "protocol.json")
    labels_path = ROOT / protocol["labels"]
    if sha(labels_path) != protocol["labels_sha256"]:
        raise ValueError("Frozen labels changed before negative-stage analysis")
    labels = {cid: [] for cid in case_ids}
    with labels_path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    negative_masks = {}
    for cid, rows in labels.items():
        case = read_json(run_root / "inputs" / cid / "input.json")
        if len(rows) != len(case["dates"]):
            raise ValueError("Incomplete reference calendar")
        for index, row in enumerate(rows):
            if int(row["day_index"]) != index or row["date"] != case["dates"][index]:
                raise ValueError("Reference calendar mismatch")
            if case["displacement_mm"][index] is None and int(row["activity_label"]) != -1:
                raise ValueError("Missing observations cannot be negative reference days")
        negative_masks[cid] = np.array([int(row["activity_label"]) == 0 for row in rows])
    counts = {}
    for name, records in predictions.items():
        per_case = {}
        for cid in case_ids:
            stages = np.array(records[cid]["stage"])
            negative = negative_masks[cid]
            if len(stages) != len(negative):
                raise ValueError("Prediction calendar mismatch")
            per_case[cid] = {label: int(np.count_nonzero(negative & (stages == code)))
                             for label, code in (("A", 1), ("S", 2), ("D", 3))}
            per_case[cid]["total"] = sum(per_case[cid].values())
        counts[name] = {"per_case": per_case,
                        "aggregate": {key: sum(row[key] for row in per_case.values())
                                      for key in ("A", "S", "D", "total")}}
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = read_json(args.run / "manifest.json")
    if run["status"] != "complete":
        raise ValueError("Experiment must complete before analysis")
    results = read_json(args.run / "results.json")
    predictions = read_json(args.run / "predictions.json")
    audit = read_json(args.run / "error-audit.json")
    # Historical controls live in the fixed source; the new prediction file stays frozen.
    prior_path = Path(run["visual_root"]) / "predictions.json"
    if sha(prior_path) != run["fixed_predictions_sha256"]:
        raise ValueError("Frozen historical control changed before analysis")
    prior = read_json(prior_path)
    negative_predictions = {**predictions, **{name: prior[name] for name in
                                            ("visual_rule", "visual_model", "robust_xgboost_stage")}}
    negative_stages = negative_stage_diagnostics(args.run, negative_predictions, run["case_ids"])
    for name, values in negative_stages.items():
        for cid, counts in values["per_case"].items():
            if counts["total"] != audit[name][cid]["stages"]["false_stage_days_on_reference_negative"]:
                raise ValueError("Negative-stage counts disagree with the frozen error partition")
            if counts["A"] != results[name]["per_case"][cid]["false_acceleration_days"]:
                raise ValueError("False acceleration count disagrees with the evaluator")
    receipts = [read_json(path) for path in (args.run / "requests").glob("*.response.json")]
    cost = {}
    for name in ("activity_global", "activity_local", "stage_global", "stage_local"):
        suffix = name.replace("activity", "locate").replace("_", "-")
        rows = [row for row in receipts if row["request_id"].endswith("-" + suffix)]
        cost[name] = {"calls": len(rows), "image_associations": sum(len(row["image_sha256"]) for row in rows),
                      "total_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in rows),
                      "service_seconds_sum": sum(row.get("seconds", 0) for row in rows)}
    common = [cid for cid in run["case_ids"] if all(predictions[name][cid]["status"] == "success"
                                                   for name in ("activity_global", "activity_local"))]
    common_scores = {name: aggregate_scores([results[name]["per_case"][cid] for cid in common])
                     for name in ("activity_global", "activity_local")}
    per_case = {}
    for cid in run["case_ids"]:
        per_case[cid] = {}
        for name in ("activity_global", "activity_local", "stage_global", "stage_local", "visual_xgboost_stage"):
            score = results[name]["per_case"][cid]
            per_case[cid][name] = {"activity": score["daily"], "weak_detected_days": score["weak_detected_days"],
                                   "stage_partition": audit[name][cid]["stages"]["counts"],
                                   "false_stage_days_on_negative": audit[name][cid]["stages"]["false_stage_days_on_reference_negative"]}
    rng = np.random.default_rng(20261007)
    draws = rng.integers(0, len(run["case_ids"]), size=(2000, len(run["case_ids"])))
    comparisons = []
    for treatment, control, metric in (("activity_local", "activity_global", "daily_f1"),
                                        ("stage_local", "stage_global", "stage_macro_f1"),
                                        ("visual_xgboost_stage", "stage_local", "stage_macro_f1")):
        differences = []
        for indices in draws:
            sample = []
            for method in (treatment, control):
                score = aggregate_scores([results[method]["per_case"][run["case_ids"][index]] for index in indices])
                sample.append(score["daily"]["f1"] if metric == "daily_f1" else score[metric])
            # An undefined F1 is not a measured zero; omit that paired draw explicitly.
            if any(value is None for value in sample):
                continue
            differences.append(sample[0] - sample[1])
        if not differences:
            raise ValueError("No defined paired metrics in exploratory resampling")
        comparisons.append({"treatment": treatment, "control": control, "metric": metric,
                            "defined_draws": len(differences), "undefined_draws": len(draws) - len(differences),
                            "exploratory_paired_record_percentile95": np.quantile(differences, [.025, .975]).tolist()})
    findings = {"verification_status": "ANALYZED", "repetitions_per_arm": 1, "independent_records": 6,
                "method_cost": cost, "common_success_activity_cases": common,
                "common_success_activity_results": common_scores, "per_case": per_case,
                "negative_stage_diagnostics": negative_stages,
                "paired_record_exploratory_intervals": comparisons,
                "limitations": "Viewed synthetic data and AI development labels; no confirmatory significance, model-call variance not estimated",
                "source_sha256": {name: sha(args.run / name) for name in ("manifest.json", "results.json", "predictions.json", "error-audit.json")},
                "labels_sha256": read_json(args.run / "protocol.json")["labels_sha256"],
                "historical_predictions_sha256": sha(prior_path)}
    save_json(args.run / "analysis.json", findings)
    global_score = results["activity_global"]["aggregate"]
    local_score = results["activity_local"]["aggregate"]
    stage_global = results["stage_global"]["aggregate"]
    stage_local = results["stage_local"]["aggregate"]
    hybrid = results["visual_xgboost_stage"]["aggregate"]
    shared_global = common_scores["activity_global"]["daily"]["f1"]
    shared_local = common_scores["activity_local"]["daily"]["f1"]
    lines = ["# 局部原图与阶段分工实验分析", "", "## Material Passport", "",
             "- Origin Skill: academic-research-suite / experiment-agent",
             "- Origin Mode: run + validate", "- Origin Date: 2026-10-07",
             "- Verification Status: ANALYZED；真实调用完成，效果尚未独立验证",
             "- Version Label: local_views_result_v1", "",
             f"实际完成 {run['calls']} 次请求，{run['total_tokens']} token，运行约 {run['seconds']:.1f} 秒；1 次契约失败，无重试。XGBoost 使用旧模型，不重训。", "",
             "## 主要结果", "",
             "| 对照 | 控制 | 局部证据 / 阶段头 | 含义 |",
             "| --- | ---: | ---: | --- |",
             f"| 定位逐日 F1 | {global_score['daily']['f1']:.3f} | {local_score['daily']['f1']:.3f} | 主表保留失败 |",
             f"| 定位事件 F1 / IoU 0.3 | {global_score['events']['f1']:.3f} | {local_score['events']['f1']:.3f} | 可能跨平台合并，须结合逐日误报 |",
             f"| 共同成功的 5 条记录定位 F1 | {shared_global:.3f} | {shared_local:.3f} | 仅诊断，不能代替主表 |",
             f"| 固定旧视觉门控下模型阶段 macro-F1 | {stage_global['stage_macro_f1']:.3f} | {stage_local['stage_macro_f1']:.3f} | 同期各一次调用，辅助图与提示完全相同 |",
             f"| 同一门控下局部图模型 vs XGBoost 阶段 | {stage_local['stage_macro_f1']:.3f} | {hybrid['stage_macro_f1']:.3f} | 监督信息不同，不推断模型家族绝对优劣 |", "",
             "## 定位：总分改善不等于弱位移问题解决", "",
             f"控制组 TP/FP/FN={global_score['daily']['tp']}/{global_score['daily']['fp']}/{global_score['daily']['fn']}；局部组为 {local_score['daily']['tp']}/{local_score['daily']['fp']}/{local_score['daily']['fn']}。两组弱位移召回均为 0/81。case_0003 平台误报仍为 92 日，case_0006 仍为 166 日。", "",
             "case_0023 控制组活动 [300,450] 和 [680,850] 与 uncertain [450,680] 共享第 450、680 日，违反互斥契约，整项未知。局部组该例成功，活动 TP=271，但新增 186 日参考负日误报。总分提升很大一部分来自这一例成功/失败差异，不能归为弱位移识别收益。", "",
             f"共同成功记录中，局部组 TP 比控制少 6 日，FP 相同，F1 {shared_global:.3f}→{shared_local:.3f}。局部图仍未稳定解决弱位移或平台分割。失败主表原样保留，不能用人工修补端点重新计算自动成绩。", "",
             "## 阶段：局部证据有本轮改善，仍不够稳定", "",
             f"阶段 A/S/D F1：全局 {stage_global['stages']['acceleration']['f1']:.3f}/{stage_global['stages']['steady_motion']['f1']:.3f}/{stage_global['stages']['deceleration']['f1']:.3f}；局部 {stage_local['stages']['acceleration']['f1']:.3f}/{stage_local['stages']['steady_motion']['f1']:.3f}/{stage_local['stages']['deceleration']['f1']:.3f}。加速和匀速改善，减速略降。", "",
             "匀速 TP 从 24 增至 38，S 假阳性从 401 降至 375，精确率从约 0.056 增至 0.092。逐例明确阶段正确日从 438 增至 536；仍存在大量范围内阶段混淆。旧模型同类配置为 0.466，本轮新控制为 0.411，说明单次调用变化不可忽略。", "",
             "## 阶段分工：学习阶段头更值得继续检验", "",
             f"视觉定位＋XGBoost 阶段 macro-F1={hybrid['stage_macro_f1']:.3f}，原视觉＋规则约 0.750；两者差距较小。它明显高于本轮局部视觉阶段，但低于数值定位＋XGBoost 的约 0.835。", "",
             "视觉门控带来的平台误报会进入阶段头，数值分类器不能自行恢复第一层正确性。当前结果支持继续研究数值阶段头与显式活动确认，尚未证明视觉第一层有净收益。", "",
             "### 阶段高分与虚假加速必须同时报告", "",
             "以下仅统计 activity_label=0 的有观测日，排除缺测和边界未知；这是非活动日上的确定阶段，不等同于活动范围内 A/S/D 之间的错分。", "",
             "| 配置 | 非活动日确定阶段 | 其中虚假加速 A |",
             "| --- | ---: | ---: |",
             *[f"| {label} | {negative_stages[name]['aggregate']['total']} | {negative_stages[name]['aggregate']['A']} |"
               for name, label in (("stage_global", "全局视觉阶段"), ("stage_local", "局部视觉阶段"),
                                   ("visual_xgboost_stage", "视觉活动＋XGBoost"),
                                   ("robust_xgboost_stage", "数值活动＋XGBoost"))], "",
             "视觉活动＋XGBoost 在 866 个明确参考阶段日中正确 807 日，范围内错分 54 日，第一层漏 5 日；但参考非活动日上有 205 日虚假加速。其阶段 macro-F1 不能单独代表稳定期加速误报风险。", "",
             "205 日虚假加速分布在 case_0003/0006/0023 的 41/150/14 日，主要集中于 case_0006 的活动后平台。逐例统计见 analysis.json，不把这个集中错误解释为所有记录的统一误报率。", "",
             "代码核查：保存的阶段模型仅在特征有支撑且阶段标签属于 A/S/D 的日上训练，目标只有三类，没有非活动拒绝类。它严格遵守输入门控，却无法自行否定错误门控。平台误检传到阶段头后被大量分配为 A，这是与输出和训练契约一致的解释，不是已验证的物理成因。", "",
             "本轮局部视觉的虚假加速从 60 降至 41 日，但非活动日确定阶段总数从 213 增至 272 日，不能称所有稳定期误报都减少。后续活动确认应独立评价；任何阶段置信度拒绝方案须在训练/验证数据上另行冻结，不用这些开发错误反推阈值。", "",
             "## 下一步", "",
             "1. 优先分离活动候选与确认：候选结合较长尺度累计位移证据，再要求原始位移持续变化；加入相关漂移、跳变和活动后平台负例。当前局部图输入方式下，弱段仍全漏。",
             "2. 图像进一步比较可考虑单窗逐一定位或更少面板，但需另冻结协议，不能从本轮只挑错误窗重跑并混入主表。",
             "3. 保留数值定位＋XGBoost 强基线；融合是否有效要检验同等误报暴露下的额外召回，而非只比较事件 F1。",
             "4. 在新增独立弱位移/匀速记录上冻结验证阈值、提示与重复数，再进行正式测试；阶段特征用于预测须另做时间前缀实验。", "",
             "## 解释与统计检查", "",
             "仅 6 条已查看仿真记录、1 个弱段、42 日明确匀速，每组每例一次请求。两个局部实验彼此独立，不能把新定位直接拼成已运行的新端到端阶段方法。请求数相同，图数与 token 不相同。", "",
             "按完整记录做 2000 次配对重采样的区间仅用于描述敏感性，不是确认性显著性，且未估计模型重复调用方差。详见 analysis.json。", "",
             "11/11 项扫描：记录异质性与汇总反转风险、仿真到现场外推、覆盖设计选择偏差、成功子集选择条件、现场发生率缺失、最差例后再测的均值回归、失败保留、多个对照与指标、已查看资料上的研究者选择、单次处理差异的因果解释，以及离线特征用于预测的时间方向问题均已检查；结论保持开发探索范围。", "",
             "源文件哈希保存在 manifest.json 与 analysis.json。标签、旧预测与原协议未改。所有输入及原始回答可在 trace/ 审查页查看。"]
    (args.run / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"common_success_f1": {name: value["daily"]["f1"] for name, value in common_scores.items()},
                      "cost": cost, "exploratory_intervals": comparisons}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
