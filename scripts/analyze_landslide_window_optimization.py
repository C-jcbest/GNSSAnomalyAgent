"""Audit and describe a completed frozen visual-window experiment without model calls."""
from __future__ import annotations

import argparse
import ast
import csv
import html
import json
from pathlib import Path

import numpy as np
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import METHODS, read_json

from gnss_sim.landslide_evaluation import aggregate_scores, runs
from gnss_sim.landslide_window_experiment import parse_window_activity


def verify_execution(out, manifest):
    for name, expected in manifest["evidence_sha256"].items():
        if sha(out / name) != expected:
            raise ValueError(f"Frozen evidence changed: {name}")
    for name, expected in manifest["source_sha256"].items():
        if sha(out / "source" / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
        if sha(ROOT / name) == expected:
            continue
        correction = read_json(out / "reporting-correction.json")
        if (name != correction["source_path"] or expected != correction["execution_sha256"]
                or sha(ROOT / name) != correction["reporting_sha256"]):
            raise ValueError(f"Undeclared source change: {name}")
        original = ast.parse((out / "source" / name).read_text(encoding="utf-8"))
        current = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        # The declared post-inference correction is limited to report generation.
        for tree in (original, current):
            tree.body = [node for node in tree.body
                         if not isinstance(node, ast.FunctionDef) or node.name != "report"]
        if ast.dump(original) != ast.dump(current):
            raise ValueError("Reporting correction changed inference or module setup")
    if sha(out / "review-packets.json") != manifest["review_packets_sha256"]:
        raise ValueError("Review candidates changed after their freeze")
    outputs = read_json(out / "screen-results.json")
    review_outputs = out / "review-results.json"
    if manifest["review_candidate_count"]:
        outputs += read_json(review_outputs)
    by_request = {row["request_id"]: row for row in outputs}
    packets = read_json(out / "screen-packets.json") + read_json(out / "review-packets.json")
    failures = read_json(out / "failures.json")
    audit = {"calls": len(packets), "verified_image_associations": 0,
             "failure_types": {}, "failed_calls_by_method": {},
             "returned_models": {}, "finish_reasons": {}}
    for packet in packets:
        request_id = packet["request_id"]
        started = read_json(out / "requests" / (request_id + ".started.json"))
        receipt = read_json(out / "requests" / (request_id + ".response.json"))
        if packet["prompt"] != started["prompt"] or started["model"] != manifest["model"]:
            raise ValueError("Actual prompt/model differs from packet")
        for name, expected in packet["image_sha256"].items():
            if sha(out / name) != expected or started["image_sha256"][Path(name).name] != expected:
                raise ValueError("Actual image differs from packet")
            audit["verified_image_associations"] += 1
        model = str(receipt.get("returned_model"))
        audit["returned_models"][model] = audit["returned_models"].get(model, 0) + 1
        reason = str(receipt.get("finish_reason"))
        audit["finish_reasons"][reason] = audit["finish_reasons"].get(reason, 0) + 1
        row = by_request[request_id]
        if row["status"] == "failed":
            if request_id not in failures or any(value != -1 for value in row["activity"]):
                raise ValueError("A failed response was silently repaired")
            failure_type = "output_contract" if receipt["status"] == "returned_json" else "request_or_json"
            audit["failure_types"][failure_type] = audit["failure_types"].get(failure_type, 0) + 1
            method = packet["method"]
            audit["failed_calls_by_method"][method] = audit["failed_calls_by_method"].get(method, 0) + 1
        else:
            case = read_json(out / "inputs" / packet["case_id"] / "input.json")
            observed = np.array([value is not None for value in case["displacement_mm"]])
            replay = parse_window_activity(json.loads(receipt["output"]), observed, packet["start"], packet["stop"])
            if replay.tolist() != row["activity"]:
                raise ValueError("Saved window result differs from actual response")
    if len(packets) != manifest["new_calls"] or len(by_request) != len(packets):
        raise ValueError("Scheduled and actual request counts differ")
    return audit


def paired_intervals(results, case_ids):
    """Record-level descriptive resampling; days are not independent samples."""
    rng = np.random.default_rng(20261008)
    draws = rng.integers(0, len(case_ids), size=(2000, len(case_ids)))
    comparisons = []
    for treatment, control in (("window_single", "window_atlas"),
                               ("window_single_review", "window_single")):
        values = {"activity_f1": [], "stage_macro_f1": []}
        for indices in draws:
            scores = [aggregate_scores([results[method]["per_case"][case_ids[index]] for index in indices])
                      for method in (treatment, control)]
            for metric in values:
                pair = [score["daily"]["f1"] if metric == "activity_f1" else score[metric] for score in scores]
                if all(value is not None for value in pair):
                    values[metric].append(pair[0] - pair[1])
        for metric, differences in values.items():
            comparisons.append({"treatment": treatment, "control": control, "metric": metric,
                                "defined_draws": len(differences), "undefined_draws": len(draws) - len(differences),
                                "descriptive_paired_record_percentile95": np.quantile(differences, [.025, .975]).tolist()})
    return comparisons


def weak_window_diagnostic(out, labels, predictions):
    """Show saved per-window evidence and aggregation loss without extra inference."""
    import matplotlib.pyplot as plt

    case_id = "case_0003"
    weak = np.array([row["feature"] == "slow_displacement" for row in labels[case_id]])
    window_rows = [row for row in read_json(out / "screen-results.json")
                   if row["case_id"] == case_id and weak[row["start"]:row["stop"]].any()]
    evidence = []
    for row in window_rows:
        covered = weak.copy()
        covered[:row["start"]] = False
        covered[row["stop"]:] = False
        activity = np.array(row["activity"])
        receipt = read_json(out / "requests" / (row["request_id"] + ".response.json"))
        evidence.append({"request_id": row["request_id"], "method": row["method"],
                         "target_inclusive": [row["start"], row["stop"] - 1],
                         "reference_weak_days_in_target": int(covered.sum()),
                         "positive": int(np.count_nonzero(covered & (activity == 1))),
                         "negative": int(np.count_nonzero(covered & (activity == 0))),
                         "unknown": int(np.count_nonzero(covered & (activity == -1))),
                         "actual_output": json.loads(receipt["output"])})
    selected = [row for row in window_rows if row["method"] == "window_single"]
    start = min(row["start"] for row in selected)
    stop = max(row["stop"] for row in selected)
    case = read_json(out / "inputs" / case_id / "input.json")
    values = np.array([row if row is not None else [np.nan] * 3 for row in case["displacement_mm"]])
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, panels = plt.subplots(8, 1, figsize=(14, 8.5), sharex=True,
                               gridspec_kw={"height_ratios": [3, 3, 3, 1, 1, 1, 1, 1]})
    for axis, name in enumerate(("N", "E", "U")):
        panels[axis].plot(np.arange(start, stop), values[start:stop, axis], linewidth=.8)
        panels[axis].set_ylabel(name + " (mm)")
        panels[axis].grid(alpha=.2)
        for left, right in runs(weak):
            panels[axis].axvspan(left, right - 1, alpha=.13, color="#d8b149")
    bands = [(f"单窗 {row['start']}–{row['stop'] - 1}", np.array(row["activity"]), row["start"], row["stop"])
             for row in selected]
    final = np.array(predictions["window_single"][case_id]["activity"])
    reference = np.array([int(row["activity_label"]) for row in labels[case_id]])
    bands += [("严格多数票", final, start, stop), ("AI开发参考", reference, start, stop)]
    for panel, (name, activity, left, right) in zip(panels[3:], bands):
        panel.set_facecolor("#f4eada")
        panel.axvspan(left - .5, right - .5, color="white")
        for code, color in ((-1, "#b9bec4"), (1, "#3b9780")):
            mask = activity == code
            mask[:left] = False
            mask[right:] = False
            for first, last in runs(mask):
                panel.axvspan(first - .5, last - .5, color=color)
        panel.set_yticks([])
        panel.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    panels[-1].set_xlim(start, stop - 1)
    panels[-1].set_xlabel("原始日索引；绿：确认活动；灰：未知；白：未确认；米色：目标窗外。黄色仅为开发参考弱段。")
    fig.suptitle("case_0003：局部已看见趋势，但重叠窗冲突使弱段未通过多数票｜事后诊断图，未送模")
    fig.subplots_adjust(left=.18, right=.98, top=.94, bottom=.08, hspace=.18)
    fig.savefig(out / "case_0003-weak-votes.png", dpi=140)
    plt.close(fig)
    return evidence


def analyze(out):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("All inference must finish before analysis")
    audit = verify_execution(out, manifest)
    results = read_json(out / "results.json")
    predictions = read_json(out / "predictions.json")
    labels = {case_id: [] for case_id in manifest["case_ids"]}
    with (out / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    details = {}
    changes = {}
    for case_id, rows in labels.items():
        actual = np.array([int(row["activity_label"]) for row in rows])
        case = read_json(out / "inputs" / case_id / "input.json")
        observed = np.array([row is not None for row in case["displacement_mm"]])
        if any(int(row["day_index"]) != index or row["date"] != case["dates"][index]
               for index, row in enumerate(rows)):
            raise ValueError("Reference calendar mismatch")
        details[case_id] = {}
        for name, records in predictions.items():
            activity = np.array(records[case_id]["activity"])
            stage = np.array(records[case_id]["stage"])
            if np.any((stage > 0) & (activity != 1)) or np.any(activity[~observed] != -1):
                raise ValueError("Predictions violate raw-activity or missing-data masks")
            details[case_id][name] = {
                "daily": results[name]["per_case"][case_id]["daily"],
                "weak_detected_days": results[name]["per_case"][case_id]["weak_detected_days"],
                "positive_unknown_days": int(np.count_nonzero((actual == 1) & (activity == -1))),
                "negative_unknown_days": int(np.count_nonzero((actual == 0) & (activity == -1))),
                "false_activity_intervals_half_open": runs((actual == 0) & (activity == 1)),
                "missed_activity_intervals_half_open": runs((actual == 1) & (activity != 1)),
                "false_acceleration_days": int(np.count_nonzero((actual == 0) & (stage == 1))),
            }
        before = np.array(predictions["window_single"][case_id]["activity"])
        after = np.array(predictions["window_single_review"][case_id]["activity"])
        changes[case_id] = {
            "removed_false_positive_days": int(np.count_nonzero((actual == 0) & (before == 1) & (after != 1))),
            "added_false_positive_days": int(np.count_nonzero((actual == 0) & (before != 1) & (after == 1))),
            "lost_true_positive_days": int(np.count_nonzero((actual == 1) & (before == 1) & (after != 1))),
            "gained_true_positive_days": int(np.count_nonzero((actual == 1) & (before != 1) & (after == 1))),
            "changed_days": int(np.count_nonzero(before != after)),
            "removed_fp_to_negative": int(np.count_nonzero((actual == 0) & (before == 1) & (after == 0))),
            "removed_fp_to_unknown": int(np.count_nonzero((actual == 0) & (before == 1) & (after == -1))),
            "lost_tp_to_negative": int(np.count_nonzero((actual == 1) & (before == 1) & (after == 0))),
            "lost_tp_to_unknown": int(np.count_nonzero((actual == 1) & (before == 1) & (after == -1))),
        }
    intervals = paired_intervals(results, manifest["case_ids"])
    weak_evidence = weak_window_diagnostic(out, labels, predictions)
    analysis = {"verification_status": "ANALYZED", "execution_audit": audit,
                "per_case": details, "review_changes": changes,
                "descriptive_resampling": intervals, "model_repetitions": 1,
                "weak_window_evidence": weak_evidence,
                "limitations": "Six viewed synthetic records; correlated AI-assisted labels; one weak episode; offline. No confirmatory inference.",
                "source_sha256": {name: sha(out / name) for name in ("results.json", "predictions.json", "manifest.json")},
                "analysis_source_sha256": sha(Path(__file__))}
    save_json(out / "analysis.json", analysis)
    lines = ["# 逐窗视觉输入与活动复核：实际优化结果", "", "## Material Passport", "",
             "- Origin Skill: academic-research-suite / experiment-agent",
             "- Origin Mode: run + descriptive validation", "- Origin Date: 2026-10-08",
             "- Verification Status: ANALYZED；实际调用和契约重放已核查；效果未独立验证",
             "- Version Label: window_optimization_v1", "",
             "## 固定条件和实际执行", "",
             "六条既有开发记录，180日窗、90日步长、每例12窗。逐窗图册与单窗各72次；相同全局图、目标提示、模型和尺度规则。单窗分辨率1536×945；图册2592×945包含四窗。每个单窗活动候选再做一次纯视觉复核，不根据评价标签挑位置。所有端到端配置共用保存的XGBoost阶段头，不重训。", "",
             f"实际 {manifest['new_calls']} 次请求，{manifest['total_tokens']} token，{manifest['seconds']:.1f} 秒。候选复核 {manifest['review_candidate_count']} 次；失败 {manifest['failures']} 次，无重试。失败窗口保留未知并进入多数票分母，平票未知；负日不是未知，复核不能扩到候选外。", "",
             "## 主表：包含全部失败", "",
             "| 配置 | 活动P | 活动R | 活动F1 | FP日 | FN日 | 阶段macro-F1 | 弱段检出/81日 | 虚假A日 | 覆盖 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, record in results.items():
        score = record["aggregate"]
        daily = score["daily"]
        lines.append(f"| {METHODS[name]} | {daily['precision']:.3f} | {daily['recall']:.3f} | {daily['f1']:.3f} | {daily['fp']} | {daily['fn']} | {score['stage_macro_f1']:.3f} | {score['weak_detected_days']} | {score['false_acceleration_days']} | {score['coverage']:.3f} |")
    lines += ["", "历史整记录调用预算不同，不能把它与逐窗的差异全部归因于图片布局。数值门控为保留的强基线。", "",
              "## 逐例与复核净变化", "",
              "| 记录 | 图册 TP/FP/FN | 单窗 TP/FP/FN | 复核 TP/FP/FN | 复核移除FP | 复核新增FP | 复核损失TP | 复核新增TP |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for case_id in manifest["case_ids"]:
        cells = []
        for method in ("window_atlas", "window_single", "window_single_review"):
            daily = details[case_id][method]["daily"]
            cells.append("/".join(str(daily[key]) for key in ("tp", "fp", "fn")))
        change = changes[case_id]
        values = [change[key] for key in ("removed_false_positive_days", "added_false_positive_days", "lost_true_positive_days", "gained_true_positive_days")]
        lines.append("| " + " | ".join([case_id] + cells + [str(value) for value in values]) + " |")
    lines += ["", "## 调用成本与执行核查", "", "```json",
              json.dumps(read_json(out / "costs.json"), ensure_ascii=False, indent=2), "```", "",
              f"核查全部 {audit['calls']} 次实际请求、{audit['verified_image_associations']} 次图片关联，完整提示和图片哈希均与冻结包一致；成功输出严格重放，失败没有补修。所有确定阶段属于预测活动且缺测未知。失败分类、服务返回模型名、finish_reason详见analysis.json。", "",
              "## 敏感性与适用限制", "",
              "按六条完整记录配对重采样2000次，未按日重采样；95%分位区间仅描述当前记录组成敏感性，不是确认性显著性或模型重复调用置信度。", "",
              "```json", json.dumps(intervals, ensure_ascii=False, indent=2), "```", "",
              "11/11项解释检查：记录汇总异质性、仿真现场外推、开发选择偏差、成功子集筛选、现场基率缺失、错误后优化的均值回归、失败/弃判保留、多对照多指标、事后选配置、单次调用因果归属、离线未来信息用于预测的时间方向均已检查。没有把开发收益解释为现场泛化、显著优越或在线预警。", "",
              "当前参考可评分5080日，正1287、负3793；12活动事件；仅81日单一弱段、42日匀速。未知和缺测不评分，预测未知正日计FN；覆盖另列。事件IoU≥0.3一对一匹配，不做PA；阶段macro-F1包含第一层漏检和负日阶段误报。", "",
              "审查入口：[主结果](index.html)、[全部实际送模图/提示/原始回答](trace.html)、[机器可读分析](analysis.json)。观察摘要evidence来自模型可见输出，未伪造内部思维链。"]
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    render_analysis_page(out, results, manifest, weak_evidence)
    print(json.dumps({"audit": audit, "review_changes": changes, "intervals": intervals}, ensure_ascii=False), flush=True)


def render_analysis_page(out, results, manifest, weak_evidence):
    table = []
    for method in ("historical_visual_gate", "window_atlas", "window_single", "window_single_review", "numeric_xgboost"):
        score = results[method]["aggregate"]
        cells = [METHODS[method], f'{score["daily"]["f1"]:.3f}',
                 score["daily"]["fp"], score["daily"]["fn"],
                 f'{score["stage_macro_f1"]:.3f}', f'{score["weak_detected_days"]}/81',
                 score["false_acceleration_days"], f'{100 * score["coverage"]:.1f}%']
        table.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in cells) + "</tr>")
    weak_rows = []
    for row in weak_evidence:
        payload = row["actual_output"]
        weak_rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in
                         [row["request_id"], row["reference_weak_days_in_target"],
                          row["positive"], row["negative"], row["unknown"], payload["evidence"]]) + "</tr>")
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>视觉优化：效果与瓶颈</title><style>
body{{font:16px system-ui,"Microsoft YaHei";background:#f4f7f5;color:#213c35;margin:0}}main{{max-width:1300px;margin:auto;padding:30px}}
h1{{font-size:30px}}h2{{margin-top:34px;font-size:22px}}p{{line-height:1.9;max-width:1120px}}a{{color:#16675a}}table{{border-collapse:collapse;width:100%;background:white}}
th,td{{border:1px solid #cddbd3;padding:11px;text-align:left}}td{{vertical-align:top}}.scroll{{overflow-x:auto}}img{{width:100%;background:white}}
.lead{{border-left:5px solid #2f8774;padding:12px 20px;background:white}}.tag{{font-size:13px;letter-spacing:1px;color:#4b6e62}}
.scroll table{{min-width:900px}}td:first-child{{min-width:150px}}
nav{{display:flex;flex-wrap:wrap;gap:18px;margin:22px 0}}details{{margin-top:20px;background:white;padding:14px}}summary{{cursor:pointer}}
</style><main><div class="tag">实际开发实验 · 2026-10-08 · 六条仿真记录</div><h1>单窗输入减少误报；弱活动仍受跨窗冲突影响</h1>
<div class="lead"><p>同调用数比较中，单窗的活动F1从0.935到0.952，误报129→9日，漏检45→110日；覆盖64.7%→92.8%。
这轮支持单窗作为更保守的视觉定位配置，尚不能说明所有异常识别都更好。额外候选复核没有净收益。</p></div>
<nav><a href="index.html">完整成绩与六例原始趋势/阶段图</a><a href="trace.html">157次真实送模图、完整提示、观察摘要和回答</a>
<a href="analysis.md">详细机器分析</a><a href="analysis.json">核查与逐例数据</a><a href="protocol.json">冻结协议</a></nav>
<h2>共同阶段头下的主结果</h2><div class="scroll"><table><tr><th>配置</th><th>活动F1</th><th>FP日</th><th>FN日</th><th>阶段macro-F1</th><th>弱段检出</th><th>虚假A日</th><th>覆盖</th></tr>{''.join(table)}</table></div>
<p>历史配置预算和提示不同，只作参考。阶段头均为原先保存的XGBoost；它没有重训，也不等于视觉模型自身的阶段识别。未知正日计FN、未知负日不算正确阴性，覆盖必须一起看。强数值基线阶段分数仍略高于单窗。</p>
<h2>弱段为什么仍漏：看见趋势不等于最后确认</h2>
<p>case_0003的参考弱段为775–855日。单窗720–899判断全窗活动；630–809判断没有活动；810–989虽然描述出N上升，仍将其置为待定。
每个弱段日期只获一张窗的确认，在两张重叠窗的严格多数规则下变为未知。因此单窗最终弱段召回0/81；图册的后一个窗口也确认活动，最终保留46/81。
这是这轮暴露的跨窗一致性与聚合瓶颈，不能只解释为图像不清楚。</p>
<a href="case_0003-weak-votes.png"><img src="case_0003-weak-votes.png" alt="原始位移与逐窗判断、多数票和AI参考的对比"></a>
<p>上图为事后诊断，未送给模型。黄色是AI开发参考，灰色为未知；全部实际送模图片均在调用审查页，无标签或导数。</p>
<details><summary>逐窗真实观察摘要与弱段投票计数</summary><div class="scroll"><table><tr><th>实际请求</th><th>目标内弱日</th><th>确认</th><th>否定</th><th>未知</th><th>模型可见输出evidence</th></tr>{''.join(weak_rows)}</table></div></details>
<h2>为什么不采用这版追加复核</h2><p>13次复核没有减少FP。case_0002的426–430被模型描述为缺口，但实际只有429缺测，426/427/428/430都有观测，四日真实活动被删。
case_0023恢复三日，净损失一日；活动事件F1从0.957降到0.917。阶段macro-F1微增0.830→0.832不能抵消这一问题。</p>
<h2>后续可检验的优化</h2><p>保留单窗输入，下一轮让各窗的阳性和冲突片段先进入候选池，再用原始位移独立确认，避免多数票在复核前丢掉弱段。
候选不直接赋活动或阶段；只有确认后才进入固定阶段头。另行冻结协议、覆盖全部候选和正常背景，再用未查看记录检验。这里没有改投票规则重算主表。</p>
<h2>执行与证据边界</h2><p>{manifest['new_calls']}次真实调用、{manifest['total_tokens']} token、约{manifest['seconds']/60:.1f}分钟；1次图册范围契约失败，无重试。
单窗72次/293034 token；图册72次/364062 token；复核额外13次/52197 token。已核查314次图片关联与全部完整提示。
推理完成后仅修正整批标签读取和报表中文字形；冻结执行源码、预测与修正记录分别保存，推理函数AST未变。</p>
<p>六条记录均已查看，标签为AI辅助参考，仅一个弱活动段。记录级配对重采样的F1差值区间跨零，尚未证明稳定优越；这不是独立盲测、专家金标准或在线预警验证。</p>
</main></html>'''
    (out / "analysis.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    analyze(parser.parse_args().run)
