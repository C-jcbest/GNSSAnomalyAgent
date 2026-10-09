"""Read-only diagnostics and evidence pages for the frozen 2x2 axis experiment."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_landslide_axis_experiment import ARM_NAMES, METHODS
from run_landslide_candidate_confirmation import verify_freeze
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json

from gnss_sim.landslide_axis_experiment import ARMS, FACTORS, axis_prompt, prompt_arm
from gnss_sim.landslide_boundary_experiment import boundary_blocks


def analyze(out):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("All 136 calls and official scoring are required")
    verify_freeze(out, manifest)
    parent = ROOT / read_json(out / "protocol.json")["source_run"]
    verify_freeze(parent, read_json(parent / "manifest.json"))
    results = read_json(out / "results.json")
    costs = read_json(out / "costs.json")
    analysis = read_json(out / "analysis.json")
    predictions = read_json(out / "predictions.json")
    outputs = read_json(out / "outputs.json")
    plot_audits = read_json(out / "plot-audit.json")
    references = {case_id: [] for case_id in manifest["case_ids"]}
    with (out / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["case_id"] in references:
                references[row["case_id"]].append(int(row["activity_label"]))
    references = {case_id: np.array(rows) for case_id, rows in references.items()}
    observed = {
        case_id: np.array([item is not None for item in
                          read_json(out / f"inputs/{case_id}/input.json")["displacement_mm"]])
        for case_id in manifest["case_ids"]
    }
    groups, returned_models, onset = {}, {}, {}
    for row in outputs:
        receipt = read_json(out / "requests" / (row["request_id"] + ".response.json"))
        returned = str(receipt.get("returned_model"))
        returned_models[returned] = returned_models.get(returned, 0) + 1
        try:
            payload = json.loads(receipt.get("output", ""))
        except (ValueError, TypeError):
            payload = None
        start, stop = row["start"], row["stop"]
        reference = references[row["case_id"]][start:stop]
        activity = np.array(row["activity"])[start:stop]
        target_counts = {
            "tp": int(np.count_nonzero((reference == 1) & (activity == 1))),
            "fp": int(np.count_nonzero((reference == 0) & (activity == 1))),
            "fn": int(np.count_nonzero((reference == 1) & (activity != 1))),
            "confirmed_reference_unknown_observed": int(np.count_nonzero(
                (reference == -1) & (activity == 1) & observed[row["case_id"]][start:stop])),
        }
        key = f"{row['case_id']}-{start}"
        groups.setdefault(key, []).append({
            "request_id": row["request_id"], "method": row["method"], "arm": row["arm"],
            "rep": row["rep"], "case_id": row["case_id"], "start": start, "stop": stop,
            "status": row["status"], "images": row["images"], "prompt": row["prompt"],
            "image_sha256": row["image_sha256"], "raw_payload": payload,
            "target_counts": target_counts,
        })
        count = onset.setdefault(row["method"], {
            "successful_nonempty": 0, "start_at_target": 0, "start_at_internal_block": 0,
            "end_at_internal_block": 0, "start_at_printed_tick": 0,
        })
        if row["status"] == "success" and payload["activity"]:
            first, last = payload["activity"][0][0], payload["activity"][-1][1]
            blocks = boundary_blocks(start, stop, prompt_arm(row["arm"]))
            audit = next(item for item in plot_audits
                         if item["case_id"] == row["case_id"] and item["start"] == start)
            ticks = audit["ticks"][str(FACTORS[row["arm"]][1])]
            count["successful_nonempty"] += 1
            count["start_at_target"] += int(first == start)
            count["start_at_internal_block"] += int(first in {left for left, _ in blocks[1:]})
            count["end_at_internal_block"] += int(last in {right - 1 for _, right in blocks[:-1]})
            count["start_at_printed_tick"] += int(first in ticks)
    for group in groups.values():
        by_method = {row["method"]: row for row in group}
        if len(group) != 8 or set(by_method) != {
                f"{arm}_rep{rep}" for arm in ARMS for rep in range(2)}:
            raise ValueError("Every target needs all eight actual answers")
        for rep in range(2):
            for arm in ARMS:
                row = by_method[f"{arm}_rep{rep}"]
                if row["prompt"] != axis_prompt(arm, row["start"], row["stop"]):
                    raise ValueError("Actual prompt does not match its factorial condition")
                global_hash = row["image_sha256"][row["images"][0]]
                if global_hash != group[0]["image_sha256"][group[0]["images"][0]]:
                    raise ValueError("Global raw evidence differs between arms")
                peer = by_method[f"{arm}_rep{1 - rep}"]
                if peer["image_sha256"] != row["image_sha256"]:
                    raise ValueError("Repeats use different images")
        group.sort(key=lambda row: list(METHODS).index(row["method"]))
    failure_separation = []
    for rep in range(2):
        common = []
        excluded = []
        common_counts = {arm: {field: 0 for field in ("tp", "fp", "fn")} for arm in ARMS}
        for key, group in groups.items():
            selected = {row["arm"]: row for row in group if row["rep"] == rep}
            if all(row["status"] == "success" for row in selected.values()):
                common.append(key)
                for arm, row in selected.items():
                    for field in common_counts[arm]:
                        common_counts[arm][field] += row["target_counts"][field]
            else:
                excluded.append({"target": key, "per_arm": {
                    arm: {"status": row["status"], **row["target_counts"]}
                    for arm, row in selected.items()
                }})
        failure_separation.append({
            "rep": rep, "common_success_target_count": len(common),
            "common_success_target_keys": common, "common_target_counts": common_counts,
            "excluded_target_diagnostics": excluded,
            "meaning": "Conditional error-burden diagnostic only. Official complete-record scores unchanged; no repaired or success-only headline scores",
        })
    unscored = {}
    for method, records in predictions.items():
        per_case = {}
        for case_id in manifest["case_ids"]:
            activity = np.array(records[case_id]["activity"])
            per_case[case_id] = int(np.count_nonzero(
                observed[case_id] & (references[case_id] == -1) & (activity == 1)))
        unscored[method] = {"per_case": per_case, "total": sum(per_case.values())}
    alignment = {}
    for arm in ARMS:
        alignment[arm] = {
            "internal_starts": sum(len(item["alignment"][arm]["internal_starts"])
                                   for item in plot_audits),
            "aligned_starts": sum(len(item["alignment"][arm]["aligned_starts"])
                                  for item in plot_audits),
        }
    close_end_ticks = []
    for item in plot_audits:
        for phase, ticks in item["ticks"].items():
            if len(ticks) >= 2 and ticks[-1] - ticks[-2] < 10:
                close_end_ticks.append({
                    "case_id": item["case_id"], "target_start": item["start"],
                    "tick_phase": int(phase), "last_pair": ticks[-2:],
                    "gap_days": ticks[-1] - ticks[-2],
                })
    summary = {}
    for arm in ARMS:
        methods = [f"{arm}_rep{rep}" for rep in range(2)]
        summary[arm] = {
            "activity_f1": [results[name]["aggregate"]["daily"]["f1"] for name in methods],
            "stage_macro_f1": [results[name]["aggregate"]["stage_macro_f1"] for name in methods],
            "fp": [results[name]["aggregate"]["daily"]["fp"] for name in methods],
            "fn": [results[name]["aggregate"]["daily"]["fn"] for name in methods],
            "weak_days": [results[name]["aggregate"]["weak_detected_days"] for name in methods],
            "false_A": [results[name]["aggregate"]["false_acceleration_days"] for name in methods],
            "failures": [costs[name]["failed_targets"] for name in methods],
            "new_tokens_total": sum(costs[name]["new_tokens"] for name in methods),
            "service_seconds_total": sum(costs[name]["service_seconds_sum"] for name in methods),
        }
    per_record_effects = []
    for rep in range(2):
        for case_id in manifest["case_ids"]:
            records = [results[f"{arm}_rep{rep}"]["per_case"][case_id] for arm in ARMS]
            fields = {}
            for metric in ("tp", "fp", "fn", "f1"):
                values = [record["daily"][metric] for record in records]
                if any(value is None for value in values):
                    fields[metric] = None
                    continue
                x00, x01, x10, x11 = values
                fields[metric] = {
                    "tick15_minus_tick0_at_block0": x01 - x00,
                    "tick15_minus_tick0_at_block15": x11 - x10,
                    "difference_of_differences": (x11 - x10) - (x01 - x00),
                    "aligned_mean_minus_unaligned_mean": (x01 + x10 - x00 - x11) / 2,
                }
            per_record_effects.append({"case_id": case_id, "rep": rep, "daily_differences": fields})
    statistical_risks = {
        "coverage": "11/11 checked; descriptive analysis, no inferential tests or confidence intervals",
        "Simpson": "Record-specific differences reported; aggregate direction does not imply every record improves",
        "ecological": "No inference from aggregate F1 to precise physical onsets or individual field stations",
        "Berkson": "Frozen candidate filtering limits target diagnostics; complete six records scored",
        "collider": "Successful-nonempty onset summaries are conditional; never replace full scores",
        "base_rate": "5080 known days: 1287 positive and 3793 negative; unknown and missing excluded",
        "regression_to_mean": "All targets in both fixed passes; no selective reruns of extreme errors",
        "survivorship": "All failed targets remain unknown in official scores",
        "look_elsewhere": "All four configurations and both repeats reported; no best-phase superiority claim",
        "forking_paths": "Inference factors frozen before calls, but this is repeated development search",
        "causation": "Tick labels and vertical grids change jointly; no internal-attention causation claim",
        "reverse_causality": "Offline retrospective ranges and centered stages are not prospective prediction",
    }
    diagnostics = {
        "status": "ANALYZED", "summary": summary, "alignment": alignment,
        "actual_target_answers": groups, "onset_behavior": onset,
        "close_end_ticks_under_10_days": close_end_ticks,
        "image_review": {
            "scope": "Same-session posthoc visual checks; not a blind reference review",
            "confirmed_end_label_overlap": [
                {"case_id": "case_0003", "target_start": 238, "tick_phase": 0,
                 "labels": [403, 404]},
                {"case_id": "case_0023", "target_start": 0, "tick_phase": 15,
                 "labels": [75, 76]},
            ],
            "consequence": "Tick intervention also affects end-label spacing; keep original inputs and scores. Do not attribute all differences to alignment alone",
        },
        "confirmed_on_reference_unknown_observed_days": unscored,
        "returned_models": returned_models, "verified_target_groups": len(groups),
        "per_record_conditional_differences": per_record_effects,
        "failure_separation_diagnostic": failure_separation,
        "statistical_risk_review": statistical_risks,
        "official_sha256": {name: sha(out / name) for name in (
            "manifest.json", "outputs.json", "predictions.json", "results.json", "analysis.json",
            "costs.json", "execution-audit.json")},
        "analysis_script_sha256": sha(Path(__file__)), "limits": manifest["scope"],
    }
    save_json(out / "diagnostics.json", diagnostics)
    plot_summary(out, summary)
    write_page(out, manifest, results, costs, analysis, diagnostics, plot_audits)
    print(json.dumps({"summary": summary, "alignment": alignment,
                      "checks": analysis["both_repeats_pass_by_arm"],
                      "effects": analysis["factor_effects"]}, ensure_ascii=False))


def plot_summary(out, summary):
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, panels = plt.subplots(1, 3, figsize=(14, 4.6))
    settings = [
        ("activity_f1", "活动逐日F1（包含失败）"),
        ("fp", "已知负日上的FP日数"),
        ("stage_macro_f1", "端到端阶段macro-F1"),
    ]
    colors = ["#147d72", "#cc6d4b"]
    for panel, (field, title) in zip(panels, settings):
        for rep in range(2):
            values = [summary[arm][field][rep] for arm in ARMS]
            positions = np.arange(4) + (-0.08 if rep == 0 else 0.08)
            panel.scatter(positions, values, color=colors[rep], marker="o" if rep == 0 else "s",
                          s=50, label=f"重复{rep + 1}")
        panel.set_xticks(np.arange(4), ["P0/T0", "P0/T15", "P15/T0", "P15/T15"])
        panel.set_title(title)
        panel.grid(axis="y", alpha=.2)
        panel.legend()
    fig.suptitle("分块×刻度交叉开发实验：每点为一次完整遍历，非独立样本／置信区间")
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=160)
    plt.close(fig)


def escape_json(value):
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def write_page(out, manifest, results, costs, analysis, diagnostics, plot_audits):
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>分块与刻度交叉实验分析</title><style>
body{font:16px/1.7 system-ui;background:#f0f4f8;color:#172638;max-width:1450px;margin:28px auto;padding:0 20px}
h1{font-size:30px}h2{font-size:23px}section,details{background:white;padding:22px;margin:17px 0;border:1px solid #dbe4ed;border-radius:10px}
.scroll{overflow:auto}table{border-collapse:collapse;min-width:100%;white-space:nowrap}
th,td{padding:10px 13px;text-align:right;border-bottom:1px solid #dbe4ed}th:first-child,td:first-child{text-align:left}
a{color:#125fa7}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fb;padding:16px;font-size:13px}
summary{cursor:pointer;font-weight:600}img{max-width:100%;height:auto}.note{border-left:4px solid #d59228;padding-left:15px}
.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}@media(max-width:850px){.pair{grid-template-columns:1fr}}
</style></head><body><h1>文字分块 × 图片刻度：因素分离与真实证据</h1>
<p><a href="index.html">全成绩与活动／阶段图</a> · <a href="trace.html">136次实际图、提示与回答</a> ·
<a href="protocol.json">冻结协议</a> · <a href="plot-audit.json">图形不变性审计</a> · <a href="diagnostics.json">完整诊断</a></p>'''
    page += (f'<p>真实新增{manifest["calls"]}次，{manifest["total_tokens"]:,} token，'
             f'失败{manifest["failures"]}次。17个候选×四组×两遍；原B1严格契约固定，'
             '分别改变文字块与局部图刻度的0／15日相位。相位0图逐字节等于历史PNG，'
             '曲线、纵轴和面板位置保持一致。下面每栏依次为重复1／重复2。</p>')
    columns = ["配置", "活动F1", "阶段macro-F1", "FP日", "FN日", "弱日", "虚假A", "失败", "总token"]
    page += '<section><h2>两遍完整成绩</h2><div class="scroll"><table><tr>'
    page += ''.join(f'<th>{column}</th>' for column in columns) + '</tr>'
    for arm in ARMS:
        row = diagnostics["summary"][arm]
        values = [ARM_NAMES[arm]]
        for field in ("activity_f1", "stage_macro_f1", "fp", "fn", "weak_days", "false_A", "failures"):
            if field in {"activity_f1", "stage_macro_f1"}:
                values.append(' / '.join(f'{value:.6f}' for value in row[field]))
            else:
                values.append(' / '.join(str(value) for value in row[field]))
        values.append(f'{row["new_tokens_total"]:,}')
        page += '<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in values) + '</tr>'
    page += '</table></div><p>各替代组两遍均通过冻结检查：' + escape_json(analysis["both_repeats_pass_by_arm"]) + '。</p>'
    page += '<img src="summary.png" alt="活动、误报和阶段的两遍完整成绩">'
    for method in ("strict_single", "numeric_reference"):
        score = results[method]["aggregate"]
        page += (f'<p>{METHODS[method]}：活动F1 {score["daily"]["f1"]:.6f}，'
                 f'阶段macro-F1 {score["stage_macro_f1"]:.6f}，FP {score["daily"]["fp"]}日，'
                 f'弱日 {score["weak_detected_days"]}。历史参考不作为同期因子对照。</p>')
    page += '<p class="note">参考未知／缺测不评分，预测未知在正日计FN；失败整目标未知，仍计入主成绩。'
    page += '六例均已查看，标签为接触过生成真值的AI辅助参考；单个弱段和两遍回答不能证明独立泛化。</p></section>'
    page += '<section><h2>这轮能说明什么</h2><p>原文字块配偏移刻度（P0/T15）两遍均减少误报和虚假加速，'
    page += '活动F1提高，但唯一弱段两遍均漏5日，第二遍还发生整目标拒判。'
    page += '它在活动后平台目标case_0006 630–989中把提前起点移到710日，FP从45／35降到0／0；'
    page += '同样的保守行为在弱位移目标把起点移到780日，产生5个FN。不能把少误报等同于全面改善。</p>'
    page += '<p>第二遍P0/T15相对P0/T0净少20个TP：失败目标在对照中有74个TP，拒判后全部丢失；'
    page += '其余16个共同成功目标净增加54个TP。112个正日属于失败目标，其中对照也有38个FN。'
    page += '这解释失败如何影响完整结果，条件诊断不用于回填或生成“修复后”分数。</p>'
    page += '<p>对齐组合的活动F1均值差第一遍+0.043230，第二遍−0.004324；第二遍含上述失败。'
    page += '四组的阶段macro-F1均低于历史严格单窗0.829918和强数值0.834878。'
    page += '因此尚未证明稳定的对齐优势或端到端阶段优势，也不从单遍0.968885挑出最终方法。</p>'
    page += '<p>下一步先在新绘图协议中统一处理刻度标签碰撞，固定少量呈现配置；'
    page += '用新增开发背景和独立原始位移参考检验弱运动／平台权衡。若检验数值边界核验，'
    page += '保持视觉候选与阶段头固定，只增加同源稳健位移证据；收益仍需实测。</p></section>'
    page += '<section><h2>条件差值与交互</h2><p>刻度效应在两种文字相位下分别计算。'
    page += '若改变刻度的效果方向随文字相位变化，不能解释为统一的“刻度偏移更好”。'
    page += '对齐均值差只作描述，不合并重复日为独立样本，不进行显著性检验或内部注意机制因果推断。</p>'
    page += '<div class="scroll"><table><tr><th>重复</th><th>指标</th><th>刻度15−0｜文字0</th><th>刻度15−0｜文字15</th><th>差值之差</th><th>对齐均值−非对齐均值</th></tr>'
    for repeat in analysis["factor_effects"]:
        for metric, effect in repeat["descriptive_conditional_differences"].items():
            fields = ["tick15_minus_tick0_at_block0", "tick15_minus_tick0_at_block15",
                      "difference_of_differences", "aligned_mean_minus_unaligned_mean"]
            values = [str(repeat["rep"] + 1), metric, *[f'{effect[field]:+.6f}' for field in fields]]
            page += '<tr>' + ''.join(f'<td>{value}</td>' for value in values) + '</tr>'
    page += '</table></div><h3>实际内部块起点与刻度对齐数</h3><pre>' + escape_json(diagnostics["alignment"]) + '</pre>'
    page += '<p>多数局部图有45日上下文，因此对齐组合主要是文字0／刻度15与文字15／刻度0；'
    page += '首尾截断产生例外。文字相位15有104个块，原相位有94个块，短首尾块使观察条数和一致性检查机会改变。'
    page += '刻度干预也改变垂直网格，图和文字因素已交叉，但这些因素内部的视觉／契约元素未进一步分离。</p>'
    page += '<p class="note">事后图片检查发现末日刻度保留会造成部分标签过近：'
    page += '原相位case_0003末端403／404、新相位case_0023末端75／76肉眼确认重叠。'
    page += '本轮保留实际输入及分数，不临时换图；刻度改变同时影响网格与末端文字间距，'
    page += '所以不能把全部差异归因于纯对齐机制。后续绘图应在新协议中统一处理标签碰撞。</p>'
    page += '<details><summary>末端相邻刻度不足10日的图（不等同于全部标签重叠）</summary><pre>'
    page += escape_json(diagnostics["close_end_ticks_under_10_days"]) + '</pre></details>'
    page += '<details><summary>逐记录条件差值（无正样本时F1不可算）</summary><pre>'
    page += escape_json(diagnostics["per_record_conditional_differences"]) + '</pre></details></section>'
    page += '<section><h2>失败与重复分歧</h2><pre>' + escape_json({
        "failures": analysis["failure_diagnostics"], "repeat_disagreement": analysis["repeat_disagreement"],
        "onset_behavior": diagnostics["onset_behavior"],
    }) + '</pre><p>成功非空起点统计是条件诊断，不能以此替换完整评分。</p>'
    page += '<details><summary>分开契约失败与共同成功目标的TP/FP/FN（条件诊断，非正式新成绩）</summary><pre>'
    page += escape_json(diagnostics["failure_separation_diagnostic"]) + '</pre></details></section>'
    page += '<section><h2>逐目标八份实际回答</h2><p>每个目标展示两张不同刻度的真实局部输入；'
    page += '对应全局图也实际送模。表中区间来自原回答，失败区间不是有效预测；逐目标TP/FP/FN依据冻结参考计算。</p>'
    for audit in plot_audits:
        key = f'{audit["case_id"]}-{audit["start"]}'
        group = diagnostics["actual_target_answers"][key]
        page += f'<details id="{key}"><summary>{audit["case_id"]} · 目标{audit["start"]}–{audit["stop"] - 1}</summary>'
        page += '<div class="pair">'
        for phase in (0, 15):
            path = audit["local_images"][str(phase)]
            page += (f'<div><h3>图刻度相位{phase}（实际送模）</h3><a href="{path}">'
                     f'<img loading="lazy" src="{path}" alt="{key}刻度{phase}"></a>'
                     f'<p>刻度：{audit["ticks"][str(phase)]}</p></div>')
        page += '</div><details><summary>共享全局原图（实际送模）</summary>'
        page += f'<img loading="lazy" src="{audit["global_image"]}" alt="{key}全局原图"></details>'
        page += '<div class="scroll"><table><tr><th>方法／重复</th><th>状态</th><th>原activity</th><th>原uncertain</th><th>TP</th><th>FP</th><th>FN</th><th>未知参考确认</th></tr>'
        for row in group:
            payload = row["raw_payload"] if isinstance(row["raw_payload"], dict) else {}
            counts = row["target_counts"]
            values = [METHODS[row["method"]], row["status"], str(payload.get("activity")),
                      str(payload.get("uncertain")), *[str(counts[field]) for field in (
                          "tp", "fp", "fn", "confirmed_reference_unknown_observed")]]
            page += '<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in values) + '</tr>'
        page += '</table></div>'
        for row in group:
            page += '<details><summary>' + html.escape(row["request_id"]) + ' · 完整实际提示／原回答</summary>'
            page += '<h3>实际提示</h3><pre>' + html.escape(row["prompt"]) + '</pre>'
            page += '<h3>原回答</h3><pre>' + escape_json(row["raw_payload"]) + '</pre>'
            page += f'<a href="requests/{row["request_id"]}.response.json">完整请求响应记录</a></details>'
        page += '</details>'
    page += '</section><section><h2>有观测但参考未知的确认</h2><pre>'
    page += escape_json(diagnostics["confirmed_on_reference_unknown_observed_days"])
    page += '</pre><p>这些日期未评分。不能把零FP解释为未知边界上的提前日期正确。</p></section>'
    page += '<section><h2>逐记录变化与冻结检查</h2><pre>' + escape_json({
        "checks": analysis["development_checks"], "paired": analysis["paired_known_day_changes"],
    }) + '</pre></section><details><summary>统计解释风险：11项检查</summary><pre>'
    page += escape_json(diagnostics["statistical_risk_review"]) + '</pre></details>'
    page += '<p>先原始位移确认活动，再在活动内部划分阶段。评分图在index页，未送模型；'
    page += '离线居中特征含未来数据，本轮不检验在线预警或预测提升。全部实际回答保留，不补造隐藏思维链。</p></body></html>'
    (out / "analysis.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    analyze(parser.parse_args().out)
