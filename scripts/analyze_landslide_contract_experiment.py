"""Present complete contract trials and raw observations without repairing predictions."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from run_landslide_candidate_confirmation import verify_freeze
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_contract_experiment import ARM_NAMES, METHODS
from run_landslide_window_optimization import read_json

from gnss_sim.landslide_confirmation_prompt_experiment import target_blocks
from gnss_sim.landslide_contract_experiment import ARMS


def analyze(out):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("All 68 requests and official scoring are required")
    verify_freeze(out, manifest)
    parent = ROOT / read_json(out / "protocol.json")["source_run"]
    verify_freeze(parent, read_json(parent / "manifest.json"))
    results, costs = read_json(out / "results.json"), read_json(out / "costs.json")
    analysis = read_json(out / "analysis.json")
    outputs = read_json(out / "outputs.json")
    predictions = read_json(out / "predictions.json")
    references = {case_id: [] for case_id in manifest["case_ids"]}
    with (out / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["case_id"] in references:
                references[row["case_id"]].append(int(row["activity_label"]))
    observed_by_case = {
        case_id: np.array([item is not None for item in read_json(out / f"inputs/{case_id}/input.json")["displacement_mm"]])
        for case_id in manifest["case_ids"]
    }
    unscored_activity = {}
    for method, records in predictions.items():
        per_case = {}
        for case_id in manifest["case_ids"]:
            observed = observed_by_case[case_id]
            reference = np.array(references[case_id])
            activity = np.array(records[case_id]["activity"])
            per_case[case_id] = int(np.count_nonzero(observed & (reference == -1) & (activity == 1)))
        unscored_activity[method] = {"per_case": per_case, "total": sum(per_case.values())}
    groups, models, onset = {}, {}, {}
    for row in outputs:
        receipt = read_json(out / "requests" / (row["request_id"] + ".response.json"))
        returned = str(receipt.get("returned_model"))
        models[returned] = models.get(returned, 0) + 1
        payload = None
        try:
            payload = json.loads(receipt.get("output", ""))
        except (ValueError, TypeError):
            pass
        group = groups.setdefault(f"{row['case_id']}-{row['start']}", [])
        group.append({
            "request_id": row["request_id"], "case_id": row["case_id"], "method": row["method"],
            "start": row["start"], "stop": row["stop"], "status": row["status"],
            "image_sha256": row["image_sha256"], "images": row["images"], "prompt": row["prompt"],
            "raw_payload": payload, "actual_output": receipt.get("output"),
        })
        count = onset.setdefault(row["method"], {"successful_nonempty": 0, "start_at_target": 0,
                                                 "start_at_internal_block": 0, "end_at_internal_block": 0})
        if row["status"] == "success" and payload["activity"]:
            first, last = payload["activity"][0][0], payload["activity"][-1][1]
            blocks = target_blocks(row["start"], row["stop"])
            count["successful_nonempty"] += 1
            count["start_at_target"] += int(first == row["start"])
            count["start_at_internal_block"] += int(first in {left for left, _ in blocks[1:]})
            count["end_at_internal_block"] += int(last in {right - 1 for _, right in blocks[:-1]})
    for group in groups.values():
        if len(group) != 4 or any(row["image_sha256"] != group[0]["image_sha256"] for row in group):
            raise ValueError("Four same-image answers are required for every original target")
        group.sort(key=lambda row: list(METHODS).index(row["method"]))
    summary = {}
    for arm in ARMS:
        methods = [f"{arm}_rep{rep}" for rep in range(2)]
        summary[arm] = {
            "daily_f1_per_repeat": [results[name]["aggregate"]["daily"]["f1"] for name in methods],
            "stage_macro_f1_per_repeat": [results[name]["aggregate"]["stage_macro_f1"] for name in methods],
            "fp_per_repeat": [results[name]["aggregate"]["daily"]["fp"] for name in methods],
            "fn_per_repeat": [results[name]["aggregate"]["daily"]["fn"] for name in methods],
            "weak_days_per_repeat": [results[name]["aggregate"]["weak_detected_days"] for name in methods],
            "false_A_per_repeat": [results[name]["aggregate"]["false_acceleration_days"] for name in methods],
            "failed_targets_per_repeat": [costs[name]["failed_targets"] for name in methods],
            "new_tokens_total": sum(costs[name]["new_tokens"] for name in methods),
        }
    common_success = []
    for target, group in groups.items():
        by_method = {row["method"]: row for row in group}
        for rep in range(2):
            control, treatment = [by_method[f"{arm}_rep{rep}"] for arm in ARMS]
            item = {"target": target, "rep": rep, "control_status": control["status"],
                    "treatment_status": treatment["status"],
                    "control_activity": control["raw_payload"].get("activity") if isinstance(control["raw_payload"], dict) else None,
                    "treatment_activity": treatment["raw_payload"].get("activity") if isinstance(treatment["raw_payload"], dict) else None}
            item["both_successful_nonempty"] = bool(
                control["status"] == treatment["status"] == "success"
                and item["control_activity"] and item["treatment_activity"]
            )
            if item["both_successful_nonempty"]:
                item["first_day_delta"] = item["treatment_activity"][0][0] - item["control_activity"][0][0]
                item["last_day_delta"] = item["treatment_activity"][-1][1] - item["control_activity"][-1][1]
            common_success.append(item)
    semantic = read_json(out / "semantic-review.json") if (out / "semantic-review.json").exists() else None
    if semantic is not None:
        if {row["request_id"] for row in semantic["reviews"]} != {row["request_id"] for row in outputs}:
            raise ValueError("Semantic review must preserve all 68 calls")
        for row in semantic["reviews"]:
            receipt = out / "requests" / (row["request_id"] + ".response.json")
            if sha(receipt) != row["receipt_sha256"]:
                raise ValueError("Semantic review receipt changed")
    diagnostics = {
        "status": "ANALYZED", "summary": summary, "actual_target_answers": groups,
        "onset_behavior": onset, "common_success_boundary_diagnostic": common_success,
        "confirmed_on_reference_unknown_observed_days": unscored_activity,
        "semantic_review": semantic, "returned_models": models, "same_image_target_groups": len(groups),
        "official_sha256": {name: sha(out / name) for name in
                            ("manifest.json", "outputs.json", "predictions.json", "results.json", "analysis.json")},
        "analysis_script_sha256": sha(Path(__file__)), "limits": manifest["scope"],
    }
    save_json(out / "diagnostics.json", diagnostics)
    write_page(out, manifest, results, costs, analysis, diagnostics)
    print(json.dumps({"summary": summary, "checks": analysis["development_checks"],
                      "failures": analysis["failure_diagnostics"], "onset": onset}, ensure_ascii=False))


def write_page(out, manifest, results, costs, analysis, diagnostics):
    page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>活动契约优化分析</title><style>body{font:16px/1.7 system-ui;background:#f0f4f8;color:#172638;max-width:1450px;margin:28px auto;padding:0 20px}h1{font-size:30px}h2{font-size:23px}section,details{background:white;padding:22px;margin:17px 0;border:1px solid #dbe4ed;border-radius:10px}.scroll{overflow:auto}table{border-collapse:collapse;min-width:100%;white-space:nowrap}th,td{padding:10px 13px;text-align:right;border-bottom:1px solid #dbe4ed}th:first-child,td:first-child{text-align:left}a{color:#125fa7}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fb;padding:16px;font-size:13px}summary{cursor:pointer;font-weight:600}img{max-width:100%;height:auto}.note{border-left:4px solid #d59228;padding-left:15px}</style></head><body><h1>活动输出契约简化：效果与错误分析</h1><p><a href="index.html">全部成绩与活动／阶段图</a> · <a href="trace.html">全部68次实际图片、提示与回答</a> · <a href="protocol.json">冻结协议</a> · <a href="diagnostics.json">完整诊断</a></p>'
    page += f'<section><h2>完整重复结果</h2><p>{manifest["calls"]}次真实调用、{manifest["total_tokens"]:,} token、{manifest["failures"]}次目标失败。原图、候选、30日相位0分块、模型、阶段头和标签固定。C0逐字沿用B1；C1仅移除块state与first_activity_day，保留观察文字和最终activity／uncertain。两遍分开报告，不取最好回答。</p><div class="scroll"><table><tr><th>配置</th><th>活动F1</th><th>FP日</th><th>FN日</th><th>阶段macro-F1</th><th>弱日</th><th>虚假A</th><th>目标失败</th><th>新token</th></tr>'
    for arm in ARMS:
        row = diagnostics["summary"][arm]
        values = [ARM_NAMES[arm], ' / '.join(f'{value:.6f}' for value in row["daily_f1_per_repeat"]),
                  ' / '.join(map(str, row["fp_per_repeat"])), ' / '.join(map(str, row["fn_per_repeat"])),
                  ' / '.join(f'{value:.6f}' for value in row["stage_macro_f1_per_repeat"]),
                  ' / '.join(f'{value}/81' for value in row["weak_days_per_repeat"]),
                  ' / '.join(map(str, row["false_A_per_repeat"])),
                  ' / '.join(map(str, row["failed_targets_per_repeat"])), f'{row["new_tokens_total"]:,}']
        page += '<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in values) + '</tr>'
    page += '</table></div><p>各栏依次为重复1／重复2。两遍是否均满足冻结开发判据：' + ('是' if analysis["both_repeats_pass"] else '否') + '。</p></section>'
    control, treatment = [diagnostics["summary"][arm] for arm in ARMS]
    fp_delta = [after - before for before, after in zip(control["fp_per_repeat"], treatment["fp_per_repeat"])]
    false_a_delta = [after - before for before, after in zip(control["false_A_per_repeat"], treatment["false_A_per_repeat"])]
    strict_score = results["strict_single"]["aggregate"]
    numeric_score = results["numeric_reference"]["aggregate"]
    page += f'<section><h2>本轮判断</h2><p>C1相对同期C0的误报日变化为{fp_delta[0]:+d}／{fp_delta[1]:+d}，虚假加速日变化为{false_a_delta[0]:+d}／{false_a_delta[1]:+d}。召回、拒判和误报必须一起解释，未通过冻结检查时不直接替换主方法。</p>'
    for row in analysis["paired_known_day_changes"]:
        net_tp = sum(case["gained_TP"] - case["lost_TP"] for case in row["per_case"].values())
        control_failures = [failure for failure in analysis["failure_diagnostics"]
                            if failure["method"] == f'{ARMS[0]}_rep{row["rep"]}']
        failed_positive = sum(failure["reference_positive_days"] for failure in control_failures)
        page += f'<p>重复{row["rep"] + 1}的C0有{len(control_failures)}个拒判目标，合计{failed_positive}参考正日未知计FN；全部记录C1−C0净增{net_tp}个TP。须结合逐记录／目标区间分解，不能将全部收益归因于更准确的读图。</p>'
    page += f'<p>历史严格单窗活动F1为{strict_score["daily"]["f1"]:.6f}、阶段macro-F1为{strict_score["stage_macro_f1"]:.6f}，强数值阶段macro-F1为{numeric_score["stage_macro_f1"]:.6f}。不以活动F1接近严格单窗推定完整阶段优势。下一步把文字分块与图片刻度位置分别控制，输出契约作为冻结因素；独立数据及观察参考继续准备。</p></section>'
    page += '<section><h2>开发判据与流程收益</h2><p class="note">删除字段同时减少机械交叉约束，因此少拒判不能单独解释为视觉理解更准确。判断还必须检查完整FP／FN、弱活动及端到端阶段；文字一致性审查不改预测或补算修复分数。</p><div class="scroll"><table><tr><th>重复</th><th>C1−C0活动F1</th><th>阶段macro-F1</th><th>通过冻结检查</th></tr>'
    for row in analysis["development_checks"]:
        page += f'<tr><td>{row["rep"] + 1}</td><td>{row["activity_f1_delta"]:+.6f}</td><td>{row["stage_macro_f1_delta"]:+.6f}</td><td>{"是" if row["passes_frozen_check"] else "否"}</td></tr>'
    page += '</table></div><p>同期C0用于因素比较，历史B1／B2、严格单窗和强数值仍列全部成绩；不把历史最好一次当同期对照。共同5080已知日，参考未知／缺测不评；失败与预测未知在正日计FN，覆盖另报。事件匹配不是精确范围正确。覆盖100%不是准确率100%；参考未知日期的活动预测也不能从零FP反推正确。</p></section>'
    page += '<section><h2>拒判及正日代价</h2>'
    if not analysis["failure_diagnostics"]:
        page += '<p>本轮没有目标级契约失败。两组都没有失败时，不能把历史失败这次消失归功于C1。</p>'
    for row in analysis["failure_diagnostics"]:
        page += f'<details><summary>{html.escape(row["request_id"])} · {row["reference_positive_days"]}正日未知计FN</summary><pre>{html.escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre></details>'
    page += '</section><section><h2>重复分歧与边界行为</h2><p>0／1／未知差异包含拒判；成功非空起点诊断具有条件性，正式评分未筛掉失败／空回答。</p><pre>' + html.escape(json.dumps({"repeat_disagreement": analysis["repeat_disagreement"], "onset_behavior": diagnostics["onset_behavior"]}, ensure_ascii=False, indent=2)) + '</pre></section>'
    page += '<section><h2>全部17目标：同图四份回答</h2><p>展开后可见共同原始位移图、各回答区间、逐块观察及完整提示；这些观察是模型实际输出。参考／预测叠加图只用于事后评分，没有送给模型。</p></section>'
    for target, group in diagnostics["actual_target_answers"].items():
        first = group[0]
        page += f'<details><summary>{html.escape(target)} · 目标{first["start"]}–{first["stop"] - 1}</summary>'
        for image in first["images"]:
            page += f'<a href="{image}"><img loading="lazy" src="{image}" alt="共同实际送模原图"></a>'
        page += '<div class="scroll"><table><tr><th>配置</th><th>状态</th><th>原始activity（闭区间）</th><th>原始uncertain</th></tr>'
        for row in group:
            payload = row["raw_payload"] if isinstance(row["raw_payload"], dict) else {}
            values = [METHODS[row["method"]], row["status"], str(payload.get("activity")), str(payload.get("uncertain"))]
            page += '<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in values) + '</tr>'
        page += '</table></div>'
        for row in group:
            page += '<details><summary>' + html.escape(METHODS[row["method"]] + ' · 实际逐块观察及回答') + '</summary><pre>' + html.escape(row["actual_output"] or "未返回回答") + '</pre><details><summary>完整实际提示</summary><pre>' + html.escape(row["prompt"]) + '</pre></details></details>'
        page += '</details>'
    page += '<section><h2>逐记录已知日期增失</h2><pre>' + html.escape(json.dumps(analysis["paired_known_day_changes"], ensure_ascii=False, indent=2)) + '</pre></section>'
    page += '<section><h2>参考未知日期上的确认（未评分）</h2><p>以下只统计有原始观测、参考活动仍未知的日期。预测延伸至这些日期不计正式FP或TP，须另行复核，不能因为正式FP为零就认为提前起点正确。</p><div class="scroll"><table><tr><th>配置</th><th>确认于参考未知的观测日</th></tr>'
    for method, row in diagnostics["confirmed_on_reference_unknown_observed_days"].items():
        page += f'<tr><td>{html.escape(METHODS[method])}</td><td>{row["total"]}</td></tr>'
    page += '</table></div><details><summary>逐记录未评分日期计数</summary><pre>' + html.escape(json.dumps(diagnostics["confirmed_on_reference_unknown_observed_days"], ensure_ascii=False, indent=2)) + '</pre></details></section>'
    semantic = diagnostics["semantic_review"]
    page += '<section><h2>文字与区间一致性审查</h2><p>同会话AI事后审查，只判断回答文字与其区间是否明显相互矛盾，不建立新真值，不验证原图物理真实性，不影响原评分。</p>'
    if semantic is None:
        page += '<p>尚未完成；不以无机器失败声称文字一致。</p>'
    else:
        page += '<pre>' + html.escape(json.dumps(semantic["summary"], ensure_ascii=False, indent=2)) + '</pre>'
        for row in semantic["reviews"]:
            if row["verdict"] != "no_clear_text_range_conflict":
                page += '<p><strong>' + html.escape(row["request_id"]) + '</strong>：' + html.escape(row["notes"]) + '</p>'
        page += '<details><summary>全部68份文字审查记录及来源哈希</summary><pre>' + html.escape(json.dumps(semantic, ensure_ascii=False, indent=2)) + '</pre></details>'
    page += '</section>'
    page += '<section><h2>适用边界</h2><p>六条已查看仿真记录、相关AI辅助标签、唯一81日弱段与两个重复，均属于开发分析；离线居中特征含未来信息。最终方法与准确率优势仍需未查看数据及独立观察参考。块列表／图片刻度对齐留待另一次预定试验，本轮没有改图。</p></section></body></html>'
    (out / "analysis.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    analyze(parser.parse_args().out)
