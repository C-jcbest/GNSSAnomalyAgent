"""Describe repeated visual-boundary answers without repairing official predictions."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from run_landslide_boundary_experiment import ARM_NAMES, METHODS
from run_landslide_candidate_confirmation import verify_freeze
from run_landslide_comparison import ROOT, save_json, sha

from gnss_sim.landslide_boundary_experiment import ARMS, boundary_blocks
from gnss_sim.landslide_evaluation import runs


def analyze(out):
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("All 102 requests and scoring must finish before interpretation")
    verify_freeze(out, manifest)

    def read(name):
        return json.loads((out / name).read_text(encoding="utf-8"))

    results, predictions = read("results.json"), read("predictions.json")
    costs, analysis = read("costs.json"), read("analysis.json")
    outputs = read("outputs.json")
    prompt_source = ROOT / read("protocol.json")["source_run"]
    candidate_source = ROOT / json.loads((prompt_source / "protocol.json").read_text(encoding="utf-8"))["source_run"]
    candidate_manifest = json.loads((candidate_source / "manifest.json").read_text(encoding="utf-8"))
    verify_freeze(candidate_source, candidate_manifest)
    candidate_protocol = json.loads((candidate_source / "protocol.json").read_text(encoding="utf-8"))
    context_days = candidate_protocol["review_context_days"]
    if context_days != 45:
        raise ValueError("Historical raw-image context differs from inspected renderer")
    references = {case_id: [] for case_id in manifest["case_ids"]}
    with (out / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["case_id"] in references:
                references[row["case_id"]].append(int(row["activity_label"]))
    per_case_errors = {}
    for method, records in predictions.items():
        per_case_errors[method] = {}
        for case_id, record in records.items():
            actual = np.array(references[case_id])
            observed = np.array([item is not None for item in read(f"inputs/{case_id}/input.json")["displacement_mm"]])
            activity, stage = np.array(record["activity"]), np.array(record["stage"])
            if (np.any(activity[~observed] != -1) or np.any(stage[~observed] != -1)
                    or np.any((stage > 0) & (activity != 1))):
                raise ValueError("Missing-date mask or activity-to-stage gate changed")
            per_case_errors[method][case_id] = {
                "fp_half_open": runs((actual == 0) & (activity == 1)),
                "fn_half_open": runs((actual == 1) & (activity != 1)),
                "false_A_days": int(np.count_nonzero((actual == 0) & (stage == 1))),
            }
        daily = results[method]["aggregate"]["daily"]
        for label in ("fp", "fn"):
            total = sum(right - left for case in per_case_errors[method].values()
                        for left, right in case[label + "_half_open"])
            if total != daily[label]:
                raise ValueError("Error ranges do not conserve official counts")
    groups, failure_diagnostics, models = {}, [], {}
    for row in outputs:
        receipt = read(f"requests/{row['request_id']}.response.json")
        returned = str(receipt.get("returned_model"))
        models[returned] = models.get(returned, 0) + 1
        key = f"{row['case_id']}-{row['start']}"
        payload = None
        try:
            payload = json.loads(receipt.get("output", ""))
        except (ValueError, TypeError):
            pass
        groups.setdefault(key, []).append({
            "request_id": row["request_id"], "case_id": row["case_id"],
            "method": row["method"], "status": row["status"],
            "start": row["start"], "stop": row["stop"], "images": row["images"],
            "image_sha256": row["image_sha256"], "prompt": row["prompt"],
            "actual_output": receipt.get("output"),
            "raw_activity": payload.get("activity") if isinstance(payload, dict) else None,
        })
        if row["status"] != "failed":
            continue
        actual = np.array(references[row["case_id"]])[row["start"]:row["stop"]]
        failure_diagnostics.append({
            "request_id": row["request_id"], "method": row["method"],
            "target_half_open": [row["start"], row["stop"]],
            "error": read("failures.json")[row["request_id"]],
            "reference_positive_days_in_failed_target": int(np.count_nonzero(actual == 1)),
            "raw_invalid_activity": payload.get("activity") if isinstance(payload, dict) else None,
            "note": "Official target remains unknown; no repaired output or counterfactual score",
        })
    for group in groups.values():
        if len(group) != 6 or any(row["image_sha256"] != group[0]["image_sha256"] for row in group):
            raise ValueError("Six-arm/repeat target images differ or a result is missing")
        group.sort(key=lambda row: list(METHODS).index(row["method"]))
    phase_boundary_changes = []
    grid_alignment = []
    for target, group in groups.items():
        by_method = {row["method"]: row for row in group}
        start, stop = group[0]["start"], group[0]["stop"]
        days = len(references[group[0]["case_id"]])
        plot_start, plot_stop = max(0, start - context_days), min(days, stop + context_days)
        ticks = set(range(plot_start, plot_stop, 30)) | {plot_stop - 1}
        for arm in ("b1_endpoints", "b2_phase15"):
            internal_starts = [left for left, _ in boundary_blocks(start, stop, arm)[1:]]
            grid_alignment.append({"target": target, "arm": arm,
                                   "internal_block_starts": internal_starts,
                                   "starts_on_printed_image_tick": [left for left in internal_starts if left in ticks]})
        for rep in range(2):
            before, after = [by_method[f"{arm}_rep{rep}"] for arm in ("b1_endpoints", "b2_phase15")]
            item = {"target": target, "rep": rep,
                    "b1_status": before["status"], "b2_status": after["status"],
                    "b1_activity": before["raw_activity"], "b2_activity": after["raw_activity"]}
            if (before["status"] == after["status"] == "success"
                    and before["raw_activity"] and after["raw_activity"]):
                first_before, first_after = before["raw_activity"][0][0], after["raw_activity"][0][0]
                item["first_day_delta"] = first_after - first_before
                item["last_day_delta"] = after["raw_activity"][-1][1] - before["raw_activity"][-1][1]
                for prefix, arm, first in (("b1", "b1_endpoints", first_before),
                                           ("b2", "b2_phase15", first_after)):
                    grid = boundary_blocks(before["start"], before["stop"], arm)
                    item[prefix + "_first_is_internal_grid_edge"] = first in {left for left, _ in grid[1:]}
            phase_boundary_changes.append(item)
    summary = {}
    for arm in ARMS:
        methods = [f"{arm}_rep{rep}" for rep in range(2)]
        summary[arm] = {
            "daily_f1_per_repeat": [results[name]["aggregate"]["daily"]["f1"] for name in methods],
            "fp_per_repeat": [results[name]["aggregate"]["daily"]["fp"] for name in methods],
            "fn_per_repeat": [results[name]["aggregate"]["daily"]["fn"] for name in methods],
            "weak_days_per_repeat": [results[name]["aggregate"]["weak_detected_days"] for name in methods],
            "false_A_per_repeat": [results[name]["aggregate"]["false_acceleration_days"] for name in methods],
            "failed_targets_per_repeat": [costs[name]["failed_targets"] for name in methods],
            "new_tokens_total": sum(costs[name]["new_tokens"] for name in methods),
        }
    paired_changes = []
    for rep in range(2):
        for treatment_arm, control_arm in (("b1_endpoints", "b0_blocks"),
                                           ("b2_phase15", "b1_endpoints")):
            treatment, control = f"{treatment_arm}_rep{rep}", f"{control_arm}_rep{rep}"
            changes = {}
            for case_id, reference in references.items():
                actual = np.array(reference)
                before = np.array(predictions[control][case_id]["activity"])
                after = np.array(predictions[treatment][case_id]["activity"])
                changes[case_id] = {
                    "gained_TP": int(np.count_nonzero((actual == 1) & (before != 1) & (after == 1))),
                    "lost_TP": int(np.count_nonzero((actual == 1) & (before == 1) & (after != 1))),
                    "removed_FP": int(np.count_nonzero((actual == 0) & (before == 1) & (after != 1))),
                    "added_FP": int(np.count_nonzero((actual == 0) & (before != 1) & (after == 1))),
                }
            before_score = results[control]["aggregate"]["daily"]
            after_score = results[treatment]["aggregate"]["daily"]
            tp_change = sum(change["gained_TP"] - change["lost_TP"] for change in changes.values())
            fp_change = sum(change["added_FP"] - change["removed_FP"] for change in changes.values())
            if (tp_change != after_score["tp"] - before_score["tp"]
                    or fp_change != after_score["fp"] - before_score["fp"]):
                raise ValueError("Paired error changes do not conserve official counts")
            paired_changes.append({"treatment": treatment, "control": control, "per_case": changes})
    diagnostics = {
        "status": "ANALYZED", "summary": summary, "actual_target_answers": groups,
        "failure_diagnostics": failure_diagnostics, "per_case_errors": per_case_errors,
        "paired_known_day_changes": paired_changes,
        "phase_boundary_changes": phase_boundary_changes,
        "printed_tick_alignment": grid_alignment,
        "alignment_diagnostic_provenance": {
            "candidate_protocol_sha256": sha(candidate_source / "protocol.json"),
            "frozen_renderer_sha256": sha(out / "source/scripts/run_landslide_window_optimization.py"),
            "context_days": context_days,
            "tick_rule": "range(max(0,start-45), min(days,stop+45),30) plus plot_stop-1",
            "interpretation": "Post-protocol diagnostic; phase15 was fixed before this observation, no image or schedule changes",
        },
        "returned_models": models, "same_image_target_groups": len(groups),
        "count_conservation_verified": True, "missing_and_stage_masks_verified": True,
        "official_sha256": {name: sha(out / name) for name in
                            ("manifest.json", "outputs.json", "predictions.json", "results.json")},
        "analysis_script_sha256": sha(Path(__file__)), "limits": manifest["scope"],
    }
    save_json(out / "diagnostics.json", diagnostics)
    write_page(out, manifest, results, analysis, diagnostics)
    print(json.dumps({"summary": summary, "failures": failure_diagnostics}, ensure_ascii=False))


def write_page(out, manifest, results, analysis, diagnostics):
    style = 'body{font:16px system-ui,"Microsoft YaHei";margin:24px auto;padding:0 20px;max-width:1500px;color:#203e34;background:#f4f7f5}p{line-height:1.9}section{background:white;padding:22px;margin:22px 0;border:1px solid #d8e3df}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f4f7f5;padding:16px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #d8e3df;white-space:nowrap;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}summary{cursor:pointer;font-weight:600}a{color:#176757}'
    page = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>端点与分块：实际重复分析</title><style>{style}</style></head><body><h1>端点与分块：两次实际重复的效果</h1><p><a href="index.html">完整结果与六例活动／阶段图</a> · <a href="trace.html">102次完整调用</a> · <a href="diagnostics.json">全部17目标的配对回答</a></p>'
    page += f'<section><h2>全部重复保留</h2><p>{manifest["calls"]}次真实调用、{manifest["total_tokens"]:,} token、{manifest["failures"]}次目标失败；三种配置分别完整重复两次。B0原分块；B1只附加端点检查与观测窗口截断说明；B2只在B1基础上改变分块列表相位。原始图与阶段头不变。</p><p>两次重复不构成两个独立数据集，不挑最好回答；下面所有范围均依次列出重复1／重复2，不能把它们当作统计置信区间。</p><div class="scroll"><table><tr><th>配置</th><th>活动F1</th><th>FP日</th><th>FN日</th><th>弱日</th><th>虚假A</th><th>目标失败</th><th>两次新token</th></tr>'
    for arm, summary in diagnostics["summary"].items():
        values = [ARM_NAMES[arm], " / ".join(f'{value:.6f}' for value in summary["daily_f1_per_repeat"])]
        for key in ("fp_per_repeat", "fn_per_repeat", "weak_days_per_repeat", "false_A_per_repeat", "failed_targets_per_repeat"):
            values.append(" / ".join(str(value) for value in summary[key]))
        values.append(f'{summary["new_tokens_total"]:,}')
        page += '<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in values) + '</tr>'
    strict = results["strict_single"]["aggregate"]
    numeric = results["numeric_reference"]["aggregate"]
    check_text = "是" if analysis["both_repeats_pass"] else "否"
    page += f'</table></div><p>历史严格单窗活动F1={strict["daily"]["f1"]:.6f}，强数值活动F1={numeric["daily"]["f1"]:.6f}；两者弱日均0/81。历史与同期重复分开，不直接用历史成功/失败替代同期对照。</p><p>B1是否两次均满足冻结开发检查：{check_text}。失败正日计FN，覆盖与端到端阶段完整保留。</p></section>'
    page += '<section><h2>本轮结论：相位组有开发收益，仍没有稳定优势</h2><p>B1未通过两个重复的开发检查：第一遍活动F1低于B0且新增一个块状态矛盾失败；第二遍F1提高但FP也增加，弱段从81日降到76日，实际起点780遗漏775–779日。只增加坐标自检没有稳定解决原始位移范围判断。</p><p>B2两遍F1均高于同期B0、弱段均81/81、34次没有契约失败；但FP为51／97日，虚假A为45／80日。对历史严格单窗，第一遍F1略高、第二遍更低；阶段macro-F1约0.791／0.788，仍低于严格单窗约0.830和强数值约0.835。不能只挑0.955的一遍宣布优越。</p><p>同图同配置两遍有269–584个观测日的状态差异（包含未知）。B1与B2共27个双方成功非空目标对中，19对起点改变，其中13对各自落在相应的内部块边界。这个行为诊断支持检查分块依赖，但两个重复不能完全分离相位因素和随机回答波动。</p><p>102次没有端点越界，两个失败均为块状态与最终活动矛盾，分别造成71／155个正日未知。下一步优先降低重复表达同一活动事实的输出负担，并独立检验图片刻度与块列表的对齐关系；历史和本轮失败不修剪、不回填。</p></section>'
    page += '<section><h2>失败与重复／相位分歧</h2><p>失败目标的下列正日全部按正式未知计FN，原始无效回答只用于诊断，没有裁剪或重跑。</p><pre>' + html.escape(json.dumps(diagnostics["failure_diagnostics"], ensure_ascii=False, indent=2)) + '</pre><details><summary>完整起止行为、开发检查及逐记录分歧</summary><pre>' + html.escape(json.dumps(analysis, ensure_ascii=False, indent=2)) + '</pre></details></section>'
    page += '<section><h2>同图回答差异：重复与分块扰动</h2><p>下表统计原图有观测日期的活动状态差异，包含0／1／未知之间的差异；不是参考标签准确率。相位差异同时可能受调用波动影响，只有两个重复不能完全分离两者。</p><div class="scroll"><table><tr><th>比较</th><th>状态不同日</th><th>阳性交集日</th><th>阳性并集日</th></tr>'
    for pair in analysis["repeat_and_phase_disagreement"]:
        records = pair["per_case"].values()
        disagreement = sum(row["observed_state_disagreement_days"] for row in records)
        intersection = sum(row["observed_positive_intersection_days"] for row in records)
        union = sum(row["observed_positive_union_days"] for row in records)
        page += f'<tr><td>{html.escape(METHODS[pair["left"]])} / {html.escape(METHODS[pair["right"]])}</td><td>{disagreement}</td><td>{intersection}</td><td>{union}</td></tr>'
    page += '</table></div><details><summary>全部17目标、两次相位对照的起止变化</summary><pre>' + html.escape(json.dumps(diagnostics["phase_boundary_changes"], ensure_ascii=False, indent=2)) + '</pre></details></section>'
    page += '<section><h2>列表分块与图片刻度的对齐关系</h2><p>事后查阅冻结绘图代码：目标图有45日上下文，横轴从图起点每30日标刻度。因此偏移15日还改变了观察块与固定图片刻度的相对对齐，不能将相位差异全归为纯内部边界识别能力。这个发现未用于选择本轮相位或重画图片。</p><details><summary>全部目标的对齐计数及来源</summary><pre>' + html.escape(json.dumps({"alignment": diagnostics["printed_tick_alignment"], "provenance": diagnostics["alignment_diagnostic_provenance"]}, ensure_ascii=False, indent=2)) + '</pre></details></section>'
    for key in ("case_0003-720", "case_0006-630", "case_0007-450", "case_0023-630", "case_0002-915", "case_0003-450"):
        answers = diagnostics["actual_target_answers"][key]
        page += f'<section><h2>实际同图判断 · {key}</h2><p>以下六份均为实际回答。图片未带标签，raw observation不是隐藏推理；参考只在结果图中事后比较。</p><img loading="lazy" src="{answers[0]["images"][1]}" alt="{key}实际送模目标图">'
        for answer in answers:
            page += f'<details><summary>{html.escape(METHODS[answer["method"]])} · {answer["status"]} · activity={html.escape(str(answer["raw_activity"]))}</summary><h3>原始回答</h3><pre>{html.escape(answer["actual_output"] or "无返回")}</pre><h3>完整实际提示</h3><pre>{html.escape(answer["prompt"])}</pre></details>'
        page += '</section>'
    page += '<section><h2>全部错误范围</h2><details><summary>逐配置、重复、记录的FP/FN与虚假A</summary><pre>' + html.escape(json.dumps(diagnostics["per_case_errors"], ensure_ascii=False, indent=2)) + '</pre></details></section>'
    page += '<p>六条已查看仿真、相关AI参考、一个弱段；两次调用不能证明稳定成功概率。离线特征含未来数据，不代表现场或在线预警。总体优势仍需独立资料。</p></body></html>'
    (out / "analysis.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    analyze(parser.parse_args().run)
