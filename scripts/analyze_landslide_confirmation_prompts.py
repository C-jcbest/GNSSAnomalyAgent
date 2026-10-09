"""Interpret the complete prompt ablation with paired errors and actual block answers."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from run_landslide_candidate_confirmation import verify_freeze
from run_landslide_comparison import save_json, sha
from run_landslide_confirmation_prompts import METHODS

from gnss_sim.landslide_confirmation_prompt_experiment import ARMS, target_blocks
from gnss_sim.landslide_evaluation import runs


def analyze(out):
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Interpretation requires every scheduled call and scoring")
    verify_freeze(out, manifest)
    def read(name):
        return json.loads((out / name).read_text(encoding="utf-8"))
    predictions, results = read("predictions.json"), read("results.json")
    outputs, costs = read("outputs.json"), read("costs.json")
    labels = {case_id: [] for case_id in manifest["case_ids"]}
    with (out / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    errors, changes = {}, {}
    comparisons = [("r1_neutral", "r0_candidate"), ("r2_blocks", "r1_neutral"), ("r2_blocks", "r0_candidate")]
    for case_id, rows in labels.items():
        actual = np.array([int(row["activity_label"]) for row in rows])
        case = read(f"inputs/{case_id}/input.json")
        observed = np.array([row is not None for row in case["displacement_mm"]])
        errors[case_id], changes[case_id] = {}, {}
        for arm in ARMS:
            activity = np.array(predictions[arm][case_id]["activity"])
            stage = np.array(predictions[arm][case_id]["stage"])
            if (np.any(activity[~observed] != -1) or np.any(stage[~observed] != -1)
                    or np.any((stage > 0) & (activity != 1))):
                raise ValueError("Missing mask or stage gate failed")
            errors[case_id][arm] = {"fp_half_open": runs((actual == 0) & (activity == 1)),
                                    "fn_half_open": runs((actual == 1) & (activity != 1)),
                                    "false_A_days": int(np.count_nonzero((actual == 0) & (stage == 1)))}
        for treatment, control in comparisons:
            before = np.array(predictions[control][case_id]["activity"])
            after = np.array(predictions[treatment][case_id]["activity"])
            changes[case_id][treatment + "_minus_" + control] = {
                "gained_TP": int(np.count_nonzero((actual == 1) & (before != 1) & (after == 1))),
                "lost_TP": int(np.count_nonzero((actual == 1) & (before == 1) & (after != 1))),
                "removed_FP": int(np.count_nonzero((actual == 0) & (before == 1) & (after != 1))),
                "added_FP": int(np.count_nonzero((actual == 0) & (before != 1) & (after == 1))),
            }
    paired_images = {}
    onset = {arm: {"successful_nonempty": 0, "starts_at_target": 0, "starts_later": 0,
                   "first_on_fixed_block_edge": 0, "shifted_on_fixed_block_edge": 0}
             for arm in ARMS}
    returned_models = {}
    for output in outputs:
        key = (output["case_id"], output["start"], output["stop"])
        paired_images.setdefault(key, []).append(output["image_sha256"])
        receipt = read("requests/" + output["request_id"] + ".response.json")
        returned_model = str(receipt.get("returned_model"))
        returned_models[returned_model] = returned_models.get(returned_model, 0) + 1
        if output["status"] != "success":
            continue
        payload = json.loads(receipt["output"])
        if not payload["activity"]:
            continue
        first = payload["activity"][0][0]
        arm = output["method"]
        onset[arm]["successful_nonempty"] += 1
        onset[arm]["starts_at_target"] += int(first == output["start"])
        onset[arm]["starts_later"] += int(first > output["start"])
        block_starts = {left for left, _ in target_blocks(output["start"], output["stop"])}
        onset[arm]["first_on_fixed_block_edge"] += int(first in block_starts)
        onset[arm]["shifted_on_fixed_block_edge"] += int(first > output["start"] and first in block_starts)
    if len(paired_images) != 17 or any(len(group) != 3 or group[0] != group[1] or group[0] != group[2] for group in paired_images.values()):
        raise ValueError("All three arms must review every target with identical images")
    failure_diagnostics = []
    for output in outputs:
        if output["status"] != "failed":
            continue
        actual = np.array([int(row["activity_label"]) for row in labels[output["case_id"]]])
        start, stop = output["start"], output["stop"]
        receipt = read("requests/" + output["request_id"] + ".response.json")
        payload = json.loads(receipt["output"])
        claims_inside_target = np.zeros(stop - start, dtype=bool)
        for left, right in payload.get("activity", []):
            inside_left, inside_right = max(start, left), min(stop, right + 1)
            if inside_left < inside_right:
                claims_inside_target[inside_left - start:inside_right - start] = True
        failure_diagnostics.append({"request_id": output["request_id"], "target_half_open": [start, stop],
                                    "raw_invalid_activity": payload.get("activity"),
                                    "reference_positive_days_in_failed_target": int(np.count_nonzero(actual[start:stop] == 1)),
                                    "raw_claimed_reference_positive_days_inside_target": int(np.count_nonzero((actual[start:stop] == 1) & claims_inside_target)),
                                    "raw_claimed_reference_negative_days_inside_target": int(np.count_nonzero((actual[start:stop] == 0) & claims_inside_target)),
                                    "interpretation": "Describes the invalid raw answer within the target; no repaired prediction or deployable score is produced"})
    selected = {}
    for case_id, start in (("case_0003", 720), ("case_0006", 630), ("case_0023", 630),
                           ("case_0002", 915), ("case_0007", 450), ("case_0003", 450)):
        selected[f"{case_id}-{start}"] = []
        for output in outputs:
            if output["case_id"] == case_id and output["start"] == start:
                receipt = read("requests/" + output["request_id"] + ".response.json")
                selected[f"{case_id}-{start}"].append({"request_id": output["request_id"], "method": output["method"],
                                                       "images": output["images"], "prompt": output["prompt"],
                                                       "status": output["status"], "actual_output": receipt.get("output")})
        selected[f"{case_id}-{start}"].sort(key=lambda answer: ARMS.index(answer["method"]))
    for arm in ARMS:
        daily = results[arm]["aggregate"]["daily"]
        fp_days = sum(right - left for case in errors.values() for left, right in case[arm]["fp_half_open"])
        fn_days = sum(right - left for case in errors.values() for left, right in case[arm]["fn_half_open"])
        if fp_days != daily["fp"] or fn_days != daily["fn"]:
            raise ValueError("Error intervals do not conserve the official FP/FN counts")
    for treatment, control in comparisons:
        pair = treatment + "_minus_" + control
        tp_change = sum(case[pair]["gained_TP"] - case[pair]["lost_TP"] for case in changes.values())
        fp_change = sum(case[pair]["added_FP"] - case[pair]["removed_FP"] for case in changes.values())
        before = results[control]["aggregate"]["daily"]
        after = results[treatment]["aggregate"]["daily"]
        if tp_change != after["tp"] - before["tp"] or fp_change != after["fp"] - before["fp"]:
            raise ValueError("Paired changes do not conserve the official counts")
    diagnostics = {"status": "ANALYZED", "per_case_errors": errors, "per_case_paired_changes": changes,
                   "onset_behavior": onset, "same_image_targets": len(paired_images), "returned_models": returned_models,
                   "selected_actual_answers": selected,
                   "failure_diagnostics": failure_diagnostics,
                   "count_conservation_verified": True,
                   "source_sha256": {name: sha(out / name) for name in ("predictions.json", "results.json", "outputs.json", "manifest.json")},
                   "analysis_script_sha256": sha(Path(__file__)), "limits": manifest["scope"]}
    save_json(out / "diagnostics.json", diagnostics)
    write_page(out, manifest, results, costs, diagnostics)
    print(json.dumps({"onset": onset, "changes": changes}, ensure_ascii=False))


def write_page(out, manifest, results, costs, diagnostics):
    frozen_analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    r0 = results["r0_candidate"]["aggregate"]
    r1 = results["r1_neutral"]["aggregate"]
    r2 = results["r2_blocks"]["aggregate"]
    summary = (f"同期R0／R1／R2的活动F1为{r0['daily']['f1']:.3f}／{r1['daily']['f1']:.3f}／{r2['daily']['f1']:.3f}；"
               f"FP为{r0['daily']['fp']}／{r1['daily']['fp']}／{r2['daily']['fp']}日，"
               f"弱活动为{r0['weak_detected_days']}／{r1['weak_detected_days']}／{r2['weak_detected_days']}日（共81日）。")
    findings = [
        "分块观察明显减少平台与漂移误报，但漏检和端点契约失败仍突出，尚未形成整体优势；当前不替代严格单窗配置。",
        "R1中性说明主要拒绝了case_0002的46日漂移误报；其14份成功非空回答仍全部沿用目标起点。R2才出现候选内部起变重定位：13份成功非空回答中8份后移，但其中7份落在固定30日块边界。后移不自动代表定位正确。",
        "R2的190个漏检日中，100日来自case_0007最终活动端点越界1日导致整目标未知，另外90日仍未被确认。无效回答仅作错误诊断，不人工修剪、不给修复成绩。所有51份回答均正常stop，未发生输出截断。",
        f"R1和R2均通过预先冻结的开发检查（FP下降、保留81弱日、活动F1不低于R0）；但R2活动F1仅比R0高{r2['daily']['f1'] - r0['daily']['f1']:.6f}，低于R1，阶段macro-F1也低于R0。开发检查通过不能等同于总体优越。",
        "下一步分别验证坐标表达的可靠性、块边界依赖与正常运动的过早起止；采用新冻结协议保留全部失败与重复结果，再在独立资料上比较。当前不追加调用或按本批移块挑最好结果。",
    ]
    case_notes = {
        "case_0003-720": "R2先把720–749日判为平台，活动输出750–880日；775–855日的81个弱活动参考日全部保留，前段30个FP删除。这只验证一个弱段，不能外推弱活动总体召回。",
        "case_0006-630": "R2记录630–689日stationary、690–719日mixed，并在观察中指出约705日起变，最终活动705–890日。630–704日的75个平台FP及其虚假A全部消除，本例TP不减、FN为0；这是成功的块内定位案例。",
        "case_0023-630": "R2输出660–809日，比R1少30个前段FP，但660–669日仍有10个FP，810–835日新增26个活动漏检。开始收缩与过早结束同时存在。",
        "case_0002-915": "R1和R2均未确认活动，46个漂移FP删除；R2将945–960日保留uncertain。这个案例的收益不应全部归功于分块，因为中性说明已经拒绝了该目标。",
        "case_0007-450": "目标为450–659日（闭区间），最后块也止于659日，但R2最终activity为520–660日。越界1日导致整目标未知，该目标内100个活动参考日全部计FN。原始无效回答在目标内声称了全部100个正日、0个负日，仅用于区分契约错误与视觉漏识，不产生可部署或修复成绩。",
        "case_0003-450": "R2活动在590日结束，591–610日共20个已确认活动参考日漏检；此回答契约有效。说明剩余漏检包含真正的可见活动范围判断错误，并非全由端点失败造成。",
    }
    columns = ["配置", "TP", "FP", "FN", "活动F1", "阶段macro-F1", "弱日", "虚假A", "覆盖"]
    table, markdown = [], ["# 确认提示因素：实际效果分析", "", "## Material Passport", "",
                           "- Origin Skill: academic-research-suite / experiment-agent", "- Origin Mode: run + descriptive validation",
                           "- Origin Date: 2026-10-08", "- Verification Status: ANALYZED；真实同图执行，效果未独立验证",
                           "- Version Label: confirmation_prompt_v1", "", summary, "",
                           *findings, "",
                           "| " + " | ".join(columns) + " |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, record in results.items():
        score = record["aggregate"]
        values = [METHODS[arm], str(score["daily"]["tp"]), str(score["daily"]["fp"]), str(score["daily"]["fn"]),
                  f"{score['daily']['f1']:.6f}", f"{score['stage_macro_f1']:.6f}", str(score["weak_detected_days"]),
                  str(score["false_acceleration_days"]), f"{score['coverage']:.2%}"]
        table.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>")
        markdown.append("| " + " | ".join(values) + " |")
    markdown += ["", "## 选定案例的事后诊断", "",
                 *[f"- {name}：{note}" for name, note in case_notes.items()], "",
                 "## 开发判据与描述区间", "", "```json", json.dumps(frozen_analysis, ensure_ascii=False, indent=2), "```", "",
                 "## 逐例配对变化", "", "```json", json.dumps(diagnostics["per_case_paired_changes"], ensure_ascii=False, indent=2), "```", "",
                 "## 起点行为与成本", "", "```json", json.dumps({"onset": diagnostics["onset_behavior"], "costs": costs}, ensure_ascii=False, indent=2), "```", "",
                 "## 失败影响（原始回答描述，不修复预测）", "", "```json", json.dumps(diagnostics["failure_diagnostics"], ensure_ascii=False, indent=2), "```", "",
                 "## 实际回答", "", "```json", json.dumps(diagnostics["selected_actual_answers"], ensure_ascii=False, indent=2), "```", "",
                 "R2改变结构化可见观察与输出校验，是组合因素；同图同模型不代表同输出token预算。首活动日是否恰好落在30日块边界另报，防止用块边界代替真实可见起变。当前不按块移位重新挑最佳结果。", "",
                 "所有失败保留，预测未知在正日计FN，覆盖另报；不能只看成功调用或事件F1。六条已查看仿真、AI相关标签、一个弱段、每目标每组单次；记录级分位区间仅描述，不证明调用稳定性。A/S/D参考活动支持866日，同时计入可评分非活动日的阶段FP；81慢日单独评分，不强补过渡／未定标签。", "",
                 "11/11方法风险检查延续冻结报告；本分析无新增模型调用，无预测裁剪、重训、阈值选择或标签修改。正式优势、现场泛化和在线预测尚未验证。", ""]
    (out / "analysis.md").write_text("\n".join(markdown), encoding="utf-8")
    style = 'body{font:16px system-ui,"Microsoft YaHei";margin:24px auto;padding:0 20px;max-width:1450px;color:#203e34;background:#f4f7f5}p{line-height:1.9}section{background:white;padding:22px;margin:22px 0;border:1px solid #d8e3df}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f4f7f5;padding:16px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #d8e3df;white-space:nowrap;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}summary{cursor:pointer;font-weight:600}a{color:#176757}'
    page = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>确认提示优化效果分析</title><style>{style}</style></head><body><h1>确认提示优化：真实效果与边界</h1>'
    page += '<p><a href="index.html">全部结果及六例活动／阶段图</a> · <a href="trace.html">51次实际调用</a> · <a href="analysis.md">完整错误分析</a> · <a href="diagnostics.json">逐例配对与原始回答</a></p>'
    page += f'<section><h2>同期主比较</h2><p>{html.escape(summary)}</p><p>实际{manifest["calls"]}次新调用、{manifest["total_tokens"]:,} token、失败{manifest["failures"]}次，无重试。各组使用相同17候选、相同PNG、相同模型参数和保存的阶段头。</p>'
    page += "".join(f'<p>{html.escape(finding)}</p>' for finding in findings)
    headers = "".join(f"<th>{name}</th>" for name in columns)
    page += f'<div class="scroll"><table><tr>{headers}</tr>{"".join(table)}</table></div></section>'
    page += '<section><h2>冻结开发检查与记录级描述区间</h2><p>R1／R2均通过冻结开发检查。R1−R0活动F1差的95%描述分位区间为[-0.002202, 0.019590]；R2−R1为[-0.111420, 0.077356]，均跨零。六条开发记录重采样不能估计模型调用方差，也不证明正式优势。全部2000次抽样均有定义。</p><details><summary>完整冻结检查与区间记录</summary><pre>' + html.escape(json.dumps(frozen_analysis, ensure_ascii=False, indent=2)) + '</pre></details></section>'
    page += '<section><h2>起点行为与实际预算</h2><p>R0非空15/15、R1非空14/14均沿用目标起点。R2非空13份，8份起点后移，其中7份在30日块边界。起点晚于目标可说明开始重新定位；落在块边界也可能是时间离散化，不能仅凭这个计数宣称定位更准确。</p><div class="scroll"><table><tr><th>配置</th><th>新调用</th><th>新token</th><th>服务秒合计</th><th>完整流程token</th><th>失败</th></tr>'
    for arm in ARMS:
        cost = costs[arm]
        page += f'<tr><td>{html.escape(METHODS[arm])}</td><td>{cost["new_calls"]}</td><td>{cost["new_tokens"]:,}</td><td>{cost["service_seconds_sum"]:.3f}</td><td>{cost["pipeline_tokens"]:,}</td><td>{cost["contract_or_transport_failures"]}</td></tr>'
    page += '</table></div><p>每组完整流程89次，含72次定位缓存；缓存没有重跑。R2结构化输出较长，新token比R1多16.48%，本次服务秒合计约两倍。完整流程token是缓存＋本轮的计数，不是本轮新增消耗。</p><details><summary>完整起点与成本记录</summary><pre>' + html.escape(json.dumps({"onset": diagnostics["onset_behavior"], "costs": costs}, ensure_ascii=False, indent=2)) + '</pre></details></section>'
    page += '<section><h2>失败造成的损失：不人工修剪或重跑</h2><p>case_0007目标450–659日，R2最终activity输出520–660日，越界1日导致整目标未知，100个参考活动日计FN；其他目标另有90FN。原始无效回答在目标内声称100正日、0负日，只用于错误诊断，不生成修复预测或修复分数。</p><details><summary>失败诊断计数</summary><pre>' + html.escape(json.dumps(diagnostics["failure_diagnostics"], ensure_ascii=False, indent=2)) + '</pre></details></section>'
    for name, answers in diagnostics["selected_actual_answers"].items():
        first = answers[0]
        page += f'<section><h2>同图实际判断 · {html.escape(name)}</h2><p><strong>作者事后诊断：</strong>{html.escape(case_notes[name])}</p><p>下面为实际送模原图；参考标签仅在结果图中用于事后评价。下方observation为模型实际输出，不补造隐藏思维链。</p>'
        page += f'<details><summary>全局上下文图（实际送模）</summary><a href="{first["images"][0]}"><img loading="lazy" src="{first["images"][0]}" alt="{html.escape(name)}实际送模全局图"></a></details>'
        page += f'<a href="{first["images"][1]}"><img loading="lazy" src="{first["images"][1]}" alt="{html.escape(name)}实际送模放大图"></a>'
        for answer in answers:
            page += f'<h3>{html.escape(METHODS[answer["method"]])} · {answer["status"]}</h3><pre>{html.escape(answer["actual_output"] or "未返回")}</pre>'
            page += '<details><summary>完整实际提示</summary><pre>' + html.escape(answer["prompt"]) + '</pre></details>'
        page += '</section>'
    page += '<section><h2>逐例活动误报与漏检</h2><p>以下全部错误范围使用[start, stop)半开区间，FP／FN合计与正式结果一致。未删除失败日，也未只保留选定案例。</p><details><summary>展开全部六例错误范围</summary><pre>' + html.escape(json.dumps(diagnostics["per_case_errors"], ensure_ascii=False, indent=2)) + '</pre></details></section>'
    page += '<p>仍是六例已查看仿真开发材料、相关AI参考、一个弱段和单次回答；结构化观察与校验是组合因素，不补造内部思维链。正式优势与现场／在线能力尚未验证。</p></body></html>'
    (out / "analysis.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    analyze(parser.parse_args().run)
