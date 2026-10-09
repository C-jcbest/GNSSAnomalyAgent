"""Describe completed candidate confirmation and its paired metadata ablation."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from run_landslide_candidate_confirmation import ARMS, METHODS, read_json, verify_freeze
from run_landslide_comparison import save_json, sha

from gnss_sim.landslide_evaluation import aggregate_scores, runs


def build_analysis_page(out, manifest, results, paired_targets):
    """Show frozen calls beside clearly separated post-hoc error interpretation."""
    rows = []
    for name, record in results.items():
        score = record["aggregate"]
        daily = score["daily"]
        values = [METHODS[name], f"{daily['precision']:.3f}", f"{daily['recall']:.3f}",
                  f"{daily['f1']:.3f}", str(daily["fp"]), str(daily["fn"]),
                  f"{score['stage_macro_f1']:.3f}", f"{score['weak_detected_days']}/81",
                  str(score["false_acceleration_days"]), f"{score['coverage']:.1%}"]
        cells = "".join(f"<td>{html.escape(value)}</td>" for value in values)
        rows.append(f"<tr>{cells}</tr>")

    examples = [
        ("case_0003", 720, "弱位移恢复，起点同时扩报",
         "参考弱活动775–855日全部检出；但参考非活动720–749日也被确认，共30日。"
         "候选覆盖成功不能替代确认边界准确性。"),
        ("case_0006", 630, "前段平台被整体趋势掩盖",
         "两组都返回活动630–895日。原始图前段保持平台，参考非活动630–704日共75日被确认，"
         "随后全部成为虚假加速。模型识别了后段运动，却没有重新定位前段起变。"),
        ("case_0002", 915, "同图观察摘要出现方向矛盾",
         "同一张图：原提示说N约92→90mm下降，缺测提示说N约91→94mm上升。"
         "两组均确认915–960日，46日均为参考非活动。可见摘要不能当作正确识图的保证。"),
        ("case_0002", 360, "缺测日期提示也可能删掉真实运动",
         "原提示确认360–444日，缺测提示只确认360–428日，将430–444日描述为平稳。"
         "后者新增15个参考活动漏日，其中13日原严格单窗已检出。case_0007反向多恢复17日，"
         "因此两组总体仅净增2个TP，无法确认稳定收益。"),
    ]
    cards = []
    for case_id, start, title, interpretation in examples:
        key = next(key for key in paired_targets if key[0] == case_id and key[1] == start)
        pair = paired_targets[key]
        first = pair[ARMS[0]]
        answers = []
        for arm in ARMS:
            packet = pair[arm]
            receipt = read_json(out / "requests" / (packet["request_id"] + ".response.json"))
            answers.append(
                f'<article><h3>{html.escape(METHODS[arm])}</h3>'
                f'<pre>{html.escape(receipt.get("output") or "未返回")}</pre>'
                f'<details><summary>完整送模提示 · {html.escape(packet["request_id"])}</summary>'
                f'<pre>{html.escape(packet["prompt"])}</pre></details></article>'
            )
        images = "".join(
            f'<figure><a href="{html.escape(path)}"><img loading="lazy" src="{html.escape(path)}" '
            f'alt="{case_id}实际送模原始位移图{index + 1}"></a>'
            f'<figcaption>实际送模图{index + 1} · 点击查看原尺寸</figcaption></figure>'
            for index, path in enumerate(first["images"])
        )
        cards.append(
            f'<section id="{case_id}-{start}"><h2>{html.escape(title)}</h2>'
            f'<p class="meta">{case_id} · 候选半开区间 [{key[1]}, {key[2]}) · 两组使用相同图片</p>'
            f'<p class="interpretation"><strong>事后作者分析（参考标签未送模型）：</strong>'
            f'{html.escape(interpretation)}</p>{images}<div class="answers">{"".join(answers)}</div>'
            f'<details><summary>含参考标签的活动结果图（仅事后分析）</summary>'
            f'<img loading="lazy" src="{case_id}.png" alt="{case_id}活动结果对比"></details></section>'
        )
    heads = ["配置", "活动P", "活动R", "活动F1", "FP日", "FN日", "阶段macro-F1", "弱段日", "虚假A日", "覆盖"]
    heading = "".join(f"<th>{value}</th>" for value in heads)
    overview = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>候选确认实验分析</title>
<style>body{font:16px system-ui,"Microsoft YaHei";margin:0;background:#f4f7f5;color:#203e34}
main{max-width:1440px;margin:auto;padding:28px}p{line-height:1.85}a{color:#176757}
h1{font-size:30px}h2{font-size:23px}h3{font-size:17px}.banner{padding:18px 24px;background:#fff1e2;border-left:5px solid #ad6822}
section{margin:28px 0;padding:24px;background:white;border:1px solid #d8e3df;border-radius:8px}
.meta{color:#597169}.interpretation{background:#f3f6f4;padding:16px}.table{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;text-align:right;border-bottom:1px solid #d8e3df;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{background:#e5efe9}.answers{display:grid;grid-template-columns:1fr 1fr;gap:20px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7f5;padding:16px;font-size:13px;line-height:1.65}
figure{margin:18px 0}img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#597169;margin:8px 0}
summary{cursor:pointer;padding:12px 0;font-weight:600}.pill{display:inline-block;margin-right:15px}
@media(max-width:800px){main{padding:14px}section{padding:16px}.answers{grid-template-columns:1fr}h1{font-size:25px}}
</style></head><body><main><h1>候选保留与原始位移确认：实际效果</h1>
<p><a href="index.html">主表与六例活动／阶段图</a> · <a href="trace.html">全部实际调用</a> ·
<a href="analysis.md">完整分析</a> · <a href="analysis.json">逐例错误核查</a> ·
<a href="diagnostics/index.html">进一步边界与阶段传播诊断</a></p>
<div class="banner"><strong>弱段恢复，但误报放大；当前不替换原严格单窗。</strong>
<p>弱活动检出0→81/81日；活动误报9→258日，虚假加速0→227日。活动F1由0.952降至0.902／0.903，阶段也下降。</p></div>
"""
    overview += (
        f'<p><span class="pill">新增真实调用 {manifest["calls"]} 次</span>'
        f'<span class="pill">候选 {manifest["candidate_count"]} 个 × 两组</span>'
        f'<span class="pill">{manifest["total_tokens"]:,} token</span>'
        f'<span class="pill">失败 {manifest["failures"]} 次，无重试</span></p>'
        '<p>复用72次冻结单窗定位，阳性／冲突仅保留为候选；用全局原图与带上下文的候选原图确认活动，'
        '再由同一保存的XGBoost判阶段。第二组仅增加真实缺测索引。trace共106条调用卡，含72条缓存与34条新增。</p>'
        f'<section><h2>整体收益与代价</h2><div class="table"><table><thead><tr>{heading}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        '<p>5080个可评分日：活动1287日、非活动3793日；1490日参考未知／缺测不评分。'
        '预测未知在参考活动日计漏检；覆盖单列。事件IoU≥0.3一对一匹配，无point adjustment。'
        '新组事件F1为0.960、严格单窗0.957，不能用事件略高掩盖258日误报。</p></section>'
    )
    overview += "".join(cards)
    overview += (
        '<section><h2>结论边界与下一步</h2><p>候选起点与返回活动起点多次一致，且候选前段平台被纳入活动，'
        '与候选起点锚定／整段概括相符；这只是行为观察，尚未证明模型内部机制。'
        '固定候选后，应优先检验候选内部重新定位、分块平台与运动证据，检查弱活动召回与负日误报，再作独立测试。</p>'
        '<p>六条已查看仿真记录、AI辅助标签曾接触生成真值、仅一个弱段；每候选每组单次回答，'
        '记录级描述区间跨零。当前是离线开发结果，不能证明现场泛化、稳定优势或预警能力。</p>'
        '<p>缺测提示组有一条回答越过目标窗口，原样拒绝并留未知；完整失败见'
        '<a href="failures.json">失败记录</a>。以上回答是保存的可见观察摘要，不是模型内部思维链。</p></section>'
        '</main></body></html>'
    )
    (out / "analysis.html").write_text(overview, encoding="utf-8")


def paired_record_intervals(results, case_ids):
    draws = np.random.default_rng(20261008).integers(0, len(case_ids), size=(2000, len(case_ids)))
    comparisons = []
    for treatment, control in (("proposal_visual", "strict_single_review"),
                               ("proposal_calendar", "proposal_visual")):
        differences = []
        for indices in draws:
            pair = [aggregate_scores([results[method]["per_case"][case_ids[index]] for index in indices])["daily"]["f1"]
                    for method in (treatment, control)]
            if all(value is not None for value in pair):
                differences.append(pair[0] - pair[1])
        comparisons.append({"treatment": treatment, "control": control, "defined_draws": len(differences),
                            "undefined_draws": len(draws) - len(differences),
                            "descriptive_record_percentile95": np.quantile(differences, [.025, .975]).tolist()})
    return comparisons


def analyze(out):
    manifest = read_json(out / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Scoring and inference must complete before interpretation")
    verify_freeze(out, manifest)
    results = read_json(out / "results.json")
    predictions = read_json(out / "predictions.json")
    labels = {case_id: [] for case_id in manifest["case_ids"]}
    with (out / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    changes, errors = {}, {}
    for case_id, rows in labels.items():
        actual = np.array([int(row["activity_label"]) for row in rows])
        case = read_json(out / "inputs" / case_id / "input.json")
        observed = np.array([row is not None for row in case["displacement_mm"]])
        weak = np.array([row["feature"] == "slow_displacement" for row in rows])
        changes[case_id], errors[case_id] = {}, {}
        for name, records in predictions.items():
            activity = np.array(records[case_id]["activity"])
            stage = np.array(records[case_id]["stage"])
            if np.any(activity[~observed] != -1) or np.any(stage[~observed] != -1) or np.any((stage > 0) & (activity != 1)):
                raise ValueError("Missing observations or stage gate were violated")
            errors[case_id][name] = {
                "fp_half_open": runs((actual == 0) & (activity == 1)),
                "fn_half_open": runs((actual == 1) & (activity != 1)),
                "weak_detected_half_open": runs(weak & (activity == 1)),
                "positive_unknown_days": int(np.count_nonzero((actual == 1) & (activity == -1))),
                "negative_unknown_days": int(np.count_nonzero((actual == 0) & (activity == -1))),
                "false_stage_days": int(np.count_nonzero((actual == 0) & (stage > 0))),
            }
        before = np.array(predictions["strict_single"][case_id]["activity"])
        for arm in ARMS:
            after = np.array(predictions[arm][case_id]["activity"])
            changes[case_id][arm] = {
                "gained_tp": int(np.count_nonzero((actual == 1) & (before != 1) & (after == 1))),
                "lost_tp": int(np.count_nonzero((actual == 1) & (before == 1) & (after != 1))),
                "added_fp": int(np.count_nonzero((actual == 0) & (before != 1) & (after == 1))),
                "removed_fp": int(np.count_nonzero((actual == 0) & (before == 1) & (after != 1))),
            }
    packets = read_json(out / "packets.json")
    outputs = {row["request_id"]: row for row in read_json(out / "outputs.json")}
    paired_targets = {}
    returned_models, finish_reasons = {}, {}
    for packet in packets:
        key = (packet["case_id"], packet["start"], packet["stop"])
        paired_targets.setdefault(key, {})[packet["method"]] = packet
        receipt = read_json(out / "requests" / (packet["request_id"] + ".response.json"))
        model = str(receipt.get("returned_model"))
        reason = str(receipt.get("finish_reason"))
        returned_models[model] = returned_models.get(model, 0) + 1
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    for pair in paired_targets.values():
        if set(pair) != set(ARMS) or pair[ARMS[0]]["image_sha256"] != pair[ARMS[1]]["image_sha256"]:
            raise ValueError("Calendar ablation did not use identical targets and images")
        if not pair[ARMS[1]]["prompt"].startswith(pair[ARMS[0]]["prompt"]):
            raise ValueError("Calendar ablation changed the base prompt")
    weak_key = next(key for key in paired_targets if key[0] == "case_0003" and key[1] <= 775 and key[2] >= 856)
    weak_responses = {}
    for arm, packet in paired_targets[weak_key].items():
        receipt = read_json(out / "requests" / (packet["request_id"] + ".response.json"))
        weak_responses[arm] = {"target_half_open": list(weak_key[1:]), "actual_output": receipt.get("output"),
                               "status": outputs[packet["request_id"]]["status"]}
    intervals = paired_record_intervals(results, manifest["case_ids"])
    analysis = {"verification_status": "ANALYZED", "per_case_changes_from_strict_single": changes,
                "per_case_error_intervals": errors, "weak_candidate_actual_answers": weak_responses,
                "same_image_paired_targets": len(paired_targets), "returned_models": returned_models,
                "finish_reasons": finish_reasons, "descriptive_record_intervals": intervals,
                "limits": manifest["scope"], "repetitions_per_proposal_arm": 1,
                "source_sha256": {name: sha(out / name) for name in ("predictions.json", "results.json", "manifest.json")},
                "analysis_source_sha256": sha(Path(__file__))}
    save_json(out / "analysis.json", analysis)
    lines = ["# 候选保留与原始位移确认：开发实验分析", "", "## Material Passport", "",
             "- Origin Skill: academic-research-suite / experiment-agent", "- Origin Mode: run + descriptive validation",
             "- Origin Date: 2026-10-08", "- Verification Status: ANALYZED；真实调用、同图对照和掩码已核查，效果未独立验证",
             "- Version Label: candidate_confirmation_v1", "",
             f"复用72次冻结单窗定位；17个候选、两组各17次，本轮新增{manifest['calls']}次、{manifest['total_tokens']} token、失败{manifest['failures']}次，无重试。任一阳性与成功窗负/未知冲突进入候选，不直接赋活动或阶段。两组图和目标一致，第二组只增加真实缺测索引，未提供信号数值或标签。", "",
             "## 主表（包括失败与未知）", "",
             "| 配置 | 活动P | 活动R | 活动F1 | FP日 | FN日 | 阶段macro-F1 | 弱段/81 | 虚假A日 | 覆盖 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, record in results.items():
        score = record["aggregate"]
        daily = score["daily"]
        lines.append(f"| {METHODS[name]} | {daily['precision']:.3f} | {daily['recall']:.3f} | {daily['f1']:.3f} | {daily['fp']} | {daily['fn']} | {score['stage_macro_f1']:.3f} | {score['weak_detected_days']} | {score['false_acceleration_days']} | {score['coverage']:.3f} |")
    lines += ["", "## 逐例计数", "",
              "| 记录 | 严格多数 TP/FP/FN | 候选原提示 TP/FP/FN | 候选缺测提示 TP/FP/FN |",
              "| --- | --- | --- | --- |"]
    for case_id in manifest["case_ids"]:
        cells = []
        for name in ("strict_single", *ARMS):
            daily = results[name]["per_case"][case_id]["daily"]
            cells.append("/".join(str(daily[field]) for field in ("tp", "fp", "fn")))
        lines.append("| " + " | ".join([case_id] + cells) + " |")
    lines += ["", "## 弱段实际回答", "", "```json", json.dumps(weak_responses, ensure_ascii=False, indent=2), "```", "",
              "## 候选覆盖仅为诊断", "", "```json", json.dumps(read_json(out / "candidate-support.json"), ensure_ascii=False, indent=2), "```", "",
              "候选覆盖不等于识别成功；未通过原始位移确认的日期不得赋确定阶段。确认器可能扩报活动边界或拒绝真运动，必须同时看FP/FN。", "",
              "## 成本", "", "```json", json.dumps(read_json(out / "costs.json"), ensure_ascii=False, indent=2), "```", "",
              "## 解释限制", "", "```json", json.dumps(intervals, ensure_ascii=False, indent=2), "```", "",
              "配对按六条完整记录重采样2000次，分位区间仅描述当前记录组成，不作确认性显著结论；每候选每组单次回答，未估计模型调用方差。候选原提示与历史严格多数复核的范围/图像上下文不同、调用也不同时；不能只归因于聚合算法。缺测提示两组为同期同候选同图对照，但仍受单次随机回答影响。", "",
              "11/11项风险检查：记录汇总异质性、仿真现场外推、开发选择偏差、成功子集选择、现场基率、错误后调整与均值回归、失败/弃判保留、多指标/多比较、事后选配置、单次调用因果归属、离线未来信息用于预测均已检查。没有改标签、旧成绩、旧配置或阈值，没有挑最好重跑。", "",
              "全部确定阶段在确认活动内，原始缺测两层未知；正常对照同样评分，虽然其缓存没有候选所以无需新增确认调用。网页trace.html包括72次缓存定位和全部新增确认，evidence是可见观察摘要而非内部思维链。"]
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    build_analysis_page(out, manifest, results, paired_targets)
    print(json.dumps({"changes": changes, "weak_answers": weak_responses, "intervals": intervals}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    analyze(parser.parse_args().run)
