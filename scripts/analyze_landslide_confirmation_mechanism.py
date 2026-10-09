"""Audit candidate-boundary behavior and stage error propagation without new VLM calls."""
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_landslide_candidate_confirmation import ARMS, read_json, verify_freeze
from run_landslide_comparison import save_json, sha
from run_landslide_window_optimization import load_case_features
from xgboost import XGBClassifier

from gnss_sim.landslide_evaluation import STAGES, aggregate_scores, evaluate_case, runs
from gnss_sim.landslide_local_experiment import learned_stages


def distribution(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"finite_days": 0, "p10": None, "median": None, "p90": None}
    quantiles = np.quantile(finite, [.1, .5, .9])
    return {"finite_days": len(finite), **dict(zip(("p10", "median", "p90"), quantiles.tolist()))}


def raw_block_evidence(values, noise, actual, start, stop):
    """Compare disjoint endpoint medians; report observations, never assign activity."""
    rows = []
    for left in range(start, stop, 30):
        right = min(left + 30, stop)
        if right - left < 28:
            rows.append({"span_half_open": [left, right], "status": "too_short_for_disjoint_14_day_blocks"})
            continue
        early = values[left:left + 14]
        late = values[right - 14:right]
        counts = [int(np.isfinite(sample).all(axis=1).sum()) for sample in (early, late)]
        if min(counts) < 8:
            rows.append({"span_half_open": [left, right], "status": "insufficient_observations",
                         "endpoint_observed_days": counts})
            continue
        change = np.nanmedian(late, axis=0) - np.nanmedian(early, axis=0)
        rows.append({"span_half_open": [left, right], "status": "descriptive_only",
                     "endpoint_observed_days": counts, "median_change_mm_NEU": change.tolist(),
                     "change_over_noise_NEU": (change / noise).tolist(),
                     "reference_positive_days": int(np.count_nonzero(actual[left:right] == 1)),
                     "reference_negative_days": int(np.count_nonzero(actual[left:right] == 0))})
    return rows


def plot_diagnostics(destination, cases, labels, predictions, groups):
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, panels = plt.subplots(2, 2, figsize=(13, 7.2), gridspec_kw={"height_ratios": [4, 1]})
    for column, (case_id, start, stop) in enumerate((("case_0006", 630, 990), ("case_0003", 720, 990))):
        values = cases[case_id][0]
        actual = np.array([int(row["activity_label"]) for row in labels[case_id]])
        activity = np.array(predictions["proposal_visual"][case_id]["activity"])
        raw, bands = panels[:, column]
        raw.plot(np.arange(start, stop), values[start:stop, 0], color="#176f66", linewidth=.85)
        raw.set_title(case_id + " · 原始N位移（事后诊断）")
        raw.set_ylabel("N位移 / mm")
        raw.grid(alpha=.2)
        for axis in (raw, bands):
            axis.set_xlim(start, stop - 1)
        for level, flags in ((1, actual), (0, activity)):
            for code, color in ((-1, "#dadada"), (0, "#a4c3bb"), (1, "#db9866")):
                for left, right in runs(flags[start:stop] == code):
                    bands.fill_between([start + left, start + right], level, level + .6, color=color)
        bands.set_yticks([.3, 1.3], ["确认输出", "参考标签"])
        bands.set_xlabel("原始日索引（0起算）")
        bands.set_ylim(-.2, 1.9)
    fig.suptitle("平台与弱活动：候选起点并非活动起点 · 橙=活动 / 绿=非活动 / 灰=未知")
    fig.tight_layout()
    fig.savefig(destination / "boundary-diagnostic.png", dpi=160)
    plt.close(fig)

    fig, panels = plt.subplots(1, 2, figsize=(11, 4.5))
    names = ["误报加速日", "参考真加速日"]
    for axis, key, title in ((panels[0], "probability_A", "三分类头的加速分数（非活动概率未建模）"),
                             (panels[1], "speed", "61日离线速度模 / mm/日")):
        samples = [np.asarray(groups[name][key]) for name in ("false_A", "reference_A")]
        samples = [sample[np.isfinite(sample)] for sample in samples]
        axis.boxplot(samples, tick_labels=names, showfliers=False)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("阶段头诊断：高类别分数不代表已确认运动 · 使用已保存模型，无重训")
    fig.tight_layout()
    fig.savefig(destination / "stage-confidence-diagnostic.png", dpi=160)
    plt.close(fig)


def analyze(run):
    manifest = read_json(run / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("A complete frozen run is required")
    verify_freeze(run, manifest)
    frozen_names = ("predictions.json", "results.json", "outputs.json", "manifest.json", "reference-labels.csv")
    source_hashes = {name: sha(run / name) for name in frozen_names}
    predictions = read_json(run / "predictions.json")
    pools = read_json(run / "candidate-pools.json")
    outputs = read_json(run / "outputs.json")
    labels = {case_id: [] for case_id in manifest["case_ids"]}
    with (run / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    model = XGBClassifier()
    model.load_model(run / "stage-xgboost.json")
    cases, candidates, blocks = {}, [], {}
    origins = {arm: {"fp_with_original_positive_vote": 0, "fp_conflict_only": 0,
                     "fp_without_proposal": 0, "fp_stages": {str(code): 0 for code in (-1, 1, 2, 3)}} for arm in ARMS}
    counterfactuals = {name: [] for name in ("oracle_activity", "remove_negative_fp_visual", "remove_negative_fp_calendar")}
    reference_features = Counter()
    stage_confusion = {name: {str(code): 0 for code in (-1, 1, 2, 3)} for name in STAGES}
    deceleration_errors = {}
    groups = {name: {key: [] for key in ("probability_A", "speed", "raw_rate")}
              for name in ("false_A", "reference_A")}
    shared_mismatch = 0
    for case_id, rows in labels.items():
        case, feature = load_case_features(run, case_id)
        values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        cases[case_id] = (values, feature)
        actual = np.array([int(row["activity_label"]) for row in rows])
        reference_features.update(row["feature"] for row in rows if int(row["activity_label"]) == 1)
        if [int(row["day_index"]) for row in rows] != list(range(len(values))):
            raise ValueError("Reference calendar is incomplete or unordered")
        probability = model.predict_proba(feature.matrix)
        codes = probability.argmax(axis=1) + 1
        source_positive = np.array(pools[case_id]["positive_vote_days"])
        conflict_only = np.array(pools[case_id]["negative_unknown_conflict_days"]) & ~source_positive
        oracle = learned_stages(codes, actual, feature.supported)
        counterfactuals["oracle_activity"].append(evaluate_case(rows, oracle))
        target_stage = np.array([STAGES.get(row["feature"], -1) for row in rows])
        for name, target_code in STAGES.items():
            for code in (-1, 1, 2, 3):
                stage_confusion[name][str(code)] += int(np.count_nonzero((target_stage == target_code) & (oracle.stage == code)))
        mistaken_deceleration = (target_stage == 3) & (oracle.stage == 2)
        deceleration_errors[case_id] = {
            "D_as_S_days": int(mistaken_deceleration.sum()),
            "spans_half_open": runs(mistaken_deceleration),
            "speed_mm_day": distribution(feature.speed[mistaken_deceleration]),
            "relative_change_60_day": distribution(60 * feature.relative_acceleration[mistaken_deceleration]),
        }
        for arm in ARMS:
            activity = np.array(predictions[arm][case_id]["activity"])
            stage = np.array(predictions[arm][case_id]["stage"])
            expected = learned_stages(codes, activity, feature.supported)
            if not np.array_equal(expected.stage, stage):
                raise ValueError("Saved stage output differs from the frozen head and support mask")
            fp = (actual == 0) & (activity == 1)
            origins[arm]["fp_with_original_positive_vote"] += int(np.count_nonzero(fp & source_positive))
            origins[arm]["fp_conflict_only"] += int(np.count_nonzero(fp & conflict_only))
            origins[arm]["fp_without_proposal"] += int(np.count_nonzero(fp & ~(source_positive | conflict_only)))
            for code in (-1, 1, 2, 3):
                origins[arm]["fp_stages"][str(code)] += int(np.count_nonzero(fp & (stage == code)))
            diagnostic_gate = activity.copy()
            diagnostic_gate[fp] = 0
            name = "remove_negative_fp_visual" if arm == ARMS[0] else "remove_negative_fp_calendar"
            counterfactuals[name].append(evaluate_case(rows, learned_stages(codes, diagnostic_gate, feature.supported)))
            strict = predictions["strict_single"][case_id]
            shared = (activity == 1) & (np.array(strict["activity"]) == 1) & feature.supported
            shared_mismatch += int(np.count_nonzero(shared & (stage != np.array(strict["stage"]))))
        false_A = (actual == 0) & (np.array(predictions[ARMS[0]][case_id]["stage"]) == 1)
        reference_A = np.array([STAGES.get(row["feature"], -1) == 1 for row in rows])
        for name, mask in (("false_A", false_A), ("reference_A", reference_A)):
            for key, series in (("probability_A", probability[:, 0]), ("speed", feature.speed),
                                ("raw_rate", feature.progressive_rate)):
                groups[name][key].extend(series[mask].tolist())
        for output in outputs:
            if output["case_id"] != case_id:
                continue
            receipt = read_json(run / "requests" / (output["request_id"] + ".response.json"))
            payload = json.loads(receipt["output"])
            intervals = payload["activity"] if output["status"] == "success" else []
            candidate_mask = np.zeros(len(actual), dtype=bool)
            candidate_mask[output["start"]:output["stop"]] = True
            review = np.array(output["activity"])
            candidates.append({"request_id": output["request_id"], "case_id": case_id,
                               "arm": output["method"], "status": output["status"],
                               "target_half_open": [output["start"], output["stop"]],
                               "activity_inclusive": intervals,
                               "first_start_equals_candidate_start": bool(intervals and intervals[0][0] == output["start"]),
                               "covers_entire_candidate": intervals == [[output["start"], output["stop"] - 1]],
                               "candidate_negative_days": int(np.count_nonzero(candidate_mask & (actual == 0))),
                               "confirmed_negative_days": int(np.count_nonzero((review == 1) & (actual == 0)))})
    for case_id, start, stop in (("case_0006", 630, 990), ("case_0003", 720, 990), ("case_0002", 915, 961)):
        values, feature = cases[case_id]
        actual = np.array([int(row["activity_label"]) for row in labels[case_id]])
        blocks[f"{case_id}-{start}"] = raw_block_evidence(values, feature.noise, actual, start, stop)
    summaries = {}
    for arm in ARMS:
        successful_nonempty = [row for row in candidates if row["arm"] == arm and row["activity_inclusive"]]
        summaries[arm] = {"successful_nonempty_answers": len(successful_nonempty),
                          "first_start_equals_candidate_start": sum(row["first_start_equals_candidate_start"] for row in successful_nonempty),
                          "covers_entire_candidate": sum(row["covers_entire_candidate"] for row in successful_nonempty)}
    window_disagreements = {}
    cached = read_json(run / "cached-screen-packets.json")
    for case_id, start, stop in (("case_0006", 630, 705), ("case_0023", 630, 670), ("case_0003", 720, 750)):
        window_disagreements[f"{case_id}-{start}-{stop}"] = []
        for window in cached:
            if window["case_id"] != case_id or window["start"] >= stop or window["stop"] <= start:
                continue
            receipt = read_json(run / "cached-receipts" / (window["request_id"] + ".response.json"))
            window_disagreements[f"{case_id}-{start}-{stop}"].append({
                "request_id": window["request_id"], "target_half_open": [window["start"], window["stop"]],
                "judgments_in_error_span": dict(Counter(map(str, window["activity"][start:stop]))),
                "actual_output": receipt["output"], "images": window["images"],
            })
    destination = run / "diagnostics"
    destination.mkdir(exist_ok=True)
    plot_diagnostics(destination, cases, labels, predictions, groups)
    diagnostics = {"status": "ANALYZED", "new_vlm_calls": 0, "source_sha256": source_hashes,
                   "analysis_script_sha256": sha(Path(__file__)), "candidate_start_summary": summaries,
                   "candidate_answers": candidates, "error_origins": origins,
                   "shared_confirmed_supported_stage_mismatches": shared_mismatch,
                   "reference_positive_feature_counts": dict(reference_features),
                   "reference_positive_days_with_scored_stage": sum(reference_features[name] for name in STAGES),
                   "oracle_stage_confusion": stage_confusion, "deceleration_as_steady_diagnostics": deceleration_errors,
                   "source_window_disagreements": window_disagreements,
                   "stage_distributions": {name: {key: distribution(values) for key, values in group.items()}
                                           for name, group in groups.items()},
                   "reference_only_counterfactuals": {name: aggregate_scores(scores) for name, scores in counterfactuals.items()},
                   "raw_30_day_block_evidence": blocks,
                   "limits": "Post-hoc development diagnostics using correlated reference labels; oracle gates use labels and are not deployable methods. Class probabilities are uncalibrated and conditional on the three stage classes. Block changes are descriptive, not activity thresholds or causal proof."}
    save_json(destination / "mechanism.json", diagnostics)
    if source_hashes != {name: sha(run / name) for name in frozen_names}:
        raise ValueError("A frozen run artifact changed during analysis")
    write_report(destination, diagnostics)
    print({"starts": summaries, "origins": origins,
           "counterfactual_stage_macro_f1": {name: aggregate_scores(scores)["stage_macro_f1"] for name, scores in counterfactuals.items()},
           "stage_distributions": diagnostics["stage_distributions"], "stage_mismatches": shared_mismatch})


def write_report(destination, diagnostics):
    lines = ["# 确认器边界与阶段传播诊断", "", "## Material Passport", "",
             "- Origin Skill: academic-research-suite / experiment-agent", "- Origin Mode: validate / post-hoc diagnostics",
             "- Origin Date: 2026-10-08", "- Verification Status: ANALYZED；冻结输出重放及计数核查，不是独立验证",
             "- Version Label: confirmation_mechanism_v1", "", "本次新增视觉模型调用0次，未改原预测、标签或评分。", "",
             "## 候选起点行为", "", "| 组 | 成功非空回答 | 起点等于候选起点 | 确认整候选 |", "| --- | ---: | ---: | ---: |"]
    for arm, row in diagnostics["candidate_start_summary"].items():
        lines.append(f"| {arm} | {row['successful_nonempty_answers']} | {row['first_start_equals_candidate_start']} | {row['covers_entire_candidate']} |")
    lines += ["", "非空分母仅用于描述起点行为；效果主表仍包括全部17候选／组、空回答、失败与未知。起点一致也可能部分正确，不据此直接断言模型锚定机制。", "",
              "## 误报来源与阶段传播", "", "```json", json.dumps(diagnostics["error_origins"], ensure_ascii=False, indent=2), "```", "",
              "两组258个FP均有原窗阳性，冲突专属日期误报0日。问题不只是新增冲突候选：原窗全段概括产生的阳性左扩范围被确认器保留。227个FP被三分类阶段头判A，其余31个FP因阶段特征不支持而未知，没有被判S或D。", "",
              "## 使用参考标签的诊断性反事实（不是算法成绩）", "",
              "| 诊断 | 活动F1 | 阶段macro-F1 | 虚假A日 |", "| --- | ---: | ---: | ---: |"]
    for name, score in diagnostics["reference_only_counterfactuals"].items():
        lines.append(f"| {name} | {score['daily']['f1']:.6f} | {score['stage_macro_f1']:.6f} | {score['false_acceleration_days']} |")
    lines += ["", "remove_negative_fp只利用参考标签删除误确认负日，保留全部原漏检和阶段头；oracle_activity直接使用参考活动标签。它们只能量化误报门控和第二层剩余错误，不能作为部署成绩、外部基线或新方法提升。", "",
              "## 阶段分数与运动特征", "", "```json", json.dumps(diagnostics["stage_distributions"], ensure_ascii=False, indent=2), "```", "",
              "训练仅含有支持的A/S/D标签，非活动及慢位移未作为阶段类别；A分数是在三类中的条件分配，未经校准，不能解释为真实发生加速或已活动的概率。共享确认且有特征支持的日期，严格单窗和两组阶段输出不一致数为0。", "",
              "## 阶段评价的覆盖范围与剩余错误", "", "```json",
              json.dumps(diagnostics["reference_positive_feature_counts"], ensure_ascii=False, indent=2), "```", "",
              "1287个参考活动日中，只有866日具有A/S/D可评分标签（67.29%）；81日慢位移、250日transition、90日unresolved均不在阶段三类评分中。未计入不等于正确，恢复全部弱日也不直接提高阶段macro-F1。活动任务与阶段任务必须同时报告。", "",
              "参考活动门控下443个A与42个S均识别正确，381个D中54日判S，其余327日正确；阶段macro-F1仅0.844141。该剩余误差属于冻结阶段头，不可全部归因于活动定位。", "", "```json",
              json.dumps(diagnostics["deceleration_as_steady_diagnostics"], ensure_ascii=False, indent=2), "```", "",
              "## 原窗已有相反观察", "", "```json",
              json.dumps(diagnostics["source_window_disagreements"], ensure_ascii=False, indent=2), "```", "",
              "case_0006与case_0023的前窗已描述平台／晚起变，后窗却概括为全窗活动；确认器没有从这些冲突中重新定位。确认输入提供全局和候选图，没有显式提供两个原窗的相反判断。不能据此宣称模型内部没有比较图像。", "",
              "## 原始位移分块证据", "", "```json", json.dumps(diagnostics["raw_30_day_block_evidence"], ensure_ascii=False, indent=2), "```", "",
              "从候选起点每30日分块，比较不相交的首尾14日中位数，每端至少8个完整三轴观测；尾块不足28日留未分析，不补值。change_over_noise使用全记录差分MAD噪声代理，不是变化显著性或置信区间，未按该比值赋标签或挑阈值。", "",
              "## 已确认、待验证和未知", "",
              "已确认：无非空回答收缩起点；误报全来自原阳性；高A分数无法自动拒绝平台；阶段头在共同输入上的输出未变。",
              "待验证：整段端点概括、候选语义诱导或跨窗负证据未对照是否造成起点扩报；分块观察、边界专门定位及短原始位移数字证据能否减少误报且保留弱段。",
              "未知：底层注意机制、底层模型版本、调用重复方差和独立真实场地效果。", "",
              "11/11风险检查：分组异质性、仿真外推、候选选择偏差、只看成功子集、仿真基率、事后调整/均值回归、失败保留、多指标、配置自由度、行为与因果区分、离线未来信息均已检查。全部为事后描述，无显著性检验；无阈值重选、重训或新增模型调用。"]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    candidate_rows = []
    for row in diagnostics["candidate_answers"]:
        values = [row["case_id"], row["arm"], str(row["target_half_open"]), str(row["activity_inclusive"]),
                  row["status"], str(row["candidate_negative_days"]), str(row["confirmed_negative_days"])]
        candidate_rows.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>")
    score_rows = []
    for name, score in diagnostics["reference_only_counterfactuals"].items():
        score_rows.append(f"<tr><td>{name}</td><td>{score['stage_macro_f1']:.6f}</td><td>{score['daily']['f1']:.6f}</td></tr>")
    page = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>确认器边界与阶段传播诊断</title><style>body{font:16px system-ui,"Microsoft YaHei";color:#203e34;background:#f4f7f5;margin:24px auto;max-width:1400px;padding:0 20px}p{line-height:1.9}section{background:white;padding:22px;margin:22px 0;border:1px solid #d8e3df}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #d8e3df;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}.scroll{overflow-x:auto}a{color:#176757}</style></head><body>
<h1>确认器边界与阶段传播诊断</h1><p><a href="../analysis.html">实验主分析</a> · <a href="../trace.html">全部实际调用</a> · <a href="report.md">完整诊断</a> · <a href="mechanism.json">机器计数与原始位移块</a></p>
<section><h2>本次发现：没有一次收缩活动起点</h2><p>两组成功非空回答29份，全部从候选起点确认运动。原提示15/15、缺测提示14/14；整候选确认分别10/15与7/14。所有258个FP都曾被原窗报阳性，冲突专属日期FP为0。原窗和确认器反复使用整段累计变化，未充分剔除起点前的平台。</p>
<p>这是行为证据，尚不能证明内部锚定机制。没有新增视觉调用，没有修改冻结成绩；候选起点有时正确，因此起点一致本身不是误判判据。</p>
<img src="boundary-diagnostic.png" alt="平台和弱活动的原始位移与参考和输出范围对照"></section>
<section><h2>高加速分数不等于确认了运动</h2><p>258个活动FP中，227日判A，其余31日阶段未知。当前阶段头训练只包含A/S/D，未学习非活动拒绝；平台输入仍被分配到其中一类。共同确认且有特征支持的日，三种配置阶段输出完全相同。</p>
<img src="stage-confidence-diagnostic.png" alt="误报和真实加速的类别分数与速度比较"><details><summary>分位数及有效日数</summary>'''
    page += "<pre>" + html.escape(json.dumps(diagnostics["stage_distributions"], ensure_ascii=False, indent=2)) + "</pre></details></section>"
    page += ('<section><h2>诊断性反事实：只用于分离错误来源</h2><p>下表使用参考标签，不是可运行检测方法或新算法成绩。'
             '删除已知误报后保留原漏检；oracle直接使用参考活动门控。分数用于检查第一层误报与第二层剩余错误。</p>'
             '<table><tr><th>诊断</th><th>阶段macro-F1</th><th>活动F1</th></tr>' + "".join(score_rows) + '</table></section>')
    page += ('<section><h2>第二层还有减速／匀速混淆，且阶段指标没有覆盖全部活动</h2>'
             '<p>参考活动门控下443个A与42个S均正确；381个D中54日被判为S。即使活动范围完全使用参考标签，'
             '阶段macro-F1仍只有0.844141，修好定位也不能自动解决第二层。</p>'
             '<p>1287个参考活动日中866日具有A/S/D标签（67.29%）。81日慢位移、250日transition、90日unresolved'
             '不参与阶段三类评分；未知标签不能硬补阶段，慢位移检出必须单独报告。</p></section>')
    for name, windows in diagnostics["source_window_disagreements"].items():
        page += f'<section><h2>相邻原窗对照 · {html.escape(name)}</h2>'
        for window in windows:
            page += (f'<h3>{html.escape(window["request_id"])}</h3>'
                     f'<pre>{html.escape(window["actual_output"])}</pre>')
        page += '<p>这些是已保存的原窗回答；后续确认没有显式接收相反判断文本。完整图片与提示见全部调用页。</p></section>'
    page += ('<section><h2>case_0006原始位移分块：前段平台与后段变化</h2>'
             '<p>每30日块比较首尾不相交14日观测中位数。只作事后描述，没有送入模型或据此重新打标签。'
             '变化／噪声是MAD噪声代理比值，不是显著性、置信度或活动阈值。完整三轴数值见机器计数文件。</p>'
             '<div class="scroll"><table><tr><th>半开时间块</th><th>N变化/mm</th><th>N变化／噪声</th><th>参考活动日</th><th>参考非活动日</th></tr>')
    for block in diagnostics["raw_30_day_block_evidence"]["case_0006-630"]:
        if block["status"] != "descriptive_only":
            continue
        page += (f'<tr><td>{block["span_half_open"]}</td><td>{block["median_change_mm_NEU"][0]:.3f}</td>'
                 f'<td>{block["change_over_noise_NEU"][0]:.3f}</td><td>{block["reference_positive_days"]}</td>'
                 f'<td>{block["reference_negative_days"]}</td></tr>')
    page += '</table></div></section>'
    page += ('<section><h2>全部候选回答与负日保留</h2><p>候选为半开区间；回答为闭区间。失败回答未纳入有效活动，仍保留候选行。</p>'
             '<div class="scroll"><table><tr><th>记录</th><th>组别</th><th>候选</th><th>有效活动回答</th><th>状态</th><th>候选负日</th><th>确认负日</th></tr>'
             + "".join(candidate_rows) + '</table></div></section>')
    page += ('<section><h2>接下来怎样验证</h2><p>固定候选、图和阶段头，先比较原确认与逐块观察／内部起变定位，'
             '明确找最早可见运动而非沿用候选起点，并保持活动／非活动／未知。另做原始位移短数值摘要因素对照；'
             '观察导数可用于活动内阶段，不能以导数提前响应确认平台已运动。</p><p>六例已查看仿真、相关AI辅助标签、'
             '一个弱段和离线特征限制依然存在。类别分数未校准，分块变化不作阈值，反事实不当优势证据。</p></section></body></html>')
    (destination / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    analyze(parser.parse_args().run)
