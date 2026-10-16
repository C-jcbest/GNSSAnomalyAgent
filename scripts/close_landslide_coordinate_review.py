"""Preserve the failed pilot, correct its design record and update the focused draft."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from finalize_landslide_adaptive_evidence import verify_manifest
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_coordinate_review import verify
from run_landslide_window_optimization import read_json
from summarize_landslide_missed_motion_repair import write_rows

from gnss_sim.landslide_evaluation import aggregate_scores


def table(rows):
    text = ["| 方法 | P | R | F1 | 弱R | 覆盖 | 事件F1(.3/.5) | 起/止MAE(日) | 内部/端到端ASD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        def number(key):
            value = row[key]
            return "未定义" if value is None or value == "" else f"{float(value):.4f}"
        text.append(f"| {row['method']} | {number('precision')} | {number('recall')} | {number('f1')} | "
                    f"{number('weak_recall')} | {number('coverage')} | {number('event_f1_iou03')}/{number('event_f1_iou05')} | "
                    f"{number('onset_mae_matched_days')}/{number('offset_mae_matched_days')} | "
                    f"{number('internal_stage_macro_f1')}/{number('stage_macro_f1')} |")
    return "\n".join(text)


def close(run, out):
    verify(run)
    analysis = run / "analysis-v1"
    verify_manifest(analysis)
    config = read_json(run / "protocol.json")
    data = Path(config["data"])
    predictions = read_json(analysis / "predictions.json")
    results = read_json(analysis / "results.json")
    with (analysis / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["method"]: row for row in rows}
    families = {cid: read_json(data / "mechanism-reference" / cid / "truth.json")["family"]
                for cid in config["case_ids"]}
    population = dict(Counter(families.values()))
    if population != {"quiet": 2, "weak_constant": 2, "reactivation": 2}:
        raise ValueError("Actual failed-pilot population differs from its correction")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    shutil.copyfile(ROOT / "tests/test_landslide_coordinate_review.py", out / "business-tests.py")
    doc = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    original = doc.read_bytes()
    (out / "manuscript-v20-original.md").write_bytes(original)
    group_rows, invariant_checks = [], 0
    for method, values in predictions.items():
        for family in sorted(population):
            aggregate = aggregate_scores([results[method]["per_case"][cid]
                                          for cid in config["case_ids"] if families[cid] == family])
            group_rows.append({"method": method, "family": family, **aggregate["daily"],
                               "weak_recall": aggregate["weak_recall"], "coverage": aggregate["coverage"]})
        if method.startswith("coordinate_consensus"):
            raw = predictions[f"raw_r{method[-1]}"]
            for cid, value in values.items():
                activity = np.array(value["activity"])
                before = np.array(raw[cid]["activity"])
                if np.any((activity >= 0) & (activity != before)):
                    raise ValueError("The frozen consensus is no longer an abstention-only operator")
                invariant_checks += 1
    event_rows = []
    for method, values in results.items():
        for threshold, key in ((.3, "boundaries_iou03"), (.5, "boundaries_iou05")):
            for cid, detail in values[key].items():
                event_rows.append({"method": method, "case_id": cid, "iou": threshold,
                                   "matched": len(detail["matched_pairs"]),
                                   "unmatched_predictions": detail["unmatched_prediction_events"],
                                   "missed": detail["missed_events"]})
    write_rows(out / "family-results.csv", group_rows)
    write_rows(out / "events.csv", event_rows)
    corrected = {
        "original_protocol_sha256": sha(run / "protocol.json"), "original_protocol_preserved": True,
        "actual_families": families, "actual_population": population,
        "population_error": "The odd-index selection spans three families twice, not six families once",
        "gate_error": "For every definite coordinate-consensus output, raw has the identical output. Consensus can only abstain. Net newly correct definite decisions over raw cannot be positive",
        "invalid_gate": "net_definite_correction_positive_vs_raw",
        "interpretation": "Do not treat failure of this unreachable gate as empirical evidence. F1 and weak recall fail independently in both repeats, and equal-cost repeat F1 is higher in both",
        "continue": False, "adopted": False, "independent_validation": False,
        "future_check": "Before freezing a trial, derive family counts from actual metadata and verify each acceptance condition is reachable by the proposed state transitions",
    }
    save_json(out / "protocol-correction.json", corrected)
    previous = read_json(ROOT / "artifacts/landslide-adaptive-review-2026-10-09/manuscript-v20/run-ledger-end.json")
    costs = read_json(analysis / "actual-costs.json")
    ledger = {
        "new_requests_started": costs["started"], "new_completed_local_receipts": costs["receipts"],
        "new_valid_model_json": costs["returned_json"], "new_reported_tokens": costs["reported_tokens"],
        "cumulative_requests_started": previous["cumulative_requests_started"] + costs["started"],
        "cumulative_local_receipts": previous["cumulative_local_receipts"] + costs["receipts"],
        "cumulative_reported_tokens_lower_bound": previous["cumulative_reported_tokens_lower_bound"] + costs["reported_tokens"],
        "new_records": 0, "pure_controlled_records": 120, "observed_residual_semisynthetic_records": 60,
        "inference_retries": 0, "failures": costs["failures"], "goal_status": "active",
        "round_status": "progress", "coordinate_method_adopted": False, "field_accuracy_confirmed": False,
        "billing": "Reported usage lower bound; all historical gaps retained",
    }
    save_json(out / "run-ledger-end.json", ledger)
    save_json(run / "run-ledger-end.json", ledger)
    notes = """# 坐标符号一致性：实际阅读范围与最近邻

2026-10-10；三次OpenAlex定向检索共18返回记录，含重复和无关结果，不作系统综述纳入数量。已读Chart Deception in Vision–Language Models: From Vulnerability to Mitigation官方arXiv2607.22600v1的摘要、引言开头、完整IV-C以及V-A/V-B开头。HTML原字节及全文机械提取文本保存；机械保存全文不等于全文阅读。

论文以忠实/误导图配对评价，包括逆坐标轴，提出图像→Vega-Lite结构化提取→图像与提取结果共同回答。它支持VLM可能受坐标表达影响这一研究动机，不保证GNSS符号反射有收益。本文x→−x且标清坐标约定不是其保留数据的逆刻度图，不作等同或作者算法复现。

作者库https://github.com/vis-nlp/visDeception，tree commit a3649527692c46d0cd06baeb86f87895216cd62d。已读完整README及ExtendedBenchmark/Mitigation/prompts.py：提取提示包含x/y轴方向、范围、图形类型和逐系列推断数值；solver收到图、提取JSON及选择题，优先可见图并返回选项/简理由。没有执行作者发布方法或模型；不依据一个提示文件声称全库无一致性复核。

获取失败保留：实验Python无fitz、bundled Python无httpx及fitz，PDF提取未执行；改取官方HTML成功。作者prompt首次urllib URLError，随后httpx成功；只属文献获取重试，模型请求无重试。paper-fetch-failure未生成是因为import在try外失败，以上说明明确补录；不填虚构PDF哈希。

本候选只用两个独立视觉回答的一致性许可，额外调用由普通重复对照匹配；不采用作者多Agent元数据架构，也不把测试时变换、一致性或弃判本身称为新颖。新机制实测两遍F1/弱R下降，停止。六记录实际三家族，登记准入含一条结构不可满足项，详见protocol-correction.json。
"""
    literature = ROOT / "artifacts/landslide-adaptive-review-2026-10-09/literature-v1/coordinate-consistency-search-v1"
    (literature / "reading-notes.md").write_text(notes, encoding="utf-8")
    save_json(literature / "delivery-manifest.json", {"files": {
        path.name: sha(path) for path in literature.iterdir() if path.is_file()}})
    selected = [indexed[f"{stem}_r{rep}"] for rep in (0, 1)
                for stem in ("raw", "reflected", "repeat_consensus", "coordinate_consensus")]
    selected += [indexed[method] for method in ("robust_rule", "pelt_rule", "xgboost", "raw_multiscale_rule",
                                              "xgboost_42_matched_support", "xgboost_84")]
    report = """# 坐标符号一致性开发试验：识别下降，停止；设计错误已更正

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Mode：run + descriptive validation
- Date：2026-10-10
- Status：completed, not adopted；仅开发，不是独立验证

仅将原始三轴位移同时取负，日期、毫米幅度、缺测和绘图尺度保持，图题标清坐标约定。复核原raw21/42冲突选定的最多两180日目标，49日权限不变。独立原图与反射图读数一致时允许确定修订，分歧未知；普通重复图使用同合并规则匹配成本。两遍同提示，不做提示/门槛搜索，原方法未修改。

**设计更正：**实际case0109/0111/0113/0115/0117/0119只覆盖quiet/weak_constant/reactivation各两条，冻结协议误写六家族各一条；未评价仪器阶跃、ASD家族或强后弱，不事后补人口。另一准入项“净确定纠错相对原图为正”不可满足：共识所有确定输出都是原图同值子集，不能增加原图没有的正确确定决策。不可达项失效不算实验证据，冻结协议和原decision.json保留，更正另存。可检验的F1、弱R及同成本对照条件也均失败，停止结论不依赖不可达项。

""" + table(selected)
    report += """

主F1 .7694/.7545，低于匹配原图.8283/.8221及普通重复.7804/.8300；弱R .7066/.5599对原图.9731/.9584。相对原图分别弃去178/190TP、104/103FP；没有新增TP或FP，净确定纠错−205/−196，另27/6正确阴性变未知。错误确定决策405→301、418→315，但覆盖1→.9261/.9285、正确确定决策3778→3573、3765→3569。由减少输出取得的错误总数下降不能称整体识别提高或无损可靠性。相对同调用成本普通重复，净确定纠错−86/−190，两遍F1均更低。

同任务强数值仍占优：XGB84 F1 .9103，42同支撑XGB .9083；规则、PELT及其余控制完整保留。模型使用同受控train24/validation12，分布和评分不变，未用本开发人口重训/调参。阶段仍同原始数据数值头在最终活动内输出，不对反射坐标计算阶段，不称原生VLM阶段创新。每类仅两条已见记录，逐日相关，不作显著性检验或普遍坐标偏差归因。

每遍主26部署调用/83101或82961token；普通重复26/82885或82825，单次原图16/48950或48852。各含6个缓存初判。实验总计60新启动/60回执/60成功JSON、204210token，无失败/重试，新记录0。原方法已有缓存也在同六记录评分，不能把本子集原方法.7601/.7431或直接.7635/.7642替代原12记录的确认结果；新提示/绘图与历史不同，识别增量只按新匹配原图解释。原控制成本未在summary.csv按子集重新归属，空值不是零成本；九数值API0，CPU未单独计时。

31配置/135780逐日行、60实际payload重放、54数值记录预测重算、全部活动/事件/边界/内部与端到端阶段指标、93家族汇总保留。复用阶段核验器的internal-stages.json固定scope文字写作selected targets，实际按全730日参考活动评分，A/S/D支撑217/637/193；本报告明确此元数据限制，不改冻结核验器。输入图和比较图已目检，缺测与活动/阶段掩码核查，12项相关业务测试及本轮Ruff通过。原源码/协议/预测/无效门槛和原v20精确字节保留，不以更换准入掩盖失败。

![坐标一致性开发比较](../artifacts/landslide-adaptive-review-2026-10-09/coordinate-review-development-v1/analysis-v1/comparison.png)

## 全部统一比较

""" + table(rows)
    report += """

## 复现与后续

仅离线重放分析（输出目录必须不存在），不重发60推理请求：

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$env:PYTHONIOENCODING='utf-8'
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/analyze_landslide_coordinate_review.py --run artifacts/landslide-adaptive-review-2026-10-09/coordinate-review-development-v1 --out artifacts/landslide-adaptive-review-2026-10-09/coordinate-review-score-replay
```

freeze.json/bindings.json绑定实际观测、图、提示、模型和执行源；prediction-freeze.json绑定回答，analysis-v1/manifest.json绑定原评分；manuscript-v21/protocol-correction.json记录上述设计更正。最近邻仅实际读取范围见literature-v1/coordinate-consistency-search-v1/reading-notes.md，未跑作者方法。本机制停止、不换单独反射组或提示救分。下一轮应先核查改动能够产生的状态转移与实际人口，再检验能够修好边界而非只弃判的观测机制；不挑新seed救本版本。

正式现场可信标签依旧缺，用户暂未提供人员，继续自主、不再索要或造专家参考；goal active。
"""
    report_path = ROOT / "docs/landslide-coordinate-review-results.md"
    report_path.write_text(report, encoding="utf-8")
    (out / "coordinate-report.md").write_text(report, encoding="utf-8")
    manuscript = original.decode("utf-8").replace("\r\n", "\n")
    manuscript = manuscript.replace("完整中文初稿v20", "完整中文初稿v21", 1)
    paragraph = "\n追加坐标符号一致性开发试验[12]时，六已见记录实际只含静稳、弱恒速和再活动各两条。两遍一致性主F1 .7694/.7545低于同提示原图.8283/.8221及同成本普通重复.7804/.8300，弱R下降，丢178/190TP而撤104/103FP，覆盖降至.9261/.9285，停止。该算子只能弃判，原登记的正净确定纠错门槛结构不可达；人口描述也写错，均以独立更正记录披露，不用这些无效设计项充当实证。其余有效指标亦失败，不事后补样本或修改门槛。完整结果见坐标复核补充报告；本方法与新种子确认分数不变。\n"
    manuscript = manuscript.replace("\n### 6.5 阶段、未知与成本", paragraph + "\n### 6.5 阶段、未知与成本", 1)
    manuscript = manuscript.replace("整个研究累计2519启动/2518回执/至少11000829已报告token，历史缺失费用保留。",
        "整个研究累计2579启动/2578回执/至少11205039已报告token，其中新增坐标开发60请求/204210token；历史缺失费用保留。", 1)
    manuscript = manuscript.replace("精确v19字节保留在manuscript-v20/manuscript-v19-original.md，所有旧源码/预测/失败不覆盖。",
        "精确v19字节保留在manuscript-v20/manuscript-v19-original.md；本轮前v20精确字节另存manuscript-v21/manuscript-v20-original.md，所有旧源码/预测/失败不覆盖。", 1)
    citation = "[12] Mahbub R, Islam M S, Laskar M T R, Rahman M, Nayeem M T, Hoque E. Chart Deception in Vision–Language Models: From Vulnerability to Mitigation. [arXiv:2607.22600v1](https://arxiv.org/html/2607.22600v1), 2026. [作者源码](https://github.com/vis-nlp/visDeception)，读取摘要/引言开头/IV-C/V-A与V-B开头及作者README/prompts.py，未复现发布效果。"
    manuscript = manuscript.replace("\n## 补充结果与完整数据", "\n" + citation + "\n\n## 补充结果与完整数据", 1)
    manuscript = manuscript.replace("- [最近邻公开源码阅读范围]", "- [坐标符号一致性负结果与设计更正](landslide-coordinate-review-results.md)\n- [最近邻公开源码阅读范围]", 1)
    doc.write_text(manuscript, encoding="utf-8")
    (out / "manuscript-v21.md").write_text(manuscript, encoding="utf-8")
    entry = """## 2026-10-10：坐标符号一致性停止，设计错误独立更正，全文v21

[坐标开发](landslide-coordinate-review-results.md)：60真实请求全成功/204210token，两遍主F1 .7694/.7545对匹配原图.8283/.8221及同成本普通重复.7804/.8300，弱R下降，丢178/190TP、撤104/103FP，净确定纠错−205/−196，停止，不换消融或提示救分。设计错误必须保留：实际六记录仅quiet/weak_constant/reactivation各两条，非六家族；共识只能弃判，登记的正净确定纠错门槛结构不可达。另存protocol-correction.json，原协议/decision/source/预测不改，停止依据其余有效F1/弱R/同成本条件，不把不可达项失败当实证。

31统一配置含九重算数值控制、135780逐日行、60payload、54数值预测、93家族结果；12相关业务测试/Ruff通过。补读VisDeception官方IV-C/部分结果和作者commit a364952的README/prompts，不复现作者模型或称坐标变换首创。v20精确原字节存manuscript-v21，全文只追加失败与成本，不扩散历史报告。累计2579启动/2578回执/至少11205039token；120受控/60半合成不变，无后台推理。goal active，正式领域参考仍未知，用户暂无人员，继续自主、不再询问或造标签。

下一轮冻结前必须从实际元数据核查人口、推导准入条件能否由状态转换达到，再做能够纠正标签/边界的小机制验证。不能只用弃判减少错误总数主张识别提高，也不重复此反射方案或挑seed救分。

"""
    for name in ("docs/project-status.md", "docs/landslide-research-continuation-prompt.md",
                 "docs/landslide-submission-roadmap.md"):
        path = ROOT / name
        existing = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        first, remaining = existing.split("\n", 1)
        path.write_text(first + "\n\n" + entry + remaining.lstrip("\n"), encoding="utf-8")
    save_json(out / "checks.json", {"actual_population_verified": population, "subset_invariant_replays": invariant_checks,
                                   "family_rows": len(group_rows), "event_rows": len(event_rows),
                                   "v20_exact_bytes_preserved": sha(out / "manuscript-v20-original.md") == sha(ROOT / "artifacts/landslide-adaptive-review-2026-10-09/manuscript-v20/manuscript-v20.md"),
                                   "new_manuscript_matches": sha(doc) == sha(out / "manuscript-v21.md"),
                                   "goal_status": "active"})
    save_json(out / "manifest.json", {"files": {str(path.relative_to(out)): sha(path)
                                               for path in out.rglob("*") if path.is_file()}})
    print({"ledger": ledger, "population": population, "stopped": True}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    close(args.run, args.out)
