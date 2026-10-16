"""Export transparent stage-study tables and preserve the exact previous manuscript."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json

from gnss_sim.landslide_evaluation import score_counts


def number(value):
    return f"{value:.4f}" if value is not None else "—"


def table_metrics(run, out):
    source = Path(read_json(run / "protocol.json")["source"])
    predictions = read_json(run / "analysis-v3/combined-predictions.json")
    results = read_json(run / "analysis-v3/combined-results.json")
    labels = {cid: [] for cid in next(iter(predictions.values()))}
    with (source / "reference-subset.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["case_id"]].append(row)
    metrics, boundaries = {}, {}
    for method, cases in predictions.items():
        weak_total, weak_detected = 0, 0
        details = {}
        for cid, value in cases.items():
            activity = np.asarray(value["activity"])
            weak = np.asarray([int(row["weak_signal"]) == 1 for row in labels[cid]])
            weak_total += int(weak.sum())
            weak_detected += int((weak & (activity == 1)).sum())
            details[cid] = {str(threshold): boundary_errors(labels[cid], activity, threshold)
                            for threshold in (0.3, 0.5)}
        boundaries[method] = details
        matched = [pair for case in details.values() for pair in case["0.3"]["matched_pairs"]]
        if len(matched) != results[method]["aggregate"]["events"]["tp"]:
            raise ValueError("Independent event matching differs")
        matched05 = sum(len(case["0.5"]["matched_pairs"]) for case in details.values())
        unmatched05 = sum(case["0.5"]["unmatched_prediction_events"] for case in details.values())
        missed05 = sum(case["0.5"]["missed_events"] for case in details.values())
        aggregate = results[method]["aggregate"]
        metrics[method] = {
            **aggregate["daily"], "event_f1_iou03": aggregate["events"]["f1"],
            "event_f1_iou05": score_counts(matched05, unmatched05, missed05)["f1"],
            "matched_events": len(matched),
            "onset_mae_days": float(np.mean([abs(pair["onset_error_days"]) for pair in matched])) if matched else None,
            "offset_mae_days": float(np.mean([abs(pair["offset_error_days"]) for pair in matched])) if matched else None,
            "weak_days": weak_total, "weak_detected_days": weak_detected,
            "weak_recall": weak_detected / weak_total if weak_total else None,
            "stage_macro_f1": aggregate["stage_macro_f1"],
            "internal_macro_f1": results[method]["independent_internal"]["macro_f1"],
            "internal_coverage": results[method]["independent_internal"]["definite_stage_coverage"],
        }
    save_json(out / (run.name + "-metrics.json"), metrics)
    save_json(out / (run.name + "-boundaries.json"), boundaries)
    with (out / (run.name + "-metrics.csv")).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *next(iter(metrics.values()))])
        writer.writeheader()
        writer.writerows({"method": method, **row} for method, row in metrics.items())
    return metrics


def method_table(metrics):
    lines = ["| 配置 | 活动P | R | F1 | 弱R | 事件F1 .3/.5 | 起点/终点MAE日 | 端到端阶段宏F1 | 真活动内宏F1 | 阶段覆盖 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, row in metrics.items():
        fields = [method, *[number(row[key]) for key in ("precision", "recall", "f1", "weak_recall")],
                  number(row["event_f1_iou03"]) + "/" + number(row["event_f1_iou05"]),
                  number(row["onset_mae_days"]) + "/" + number(row["offset_mae_days"]),
                  *[number(row[key]) for key in ("stage_macro_f1", "internal_macro_f1", "internal_coverage")]]
        lines.append("| " + " | ".join(fields) + " |")
    return "\n".join(lines)


def main():
    root = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
    out = root / "manuscript-v12"
    out.mkdir(parents=True, exist_ok=False)
    manuscript = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    original = manuscript.read_bytes()
    (out / "manuscript-v11-original.md").write_bytes(original)
    development = root / "stage-contrast-review-development-v2"
    confirmation = root / "stage-contrast-confirmation-review-v1"
    dev_metrics, confirm_metrics = [table_metrics(run, out) for run in (development, confirmation)]
    receipts = []
    run_names = ["stage-contrast-review-development-v1", "stage-contrast-review-development-v2",
                 "stage-contrast-confirmation-activity-v1", "stage-contrast-confirmation-native-v1",
                 "stage-contrast-confirmation-review-v1"]
    for name in run_names:
        receipts.extend(read_json(path) for path in (root / name / "requests").glob("*.response.json"))
    requests_started = sum(len(list((root / name / "requests").glob("*.started.json"))) for name in run_names)
    tokens = sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts)
    successful = sum(row["status"] == "returned_json" for row in receipts)
    ledger = {"new_requests_started": requests_started, "new_completed_local_receipts": len(receipts),
              "new_valid_model_json": successful, "new_reported_tokens": tokens,
              "provider_rejections": sum(row.get("http_status") == 400 for row in receipts),
              "connection_failures": sum(row.get("error_type") == "ConnectError" for row in receipts),
              "cumulative_requests_started": 1663 + requests_started,
              "cumulative_local_receipts": 1662 + len(receipts),
              "cumulative_reported_tokens_lower_bound": 7446924 + tokens,
              "pure_controlled_records": 108, "observed_residual_semisynthetic_records": 60,
              "billing": "No exact total: earlier interrupted receipt and current connection failures have no usage"}
    save_json(out / "run-ledger-end.json", ledger)
    method = """阶段复核使用既定活动内三个14日原始NEU中位数块，中心相距42日。前后位移变化的范数除以42得到两侧速率；仅在全部已观测证据日均已确认活动、每块至少8完整观测、最大缺口不超过7日且两次位移方向一致时可用。变化门槛为两侧最大速率的25%与2倍全记录日差MAD噪声速率代理二者较大值；它不是置信区间。A遇明确降速、D遇明确升速、S遇明确变化才构成矛盾。同一连续既有阶段片段累计至少14矛盾日才允许选窗，不是14连续矛盾日。最多复核一个活动，原始全局图、局部原图及61日辅助图保留；数值表只提供同源测量，不提供参考类别。主方案只改写原矛盾日期，替换类别仍矛盾则未知；活动与缺测不变。"""
    report = f"""# 活动内原始速率矛盾驱动的阶段复核：开发与新冻结确认

日期2026-10-09。ARS实验run/validate；Verification Status：保存预测、请求和评分已验证，领域观测准确性未验证。

开发显示局部阶段修正价值，但冻结选择器在新12条记录两遍均0触发，未确认整体阶段改进。停止扩展这套选择器；不降低门槛、切换事后固定组或重复推理救分。原活动主方法及年参照负结果不改。

## 机制与选择顺序

{method}

原双尺度21/42主探针在0037—0048两遍均0干预，未通过。raw42作为已登记单尺度消融，分别撤回85/97错误阶段日，损失5/0正确日；据此另冻结42日假设。第二已见开发0049—0060撤回105/84错误日、无正确损失，获准局部模型开发。两个开发批均已暴露，不称独立。单尺度不是原双尺度假设的通过结果。

## 完整开发与消融

0049—0060全部12条两遍，34有效局部请求、197477token，0契约失败。主数值反馈修正105/46错误日，无正确阶段损失，真活动内宏F1 .2471/.2810→.3060/.3156；仅弃判为.2497/.2846。纯图为.2856/.3405，反馈一遍较好、一遍较差，没有稳定增量证据。精确范围外全保留；整目标更新修好262/159日，却损失83/30正确日，宏F1更高不能掩盖损害。第二遍去veto会将38个既有错误改成另一错误，veto保留未知，不能叫修正正确。固定与自适应选中的相关目标相同，纯图效果相同，只能比较实际调用预算，不能宣称选窗准确性优势。

所有24配置如下；两个普通fixed full为事后补全对照，使用此前真实冻结回答，没有新调用，不参与选择主方法。

{method_table(dev_metrics)}

## 新冻结确认

0097—0108在上述主方法、策略与源码哈希冻结后生成，6预定义家族各2条、730日，延续原生成分布并使用新的独立随机流。原train24/validation12拟合及调参的规则、PELT、XGBoost和两个同原始长证据控制不改变；这些新标签未用于拟合。没有访问旧封存测试。累计108纯受控记录与60观测残差半合成记录分别计数。

新主方法两遍0个选定活动、0次额外自适应请求，阶段与初始完全相同。端到端宏F1 .3961/.4288，真活动内 .4037/.4364；活动F1 .7276/.7185不变。原始对比在14日条件之前仅case0100有2/10个矛盾日，其他记录0。case0102没有可用42日活动内证据；case0106第一遍原生阶段因连接失败全未知，该选择器也不覆盖原生unknown。这是选择器未提供纠错机会，不证明局部复核本身无效。

36局部尝试全部属于固定对照池，33成功、3ConnectError，183704已报告token；初始直接活动24成功、59421token；原生阶段18尝试17成功、1ConnectError、89604token。全部未知与失败进入完整分母，无重试。ordinary fixed numeric第二遍损失268正确日，其中260来自0100/0107连接失败、8来自已成功目标；不能把268全归为模型内容损害。fixed image第二遍损失134日均来自已有回答内容。post-hoc fixed组不是预冻结自适应主方法的独立确认，也不以其中最好一遍替换主结论。

{method_table(confirm_metrics)}

共同真活动内A/S/D分母由逐日参考及缺测确定。弱召回从weak_signal属性计算，旧通用评分器的weak_days=0仅因其另用slow_displacement字段，不作为本任务弱人口；完整弱分母另存metrics。事件按已知可评分日IoU .3/.5一对一匹配，不做point-adjustment；边界MAE只在匹配事件上计算，端点为可评分参考活动端点，漏事件与未匹配预测均保存。24配置逐日预测、逐类P/R/F1、参考未知及全记录图都公开，不只保留成功子集。XGBoost新批真活动内宏F1 .8531仍明显更高，阶段修正开发收益不等于活动改善或强数值优势。

## 成本、失败与可复现性

本轮{requests_started}启动／{len(receipts)}本地回执／{successful}成功模型JSON，已报告{tokens}token；含34条首轮HTTP400拒绝和4连接失败。累计{ledger['cumulative_requests_started']}启动／{ledger['cumulative_local_receipts']}回执／至少{ledger['cumulative_reported_tokens_lower_bound']}token，未收到usage的费用未知，不报告精确总账或货币。

失败版本全部保留：原stage probe误纳解除门控数值消融，0API；模型v1提示未含字面JSON而34次HTTP400，没有模型回答，不用于内容效果；分析v1错误匹配no_veto名称；确认分析v1/v2沿用只接受成功回执的核验器，未改预测，v4核验全部失败绑定。修复均另存版本，不覆盖冻结源。连接失败不作模型正确弃判，也不删除评分日期。

开发与新确认各24配置、216活动／权限检查、210240逐日CSV行及12同轴全记录图；实际回执34＋36全部绑定，旧图与源码哈希核对。新阶段证据6测试通过，Ruff通过。保存输出可确定重放，不保证远程模型再次产生相同回答。

## 最近邻与解释边界

新增10条定向arXiv记录及TSHarness方法、时间序列Agent综述相关段落，阅读范围见literature-v1/stage-consistency-search-v1/reading-notes.md。有限循环、受保护修订、数值感知与语义分工都已有先例；不把更少工具或提示数当创新。TimeEvo先前源码阅读边界保持。当前阶段候选没有新批实测收益，不能升格为已验证贡献或独立现场结论。

11项解释检查均处理：总体与逐例并报；日期不作独立站点；不按成功／矛盾支撑筛主分母；A/S/D基率保存；同期两遍不挑最高；所有失败保留；全部尺度和消融公开、无显著性；post-hoc fixed独立标注；测量同源不当独立证据；居中辅助及全记录只用于回顾；不以阶段未知反推无活动。开发两遍、12新记录和三/四个连接失败都不足以估计随机推理的完整分布。

## 复现与产物

复现Python：artifacts/reproduction-2026-10-07/environment/Scripts/python.exe；PYTHONPATH设src与scripts，编码UTF-8。只读表导出及核验：scripts/summarize_landslide_stage_contrast_review_v4.py，输入已完成run、同训练additional-baselines及排他新out；不再次run推理。冻结开发、新数据生成、初始活动、原生阶段和确认runner分别为scripts/probe_landslide_stage_raw_consistency_v2.py、probe_landslide_stage_contrast_transfer.py、prepare_landslide_stage_contrast_confirmation.py、run_landslide_stage_confirmation_activity.py、run_landslide_adaptive_vlm_stages.py、run_landslide_stage_contrast_confirmation.py。

最终有效结果：stage-contrast-review-development-v2/analysis-v3与stage-contrast-confirmation-review-v1/analysis-v3；完整补充指标/事件/边界CSV与台账在manuscript-v12。现有独立领域复核者仍缺，按用户授权继续自主方法研究，不造专家标签或重复请求人员。下一研究不保留这套未触发选择器，不以调低同批门槛或事后fixed替代新冻结验证。
"""
    report_path = ROOT / "docs/landslide-stage-contrast-review-results.md"
    report_path.write_text(report, encoding="utf-8")
    addition = """### 6.15 活动内原始速率矛盾驱动的原生阶段复核

阶段候选只检查已确认活动内三个14日NEU中位数块的前后位移速率。42日中心间隔、25%相对变化及2倍日差MAD噪声代理均在新记录生成前冻结；非置信界。A遇明确降速、D遇明确升速、S遇明确变化才构成矛盾，同一连续既有阶段片段累计至少14矛盾日可选最多一个活动。模型保留全局原图并增加局部原图、61日辅助图和同源数值表，只允许修订原矛盾日期；仍矛盾的替换未知。活动与缺测始终不改，因此不会产生新活动检测收益。

原双尺度开发0干预，随后从其raw42消融形成新假设。0049—0060两遍34有效局部请求／197477token，主数值反馈修好105/46阶段日，没有改坏正确日；真活动内宏F1由.2471/.2810提高到.3060/.3156，高于仅弃判.2497/.2846。纯图为.2856/.3405，反馈没有稳定增量；整目标更新为.3421/.3609，却损失83/30正确阶段日。范围保护表现为修好与改坏的取舍，不能只挑最高F1。固定与自适应相关观察目标相同，不能据相同结果证明选窗准确率优势；共同XGBoost真活动内.7984仍更高。两个已见批次只作开发证据。

### 6.16 新阶段冻结确认：选择器无观察机会

预冻结主数值反馈、42日证据、14矛盾日、最多一次复核、精确权限及veto后，新增0097—0108十二条730日受控记录，六家族各两条独立随机流。旧train24/validation12模型与参数保留，因为生成分布相同；新标签未用于拟合或选择。纯受控总108与P227残差半合成60分别计数，仍没有独立专家观测参考。

主方案两遍0触发，真活动内宏F1 .4037/.4364与原生阶段完全相同；端到端.3961/.4288、活动F1 .7276/.7185也不变，未通过冻结继续条件。14日条件之前仅0100有2/10个矛盾日，其余0；0102没有42日活动内证据，0106第一遍原生阶段连接失败全未知，选择器不覆盖unknown。故停止扩展该选择器，不降低同批门槛或改选fixed。不触发未检验局部纠错本身，却已说明当前完整策略不能迁移开发收益。

实际36局部尝试均为固定对照池，33成功、3连接失败；初始活动24成功、原生阶段18尝试17成功另1连接失败，无重试。事后补全ordinary fixed纯图真活动内宏F1 .4727/.4922，修好141/227日并损失12/134正确日；fixed数值为.4938/.4019，修好208/153日并损失24/268日，第二遍268损失中的260来自两条连接失败，不归为模型内容错误。post-hoc fixed不是预冻结主方法确认，不按最佳遍升级。共同XGBoost真活动内.8531和多尺度原始规则活动F1 .8664继续显示强基线价值。

开发和确认各24统一配置，包括六个共同训练数值控制；弱属性分母、两IoU事件匹配、匹配边界误差、阶段三类及覆盖、逐日CSV和全记录图均保存。两批各216权限检查、210240导出行、12同轴图，所有失败进入分母。本轮146启动/146本地回执/108成功模型JSON，至少530206新增已报告token，含34个首版接口拒绝及4连接失败；累计1809启动/1808回执/至少7977130token，未知费用保留。

TSHarness[17]采用显式数值感知状态与有限复感知循环，时间序列Agent综述[18]讨论证据质量和对基础检测器不劣化；这些通用思想不是本文首创。当前新的阶段选择机制未获冻结新批收益，不能算已验证阶段贡献。完整24配置、源哈希及所有失败见[阶段实验报告](landslide-stage-contrast-review-results.md)。

"""
    text = original.decode("utf-8")
    first_newline = text.index("\n", text.index("> 完整中文初稿v11"))
    first_start = text.index("> 完整中文初稿v11")
    text = text[:first_start] + "> 完整中文初稿v12，2026-10-09。新增活动内原始速率矛盾的阶段开发及12条新冻结确认：开发修好105/46日、无正确损失，新批选择器两遍0触发，不升级该机制。各24配置、全部失败和强基线保留；独立领域观测参考仍缺，本稿不是投稿定稿。" + text[first_newline:]
    text = text.replace("## 7 讨论", addition + "## 7 讨论", 1)
    abstract_note = "追加阶段机制在开发修好105/46日、无正确标签损失，但新12条冻结确认两遍0触发，未获阶段收益；不将开发修正或事后固定组升级为已验证贡献。\n\n"
    text = text.replace("**关键词：**", abstract_note + "**关键词：**", 1)
    text = text.replace("## 1 引言", "A subsequent stage-repair development corrects 105/46 stage days without losing correct labels, "
                        "but the frozen selector triggers zero reviews on twelve new analytic records. "
                        "The development gain is not confirmed and this selector is not promoted.\n\n## 1 引言", 1)
    text = text.replace("## 参考文献", "新阶段候选虽在开发修正部分A/S/D错误，新记录却没有提供自适应观察机会；停止该选择器扩展，保留非劣化边界、固定观察损益和强数值优势，不把局部开发收益写成已确认方法贡献。\n\n## 参考文献", 1)
    text = text.replace("## 附录A：产物定位", "[17] Xie, J., Yuan, H., & Wang, Y. (2026). *Zero-Shot Time-Series Question Answering via Decoupled Perception and Reasoning*. arXiv:2610.04942v1. https://arxiv.org/abs/2610.04942v1\n\n"
                        "[18] Chen, Y., Qin, X., Liu, C., Wu, L., Samia, N. I., & Ding, K. (2026). *LLM Agents for Time-Series: A Survey*. arXiv:2608.26226v1. https://arxiv.org/abs/2608.26226v1\n\n## 附录A：产物定位", 1)
    text = text.replace("| 当前全文与v10原字节快照 | manuscript-v11 |", "| 阶段原始证据开发与完整消融 | stage-contrast-review-development-v2 / analysis-v3 |\n"
                        "| 新阶段冻结数据、初始活动、原生阶段与确认 | stage-contrast-confirmation-data-v1；stage-contrast-confirmation-activity-v1；stage-contrast-confirmation-native-v1；stage-contrast-confirmation-review-v1 / analysis-v3 |\n"
                        "| 当前全文与v11原字节快照、补充事件边界和成本 | manuscript-v12 |", 1)
    text += "\n新增阶段开发与冻结确认已写入6.15—6.16，未通过新批方法确认；最新有效分析为两批analysis-v3。源码、原预测、四个连接失败、34接口拒绝和全部失败分析保持原版本。\n"
    manuscript.write_text(text, encoding="utf-8")
    (out / "manuscript-v12.md").write_bytes(manuscript.read_bytes())
    status = """## 2026-10-09：阶段原始证据复核开发通过，新12条冻结确认0触发，全文v12

[阶段研究](landslide-stage-contrast-review-results.md)：原双尺度0干预，另从raw42消融形成新假设。已见开发两遍修好105/46阶段日，无正确损失；数值反馈对纯图无稳定增量，整目标会改坏83/30正确日。新0097—0108冻结确认两遍0触发、无新增阶段收益；停止该选择器，不降同批门槛、改选post-hoc fixed或重跑推理救分。ordinary fixed纯图有修正及12/134损害；fixed数值第二遍268损害中260来自连接失败，不能当内容失效。

两批各24配置、六同训练数值控制、216权限检查、210240逐日行和12全图；所有70局部回执和失败保留。12新记录延续同分布与原train24/validation12模型，纯受控108与观测残差半合成60分开。新增146启动/146本地回执/108成功模型JSON/530206已报告token，包括34接口拒绝和4ConnectError；累计1809/1808/至少7977130token，未知费用不补造。有效分析均analysis-v3。原始阶段证据6测试及Ruff通过；v12和精确v11原字节已保存。阶段主要新假设未确认，不升级主线；用户暂无领域复核者，继续自主研究，不重复索要或造标签。goal active，有实质进展，非完成或blocked。

所有本轮推理均结束，禁止重跑旧run。后续研究须面对未知阶段和活动范围不足使证据不可用的问题，先验证观察机会的纠错价值；不以同批门槛搜索替代新冻结验证。TSHarness/综述10条检索、实际原文阅读范围及失败源已记；通用受保护修订和工具分工不是创新。

"""
    for name in ("project-status.md", "landslide-research-continuation-prompt.md"):
        path = ROOT / "docs" / name
        prior = path.read_text(encoding="utf-8")
        position = prior.index("\n", prior.index("\n") + 1) + 1
        path.write_text(prior[:position] + "\n" + status + prior[position:], encoding="utf-8")
    for name in ("landslide-adaptive-review-results.md", "landslide-nearest-method-code-review.md"):
        path = ROOT / "docs" / name
        prior = path.read_text(encoding="utf-8")
        position = prior.index("\n") + 1
        path.write_text(prior[:position] + "\n最新[阶段原始证据研究](landslide-stage-contrast-review-results.md)：开发修正105/46日、无正确损失，新12条冻结确认两遍0触发，未升级该选择器；完整稿v12。TSHarness与综述说明有限证据循环、工具分工、对基础检测器不劣化已有先例，实际阅读边界见新检索notes，不将通用机制当首创。\n" + prior[position:], encoding="utf-8")
    if (out / "manuscript-v11-original.md").read_bytes() != original:
        raise ValueError("Original manuscript bytes changed")
    save_json(out / "closing-checks.json", {
        "manuscript_v11_original_sha256": sha(out / "manuscript-v11-original.md"),
        "manuscript_v12_sha256": sha(manuscript), "exported_metric_populations": [24, 24],
        "source_sha256": sha(Path(__file__)), "goal_status": "active",
    })
    print(ledger, flush=True)


if __name__ == "__main__":
    main()
