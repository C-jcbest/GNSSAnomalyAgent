"""Write result tables directly from the completed dual-target experiment."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_dual_reference_confirmation import BASE, COMPARISON
from prepare_landslide_ai_reference_holdout import verify_manifest
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json

ROOT = BASE.parents[1]
ANALYSIS = BASE / "dual-reference-confirmation-analysis-v1"
OUT = BASE / "dual-reference-confirmation-report-v2"


def read_rows(target):
    with (ANALYSIS / target / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def full_table(rows):
    lines = ["| 方法 | P | R | F1 | 事件F1 .3/.5 | 弱R | FP/FN | 起/止MAE日 | 内部/端到端阶段F1 | 调用 | token |",
             "| --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | ---: | ---: |"]
    for row in rows:
        def number(key):
            return f"{float(row[key]):.4f}" if row[key] else "—"
        lines.append(f"| {row['method']} | {number('precision')} | {number('recall')} | {number('f1')} | "
            f"{number('event_f1_iou03')}/{number('event_f1_iou05')} | {number('weak_recall')} | "
            f"{row['fp']}/{row['fn']} | {number('onset_core_mae_days')}/{number('offset_core_mae_days')} | "
            f"{number('stage_internal_macro_f1')}/{number('stage_end_to_end_macro_f1')} | "
            f"{row['requests']} | {row['reported_tokens']} |")
    return "\n".join(lines)


def figure(tables):
    selections = [("Main r0", "adaptive_no_feedback_r0__conflict_scope"),
        ("Main r1", "adaptive_no_feedback_r1__conflict_scope"),
        ("Direct r0", "direct_visual_r0"), ("Direct r1", "direct_visual_r1"),
        ("Fixed whole r0", "fixed_image_review_r0__whole_target"),
        ("Fixed whole r1", "fixed_image_review_r1__whole_target"),
        ("Same K late r0", "same_count_fixed_late_r0__conflict_scope"),
        ("Same K late r1", "same_count_fixed_late_r1__conflict_scope"),
        ("Numeric feedback r0", "adaptive_numeric_feedback_r0__conflict_scope"),
        ("Numeric feedback r1", "adaptive_numeric_feedback_r1__conflict_scope"),
        ("XGBoost", "xgboost"), ("Persistence XGBoost", "xgboost_raw_persistence"),
        ("Raw multiscale rule", "raw_multiscale_rule")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for axis, (target, rows) in zip(axes, tables.items(), strict=True):
        values = {row["method"]: row for row in rows}
        axis.barh(np.arange(len(selections)), [float(values[method]["f1"]) for _, method in selections],
            color=["#327b88" if label.startswith("Main") else "#87949b" for label, _ in selections])
        axis.set_yticks(np.arange(len(selections)), [label for label, _ in selections])
        axis.set_xlim(0, 1)
        axis.set_xlabel("Activity F1")
        axis.set_title("Author-AI definite dates" if target == "primary_ai" else "All observed latent-process dates")
        axis.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("New12 controlled records; target-matched models; both repeats retained")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=160)
    fig.savefig(OUT / "comparison.svg")
    plt.close(fig)


def write_report():
    verify_manifest(ANALYSIS)
    OUT.mkdir(parents=True, exist_ok=False)
    tables = {target: read_rows(target) for target in ("primary_ai", "secondary_process")}
    figure(tables)
    agreement = read_json(ANALYSIS / "reference-agreement.json")
    audit = read_json(COMPARISON / "analysis/audit.json")
    text = """# 未见记录的双参考确认：目标适配基线、视觉复核与位置消融

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate，当前Codex直接执行与实际视觉复核
- Origin Date: 2026-10-11
- Verification Status: 130真实请求/回执；34方法×2目标统一评分，sklearn与共享契约核验；精确重放见verification-v1
- Version Label: dual-reference-confirmation-v1

## 主要判断

原位置机制在主作者AI观测目标未完全复现：主F1 .8536/.8363，登记同次数早/晚位置四项比较通过3/4；第二遍低于同次数固定最晚.8467，净确定纠错−46，亦略低于直接视觉.8377与固定整目标.8371。全观测过程目标主F1 .8280/.8073，四比较通过4/4；该目标是生成前登记的第二目标，不覆盖主目标失败。原方法保持作为已实现、场景受限的候选，不能写两遍均稳定胜出。

数值文字反馈主目标F1 .8951/.8504、弱R .7363/.5801，比原主点值高；过程目标.8434/.7999，第二遍退化，事件分割也可能变坏。它是原登记消融，并非本轮新发明。不得根据本批选择反馈或居中最优结果替换原主要方法后仍称独立验证；本批从此可用于错误开发，下一核心机制改变须另行冻结验证。

## 冻结顺序与公平性

沿用已适配作者AI观测目标的ai-observable-calibration-v3；另从完全相同0049—0060原始输入、相同6训练/6验证划分，复用原网格与验证dailyF1+eventF1目标，重新训练/校准过程目标的规则、PELT、普通XGB、规则活动+XGB阶段、原始多尺度规则、持续证据XGB及共同阶段头。两套模型均在新输入生成之前冻结。主模型未冒称本轮重新训练，过程模型未冒用AI目标模型。训练为0049/0051/0053/0056/0058/0060，验证为0050/0052/0054/0055/0057/0059。

新seed2026101181、排列seed2026101182、stream144—155及case0145—0156在生成前登记；旧六家族各2，不重抽。当前Codex实际查看24幅原始全局/局部图，先冻结活动，再看12幅辅助图与31/61/91日数值；本批预测和潜在标签未见。原作者熟悉通用设计、历史方法，且在v30已见0133—0144过程标签；本次不是独立专家标注。复核标签与模型输出冻定后才打开本批封存逐日过程标签，具体generation-ledger家族映射未作语义读取。

原21/42日三块持续运动冲突、最多2个不相交180日观察、14日准入和49日修订权限完全不变。所有视觉配置共享同一批真实回答；第二目标只更换适配的共同阶段头与六个数值控制，视觉活动预测逐日完全相同，不重复调用。各目标分别按自己的活动参考、事件及阶段评分，未知不补齐，不比较两个目标弱R的不同分母。

## 观测复核与参考边界

8,760日、440缺测；主参考确认活动1,359、静稳5,822、未知1,579。未知中1,139有观测，不能自动当静稳。474确定阶段日为A169/S141/D164，活动内另885阶段未知。确认活动来自原始持续位移；阶段证据不足、相邻阶段跨窗混合或多尺度冲突均保留未知，例如0151整段弱活动不确定A/S/D，0145尾部和0155峰/尾支撑不足。

两参考确定活动/静稳一致98.19%，但130日AI静稳对应过程活动；1,139观测未知含353过程静稳和786活动，必须单列。过程全观测活动2,275日，AI确定活动覆盖仅59.74%。474确定A/S/D中359一致/115分歧，一致75.74%，不能用高活动一致率声称阶段参考可靠。两参考及未评分未知保留原标签。本次两个目标同时改变参考与匹配模型，不能把其成绩差异归因于单个因素。

共享编译器保留了旧2026-10-10日期和prior_generation_labels_seen=false元数据；本批实际日期与历史暴露由actual-review-provenance.json及生成前protocol明确更正，原冻结标签不改。Windows路径键导致第一次准确来源说明未写入，随后准备目录采用旧图片检查名单；该无远端调用的准备失败完整保留于preparation-metadata-failure-v1，正式使用comparison-v2。失败不涉及改标签、阈值、种子或挑重复。

## 主目标：作者AI确定观测日期

评价7,181确定日期，活动1,359，预定四个弱内部共512日。边界为确认内部端点，并非精确物理起止。所有34方法如下，r0/r1不作为独立样本合并。

"""
    text += full_table(tables["primary_ai"])
    text += """

主相对同次数最早位置净确定纠错+88/+9，相对最晚+58/−46。弱R均不下降，但第二遍F1和纠错失败。主相对固定整目标净确定纠错−10/−140，不能把F1点值增益当作全部确定判断更可靠；相对直接视觉净+216/+82，其中第二遍活动F1仍略降。未知→确定及确定→未知也影响纠错，该计数与只计阳性的F1不是同一个指标。

主比整目标修订两遍各撤32FP、TP不变，F1提高；第一遍居中范围F1 .8624更高，第二遍主.8363略高于居中.8353。全部位置比较的主目标整记录描述bootstrap区间跨0，未建立小样本显著性或唯一最优范围。

## 第二目标：全观测潜在过程

评价8,320日期、活动2,275，过程弱活动801日；规则/PELT/XGB均为过程目标重新匹配模型。边界使用受控过程端点，不能把这些误差与AI确认内部误差混称同一个真值。全部方法保留。

"""
    text += full_table(tables["secondary_process"])
    text += """

主相对同次数最早净+69/+24，最晚+32/+26，四条件通过；主两遍日F1高于直接、固定及本批全部数值控制。过程主相对固定整目标净−51/−133，仍有未知覆盖/静稳决策代价。过程数值反馈第二遍F1低于原主、事件F1 .6667低于原主.7143，未建立跨目标稳定增量。历史XGB84优于主方法及半合成失败仍有效，不能据本批较弱数值成绩宣称普遍视觉优势。

## 实际视觉错误分析

评分后实际查看0148、0152、0153、0154三轴+预测对齐图，不回写参考或预测。相对同次数最晚位置的第二遍：0148恢复36TP却增加49FP，末端E已接近平台而全局/局部活动仍延长；0152增加31个早期静稳FP，后段530—650弱活动仍未检出，部分改为未知并不能恢复召回；0154增加29FP而无新增TP。0153主撤50早期FP、增加5FP，净+45，但第二轮弱活动仍漏检。数值文字反馈在0153恢复59TP的同时增加50早期FP，故恢复局部弱尾不等于正确边界。

四图、完整144个逐记录配对纠错和所有变化区间保存在errors-v1。观察位置有修复价值，但局部回答仍会跨入平台或沿错误初判过扩张。当前应探索更明确的局部起止证据表达，先在已暴露记录验证作用；不复活已停止的整段运动保护或双反证模块，也不根据本批成绩添加新门槛。

## 阶段、成本与复现

主目标所有视觉配置内部A/S/D macroF1相同.9061，来自共同数值阶段头，不是原生VLM阶段推理创新。原主端到端阶段.8347/.8240，固定.8179/.8582，第二遍低；过程原主内部.7694/.7596、端到端.7495/.7363。两参考的阶段人口与阶段头不同，不能跨列推断哪个更准确。各A/S/D P/R/F1和支持数在results.json，未略去未知阶段。

原主每遍部署29/28调用、98,941/94,778报告token；固定36/36调用、127,419/127,517token；直接12/12调用、29,735/29,743token。数值文字反馈同调用数但114,284/109,262token。全消融共享130独特真实启动/130回执，520,696报告token，远端run墙钟358秒，无失败/重试；六数值预测局部计时不含训练和初始特征提取，不能把它当完整端到端耗时或人类复核成本。

累计2,980启动/2,979回执、至少13,067,842报告token，已有一次无回执未知token不补零；受控156条、半合成60条。独立记录划分、生成前冻结、请求payload/原始响应、全部34方法×2目标、事件IoU.3/.5、逐记录结果、边界、阶段、未知、成本与失败均保留。精确数值2文件及评分19文件重放见verification-v1，无新增远端调用。正式独立领域/现场效果仍未建立，完整研究goal保持active，不等待专家再开展授权研究。

![双目标F1，全部两遍保留](../artifacts/landslide-adaptive-review-2026-10-09/dual-reference-confirmation-report-v2/comparison.png)

## 复现入口与论文

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$env:PYTHONIOENCODING='utf-8'
# 原完成目录不能覆盖。离线重放需选择尚不存在的新目录，并保持冻结输入。
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/analyze_landslide_dual_reference_confirmation.py analyze --process-predictions artifacts/landslide-adaptive-review-2026-10-09/dual-reference-process-predictions-v1 --out artifacts/landslide-adaptive-review-2026-10-09/dual-reference-confirmation-analysis-user-replay
```

数据与视觉/数值复核：dual-reference-confirmation-data-v1；目标适配校准：ai-observable-calibration-v3和dual-reference-process-calibration-v1；实际推理：dual-reference-confirmation-comparison-v2/run；双目标评分：dual-reference-confirmation-analysis-v1；冻结复核、复现源码与所有失败目录按原名保留。完整中文初稿v31继续使用Markdown LaTeX公式。

本轮官方arXiv近期定向检索取得5条摘要，TimeEvo已读官方源码/方法的历史边界不扩大；其余主要为预测、预训练、遥感与制造数据生成，未确认直接GNSS近邻或运行新发布模型。检索XML/哈希/摘要与阅读笔记保存在literature-v1/dual-reference-search-v1，不将抽象中的发布分数移入本任务表。
"""
    (ROOT / "docs/landslide-dual-reference-confirmation-results.md").write_text(text, encoding="utf-8")
    write_json(OUT / "provenance.json", {"source_analysis_sha256": sha(ANALYSIS / "manifest.json"),
        "reference_agreement": agreement, "remote_audit": audit,
        "report_sha256": sha(ROOT / "docs/landslide-dual-reference-confirmation-results.md")})
    (OUT / "source.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    write_report()
