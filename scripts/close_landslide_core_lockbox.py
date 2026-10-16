"""Write a focused manuscript from completed frozen results and actual inspection."""

from __future__ import annotations

import csv
from pathlib import Path

from prepare_landslide_ai_reference_holdout import verify_manifest
from report_landslide_core_lockbox import full_table
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
DATA = BASE / "core-lockbox-data-v1"
RUN = BASE / "core-lockbox-confirmation-v1"
ANALYSIS = BASE / "core-lockbox-analysis-v1"
REPORT = BASE / "core-lockbox-report-v1"
OUT = BASE / "manuscript-v39"


def table_rows(target):
    with (ANALYSIS / target / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def selected_table(rows):
    numerical = [row for row in rows if not any(f"_r{rep}" in row["method"] for rep in (0, 1))]
    names = {f"direct_visual_r{rep}" for rep in (0, 1)}
    for rep in (0, 1):
        names.add(f"adaptive_no_feedback_r{rep}__conflict_scope")
        names.add(f"fixed_image_review_r{rep}__whole_target")
        names.add(f"same_count_fixed_early_r{rep}__conflict_scope")
    visual = [row for row in rows if row["method"] in names]
    return full_table(numerical + visual)


def ablation_table(primary, secondary):
    lines = [
        "| 固定机制/消融 | 过程F1两遍 | 作者目标F1两遍 | 过程弱R两遍 | 作者弱R两遍 |",
        "|---|---|---|---|---|",
    ]
    variants = [
        ("主纯图/冲突权限", "adaptive_no_feedback", "conflict_scope"),
        ("同K最早位置", "same_count_fixed_early", "conflict_scope"),
        ("同K最晚位置", "same_count_fixed_late", "conflict_scope"),
        ("主位置/整目标权限", "adaptive_no_feedback", "whole_target"),
        ("主位置/等日数居中权限", "adaptive_no_feedback", "matched_center_scope"),
        ("主位置/增加数值文字", "adaptive_numeric_feedback", "conflict_scope"),
    ]
    for label, stem, variant in variants:
        values = []
        for metric, table in (
            ("f1", primary),
            ("f1", secondary),
            ("weak_recall", primary),
            ("weak_recall", secondary),
        ):
            values.append(
                "/".join(
                    f"{float(table[f'{stem}_r{rep}__{variant}'][metric]):.4f}" for rep in (0, 1)
                )
            )
        lines.append("| " + " | ".join([label] + values) + " |")
    return "\n".join(lines)


def family_table():
    with (REPORT / "per-family.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    primary = {
        (row["family"], row["method"]): row for row in rows if row["target"] == "primary_process"
    }
    lines = [
        "| 过程家族（各4条） | 主F1两遍 | 直接F1两遍 | XGB84 F1 | 主FP两遍 | 直接FP两遍 | XGB84 FP |",
        "|---|---|---|---:|---|---|---:|",
    ]
    for family in sorted({key[0] for key in primary}):
        main = [primary[(family, f"adaptive_no_feedback_r{rep}__conflict_scope")] for rep in (0, 1)]
        direct = [primary[(family, f"direct_visual_r{rep}")] for rep in (0, 1)]
        numeric = primary[(family, "xgboost_84")]

        def f1(row):
            return f"{float(row['f1']):.4f}" if row["f1"] else "—"

        values = [
            family,
            "/".join(f1(row) for row in main),
            "/".join(f1(row) for row in direct),
            f1(numeric),
            "/".join(row["fp"] for row in main),
            "/".join(row["fp"] for row in direct),
            numeric["fp"],
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def close():
    for folder in (ANALYSIS, REPORT):
        verify_manifest(folder)
    if read_json(RUN / "state.json")["status"] != "predictions_complete":
        raise ValueError("Write final findings only after terminal inference")
    inspection = read_json(REPORT / "developer-inspection.json")
    if not inspection["images_actually_inspected"]:
        raise ValueError("Actual post-score visual inspection required")
    for name, expected in inspection["images_actually_inspected"].items():
        if sha(REPORT / name) != expected:
            raise ValueError("Inspected image changed: " + name)
    verify_manifest(REPORT, "delivery-manifest.json")
    verify_manifest(BASE / "core-lockbox-diagnostics-v1")
    primary_rows, secondary_rows = table_rows("primary_process"), table_rows("secondary_developer")
    primary = {row["method"]: row for row in primary_rows}
    secondary = {row["method"]: row for row in secondary_rows}
    decision = read_json(ANALYSIS / "decision.json")
    cost = read_json(REPORT / "cost.json")
    summary = read_json(DATA / "codex-review/compiled/summary.json")
    phase = "通过" if decision["passed"] else "未全部通过"

    def pair(table, stem, key="f1"):
        names = [f"{stem}_r{rep}__conflict_scope" for rep in (0, 1)]
        if stem == "direct_visual":
            names = [f"{stem}_r{rep}" for rep in (0, 1)]
        return "/".join(f"{float(table[name][key]):.4f}" for name in names)

    primary_main = pair(primary, "adaptive_no_feedback")
    secondary_main = pair(secondary, "adaptive_no_feedback")
    main_calls = [
        int(primary[f"adaptive_no_feedback_r{rep}__conflict_scope"]["requests"]) for rep in (0, 1)
    ]
    reductions = [100 * (1 - count / 72) for count in main_calls]
    known_reference = summary["activity_counts"]
    stage_count = summary["stage_evaluable_days"]
    manuscript_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    old = manuscript_path.read_text(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "manuscript-v38-original.md").write_bytes(manuscript_path.read_bytes())
    # Preserve the original method equations and semantics. The long development
    # history is retained verbatim in the archive and its individual result docs.
    core = old[old.index("## 3 任务定义") : old.index("## 5 实验设计")]
    bibliography = old[old.index("## 参考文献") : old.index("## 补充结果与完整数据")]
    errors = inspection["findings_markdown"]
    ablations = ablation_table(primary, secondary)
    changes = []
    with (ANALYSIS / "primary_process/corrections.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            if row["control"].startswith(
                ("direct_visual", "fixed_image_review", "same_count_fixed_early")
            ):
                changes.append(row)
    correction_table = (
        "| 对照 | 主方法 | F1差 | 净确定纠错日 | 受益记录 |\n|---|---|---:|---:|---:|\n"
    )
    correction_table += "\n".join(
        f"| {row['control']} | {row['treatment']} | {float(row['f1_difference']):+.4f} | {row['net_correct']} | {row['benefited_records']} |"
        for row in changes
    )
    related = """## 2 相关工作与技术差异

滑坡季节慢滑及GNSS相关误差提供任务背景[1,2]。PELT[7]检测变点，但一次仪器阶跃不等于持续运动；XGBoost[8]与长尺度原始规则是需要充分校准的强对照。

TAMA[3]已有正常图示例和多尺度观察，AnomaMind[4]已有定位、工具复核与有限修订，TSAD-Agents[5]讨论多模态异常判断；因此框架或再询问本身不是贡献。已核查TAMA官方多尺度源码：其所读路径由非空视觉初判触发；本方法可从视觉初判空区与原始持续运动证据的冲突提出观察需求。Refining[6]对传统检测器候选整段接受或拒绝，不能增加候选召回；本方法保留活动、静稳、未知三状态，局部回答可发现初判漏掉的运动。

后续定向核查还包括正常条件化[10,11]、图表与视觉可靠性[12—15]、数值到视觉表示的可训练桥接[16]、数值分段后视觉描述[17]和视觉硬判断的时间平滑/融合[18,19]。相关原文方法段落、作者公开代码与未运行范围均在来源报告中记录。本文不以多尺度、局部图、投票或时间解码本身主张首创，也不把任务适配分数冒充发布模型复现。技术差异集中于GNSS原始连续运动语义下、有限预算内的冲突位置选择；49日权限是固定实现约束，其独立收益须由消融验证。最近邻逐项来源记录见[最近邻源码核查](landslide-nearest-method-code-review.md)、[带符号证据来源与实验](landslide-signed-evidence-development-results.md)、[分段视觉来源与实验](landslide-partition-review-results.md)和[时间融合来源与实验](landslide-temporal-fusion-results.md)。
"""
    design = r"""## 5 实验设计

### 5.1 目标、划分与参考隔离

过程目标采用六个既有受控家族：静稳、仪器阶跃、弱恒速、A/S/D活动、再活动、强运动后弱尾。原噪声、尖峰和缺测保留，潜在速度及位移活动可追溯。准确的生成标签仅支持恢复已知设计运动；不能证明弱运动在现场可辨识，也不证明季节背景无运动。

原过程数值模型全部基于同分布24训练/12验证记录，并用原验证目标获得规则、PELT、XGBoost、42/84日长尺度规则和学习模型的合理调参机会。原robust/PELT各九候选，学习模型六候选，新信息规则九候选；按验证日F1加事件F1选参。42同支撑与84学习组使用相同有效支撑人口，区分信息尺度与更多可评价日期。最终确认继续使用适用的冻结权重，逐条重新计算预测；测试不重新拟合、不调整阈值。作者观测目标单独沿用其既有6训练/6验证标签匹配模型及阶段头，不用过程权重冒充公平对手。

最终确认在生成前固定seed2026101113、打乱seed2026101114和流168—191，生成case0169—0192共24条730日记录，六家族各四条。全部24条为主评价人口，不筛选、更换或按结果挑子集。首12个不透明编号在生成前指定作开发者观测参考，不能在其分数较好时替代完整人口。作者实际看24张原始全局/局部图和全部30日原始中位数块，先冻结活动，再看12张31/61/91日辅助图及候选速率核查阶段；该批具体预测和生成标签均未用于决定参考。作者知道历史方法结果、历史过程标签和通用生成设计，故不是独立专家或现场金标准。原始缺测、未知边界和未知阶段完整保留。

原主方法、提示、两遍模型、预算及继续条件在输入存在前固定。两遍均须比直接视觉、完整固定两窗和同次数最早固定位置取得更高日F1及正净确定纠错；对直接每遍至少两记录受益，两遍平均弱召回不低于同次数早位置，调用少于完整固定。九强数值、事件、边界、阶段及所有家族全部报告，不要求所有指标胜全部数值。条件未全部通过时不换种子、提示、重复或主配置救分；已停止扩展不借此复活。

### 5.2 对比与消融

模型为qwen3.8-flash，temperature=0.1、top_p=0.3、max_tokens=2048，请求关闭thinking，严格JSON，0重试；服务侧模型版本不能由客户端固定，两遍用于记录波动。所有视觉组共享同遍全局初判及可共享局部回答。直接视觉仅一次全局观察；固定复核在两个预定等间距位置观察；主方法选至多两个冲突窗。对每条记录、每遍的主选择次数 $K\in\{0,1,2\}$，同次数最早/最晚固定位置只取既有固定位置的前/后 $K$ 个。它们是隔离位置效应的消融，不是另外一个无需全局或数值选次数的部署算法。整目标、49日冲突范围和等可改写日数居中范围共用回答；数值文字组只改变反馈内容。完整过程比较共37配置，观测目标34配置，三种额外长尺度过程控制不冒充观测目标已校准模型。

最近邻TAMA GNSS适配在此前case0121—0132上沿用既有固定适配、共同Qwen和阶段头，所有实际正常描述及局部回答计入成本。它是同任务工作流比较，未运行作者的另一模型，也不以发布分数替代本地评分。

### 5.3 指标与统计单位

活动Precision、Recall、F1在参考已知、实际有观测的日期计算。预测未知在真实活动处计漏检，未知在静稳处不获正确阴性信用；未知参考不当静稳，其预测阳性、静稳及弃判数量另报。

$$
P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},\qquad F_1=\frac{2TP}{2TP+FP+FN}.
$$

过程弱日由预先定义的设计属性确定；观测弱内部在预测前由作者固定，不从生成标签或成功案例反推。弱阳性分母为空时记未定义，不补零或制造阳性。事件采用一对一匹配，分别报告IoU阈值0.3与0.5，不使用point adjustment。

$$
\operatorname{IoU}(I,J)=\frac{|I\cap J|}{|I\cup J|}.
$$

边界误差在对应匹配事件上统计起/止MAE；作者参考仅为确认内核，不能当精确物理起止。不同方法匹配集合不同，MAE须和事件匹配率一起解读。A/S/D报告确认内部宏F1、覆盖及端到端阶段宏F1，不能删除未知输出。共同数值阶段头用于隔离活动机制，不支撑新增视觉阶段能力。

确定纠错比较同时报告新增正确、丢失正确、净日数及受益记录。净正确并不等于新增TP减新增FP，因其还包括未知转换。配对bootstrap以整记录为抽样单位、2000次，仅为描述性区间；相关日期、同背景变体和两遍服务回答不当独立统计样本，不以区间做新的方法选择。

成本区分单方法部署调用/token与共享全消融的独特请求，不把多个共享组相加作实际总成本。报告实际推理墙钟、失败和重试；数值控制0API不等于0CPU，未分别计时的训练、渲染及特征成本不编造。
"""
    history = """### 6.1 此前冻结证据与已知负结果

| 评价批次 | 主方法F1两遍 | 直接视觉F1两遍 | 同次数早位置F1两遍 | 强数值F1 | 解释 |
|---|---|---|---|---|---|
| 新受控0109—0120 | .8163/.7581 | .7413/.7315 | 原批未登记此对照 | XGB84 .9043 | 对直接/固定有收益，弱恒速家族两遍低于直接 |
| 位置确认0121—0132 | .8474/.8794 | .7200/.7708 | .7775/.7842 | XGB84 .9155 | 同次数位置收益两遍重现，数值日F1更高 |
| 作者参考0133—0144 | .8964/.9029 | .8287/.9149 | 见完整结果 | XGB .8956 | 第二遍低于直接，非稳定普遍纠错 |
| P227半合成测试24条 | .5829/.5623 | .6722/.6330 | 原批未登记此对照 | XGB84 .7261 | 弱召回上升但背景误报大增，整体F1下降 |

位置确认0121—0132相对同次数早位置净确定纠错+318/+297；弱召回.6906/.7600对.6419/.6054；两遍登记位置条件通过。第一遍主事件F1 .6667低于TAMA适配.7407，主日F1 .8474/.8794高于适配.7741/.7708，但主FP201/233多于适配94/50，不能称无损可靠性改善。

P227由官方三分量[9]构建十个不相交730日残差背景块，按背景隔离24训练/12验证/24测试并重新校准所有强基线。只有四个测试背景，24变体不是24独立站点。其参考是已知注入，背景真实运动未知；主弱召回.8083/.7321高于直接.6178/.3910，但FP5383/5398高于2648/2269，不能用弱日收益替代整体准确性。这是保留的迁移负证据。

后续作者观测确认仍出现重复间退化。已停止的顺序追加、带符号事实卡、PELT分段视觉与连续时间融合均未通过各自登记条件；时间融合在数值支撑日的视觉净增量仅4/6日，大部分表面净收益来自填补数值弃判。本文不将它们合并进主方法，也不把失败模块数量视为创新强度。完整实验和停止原因保留于历史初稿及独立结果文档。
"""
    abstract = f"""# GNSS原始运动证据驱动的有限视觉复核：受控收益与识别边界

> 完整中文初稿v39，2026-10-11。核心方法、完整24条冻结确认及预指定12条视觉/数值参考集中呈现；历史v38原字节完整存档。公式使用Markdown LaTeX。本稿未建立现场准确率或独立专家参考，不是投稿定稿。

## 摘要

长期GNSS累计位移中的弱运动可能被历史强运动压缩，高位平台又容易被视觉模型误报为持续活动。本文提出原始持续运动证据驱动的有限视觉复核：比较三个不相交原始位移中位数块的连续同向变化，以其与全局视觉初判的冲突选择至多两个局部窗，并在固定冲突邻域内允许修订。原始数值选择观察需求，视觉回答提供活动区间，缺测、未知和活动内部A/S/D门控保持一致。统一比较直接视觉、固定观察、同次数位置及数值文字消融，以及规则、PELT和强XGBoost。此前冻结位置确认中日F1 .8474/.8794高于同次数早位置.7775/.7842，但XGB84达.9155，半合成背景迁移存在负结果。最终生成前固定的新24条完整确认中，主F1 {primary_main}，登记条件{phase}：第一遍相对完整固定复核净减少23个确定正确判断。另有两次全局字段失败，收益包含失败恢复；剔除对应失败记录的描述性分析中，第一遍相对直接视觉F1下降0.0078、净减少57日。预指定12条作者视觉与原始数值辅助参考上F1 {secondary_main}。作者参考先于预测及具体生成标签冻结，但复核者已知历史研究，不能当独立金标准。结果量化有限观察的位置收益、误报/漏检和调用代价，不主张通用Agent增益、精确物理边界、原生视觉阶段创新或现场泛化。

**关键词：** GNSS；持续位移；视觉模型；自适应观察；有限复核；弱运动

## Abstract

Weak sustained GNSS motion may be hidden by larger historical displacement, while elevated stationary plateaus can be misclassified as activity. We propose bounded visual review selected by conflict between a global visual prediction and direction-consistent changes across three disjoint raw displacement median blocks. At most two local views are acquired, with revision limited to a fixed evidence neighborhood. Numerical observations select review demand; visual responses supply activity intervals. Missing observations, abstentions and activity-gated A/S/D labels are preserved. Shared-task comparisons include direct vision, fixed and same-count placement controls, numerical-text ablation, rules, PELT and strong XGBoost. Earlier placement confirmation yields F1 .8474/.8794 versus .7775/.7842 for matched-count early placement, while XGBoost84 reaches .9155 and observed-background transfer has negative results. A final preregistered complete 24-record trial yields main F1 {primary_main}; the full continuation gate is {"passed" if decision["passed"] else "not fully passed"}. The first repeat loses 23 definite correct days relative to complete fixed review. Two global output-contract failures contribute to recovery gains; excluding the failed case descriptively gives a first-repeat F1 change of -0.0078 and a net loss of 57 correct days versus direct vision. A preselected 12-record developer visual/numerical reference yields F1 {secondary_main}. This reference is frozen before specific predictions and generation labels but is not independent expert ground truth. We report recognition and computation tradeoffs without claiming field accuracy, precise physical boundaries or novel native visual stage reasoning.

## 1 引言

累计位移有多大与当前是否仍在持续变化是两个问题。长期滑坡可出现间歇活动和季节慢滑[1]，GNSS序列还含相关误差、尖峰和缺测[2]。全局视觉能读整体形态，但长期图的尺度可能遮蔽弱尾；局部自动缩放又可能放大噪声。一次阶跃后的累计偏移与持续运动也不能混为一谈。

本文研究有限观察预算下的具体识别机制：原始连续运动证据与全局初判冲突，是否能选择更有纠错价值的观察位置？贡献包括该GNSS证据选择实现、同次数位置与反馈/权限消融、冻结新记录确认及强数值比较。权限模块的独立增益和现场适用性不预设成立；完整报告失败、未知和成本，使有效场景与识别边界可以被复查。

"""
    results = f"""## 6 实验结果

{history}

### 6.2 最终完整24条过程恢复确认

主方法日F1 {primary_main}，直接视觉{pair(primary, "direct_visual")}，同次数早位置{pair(primary, "same_count_fixed_early")}；完整预登记条件{phase}。判定只针对本次不变核心和完整过程人口，不能覆盖旧观测目标退化，也不重新确认旧权限的严格独立收益。唯一未通过项是第一遍相对完整固定复核的净确定纠错为−23日，尽管F1更高。全部37配置、六家族各四条、两遍与强数值保留，不改主组或挑较好重复。

{selected_table(primary_rows)}

{correction_table}

### 6.3 预指定12条作者观测目标

8760日历位置包含确认活动{known_reference["1"]}日、确定静稳{known_reference["0"]}日及未知/缺测{known_reference["-1"]}日，其中原始缺测{summary["missing_days"]}日；11个确认活动内部，确定阶段{stage_count}日。原始图、块中位数、活动先冻结、辅助图与阶段核查的顺序可追溯。历史编译器的reviewer.name显示旧日期2026-10-10且prior_generation_labels_seen固定为false；实际复核日期2026-10-11及已知历史过程标签的暴露由本次决策、交付manifest与actual-review-provenance.json记录，不将旧字段当独立性证明。

主F1 {secondary_main}，直接{pair(secondary, "direct_visual")}，同次数早位置{pair(secondary, "same_count_fixed_early")}。此辅助目标有自己的六个冻结数值基线和共同阶段头，完整34配置保留。未知参考预测另计，确认内核边界不是物理真值；该12条不能替换完整24条的登记判定。

{selected_table(secondary_rows)}

### 6.4 消融、家族与实际错误复核

同次数早/晚位置、纯视觉、去数值文字、整目标、冲突范围及等日数居中范围均来自冻结回答。数值文字是内容消融，数值选择观察与数值作为答案不可混称。家族表及全量逐日结果见完整报告；每家族只有四个过程记录，观测子集的家族数量随机且未人为均衡，均作描述而非新选择标准。

{ablations}

{family_table()}

{errors}

![最终完整确认与预指定作者观测子集](../artifacts/landslide-adaptive-review-2026-10-09/core-lockbox-report-v1/comparison.png)

### 6.5 阶段与计算成本

两参考的活动内部及端到端阶段指标见主表，A/S/D逐类P/R/F1、支持数和弃判覆盖见results.json。阶段头按目标匹配训练；同一目标的视觉组共享它，活动改进传到门控后的阶段输出不构成新增视觉阶段内容能力。

完整24条单遍主部署{main_calls[0]}/{main_calls[1]}调用，相对固定72/72少{reductions[0]:.1f}%/{reductions[1]:.1f}%；直接24/24调用，主局部复核增加请求成本。全消融实际{cost["requests_started"]}启动/{cost["responses"]}回执，{cost["reported_tokens"]}报告token，推理墙钟{cost["inference_wall_seconds"]:.3f}秒，任务失败{cost["task_failures"]}，无重试。共享回执不在实际总量重复累计；报告token不等于账单金额或能耗。数值0API，历史训练及各方法完整CPU未分别计时，不报零成本。
"""
    conclusion = f"""## 7 讨论与结论

有限复核的收益应由同次数位置对照证明，而非由调用次数、提示数量或工具框架推断。此前两批受控证据显示部分位置收益，强数值日F1仍更高；观测目标存在重复波动，半合成背景误报也限制迁移。最终完整新批登记条件{phase}，应与这些历史正/负证据一同解读。

本轮两次全局字段失败产生全未知初判，局部恢复的分数包含输出契约可靠性成分。排除失败的分析只描述其影响，不改变完整人口与NOT_CONFIRMED判定；第一遍正常输出记录上的净纠错未稳定为正，作者目标第二遍对固定复核也有净损失。因此不把全部F1差解释为普通视觉语义漏检的纠正。

原始正证据能提出初判空区的观察需求，却不能凭同一序列区分所有真实缓变、仪器漂移和季节背景。局部回答也可能漏掉弱运动或延长平台。未知不是确定静稳，保守边界不能当精确起止；提高覆盖或恢复数值弃判不等于视觉异常检出增量。权限约束的独立收益尚无稳定证明，本文不将它升级为单独贡献。

本研究形成一个机制清楚、次数有限且可复现的GNSS视觉复核方法，并量化相对视觉控制的场景收益、强数值优势及代价。结论范围为设计运动恢复及非独立作者观测敏感性，尚未完成真实现场独立参考验证。已停止机制不纳入方法；不据本次结果继续抽新seed、改提示或筛选子集救分。

## 8 复现与AI使用说明

本轮包含24新受控记录，受控研究累计192条；60半合成单独计数，各批开发、选择和确认身份不混合。累计{3429 + cost["requests_started"]}启动/{3427 + cost["responses"]}回执、至少{14927951 + cost["reported_tokens"]}报告token；历史两次无回执的未知成本不补零。原始观察、模型、协议、源码、作者参考时序、请求payload/原始回复、失败、37/34配置、逐记录/逐日/家族及事件边界完整保留。

开发者Codex进行了代码开发、图像与原始数值辅助复核、评分和写作。已知历史结果、历史过程标签及通用生成设计的暴露明确记录；新具体生成标签只在两遍预测全部冻定后开放评分。标签不是独立专家金标准；不声称模型已独立证实自己的正确性。

数据与复核为core-lockbox-data-v1，真实推理core-lockbox-confirmation-v1，评分core-lockbox-analysis-v1，确定性重放core-lockbox-replay-v1，图表与全量导出core-lockbox-report-v1。评分在预测前修正了12条辅助目标的请求人口计数和JSON布尔序列化，数学指标、完整24条gate和所有方法不变；原生成前源码快照及修正绑定均保留。冻结评分的数值配对筛选误用_r子串（规则/raw名称也含它），不影响九/六数值指标或登记判定。core-lockbox-diagnostics-v1以明确_r0/_r1筛选补全30个数值配对，并复现12行字段失败敏感性；与报告追加CSV字节一致。不修改冻结评分源码或重调模型。没有重新调用服务来获得更好结果。

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$env:PYTHONIOENCODING='utf-8'
# 离线重放选择尚不存在的新输出目录；不能重启已完成推理目录。
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/analyze_landslide_core_lockbox.py --data artifacts/landslide-adaptive-review-2026-10-09/core-lockbox-data-v1 --run artifacts/landslide-adaptive-review-2026-10-09/core-lockbox-confirmation-v1 --out artifacts/landslide-adaptive-review-2026-10-09/core-lockbox-user-replay
```

完整结果、所有消融与实际视觉错误见[本轮报告](landslide-core-lockbox-results.md)。此前所有公式、表、失败模块及阅读范围保存在[历史完整初稿v38](../artifacts/landslide-adaptive-review-2026-10-09/manuscript-v39/manuscript-v38-original.md)，对应结果报告继续有效。正文不再按实验尝试逐章累加；任何旧结果均未删除或追改。全部正文公式使用LaTeX行内或独立块并进行KaTeX渲染核验。

"""
    text = (
        abstract
        + related
        + "\n"
        + core
        + design
        + "\n"
        + results
        + "\n"
        + conclusion
        + bibliography
    )
    manuscript_path.write_text(text, encoding="utf-8")
    (OUT / "manuscript-v39.md").write_text(text, encoding="utf-8")
    report_text = "# 冻结核心的完整24条确认与开发者视觉辅助参考\n\n"
    report_text += f"2026-10-11。方法未改；全部登记条件{phase}，具体条件见decision.json。\n\n"
    report_text += design + "\n" + results + "\n" + conclusion
    report_text += "\n## 全量过程37配置\n\n" + full_table(primary_rows)
    report_text += "\n\n## 全量开发者34配置\n\n" + full_table(secondary_rows) + "\n"
    (ROOT / "docs/landslide-core-lockbox-results.md").write_text(report_text, encoding="utf-8")
    update = f"""## 2026-10-11：不变核心24条完整确认、实际视觉参考与聚焦初稿v39

主过程F1 {primary_main}，原始视觉/数值复核的预指定12条辅助F1 {secondary_main}；本次完整登记条件{phase}，唯一失败为第一遍对完整固定净−23日。两全局字段失败使收益含失败恢复；剔除对应失败记录的过程第一遍对直接F1差−.0078、净−57日（仅描述性，不替换完整判定）。新24条不换seed/重复/子集，原方法、强数值及同次数/反馈/权限消融全部保留；不复活已停止扩展。

实际查看24原始图、12辅助图及原始块/速率，先冻活动再判阶段；8760逐日标签、11确认内部及{stage_count}确定阶段日通过共享编译重放与图/CSV/JSON缺测掩码验证。作者有历史暴露，非独立专家。新批过程标签只在预测全冻定后评分开放。

新{cost["requests_started"]}启动/{cost["responses"]}回执、{cost["reported_tokens"]}报告token，失败{cost["task_failures"]}、0重试。完整37/34配置、所有家族/损益/逐日结果及字节精确离线重放保留。受控192/半合成60分开计数。v38原字节存档；v39集中于核心方法与证据，公式继续LaTeX。详见[完整结果](landslide-core-lockbox-results.md)。

研究goal仍active：本轮确认不能证明现场准确率、独立专家参考、稳定权限增量或通用视觉阶段创新。下方为历史记录。

"""
    for name in (
        "project-status.md",
        "landslide-adaptive-review-method.md",
        "landslide-submission-roadmap.md",
    ):
        path = ROOT / "docs" / name
        body = path.read_text(encoding="utf-8")
        first, rest = body.split("\n", 1)
        path.write_text(first + "\n\n" + update + rest.lstrip("\n"), encoding="utf-8")
    write_json(
        OUT / "content-check.json",
        {
            "old_manuscript_sha256": sha(OUT / "manuscript-v38-original.md"),
            "new_manuscript_sha256": sha(manuscript_path),
            "core_method_source_text_retained": True,
            "old_history_preserved_verbatim_in_archive": True,
            "full_primary_records": 24,
            "fixed_developer_subset": 12,
            "registered_gate_passed": decision["passed"],
            "formal_field_accuracy_verified": False,
        },
    )
    print("Wrote focused v39 manuscript from terminal experiment and actual inspection")


if __name__ == "__main__":
    close()
