"""Preserve v53 and write a focused manuscript from the registered new trial."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from probe_landslide_endpoint_fit import BASE, ROOT, csv_rows, export
from run_landslide_window_optimization import read_json
from write_landslide_claim_review_results import DOCUMENTS

from gnss_sim.artifacts import sha

NAMES = (*DOCUMENTS, "landslide-paired-persistence-results")
ANALYSIS = BASE / "paired-persistence-confirmation-analysis-v1"
REVIEW = BASE / "paired-persistence-confirmation-error-review-v1"


def verify_manifest(folder):
    for name, expected in read_json(folder / "manifest.json")["files"].items():
        if sha(folder / name) != expected:
            raise ValueError("Result report changed: " + name)


def preservation(archive):
    manifest = read_json(archive / "v53-delivery-manifest-original.json")
    mapping = []
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if name in {"docs/" + item + ".md" for item in NAMES}:
            path = archive / (Path(name).stem + "-v53-original.md")
        if sha(path) != expected:
            raise ValueError("Historical bytes changed: " + name)
        mapping.append({"original_path": name, "preserved_path": path.relative_to(ROOT).as_posix(), "sha256": expected})
    return mapping


def table(rows, names, columns):
    lookup = {row["method"]: row for row in rows}
    text = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for name in names:
        row = lookup[name]
        cells = []
        for key in columns:
            value = row[key]
            if value is None:
                cells.append("—")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        text.append("| " + " | ".join(cells) + " |")
    return "\n".join(text)


def seed_sensitivity_section():
    report = BASE / "paired-persistence-training-seed-report-v1"
    for name, expected in read_json(report / "report-freeze.json").items():
        if sha(report / name) != expected:
            raise ValueError("Seed sensitivity report changed: " + name)
    contrasts = read_json(report / "contrasts.json")
    primary = contrasts["primary_process__visual_numeric_augmentation"]
    secondary = contrasts["secondary_observation__visual_numeric_augmentation"]
    rows = []
    for values in csv_rows(report / "comparison.csv"):
        row = {"method": values["seed"]}
        row.update({key: float(value) for key, value in values.items() if key != "seed"})
        rows.append(row)
    columns = ("method", "main_f1", "augmentation_f1", "shuffled_f1", "numeric_pair_f1",
               "main_weak_recall", "main_event_f1", "secondary_main_f1", "secondary_augmentation_f1")
    return ("\n\n### 5.5 冻结参数的训练随机性敏感性\n\n"
            "在正式确认后登记五个连续训练种子20261020—20261024，对七个神经臂复用原训练中心、"
            "冻结视觉编码、原选宽度和阈值；不重新校准、不替换主模型、不重抽测试记录。"
            "这是已评分记录上的事后稳健性检查，不是新的独立确认。原20261020种子的七组权重、"
            "概率与预测精确复现，全部35头的保存权重重放也精确一致。\n\n"
            + table(rows, [row["method"] for row in rows], columns)
            + f"\n\n主过程F1相对同增广五个种子均提高，差范围"
            f"[{primary['minimum']:+.6f}, {primary['maximum']:+.6f}]；对普通头、打乱配对和纯数值配对"
            "的日F1差也全部为正。主事件F1相对增广四个种子为正、一个持平，说明事件收益仍存在训练波动。"
            f"观测参考上主对增广五种子全部下降，差范围[{secondary['minimum']:+.6f}, {secondary['maximum']:+.6f}]。"
            "该负结果具有同样一致性，不能用种子稳定性将适用范围扩展到观测参考。"
            "五个拟合共享同一24条／12条数据，不能视为五个独立测试集；种子差值范围不是置信区间，"
            "也不能消除原记录重采样区间跨零或历史方法选择偏差。\n\n"
            "新35次小头训练合计17.448秒，仅报告本次实际训练计时；没有重算视觉编码或新增API，"
            "也不据此补写未知的原校准成本。全部七臂／两参考的70行汇总和逐记录结果保留于"
            "[随机种子结果](../artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-training-seed-report-v1/report.md)。\n\n"
            "![固定参数种子敏感性及强数值基线](../artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-training-seed-report-v1/seed-sensitivity.png)\n")


def write():
    verify_manifest(ANALYSIS)
    replay = BASE / "paired-persistence-confirmation-replay-v1"
    verify_manifest(replay)
    replay_files = read_json(replay / "manifest.json")["files"]
    if read_json(ANALYSIS / "manifest.json")["files"] != replay_files:
        raise ValueError("Two complete confirmation reports differ")
    for name, expected in read_json(REVIEW / "review-freeze.json")["files"].items():
        if sha(REVIEW / name) != expected:
            raise ValueError("Completed post-score review changed: " + name)
    if read_json(REVIEW / "review-completion.json")["actual_image_review"] != "completed":
        raise ValueError("Actually inspect the post-score diagnostic images before writing")
    archive = BASE / "manuscript-v54"
    archive.mkdir(exist_ok=False)
    for name in NAMES:
        (archive / (name + "-v53-original.md")).write_bytes((ROOT / "docs" / (name + ".md")).read_bytes())
    (archive / "v53-delivery-manifest-original.json").write_bytes((BASE / "manuscript-v53/delivery-manifest.json").read_bytes())
    export(archive / "v53-preservation-before.json", preservation(archive))
    old = (archive / (DOCUMENTS[0] + "-v53-original.md")).read_text(encoding="utf-8")
    primary = read_json(ANALYSIS / "primary_process/summary.json")
    secondary = read_json(ANALYSIS / "secondary_observation/summary.json")
    development = read_json(BASE / "paired-persistence-development-analysis-v1/summary.json")
    calibration = read_json(BASE / "paired-persistence-calibration-v1/validation-summary.json")
    lookup = {row["method"]: row for row in primary}
    main = lookup["visual_numeric_pair"]
    augment = lookup["visual_numeric_augmentation"]
    plain = lookup["visual_numeric_plain"]
    numeric = lookup["numeric_pair"]
    expert = lookup["xgb_expert24"]
    decision = read_json(ANALYSIS / "decision.json")
    audit = read_json(ANALYSIS / "audit.json")
    bootstrap = read_json(ANALYSIS / "primary_process/descriptive-bootstrap.json")
    process_results = read_json(ANALYSIS / "primary_process/results.json")
    observation_results = read_json(ANALYSIS / "secondary_observation/results.json")
    correction_rows = read_json(ANALYSIS / "primary_process/corrections.json")
    net = {name: sum(row["net"] for row in correction_rows if row["control"] == name) for name in lookup}
    costs = read_json(BASE / "paired-persistence-confirmation-inference-v1/inference-cost.json")
    ai = {row["method"]: row for row in secondary}
    summary = (f"新24条过程主F1 {main['f1']:.6f}、弱R {main['weak_recall']:.6f}、事件F1(.3) {main['event_f1_iou03']:.6f}；"
               f"对plain/增广/纯数值配对的F1差{main['f1']-plain['f1']:+.6f}/"
               f"{main['f1']-augment['f1']:+.6f}/{main['f1']-numeric['f1']:+.6f}，"
               f"对plain/增广净确定纠错{net['visual_numeric_plain']:+d}/{net['visual_numeric_augmentation']:+d}。"
               f"预登记{decision['passed']}/{decision['total']}条件，**{decision['decision']}**。")
    verdict = ("本次确认支持该冻结配对方法在指定受控分布内的机制收益；仍须限定事件、边界、代价和参考性质。"
               if decision["passed"] == decision["total"] else
               "本次确认未复现全部预定机制条件；此固定候选停止升级，不换种子、阈值或最优消融救分。")
    names = ["visual_numeric_pair", "visual_numeric_augmentation", "visual_numeric_plain",
             "visual_numeric_shuffled_pairs", "visual_pair_no_numeric", "numeric_pair", "numeric_augmentation",
             "xgb_raw18", "xgb_expert24", "xgb_chart384"]
    primary_names = names + ["robust_rule", "pelt_rule", "xgboost_84", "direct_visual_r0", "direct_visual_r1",
        "fixed_image_review_r0__whole_target", "fixed_image_review_r1__whole_target",
        "adaptive_no_feedback_r0__conflict_scope", "adaptive_no_feedback_r1__conflict_scope",
        "adaptive_numeric_feedback_r0__conflict_scope", "adaptive_numeric_feedback_r1__conflict_scope"]
    daily_columns = ("method", "precision", "recall", "f1", "weak_recall", "event_f1_iou03", "event_f1_iou05")
    boundary_columns = ("method", "onset_core_mae_days", "offset_core_mae_days", "stage_internal_macro_f1",
                        "stage_end_to_end_macro_f1", "coverage", "requests", "reported_tokens")
    pairs = "\n".join(f"- 相对{name}：F1差{value['difference']:+.6f}，记录重采样区间"
                       f"{value['exploratory_95_percentile_interval']}。" for name, value in bootstrap.items())
    family_lookup = {(row["family"], row["method"]): row
                     for row in csv_rows(ANALYSIS / "primary-family-summary.csv")}
    family_rows = []
    for family in read_json(BASE / "paired-persistence-confirmation-data-v1/protocol.json")["generation"]["families"]:
        row = {"method": family, "records": 4}
        for name in ("visual_numeric_pair", "visual_numeric_augmentation", "numeric_pair", "xgboost_84"):
            value = family_lookup[family, name]["f1"]
            row[name] = float(value) if value else None
        for name, key in (("visual_numeric_pair", "main_fp"), ("visual_numeric_augmentation", "augmentation_fp")):
            row[key] = int(family_lookup[family, name]["fp"])
        family_rows.append(row)
    family_table = table(family_rows, [row["method"] for row in family_rows],
                         ("method", "records", "visual_numeric_pair", "visual_numeric_augmentation",
                          "numeric_pair", "xgboost_84", "main_fp", "augmentation_fp"))
    reference_lines = []
    for new_number, old_number in enumerate((1, 2, 3, 4, 7, 8, 41, 42, 43), 1):
        match = re.search(rf"^\[{old_number}\] .*$", old, re.MULTILINE)
        if match is None:
            raise ValueError("Verified reference absent: " + str(old_number))
        reference_lines.append(re.sub(r"^\[\d+\]", f"[{new_number}]", match.group()))
    mathematical_method = r"""## 3 方法

### 3.1 任务与可评价范围

输入为原始日历上的三维位移 $x_t\in\mathbb R^3$ 和观测可用性 $o_t$。活动标签 $a_t\in\{-1,0,1\}$ 分别表示未知、静稳和持续运动。累计位移保持高位并不延长活动；单次设备偏移本身也不构成持续活动。先确认活动，再在其内部区分加速A、近匀速S和减速D。阶段不明确时保持未知。

$$
o_t=0\ \Longrightarrow\ \widehat a_t=-1,\ \widehat s_t=-1,
\qquad \widehat s_t\in\{A,S,D\}\ \Longrightarrow\ o_t=1,\ \widehat a_t=1.
$$

本文是使用前后观测的离线回溯识别。中心图需要约前后91日语境，不能将结果当成仅使用过去数据的实时预警。

### 3.2 同净位移的持续性训练对

只有受控训练记录允许分解 $x_t=m_t+e_t$；$m_t$ 为已知训练运动分量，$e_t$ 为观测误差与干扰。中心 $c$ 的端点为 $u=c-91$、$v=c+91$。负例用一次偏移代替持续运动，保留原观测误差、缺测和首末位移：

$$
\widetilde x_t=x_t-m_t+m_u+\mathbf1_{t\ge k}(m_v-m_u),
\qquad k\in\{c-42,c,c+42\}.
$$

实现直接复制端点原观测，使 $\widetilde x_u=x_u$、$\widetilde x_v=x_v$ 精确成立。中心附近14日需至少10个原观测日且原标签均为活动；每28日固定抽样，负例阶跃位置循环指定。137对来自16个原训练parent，其中48个弱中心，不是137条独立记录。首末变化相同而持续性不同是训练干预的识别目标；每条评估记录仍只输入观测，不能读取潜在分量。噪声代理在负例上重新估计，所以并不声称数值标准化尺度完全相同。

### 3.3 冻结预训练视觉编码与活动头

每个中心保留61个三日槽，槽中至少两天三轴观测完整才绘制中位数。未观测槽断线并显示缺测掩码；三个NEU面板保留物理分量、相对日历和中心位置。图的纵向尺度不小于六倍噪声代理，先去平移偏置再绘制256像素图。冻结MobileCLIP2-S0/dfndr2b图文预训练模型的视觉分支[8]，严格使用其mean0/std1与bilinear/shortest预处理，取得512维L2编码。

$$
h_c=\frac{E_{\mathrm{frozen}}(I_c)}{\|E_{\mathrm{frozen}}(I_c)\|_2},
\qquad z_c=f_\theta([h_c,n_c]),\qquad p_c=\sigma(z_c).
$$

$n_c$ 包含18项已有raw84数值和18个缺值指示，按原抽样训练统计归一化。活动头是一个16或32单元ReLU隐层和一个输出单元。视觉分支不微调，不调用语言生成或Agent工具；本文候选属于预训练视觉与数值融合识别，不能称新增语言推理能力。

支持掩码要求中心原观测可用且至少一种已有持续速率可估；支持外未知。活动由校准阈值决定。内部阶段沿用同一数值XGBoost阶段头及已有支撑门控，没有训练原生视觉A/S/D头。

### 3.4 监督与配对约束

原中心与新增负中心进行带正类权重的BCE训练；每个监督minibatch另使用全部137训练对的平均排序项：

$$
\mathcal L_{\mathrm{pair}}=
\frac1{|\mathcal P|}\sum_{(c^+,c^-)\in\mathcal P}
\log\bigl(1+\exp(1-z_{c^+}+z_{c^-})\bigr),
\qquad \mathcal L=\mathcal L_{\mathrm{BCE}}+\mathcal L_{\mathrm{pair}}.
$$

固定12个epoch、batch128、Adam学习率0.001、weight decay0.0001、seed20261020、CPU两线程。验证只选择隐宽与活动阈值，不生成配对训练数据。Anlu[7]针对参考条件标签的干预不同；CAPS[9]作者源码已有matched/shuffled监督和BCE加softplus排序。因此本文不把配对、排序项或打乱消融称首创。待检验技术差异仅限于GNSS同误差同净位移的持续运动／单次偏移干预及预训练视觉支撑。

## 4 实验设计与评价

### 4.1 数据、开发和新记录

所有学习方法使用原24训练／12验证受控划分。原抽样训练中心7656，增广中心137；神经臂使用原stride2抽样，强XGB使用全部原已知支撑训练日并得到相同137负中心。各学习臂都有两个宽度或深度与19个阈值，共38次校准机会，按原日F1加事件F1目标选择，不能复用不适用的旧模型充当同增广基线。

case_0169—0192共24条已暴露记录用于完整开发。随后一次登记seed2026101121、排列seed2026101122、独立流192—215，生成case_0193—0216共24条、每条730日，quiet、instrument_step、weak_constant、asd、reactivation、weak_after_strong六族各四条。整批不筛选、无替换、无重抽；全部权重、阈值、画法和判定条件在观测生成前冻结。新记录不参与方法选择；其过程参考说明设计运动恢复，并非现场分布或独立专家可观测真值。

另在生成前指定前12个匿名编号，由当前Codex实际查看24幅原始全局／局部图与全部30日原数值块。活动先冻结，再实际查看12幅辅助图及31／61／91日趋势后冻结内部阶段。新具体预测与过程标签未参与这次复核，但复核者知道历史标签和通用设计。未知边界、长缺口与不一致阶段均保留。该参考只作独立于本批预测的开发者观测敏感性，不能与完整24条过程主结果合并。图、JSON与CSV使用同一活动与缺测掩码。

### 4.2 基线与消融

七个学习臂为主配对、同数据增广去配对项、原样本普通视觉数值头、打乱配对、配对去数值、纯数值配对和纯数值增广。主配对、去配对项和打乱配对使用相同增广样本，检验额外排序／配对关系；普通头仅使用原样本。模态对照检验视觉相对已有数值的增量。不同臂按同等机会各自校准，单个差异不能分离所有模型选择因素。

同增广强XGB包括18项raw84、24项expert和384项图信息对照；后者获得183个槽位置和183个可用性再加18数值，避免把图提供的额外时序信息都归于视觉编码。已有数值规则、PELT[5]、XGBoost[6]及raw42/raw84控制保留原同分布训练／验证校准并在新数据重新预测。它们与同增广新XGB的身份分别报告。

普通VLM、固定两窗工作流、原有限自适应及数值文字对照沿用原qwen3.8-flash协议，完整24条各两遍、无重试。原自适应由三块原始位移持续证据与初判冲突选至多两个180日窗，49日邻域权限及同K位置对照由同一冻结回执重放。有限工具与多尺度观察已有近邻[3,4]，不作为本候选首创。全部47个主过程方法均评价；正文显示关键对照，完整表保留。

观测敏感性中，过程训练的活动／阶段头只作为目标迁移敏感性；另列已按作者观测目标六训练／六验证拟合的六个冻结数值控制。不称过程模型已经在新观测目标获得同等校准。

### 4.3 统一指标与一次判定

在参考可评价的原观测日计算活动Precision、Recall与F1。未知预测不能删掉后再算召回，已知正例上的弃判进入漏检；缺测仍不评分。单方法零预测正类时Precision未定义，导出保持空值而不补零。

$$
P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},\qquad
F_1=\frac{2TP}{2TP+FP+FN}.
$$

事件按原episode及共同匹配器计算IoU0.3和0.5下的P/R/F1；边界MAE仅针对匹配事件，未匹配事件单列，不能把条件边界误差较低理解为总体定位更好。继续报告活动内部和端到端A/S/D宏F1、逐类支持、弃判覆盖与未知参考上的确定预测。弱运动主子集使用生成前固定家族及过程weak属性，观测子集则在预测前手工冻结，二者分母不混用。

新记录沿用十项开发门槛：日F1不低于同增广和打乱配对；对普通头至少提高0.005；对同增广日或事件F1严格改善；弱R不低于普通头与增广；对纯数值配对满足日F1不低或事件改善且日损失不超0.01；对普通头与增广均净确定纠错为正且至少两条受益记录。门槛判断方法是否继续，不替代效应量或统计检验，也不要求所有指标优于全部数值基线。

按完整记录作配对bootstrap描述波动，日期和两遍VLM输出不能视为独立样本。开发中存在大量历史候选选择，未进行多重比较校正；本稿不报告确认性p值、不将bootstrap区间解释成证明显著。固定六族覆盖比例不是现场活动率，观测未知参考也具有选择限制。
"""
    text = ("# GNSS持续运动与单次偏移的配对视觉识别：受控对照、消融与识别边界\n\n"
            "> 完整中文初稿v54，2026-10-11。主文围绕本轮冻结候选与新24条实验重写；历史v53及五个可变文档原字节存档。公式使用Markdown LaTeX，所有结论限于实际实验及所述参考范围。\n\n"
            "## 摘要\n\n长期GNSS位移中的弱持续运动与单次偏移后平台具有不同时间结构，单凭净位移或大尺度图像易混淆。"
            "本文研究一种简单的训练干预：仅在受控训练记录中，将运动分量替换为同净位移单次偏移，保留原误差与缺测；"
            "冻结图文预训练模型的视觉分支，利用小活动头和数值输入学习持续性配对。"
            "统一比较普通视觉数值头、同数据增广、打乱配对、模态消融、同增广强XGB以及直接VLM和固定/自适应工作流。"
            + summary + f" 相对expert24的日F1差{main['f1']-expert['f1']:+.6f}，"
            f"但低于原XGBoost84的{lookup['xgboost_84']['f1']:.6f}。"
            "相对同增广Precision下降、边界MAE增大；观测参考上日F1也下降。"
            + verdict + "另有预指定12条由Codex实际看图及原数值辅助冻结的观测参考，属于开发者敏感性；"
            "不证明现场准确率或独立专家一致性。阶段沿用数值头，活动门控收益不能当新增视觉阶段推理。\n\n"
            "**关键词：** GNSS；持续位移；视觉编码；配对监督；弱运动；异常识别\n\n"
            "## Abstract\n\nSustained weak GNSS motion and a single offset followed by a plateau can have similar net displacement. "
            "We test a training intervention that replaces controlled training motion with an equal-displacement offset while retaining observational errors and missing dates. "
            "A frozen pretrained vision branch supplies image embeddings to a small activity head with numerical features. "
            "Comparisons separate augmentation, pair identity and modality effects and include strong numerical, direct VLM and fixed/bounded review baselines. "
            f"On one preregistered, unfiltered 24-record controlled batch, activity F1 is {main['f1']:.4f}, weak-motion recall {main['weak_recall']:.4f}, "
            f"and event F1 at IoU 0.3 {main['event_f1_iou03']:.4f}. The F1 difference from identical augmentation is {main['f1']-augment['f1']:+.4f}; "
            f"{decision['passed']} of {decision['total']} registered continuation conditions pass. "
            "Precision and matched boundary errors deteriorate relative to augmentation; daily F1 also decreases on the secondary observation target. "
            "A separate preselected 12-record visual/numerical developer reference is frozen before exposure to specific predictions and process labels. "
            "Historical label exposure is disclosed, so this secondary reference is not independent expert ground truth. "
            "We report event, boundary and computation tradeoffs without claiming field accuracy or novel visual stage reasoning.\n\n"
            "## 1 引言\n\n长期滑坡可呈间歇慢滑，GNSS观测又含相关误差、离群点和缺测[1,2]。"
            "异常识别应判断当前是否持续运动，而不能仅判断累计位移是否偏离原点。全局图可遮蔽弱尾，局部自动缩放也会放大噪声；"
            "重新观察不一定纠错，因此需要能隔离机制的消融与完整记录评价。\n\n"
            "本研究聚焦训练时可控制的歧义：首末位移与误差尽量相同，只改变期间的持续性。"
            "假设是配对约束能帮助预训练视觉与原数值在活动识别上取得额外收益；这一假设同时接受去配对、打乱配对与纯数值检验。"
            "方法贡献的成立范围由冻结新记录结果决定，不由Agent框架、提示数量或配对损失名称决定。\n\n"
            "## 2 相关研究与差异\n\nTAMA[3]与AnomaMind[4]已有图像示例、多尺度定位和有限工具复核，"
            "因此再观察的工作流不能单独构成新贡献。PELT[5]与XGBoost[6]是必须保留的强对照，"
            "位移变化点和持续活动的定义应一致。Anlu[7]通过参考条件改变异常判定，不能直接移植为固定GNSS活动标签。"
            "CAPS[9]已有配对、打乱对照与排序目标；本方法只提出针对持续运动／一次偏移混淆的GNSS训练干预。"
            "MobileCLIP2[8]提供冻结预训练视觉编码，本研究没有复现其预训练或引入新的语言推理架构。\n\n"
            + mathematical_method + "\n\n## 5 结果\n\n### 5.1 校准与已暴露完整开发\n\n"
            "12条验证用于选择参数，不是独立效果结论。主校准F1为0.906712，对普通头+0.012543、对同增广+0.007609；"
            "expert24为0.930296，数值基线更强。完整24条已暴露开发主F1为0.873946，对普通头+0.020087、"
            "对增广+0.008252；事件F1为0.626866，低于增广0.666667，主起止MAE为26.667/24.143日。"
            "校准8/8、开发10/10门槛通过后才冻结本次新记录。开发对增广F1差的描述性区间跨零，不能称显著。\n\n"
            + table(calibration, names, daily_columns) + "\n\n"
            + table(development, primary_names, daily_columns) + "\n\n### 5.2 完整24条未参与选择的新记录\n\n"
            + summary + "\n\n" + verdict + "\n\n" + table(primary, primary_names, daily_columns) + "\n\n"
            + table(primary, primary_names, boundary_columns) + "\n\n"
            f"主事件F1与同增广的差为{main['event_f1_iou03']-augment['event_f1_iou03']:+.6f}；"
            f"主Precision {main['precision']:.6f}低于增广{augment['precision']:.6f}，"
            f"主起止MAE {main['onset_core_mae_days']:.3f}/{main['offset_core_mae_days']:.3f}日，"
            f"增广为{augment['onset_core_mae_days']:.3f}/{augment['offset_core_mae_days']:.3f}日。"
            f"主匹配23/24事件、FP19个，增广20/24、FP18个；直接VLM第一遍事件F1 "
            f"{lookup['direct_visual_r0']['event_f1_iou03']:.6f}高于主方法。"
            "日指标、事件和边界分别解释，不按单项排名声称全面改善。所有47方法、逐记录、每家族四条、"
            "事件匹配及A/S/D支持详见完整结果文件。新十个头的冻结权重、概率与预测精确重放。\n\n"
            + pairs + "\n\n相对同增广，主日TP增加324、FP增加209，11条净受益、6条净受损、7条相同。"
            "主总F1不是逐记录F1的平均。下表各家族均4条，前四个方法列为家族日F1，末两列为误报日数。"
            "quiet没有正活动；零误报时F1未定义，记为—，不能将该族F1用于整体排名。\n\n"
            + family_table + "\n\n弱匀速与强后弱续动族相对增广改善；ASD与再活动族日F1下降。"
            "这说明改进主要适用于弱运动召回，不能声称消除平台误报或普遍提升明显活动。\n\n"
            "主内部A/S/D支持为A924、S2715、D931日，内部宏F1 "
            f"{main['stage_internal_macro_f1']:.6f}、端到端{main['stage_end_to_end_macro_f1']:.6f}；"
            "内部得分仅在已知真实活动阶段评价，端到端还惩罚静稳上的假阶段。共享阶段头意味着差异来自活动门控，"
            "没有单独证实阶段分类器的视觉改进。\n\n"
            "![完整24条比较](../artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-confirmation-analysis-v1/primary_process/comparison.png)\n\n"
            "### 5.3 预指定观测参考敏感性\n\n"
            f"12条原日8760，观测参考确认活动1349日、静稳6221日、未知1190日，含378缺测日；"
            f"可评价A/S/D共532日。主活动F1为{ai['visual_numeric_pair']['f1']:.6f}，"
            f"同增广为{ai['visual_numeric_augmentation']['f1']:.6f}。"
            f"主弱召回{ai['visual_numeric_pair']['weak_recall']:.6f}高于增广"
            f"{ai['visual_numeric_augmentation']['weak_recall']:.6f}，但FP210高于109、日F1差"
            f"{ai['visual_numeric_pair']['f1']-ai['visual_numeric_augmentation']['f1']:+.6f}。"
            "该负结果保留，不能以主过程门槛通过覆盖观测目标的恶化。"
            "原活动在辅助阶段复核前冻结，若长窗支撑不足或窗口冲突，只改变阶段的未知状态，不扩活动边界。"
            "过程头的该表分数是目标迁移敏感性，不能代替主过程评价或称同目标训练优势。\n\n"
            + table(secondary, names + ["observable_target__xgboost", "observable_target__xgboost_raw_persistence",
                                       "direct_visual_r0", "fixed_image_review_r0__whole_target"], daily_columns)
            + "\n\n七学习臂在532个保守已知阶段日上的内部A/S/D宏F1均为0.993640，"
            "这不代表未知817个活动内部日也被正确分期，更不证明新视觉阶段贡献。主端到端宏F1为0.816919，"
            "低于增广0.881734，静稳上的假阶段增多；仅给内部高分会掩盖该代价。"
            + "\n\n![观测敏感性](../artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-confirmation-analysis-v1/secondary_observation/comparison.png)\n\n"
            "### 5.4 错误解释与计算成本\n\n"
            "开发实际看图与原块复核同时保留好坏案例：0183一次偏移后平台使主新增25个误报日，增广和expert24均0；"
            "0169弱检124/212高于增广98/212；0178强运动后弱续动主与增广均0/196，而expert24为65/196；"
            "0188主189/208高于增广178/208。它们按评分后极值选择，只解释错误，不是独立样本选择或盲标。"
            "训练对实际查看前四对八图，端点精确相等，误差差异在浮点范围内；缺测内阶跃仍无法仅图判仪器原因。\n\n"
            "新确认评分后实际查看0196、0209、0206、0204、0208五幅图与125个原始30日块。"
            "0209主检出182/218弱日，增广111/218，净纠错+63；0196主FP67、增广22，净纠错−31。"
            "0206后平台小偏移处主FP45、纯数值14，净纠错−30；0204两次偏移主FP22、增广15。"
            "0208弱尾主55/194、增广0，但仍漏139弱日且对纯数值净纠错−11。"
            "这些正反案例保持原参考，详见[实际复核及数值证据](../artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-confirmation-error-review-v1/actual-review.md)。\n\n"
            "![弱累积纠错案例](../artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-confirmation-error-review-v1/case_0209-diagnostic.png)\n\n"
            f"本次24条共编码{sum(row['encoded_images'] for row in costs['rows'])}图，共享绘图/编码"
            f"{sum(row['render_encoder_seconds'] for row in costs['rows']):.3f}秒，主小头"
            f"{sum(row['heads']['visual_numeric_pair'] for row in costs['rows']):.6f}秒，含加载及十头推断总墙钟"
            f"{costs['wall_seconds']:.3f}秒。编码在视觉臂间共享，不能仅报小头开销。"
            f"VLM本轮{audit['api_starts']}启动/{audit['api_responses']}回执、{audit['reported_tokens']}报告token；"
            "方法各自部署请求见表，共享响应只在实际总量计一次，报告token不等于账单或能耗。"
            "历史XGB训练及总校准耗时未完整保存，保持未知。首次下载与模型加载开销另外记录。\n\n"
            + seed_sensitivity_section() + "\n\n## 6 讨论与结论\n\n" + verdict + "\n\n"
            "与同数据增广的比较检验排序项，打乱配对检验配对身份，纯数值配对检验视觉支撑。"
            "三个比较必须共同解释，不能把增广收益全部归于视觉，也不能用对较弱普通头的优势替代与强XGB的比较。"
            "正式确认固定一个训练种子，补充五种子也共享同一批小规模六家族记录；大量历史探索增加方法选择偏差，"
            "新完整记录有助于限制该偏差，但不能提供现场普遍有效性或统计显著性证明。\n\n"
            "训练负例依赖已知合成运动分量，无法直接从未标注现场GNSS拆出同等训练对。季节背景、仪器漂移、"
            "空间共模与非GNSS形变可能改变可辨识性；此前60条观测残差半合成迁移有明显误报，"
            "并未对本候选建立成功的现场迁移。原有限Agent独立完整24条确认曾未通过可靠性条件，"
            "这些失败保留，不因新视觉方法结果而追改。下一步研究应围绕新的识别依据或外部记录，不能重复改seed救当前候选。\n\n"
            "A/S/D由同一数值头给出，本文实际新学习能力仅活动识别。边界的未知与缺测具有观测意义，"
            "不能为了完整输出填确定标签。当前没有独立领域专家或现场金标准，Codex参考只提供可追溯的辅助判断。\n\n"
            "## 7 复现、数据与AI披露\n\n"
            f"受控记录累计216条，60半合成另计。累计{4017+audit['api_starts']}启动/{4015+audit['api_responses']}回执、"
            f"至少{17245825+audit['reported_tokens']}报告token；历史两次无回执的未知成本不补零。"
            "代码开发、实际看图、数值辅助、实验评分和写作由当前Codex执行，历史标签与方法暴露明确披露。"
            "复核时间戳、36图哈希、原块、中间拟议阶段与两处S降为未知的改动均保存；"
            "十个学习头及远程VLM在参考冻结后才运行，具体过程标签在全部预测冻结后才开放。"
            "原九数值控制由准备脚本在参考复核前计算，但其具体输出未在标注时读取；因此原评分audit的"
            "reference_before_all_inference=true范围写得过宽，已另存chronology-correction.json更正，"
            "冻结源码与原结果保持以便追溯。不以AI自判冒充外部专家确认。\n\n"
            "数据位于paired-persistence-confirmation-data-v1，十头推断为paired-persistence-confirmation-inference-v1，"
            "VLM原回执为paired-persistence-confirmation-vlm-v1，评分与离线重放分别为"
            "paired-persistence-confirmation-analysis-v1和paired-persistence-confirmation-replay-v1。"
            "旧校准评分曾因独占summary同名导出失败，只从已有预测恢复，未重训；旧开发runner仅Ruff导入格式I001未通过，"
            "冻结源码保留。新确认源码Ruff与观测编译契约核验通过。\n\n"
            "两次新完整离线评分的28个输出文件逐字节相同。quiet与instrument_step没有已知A/S/D支撑，"
            "冻结评分器在其阶段覆盖均值处产生空样本警告并保存未定义值；不将它解释成0或有效阶段测量。"
            "完整24条与12条观测子集的已知阶段指标均有非空支撑。统计解释、时序更正和实际看图完成记录"
            "均在paired-persistence-confirmation-error-review-v1单独封存。\n\n"
            "```powershell\n$env:PYTHONPATH='artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-runtime-v1/site-packages;src;scripts'\n"
            "$env:PYTHONIOENCODING='utf-8'\n& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/score_landslide_paired_persistence_confirmation.py "
            "--data artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-confirmation-data-v1 "
            "--heads artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-confirmation-inference-v1 "
            "--vlm artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-confirmation-vlm-v1 "
            "--out artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-user-confirmation-replay\n```\n\n"
            "输出目录需尚不存在。离线重放不训练、不开新API、不修改原结果。"
            "完整开发与校准见[配对结果](landslide-paired-persistence-results.md)，新全部表见"
            "[过程主表](../artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-confirmation-analysis-v1/primary_process/report.md)和"
            "[观测敏感性表](../artifacts/landslide-adaptive-review-2026-10-09/paired-persistence-confirmation-analysis-v1/secondary_observation/report.md)。"
            "历史所有失败、公式、引文与报告见[封存完整v53](../artifacts/landslide-adaptive-review-2026-10-09/manuscript-v54/landslide-adaptive-review-manuscript-draft-v53-original.md)。"
            "正文公式使用LaTeX并进行KaTeX语法与渲染核验。\n\n## 参考文献\n\n" + "\n\n".join(reference_lines) + "\n")
    (ROOT / "docs" / (DOCUMENTS[0] + ".md")).write_text(text, encoding="utf-8")
    result_path = ROOT / "docs/landslide-paired-persistence-results.md"
    original_result = result_path.read_text(encoding="utf-8")
    result_path.write_text(original_result + "\n## 一次新完整24条确认\n\n" + summary + "\n\n" + verdict
        + "\n\n" + table(primary, primary_names, daily_columns) + "\n\n"
        + "预指定12条36幅图由Codex实际查看，先冻结活动再复核阶段；8760原日，活动1349、静稳6221、未知1190，"
        "缺测378，A/S/D支持532。非独立专家参考、过程模型目标迁移敏感性和另列六观测目标控制均披露。"
        "新主稿按候选最终结果重写，历史v53保存。全部47主方法、53辅助方法及完整家族结果见新评分目录；"
        "未知和全部失败保持，goal active。\n", encoding="utf-8")
    status = "\n## 2026-10-11：配对视觉一次新完整确认，初稿v54\n\n" + summary + "\n\n" + verdict
    status += ("\n\n24新受控/216累计、预指定12条36幅原始及辅助图实际看图复核；参考在本批预测与过程标签开放前冻结，"
               "历史标签暴露、非独立专家、共享数值阶段头及未知范围披露。47主方法/53观测敏感性方法统一评分；"
               "主稿围绕候选与实际新结果重写，历史v53五文档原字节保留。"
               "[主稿](landslide-adaptive-review-manuscript-draft.md)。goal active，不以论文写完替代方法目标完成。\n\n")
    for name in DOCUMENTS[1:]:
        path = ROOT / "docs" / (name + ".md")
        heading, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(heading + "\n" + status + rest, encoding="utf-8")
    export(archive / "v53-preservation-after.json", preservation(archive))
    export(archive / "progress.json", {"status": "goal_active", "decision": decision,
        "new_controlled_records": 24, "formal_field_reference": False,
        "replay_checked_files": len(replay_files), "api_accounting": audit})
    export(archive / "statistical-interpretation.json", read_json(REVIEW / "statistical-interpretation.json"))
    export(archive / "chronology-correction.json", read_json(REVIEW / "chronology-correction.json"))
    export(archive / "reported-count-check.json", {
        "primary_main_daily": process_results["visual_numeric_pair"]["aggregate"]["daily"],
        "primary_augmentation_daily": process_results["visual_numeric_augmentation"]["aggregate"]["daily"],
        "primary_main_stages": process_results["visual_numeric_pair"]["internal_stages"],
        "secondary_main_daily": observation_results["visual_numeric_pair"]["aggregate"]["daily"],
        "secondary_augmentation_daily": observation_results["visual_numeric_augmentation"]["aggregate"]["daily"],
        "all_scores_read_from_verified_frozen_results": True})
    print("Wrote focused v54;", decision["decision"], flush=True)


def seal():
    archive = BASE / "manuscript-v54"
    preservation(archive)
    for name, folder in ((DOCUMENTS[0], "formula-verification"),
                         ("landslide-paired-persistence-results", "result-formula-verification")):
        audit = read_json(archive / folder / "formula-audit.json")
        if audit["status"] != "passed" or audit["source_sha256"] != sha(ROOT / "docs" / (name + ".md")):
            raise ValueError("Formula audit must bind current delivered Markdown")
    folders = ("manuscript-v54", "paired-persistence-confirmation-data-v1",
               "paired-persistence-confirmation-inference-v1", "paired-persistence-confirmation-vlm-v1",
               "paired-persistence-confirmation-analysis-v1", "paired-persistence-confirmation-replay-v1",
               "paired-persistence-confirmation-error-review-v1", "paired-persistence-training-seed-sensitivity-v1",
               "paired-persistence-training-seed-report-v1")
    paths = [path for name in folders for path in (BASE / name).rglob("*") if path.is_file()]
    paths.extend(ROOT / "docs" / (name + ".md") for name in NAMES)
    paths.extend(ROOT / "scripts" / name for name in (
        "prepare_landslide_paired_persistence_confirmation.py", "compile_landslide_paired_confirmation_review.py",
        "run_landslide_paired_persistence_confirmation.py", "score_landslide_paired_persistence_confirmation.py",
        "write_landslide_paired_confirmation_manuscript.py", "review_landslide_paired_confirmation_errors.py",
        "check_landslide_paired_training_seed_stability.py", "report_landslide_paired_seed_stability.py"))
    paths.append(BASE / "paired-persistence-confirmation-replay-v1.console.log")
    files = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(paths))}
    export(archive / "delivery-manifest.json", {"files": files})
    for name, expected in files.items():
        if sha(ROOT / name) != expected:
            raise ValueError("New delivered bytes differ: " + name)
    print("Sealed", len(files), "files; controlled research delivery complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    if parser.parse_args().seal:
        seal()
    else:
        write()
