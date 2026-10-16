"""Write a new complete draft from frozen shared-frame results, retaining v5."""

from __future__ import annotations

from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json


def write():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    run = base / "shared-frame-calibration-v1"
    verify_manifest(run, "freeze.json")
    receipt = read_json(base / "shared-frame-replay-v1/verification.json")
    if receipt["status"] != "VERIFIED":
        raise ValueError("Do not write verified shared-frame claims before replay")
    source = root / "docs/gnss-deformation-manuscript-draft-2026-10-12-v5.md"
    target = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v6.md"
    if target.exists():
        raise FileExistsError(target)
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "# GNSS形变异常的视觉数值识别：持续性配对、目标观察与分歧复核验证",
        "# GNSS形变异常的视觉数值识别：多形态基准与持续性配对机制验证",
        1,
    )
    paragraphs = text.splitlines()
    paragraphs[2] = (
        "> 中文完整研究初稿，2026-10-13。本稿以整体形变异常为主，偏移和尖峰单独识别，弱运动与活动内部A/S/D为附加分析。本版新增训练侧绘图上下文诊断、共享坐标配对与严格同监督对照的完整48／24实验；v5保持原样。结果为已暴露开发评价，24保留参考未打开，当前尚未建立稳定的新Agent/VLM贡献，尚不作为已证明改进的算法创新投稿稿。"
    )
    text = "\n".join(paragraphs) + "\n"
    text = text.replace("deformation-manuscript-20261012-v5", "deformation-manuscript-20261013-v6")
    text = text.replace(
        "## 摘要",
        "- Shared-Frame Verification: 127组共享图外侧槽raster／缺测／无裁切、254编码、9选中模型157680概率、22方法528输出及评分精确重放；没有重训全部18模型。\n\n## 摘要",
        1,
    )
    chinese = """长期GNSS形变异常识别需要区分持续运动与仪器偏移、尖峰及活动后平台。本文研究冻结图文预训练模型视觉分支与数值特征的持续性配对学习，并在训练侧构造同净位移的运动／偏移反事实，以检验形态证据是否改善活动定位。为验证推广性，自主生成96条730日多形态记录，按完整记录划分48训练、24开发验证及24保留评价；规则、PELT、XGBoost与视觉方法使用统一任务、划分和缺测协议。在既有窄受控分布中，配对相对同增广活动F1提高约0.0225，但新增多形态验证未复制稳定整体收益。进一步发现局部物理干预虽保留外侧观测，却因独立绘图归一化改变所有127组配对的外侧槽像素。为隔离此因素，提出共享坐标持续性配对候选SC-PP，固定物理干预和监督曝光，仅使配对共享参考、跨度及噪声下限。外侧槽raster严格一致，但活动F1为0.7352，低于同监督增广0.7619，与独立坐标配对0.7353几乎相同；事件F1为0.7179，低于增广0.8235。普通视觉数值活动F1为0.7941，raw84数值树为0.7836，后者事件F1为0.8889。结果表明当前共享坐标修复不能转化为识别增益，也不证明独立归一化是旧方法失败的原因。本文保留完整对比、消融、误例及重放证据，明确配对方法的受控价值与推广限制；保留集尚未使用，AI辅助原图复核不冒称独立专家参考。"""
    start = text.index("## 摘要\n\n") + len("## 摘要\n\n")
    end = text.index("\n\n**关键词", start)
    text = text[:start] + chinese + text[end:]
    english = """Long-term GNSS deformation detection must distinguish sustained motion from instrument offsets, spikes, and post-motion plateaus. We investigate persistence-paired learning with a frozen pretrained vision branch and numerical features, using equal-displacement motion/offset counterparts on controlled training records. To evaluate generalization, we generated 96 diverse 730-day records and split complete records into 48 training, 24 exposed development-validation, and 24 reserved evaluation records. Numerical rules, PELT, XGBoost, and visual methods share the task, split, and missing-observation protocol. Pairing improved daily F1 by approximately 0.0225 over identical augmentation on an earlier narrow controlled distribution, but did not establish stable overall gains on the diverse validation set. A subsequent diagnostic found that independent chart normalization changed outer raster slots in all 127 local training pairs despite preserving the outer observations. Shared-Coordinate Persistence Pairing (SC-PP) therefore holds physical interventions and supervised exposure fixed while sharing the chart reference, span, and original noise floor. The outer raster slots become identical, but daily F1 is 0.7352 versus 0.7619 for matched augmentation and 0.7353 for independent-coordinate pairing. Event F1 is 0.7179 versus 0.8235 for augmentation. Ordinary visual–numeric learning reaches daily F1 of 0.7941; a strong raw84 numerical tree reaches daily/event F1 of 0.7836/0.8889. Repairing chart-context consistency thus does not improve detection for this configuration, nor establish independent normalization as the cause of the earlier failure. We retain comparisons, ablations, error analyses, and exact output replays. The reserved set remains unopened, and developer AI-assisted observation review is not presented as independent expert reference."""
    start = text.index("## Abstract\n\n") + len("## Abstract\n\n")
    end = text.index("\n\n## 1", start)
    text = text[:start] + english + text[end:]
    method = r"""### 3.9 共享坐标的局部反事实配对

在训练侧局部物理干预冻结后，令正负成员第$k$槽、第$j$轴的位移中位值分别为 $z_{k,j}^{+}$ 与 $z_{k,j}^{-}$。共享坐标持续性配对（SC-PP，Shared-Coordinate Persistence Pairing）使用正图中位参考，并同时覆盖双方有效值：

$$
\rho_j=\operatorname{median}_{k\in\mathcal K}z_{k,j}^{+},\qquad
S_j=\max\!\left(2\max_{a\in\{+,-\},\ k\in\mathcal K}|z_{k,j}^{a}-\rho_j|,\;6\widehat\sigma_j\right).
$$

$$
y_{k,j}^{a}=31\left(\frac12-\frac{z_{k,j}^{a}-\rho_j}{S_j}\right),\qquad a\in\{+,-\}.
$$

其中噪声代理只用原正记录；双方共用参考、跨度和噪声下限。缺测槽不画线，至少两个完整原日才可形成槽。共享跨度覆盖双方，不能以负图越界裁切制造差异；无支撑全空图保持空白。由于中心±21日之外的物理观测相同，偏移大于21日的三日槽raster和availability通道逐值相同。这里的严格相同指六通道槽像素；PNG连线可影响相邻线段，不将相同声明无条件扩展到±21日以外的所有连续画布像素。

所有新神经方法获得同一监督池、标签和顺序：原10187查询＋127共享正图＋127共享负图＋127独立坐标负图，共10568。独立正图已在原查询池。共享／独立坐标配对仅改变辅助排序所用正负图，不改变监督曝光、物理干预或优化参数。去配对、打乱关系、去视觉、去数值和纯数值增广进行匹配消融。raw84及图信息强树获得同381增广与全部30659原合法训练日。推理仍用普通原图和原数值，不读取评价真值或生成反事实。

共享坐标、背景一致性和反事实学习都不是通用首创。CAPS第3节已经明确同上下文对照隔离背景差异；VETime第3.2—3.3节已有视觉变换和时间对齐。SC-PP只检验本GNSS物理干预向图像传递时的具体上下文一致性机制，效果必须独立成立。

"""
    text = text.replace("## 4 数据与实验协议", method + "## 4 数据与实验协议", 1)
    scores = read_json(run / "summary.json")
    methods = [
        "shared_visual_numeric_pair",
        "shared_visual_numeric_augmentation",
        "independent_visual_numeric_pair",
        "shared_visual_numeric_shuffled",
        "shared_visual_pair",
        "shared_numeric_pair",
        "shared_numeric_augmentation",
        "shared_xgb_raw84_augmentation",
        "shared_xgb_chart384_augmentation",
        "visual_numeric",
        "numeric_mlp",
        "xgb_raw84",
        "pure_visual",
    ]
    indexed = {row["method"]: row for row in scores}
    table = [
        "| 方法 | P | R | 日F1 | 事件F1 .3/.5 | FP/FN | 起止MAE（日） |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for name in methods:
        row = indexed[name]
        table.append(
            f"| {name} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['event_f1_iou03']:.6f}/{row['event_f1_iou05']:.6f} | {row['fp']}/{row['fn']} | {row['onset_core_mae_days']:.3f}/{row['offset_core_mae_days']:.3f} |"
        )
    results = (
        """### 5.11 共享坐标机制：外侧槽一致但没有识别增益

训练侧127局部配对原观测外侧逐值一致，但独立归一化后127组均出现外側槽像素变化，中位外围trace差异占比0.699743，外围平均槽位移中位0.985536像素、最大13.533981像素。Codex实际打开两组最大外围差异图，确认物理上下文不变不等于图像上下文不变；这不是因果结论。

新一轮在同48／24完整划分冻结上述共享坐标方案，全部127对来自28训练父记录，新增254编码、0独立记录、0API。七神经与两强树各38配置，共342；13原对照与455历史机会保留，不在保留集选模型。所有外侧槽raster一致、原缺测保留、有效槽trace质量为1，未裁切。

"""
        + "\n".join(table)
        + """

共享主对同监督增广净损101日，对独立坐标配对仅净补1日，二者最终活动只有5个日历日不同；日F1略低而事件F1相同。主纯干扰FP51，同增广0；混合召回0.630667对0.598667，仅有子组召回收益。预设六性能条件2项成立，STOP_SHARED_FRAME。不改参数救此版本，不将打乱配对较高日F1或同增广图信息树0.798080更名为贡献。打乱组事件F1仅0.622222，也不能据日F1更高认为错配本身有效。结果否定当前共享坐标配对的总体识别改进假设，但不能推广为共享坐标对所有模型无用。

实际原图0385仍为两次水平转换后平台，共享主产生[144,162)和[271,288)活动，净增35误报日，同增广没有活动；原14日块N中位在首个转换前约0附近，转换后保持约−25mm，再次转换后约−52mm。0400的N在约550—585日持续下降后平台，E/U信号弱且波动明显；主相比增广补检3日但仍漏23个过程阳性日，不能从这一例确认过程精确起止。两例按净纠错最差／最好事后选择，只作原因分析，不修改标签或赋新增A/S/D。

配对表征距离中位由独立坐标0.064785变为共享0.044316，仅是冻结特征描述，不能据表征更接近或分数更平滑宣称可靠性提高。共享主与独立对照全日平均概率绝对差为0.003760。当前证据无法证明旧失败由归一化造成，更支持停止继续围绕同全局向量的坐标微调。

127反事实物理值重新构造，原数值重放绑定；共享图、254编码、9选中模型157680概率、22方法528输出及活动／阶段评分精确重放，没有重训全部18模型。预处理总21.128秒，其中共享编码12.228秒，训练／校准分段用时另存，不当端到端部署基准。内部阶段F1主0.452072、增广0.457088来自共同数值头及活动门控，不称新阶段能力。完整结果、八族与图见[本轮报告](../artifacts/gnss-deformation-diversity-2026-10-12/shared-frame-report-v1/report.md)，原图、槽坐标及分数诊断见shared-frame-analysis-v1。

"""
    )
    text = text.replace("## 6 讨论", results.replace("外側", "外侧") + "## 6 讨论", 1)
    text = text.replace(
        "### 6.2 可信参考与独立性",
        "共享坐标实验已经隔离训练监督曝光并实现外侧槽一致，却没有优于独立坐标或普通增广。故不得继续将这一输入问题视为旧失败的已证实原因，也不能用更小表征差异替代活动、事件和边界收益。下一轮需要新增可识别证据或适配能力的明确依据，不能只是再修改同全局视觉表示的坐标。\n\n### 6.2 可信参考与独立性",
        1,
    )
    text = text.replace(
        "应继续围绕事件与平台区分开发一个更有效的小机制",
        "共享坐标修复虽恢复了训练侧槽上下文一致，也未带来总体识别增益。应继续围绕事件与平台区分开发一个更有效的小机制",
    )
    text = text.replace(
        "保留评价文件只核验哈希，不在本稿工作中加载内容。",
        "共享坐标由prepare/train/verify/report_landslide_shared_frame.py复现，analyze_landslide_shared_frame.py保存原块和配对表征诊断；训练前协议、全部网格和结果独立冻结。保留评价文件只核验哈希，不在本稿工作中加载内容。",
        1,
    )
    target.write_text(text, encoding="utf-8")
    print(target, flush=True)


if __name__ == "__main__":
    write()
