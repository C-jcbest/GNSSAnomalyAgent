"""Update the complete research draft from the two frozen distribution experiments."""

from __future__ import annotations

from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json


def score_table(summary, methods):
    indexed = {row["method"]: row for row in summary}
    rows = [
        "| 方法 | Precision | Recall | 日F1 | 事件F1 .3/.5 | FP/FN | 起止MAE（日） | 活动内阶段F1 |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for name in methods:
        row = indexed[name]
        rows.append(
            f"| {name} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['event_f1_iou03']:.4f}/{row['event_f1_iou05']:.4f} | {row['fp']}/{row['fn']} | {row['onset_core_mae_days']:.2f}/{row['offset_core_mae_days']:.2f} | {row['stage_internal_macro_f1']:.4f} |"
        )
    return "\n".join(rows)


def write():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    mass = base / "motion-mass-probe-v1"
    visual = base / "distribution-view-calibration-v1"
    for folder in (mass, visual):
        verify_manifest(folder, "freeze.json")
    for replay in ("motion-mass-replay-v2", "distribution-view-replay-v1"):
        if read_json(base / replay / "verification.json")["status"] != "VERIFIED":
            raise ValueError("Do not write an unverified result")
    summary = read_json(visual / "summary.json")
    table = {row["method"]: row for row in summary}
    main = table["ordered_distribution_numeric"]
    duplicate = table["duplicate_motion_global"]
    permuted = table["permuted_distribution_numeric"]
    decision = read_json(visual / "decision.json")
    best = table[decision["best_numeric"]]
    cost = read_json(base / "distribution-view-encoding-v2/encoding-cost.json")
    text = (root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v6.md").read_text(
        encoding="utf-8"
    )
    text = text.replace(
        "# GNSS形变异常的视觉数值识别：多形态基准与持续性配对机制验证",
        "# GNSS形变异常的视觉数值识别：运动分布证据、多形态基准与机制验证",
        1,
    )
    start = text.index("> 中文完整研究初稿")
    end = text.index("\n\n## Material Passport", start)
    text = (
        text[:start]
        + "> 中文完整研究初稿，2026-10-13，v7。整体形变异常为主，仪器偏移／尖峰独立识别，弱运动与活动内A/S/D为子分析。本版新增运动变化分布数值探针和同信息强对照的视觉辅助图小试，历史冻结结果保持。所有新分数为暴露开发验证；保留24记录未打开，AI辅助看图不等于独立领域参考。候选名称不表示有效创新已成立。"
        + text[end:]
    )
    text = text.replace(
        "deformation-manuscript-20261013-v6", "deformation-manuscript-20261013-v7", 1
    )
    start = text.index("## 摘要\n\n") + len("## 摘要\n\n")
    end = text.index("\n\n**关键词", start)
    chinese = f"""长期GNSS形变异常识别需要区分持续运动与仪器跳变、尖峰和运动后平台。本文在冻结图文预训练视觉分支与数值活动识别的基础上，研究可隔离的运动证据机制，并自主生成96条730日多形态记录，按完整记录划分48训练、24开发验证和24保留评价。规则、PELT、XGBoost及视觉方法使用统一活动任务、阶段约束和缺测解码。既有持续性配对在窄受控分布中的收益未稳定推广；共享坐标修复也未改善识别。因此进一步从原始位移块构造多尺度运动变化分布，48／24开发日F1由raw84数值树0.7836提高至查询分布树0.8442，纯干扰误报由15日降为0日；但查询局部特征对全局分布多75TP和76FP，净−1日，未通过预设继续条件。为检验视觉组织而非仅增加信息的价值，另在固定八族16／8开发小试中提出MD-VN（Motion-Distribution Visual-Numeric Recognition）候选，将变化量分布和邻接方向一致性画为辅助图，与重复原图、打乱布局、去数值反馈及同信息数值模型比较。主方法日／事件F1为{main["f1"]:.4f}/{main["event_f1_iou03"]:.4f}，重复原图{duplicate["f1"]:.4f}/{duplicate["event_f1_iou03"]:.4f}，打乱辅助图{permuted["f1"]:.4f}/{permuted["event_f1_iou03"]:.4f}；最强数值对照{best["f1"]:.4f}/{best["event_f1_iou03"]:.4f}。七事前条件{sum(decision["checks"].values())}项成立，判定{decision["status"]}。研究已保存统一对比、消融、逐族误例和选中模型精确重放；开发收益与负结果如实报告，尚不宣称独立现场效果或稳定Agent/VLM优势。"""
    text = text[:start] + chinese + text[end:]
    text = text.replace(
        "**关键词：** GNSS；形变异常；观测干扰；持续性配对；视觉数值融合；事件识别。",
        "**关键词：** GNSS；形变异常；运动分布；观测干扰；视觉数值融合；事件识别。",
        1,
    )
    start = text.index("## Abstract\n\n") + len("## Abstract\n\n")
    end = text.index("\n\n## 1", start)
    english = f"""Long-term GNSS deformation detection must distinguish sustained motion from instrument offsets, spikes, and post-motion plateaus. We investigate isolated physical-evidence mechanisms with a frozen pretrained visual branch and numerical activity models. A controlled benchmark contains 96 diverse 730-day records, split by complete record into 48 training, 24 exposed development-validation, and 24 reserved records. Numerical rules, PELT, XGBoost, and visual methods share the task and missing-observation decoder. Previously observed narrow-distribution gains from persistence pairing did not generalize reliably, and shared chart coordinates did not improve detection. A new numerical movement-distribution probe increases development daily F1 from 0.7836 for raw84 XGBoost to 0.8442 and reduces nuisance-only false positives from 15 days to zero. However, query-local summaries add 75 true-positive and 76 false-positive days over global summaries, failing the predeclared net-correction criterion. To test visual organization rather than additional information alone, a separate fixed 16-training/8-validation pilot introduces the MD-VN candidate, which plots displacement-change magnitudes and neighboring direction agreements. It is compared with duplicated charts, permuted auxiliary layouts, removal of numerical feedback, and equally informed numerical models. Its daily/event F1 is {main["f1"]:.4f}/{main["event_f1_iou03"]:.4f}, versus {duplicate["f1"]:.4f}/{duplicate["event_f1_iou03"]:.4f} for duplicated charts, {permuted["f1"]:.4f}/{permuted["event_f1_iou03"]:.4f} for permuted auxiliary layouts, and {best["f1"]:.4f}/{best["event_f1_iou03"]:.4f} for the strongest numerical control. The pilot satisfies {sum(decision["checks"].values())} of seven predeclared conditions. Unified comparisons, ablations, family-level errors, and exact selected-output replays are retained. These are development findings; the reserved records remain unopened, and AI-assisted developer observation review is not independent domain reference."""
    text = text[:start] + english + text[end:]
    text = text.replace(
        "本文报告三个可核查的研究工作：",
        "本文持续检验小规模机制，而不将弱运动或A/S/D作为主任务。最新工作检验运动变化是否分散、方向是否持续及证据相对查询位置的分布。本文此前报告三个可核查的研究工作：",
        1,
    )
    related = """TShape[13]用局部patch形态、位置及跨patch关系识别复杂形态异常，说明时序排列和局部／全局组合本身已有先例。作者当前固定commit的TShape_model实际仅传入一个卷积核尺度，不能将论文设计与默认代码混为一谈。VAN-AD[14]将视觉MAE重构和可选normalizing flow结合，源码支持不同微调范围；其质量异常检测定义不等于本文形变阳性／仪器干扰独立任务。本轮阅读范围与精确源码commit另存，未运行两作者模型。本文运动分布辅助图只检验GNSS三轴物理证据的组织价值，不能把视觉编码、时序位置或普通拼接称为通用首创。

"""
    text = text.replace("## 3 任务与方法", related + "## 3 任务与方法", 1)
    method = r"""### 3.10 运动变化分布数值探针

令原13个14日块的噪声归一化三轴中位向量为 $\mathbf z_k$。每块至少8个完整原NEU观测，缺测保持未知，仅在相邻两个块可用时定义：

$$
\mathbf d_k=\mathbf z_{k+1}-\mathbf z_k,\qquad
m_k=\lVert\mathbf d_k\rVert_2,\qquad
M=\sum_{k\in\mathcal E}m_k,\qquad
C=\frac{\lVert\sum_{k\in\mathcal E}\mathbf d_k\rVert_2}{M}.
$$

其中 $\mathcal E$ 只包含可用邻接边。有效运动参与数和最大变化占比为：

$$
N_{\mathrm{eff}}=\frac{(\sum_{k\in\mathcal E}m_k)^2}{\sum_{k\in\mathcal E}m_k^2},\qquad
P_{\max}=\frac{\max_{k\in\mathcal E}m_k}{M}.
$$

公式用于描述可用且非零变化；零分母、不可用或样本不足按代码保留未知／指定中性值，不将其自动赋为活动。三尺度分别使用5／7／13块，得到净变化、总路径、coherence、最大占比、参与数、去最大变化后的净量及支持比例21项；目标两侧变化占比与变化质量距目标的平均距离另6项。去最大变化统计不等于删除原观测或直接扣除仪器跳变。该探针仅检验信息，不证明视觉或Agent创新。

### 3.11 运动分布视觉数值识别候选

MD-VN（Motion-Distribution Visual-Numeric Recognition，运动分布视觉数值识别）在原曲线视觉特征与原18数值基础上，新增一个辅助视图。上栏画12个变化量，下栏画11个邻接方向一致性：

$$
a_k=\frac{\mathbf d_k^{\mathsf T}\mathbf d_{k+1}}{\lVert\mathbf d_k\rVert_2\lVert\mathbf d_{k+1}\rVert_2},\qquad
S_m=\max\!\left(6,\max_{k\in\mathcal E}m_k\right).
$$

原日历位置保留，纵幅覆盖最大值，无裁切；不可用边灰色，零向量方向未知，不跨缺测连接。原图与辅助图均用同冻结MobileCLIP2视觉分支和原transform编码。活动头输入为：

$$
\mathbf h_t=\left[E(I_t)\,;\,E(J_t)\,;\,\widetilde{\mathbf n}_t\right],\qquad
p_t=\sigma\!\left(f_\theta(\mathbf h_t)\right).
$$

主／重复原图／打乱辅助图均1060维，相同小头、训练中心、标签及样本次序。去数值为1024维，辅助单图＋数值548维；同信息数值MLP获得18原值＋46辅助值／可用＋52原块profile，并以TRAIN缩放为232维。强树得到相同116维原证据，另一482维树再包含原曲线全部366描述符。全局／查询分布树也在相同16TRAIN重新拟合，不能借用48TRAIN分数。

打乱辅助图使用事前固定排列，保留全部值和缺测multiset，同时去掉原时序和两栏关系，改成rank布局，因此不解释为只去时间、其余全部像素完全相同的消融。本候选固定观察每个合格中心，不是自适应Agent运行时，也不新增语言模型或A/S/D分类机制。

"""
    text = text.replace("## 4 数据与实验协议", method + "## 4 数据与实验协议", 1)
    mass_table = score_table(
        read_json(mass / "summary.json"),
        (
            "mass_raw84",
            "global_motion_mass",
            "query_motion_mass",
            "raw_motion_profile",
            "all_motion_information",
            "visual_numeric",
        ),
    )
    methods = (
        "ordered_distribution_numeric",
        "duplicate_motion_global",
        "permuted_distribution_numeric",
        "ordered_distribution_visual",
        "auxiliary_only_numeric",
        "same_information_numeric_mlp",
        "distribution_information_xgb",
        "all_distribution_information_xgb",
        "pilot_global_mass_xgb",
        "pilot_query_mass_xgb",
        "global_numeric",
        "numeric",
        "xgb_raw12",
        "xgb_raw84",
        "xgb_chart384",
        "robust_rule",
        "pelt_rule",
    )
    visual_table = score_table(summary, methods)
    checks = "; ".join(
        name + ":" + ("通过" if value else "未通过") for name, value in decision["checks"].items()
    )
    results = f"""### 5.12 运动分布的数值信息收益与净纠错限制

完整48／24、原30659合法训练日、五树各38机会，190新配置。原raw84重新拟合逐值复现历史输出；13原方法同455机会保留。新探针72原证据、5选中模型87600概率、18方法432输出及活动／阶段评分精确重放。第一次重放因方法列表排序而失败，改为按method比较完整集合后通过，模型和分数未改。

{mass_table}

主查询分布日F1 .844167对raw84 .783572，纯干扰FP15→0，混合召回.634667→.738667；有效信息收益覆盖总体活动而不限弱运动。但对全局分布多75TP也多76FP，净−1日。预设五条件4成立，STOP_MOTION_MASS_DESCRIPTOR，停止这版查询描述符，不将全局对照改名为贡献。事后原图0389显示查询更早扩边造成净−19日，0402混合背景补回38日且无FP，不能挑后者代替整体结论。

### 5.13 同信息数值对照下的辅助视觉分布小试

按已固定八族每族2TRAIN／1VALIDATION选择16／8，同原stride3训练中心、TRAIN缩放及阶段头；六神经／四树各38机会，共380新机会，加原11方法360历史机会。与48／24属于不同训练规模，分数不可横向称为提高；完整21方法表、八族表、逐例净纠错与图在distribution-view-report-v1。

{visual_table}

七条件{sum(decision["checks"].values())}/7成立，**{decision["status"]}**，对重复／打乱净纠错{decision["net_correct"]["duplicate"]}/{decision["net_correct"]["permuted"]}日。判据逐项：{checks}。最强数值为{decision["best_numeric"]}，它是公平对照而非重新命名的视觉创新。活动内阶段分数仍由共同数值头及活动门控产生，不能称新视觉阶段贡献。

原24记录辅助证据与TRAIN缩放逐值重构，96辅助视觉编码抽样重算；10选中模型58400概率、21方法168记录输出及全部统一评分精确重放。重复组重训输出与原相同1060维重复原图组完全一致。没有重新训练全部20模型或重新编码全部{cost["new_visual_encodings"]}图。总准备／编码{cost["seconds"]:.3f}秒，训练分段时间及头参数另存；新增0独立记录／API，不能把窗口数冒充样本规模。

首次编码进度CSV重复创建失败后改成每记录独立文件，在新目录完整重启；没有改变图像、seed、loss、阈值或标签。源码／失败回执保留。保留24输入与参考未读取，不把开发者实际看图解释当作独立标签。

"""
    text = text.replace("## 6 讨论", results + "## 6 讨论", 1)
    discussion = """### 6.4 新信息收益与视觉贡献必须分开验证

运动变化分布比原局部速率更直接表述持续位移与集中偏移的差别，数值探针确有开发信息收益。但数值特征收益不是Agent/VLM创新。辅助视图与同信息数值、重复图和打乱图的比较，检验信息经过冻结视觉编码后是否仍有额外价值；不能用更好数值对照替换预设主方法。小试只有8条暴露验证记录，不能提供独立显著性结论，若继续也只扩完整开发集。视觉基础模型可能忽略量级、目标位置和两栏对应关系，这是待验证解释；本轮没有微调视觉骨干，不能认定骨干能力不足或宣称微调必然解决问题。

"""
    text = text.replace("## 7 结论", discussion + "## 7 结论", 1)
    addition = f"运动分布数值探针显示可用物理信息的新收益，却因查询特征净纠错−1日停止；辅助视觉小试通过{sum(decision['checks'].values())}/7条件，判定{decision['status']}。当前仍需一个稳定机制及未参与选择的新记录验证，方可支撑Agent/VLM算法贡献。"
    text = text.replace("## 数据、代码与AI使用说明", addition + "\n\n## 数据、代码与AI使用说明", 1)
    text = text.replace(
        "Codex参与文献定位",
        "运动分布由probe/verify/report_landslide_motion_mass.py和encode/train/verify/report_landslide_motion_distribution.py复现；模型、380网格、概率、图表及全部评分各自冻结。本轮TShape／VAN-AD官方原文和六个固定commit作者源码文件保存于morphology-primary-v1／morphology-code-source-v1，阅读边界见distribution-session-v1/reading-notes.md，未运行作者模型。\n\nCodex参与文献定位",
        1,
    )
    text += "\n[13] TShape: Rescuing Machine Learning Models from Complex Shapelet Anomalies. [arXiv:2510.00680](https://arxiv.org/abs/2510.00680), 2025. [作者代码](https://github.com/CSTCloudOps/TShape)。实际读取摘要、III-A／III-B／III-C和III-D起始，及readme.md／layyer.py固定commit，未运行作者模型。\n\n[14] VAN-AD: Visual Masked Autoencoder with Normalizing Flow for Time Series Anomaly Detection. [arXiv:2603.26842](https://arxiv.org/abs/2603.26842), 2026. [作者代码](https://github.com/PenyChen/VAN-AD)。实际读取摘要、IV overview／Forward Module起始及固定commit四个源码文件相关范围，未复现作者成绩。\n"
    target = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v7.md"
    if target.exists():
        raise FileExistsError("Do not overwrite an existing draft snapshot")
    target.write_text(text, encoding="utf-8")
    print(str(target), flush=True)


if __name__ == "__main__":
    write()
