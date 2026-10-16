"""Add the frozen NCVR experiment to the complete research draft without rescoring."""

from __future__ import annotations

from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json
from write_landslide_distribution_manuscript import score_table


def write():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    run = base / "conditional-visual-calibration-v1"
    for name in (
        "conditional-visual-calibration-v1",
        "conditional-visual-replay-v1",
        "conditional-visual-report-v1",
        "conditional-visual-primary-v1",
        "conditional-visual-code-source-v1",
    ):
        verify_manifest(base / name, "freeze.json")
    verified = read_json(base / "conditional-visual-replay-v1/verification.json")
    if verified["status"] != "VERIFIED":
        raise ValueError("The manuscript requires a successful result replay")
    decision = read_json(run / "decision.json")
    projection = read_json(run / "projection-cost.json")
    source = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v7.md"
    text = source.read_text(encoding="utf-8")
    start = text.index("> 中文完整研究初稿")
    stop = text.index("\n\n## Material Passport", start)
    text = (
        text[:start]
        + "> 中文完整研究初稿，2026-10-13，v8。整体形变异常为主，仪器偏移／尖峰独立识别且可同现；弱运动及活动内A/S/D为子分析。本版补数值证据条件化视觉残差（NCVR）的完整48／24开发实验、强对照、消融与负结果。96条730日记录规模未变，保留24内容未加载；AI辅助看图为开发者分析，尚无独立领域参考，尚未成立稳定Agent/VLM贡献。"
        + text[stop:]
    )
    text = text.replace("deformation-manuscript-20261013-v7", "deformation-manuscript-20261013-v8", 1)
    start = text.index("## 摘要\n\n") + len("## 摘要\n\n")
    stop = text.index("\n\n**关键词", start)
    abstract = "长期GNSS形变异常识别需要区分持续运动与仪器跳变、尖峰和运动后平台。本文围绕冻结图文预训练视觉分支与数值模型检验简单物理证据机制，自主生成96条730日多形态记录，按完整记录划分48训练、24暴露开发验证和24保留评价。规则、PELT、XGBoost及视觉方法使用统一活动任务、缺测解码及活动内阶段协议。运动变化分布数值探针使开发日F1由raw84树0.7836提高至查询分布树0.8442，但查询特征对全局分布净纠错−1日；另在16／8小试中，运动分布辅助图未优于打乱布局及同信息数值对照。在完整48／24上进一步检验数值证据条件化视觉残差（NCVR）：由训练数值证据无标签回归视觉表示，再将剩余表示与原数值拼接。七神经臂、四强树共418新配置，主日／事件F1为0.7933／0.8235，低于原拼接0.8057／0.8485及数值分布树0.8372／0.8485；对原拼接净纠错−37日，七预设条件仅两项成立，停止该版本。训练期望、选中模型概率及29方法统一输出均已精确重放。本文报告可追溯信息收益及机制负结果，尚不宣称稳定Agent/VLM优势、独立现场效果或可投稿的已验证算法贡献。"
    text = text[:start] + abstract + text[stop:]
    start = text.index("## Abstract\n\n") + len("## Abstract\n\n")
    stop = text.index("\n\n## 1", start)
    english = "Long-term GNSS deformation detection must distinguish sustained motion from instrument offsets, spikes, and post-motion plateaus. We test isolated physical-evidence mechanisms combining a frozen pretrained visual branch and numerical activity models. The controlled benchmark contains 96 diverse 730-day records, split by complete record into 48 training, 24 exposed development-validation, and 24 reserved records. Rules, PELT, XGBoost, and visual methods share activity, missing-observation decoding, and within-activity stage protocols. Numerical movement-distribution descriptors increase development daily F1 from 0.7836 to 0.8442, but query-local descriptors introduce one net additional error compared with global descriptors. An auxiliary distribution chart fails to outperform permuted layouts and equally informed numerical models in a separate 16/8 pilot. We then test Numerical-Conditioned Visual Residuals (NCVR) on the full 48/24 development population: a label-free TRAIN-only regression predicts visual embeddings from numerical evidence, and the residual is fused with the unchanged numerical inputs. Seven neural arms and four strong trees receive 418 new configurations. NCVR attains daily/event F1 of 0.7933/0.8235, below raw concatenation at 0.8057/0.8485 and a movement-distribution tree at 0.8372/0.8485. It introduces 37 net additional day errors versus raw fusion and meets only two of seven predeclared conditions. The version is stopped. The expectation fits, selected probabilities, and all 29 methods are exactly replayed. The reserved records remain unopened; AI-assisted developer review does not constitute independent domain reference. These results support traceable numerical information gains and mechanistic negative findings, rather than a validated general Agent/VLM advantage."
    text = text[:start] + english + text[stop:]
    related = """ResComEmb[15]从多粒度视觉token的分层重复性压缩出发，优先保留具有新证据的细粒度表示；本轮只读摘要与RHC概览、coarse-anchor set及intra-resolution assessor起始，不称全文核查。VTBench[16]研究图像和原序列的融合，本文普通拼接不能作为通用创新。固定作者commit中的FusionModule支持concat和输入无关的全局可训练softmax分支权重；该静态路径不能直接等同论文描述的动态attention，也不据静态片段宣布作者算法错误。两作者方法均未在本GNSS任务实际运行。NCVR与这些先例的差异只在数值运动证据条件化的视觉残差假设，回归、归一化与残差操作本身不是首创。

"""
    text = text.replace("## 3 任务与方法", related + "## 3 任务与方法", 1)
    method = r"""### 3.12 数值证据条件化视觉残差候选

NCVR（Numerical-Conditioned Visual Residuals）检验冻结全局视觉表示中的冗余是否妨碍证据利用。输入数值为原18项及运动分布21项，共39项；仅用10187个合法TRAIN中心拟合缩放、插补与不可用指示，形成78维向量 $\mathbf u_t$。令512维原视觉表示为 $\mathbf v_t$，固定正则系数1，无活动标签的表示期望为：

$$
(\widehat{\mathbf W},\widehat{\mathbf b})
=\mathop{\arg\min}_{\mathbf W,\mathbf b}
\sum_{t\in\mathcal T_{\mathrm{train}}}
\left\|\mathbf v_t-\mathbf W\mathbf u_t-\mathbf b\right\|_2^2
+\|\mathbf W\|_F^2.
$$

VALIDATION不重拟合期望、不逐记录重新中心化。残差单位化与融合活动头定义为：

$$
\mathbf r_t=\mathbf v_t-\widehat{\mathbf W}\mathbf u_t-\widehat{\mathbf b},\qquad
\widetilde{\mathbf r}_t=\frac{\mathbf r_t}{\max(\|\mathbf r_t\|_2,10^{-12})},\qquad
p_t=\sigma\!\left(h_\theta([\widetilde{\mathbf r}_t;\mathbf u_t])\right).
$$

该可预测分量也可能包含真实运动，不能全称仪器背景或因果分解。原拼接、仅TRAIN均值中心化后单位化、打乱数值与视觉对应后拟合、去单位化、去数值与数值MLP为关键消融。五主要融合臂均590维，保持训练曝光、顺序、seed及16／32宽头一致；不修改原图、不调用语言模型，属于视觉数值识别候选，而非已实现Agent复核。

四强树使用同源证据：全局分布39维、嵌入与运动551维、完整原图及块证据457维、全信息加嵌入969维。纯数值树使用30659合法训练日；带视觉嵌入树只能使用实际编码的10187中心，不能把未编码零向量当成真实输入。公平性既包括信息，也明确记录监督量差异。

"""
    text = text.replace("## 4 数据与实验协议", method + "## 4 数据与实验协议", 1)
    methods = (
        "conditional_visual_motion", "raw_visual_motion", "centered_visual_motion",
        "shuffled_visual_motion", "unnormalized_visual_motion", "conditional_visual_only",
        "motion_numeric_mlp", "refit_global_motion_mass", "embedding_motion_xgb",
        "raw_motion_information_xgb", "all_visual_motion_xgb", "query_motion_mass",
        "visual_numeric", "numeric_mlp", "mass_raw84", "xgb_chart384", "pure_visual",
        "fixed_blend", "raw_evidence_review", "generic_score_stacker", "robust_rule", "pelt_rule",
    )
    table = score_table(read_json(run / "summary.json"), methods)
    result = f"""### 5.14 条件化视觉残差：相同容量与强数值下的负结果

七神经／四树各38机会，418新配置，加18原方法共29方法。全局分布树重新训练逐值复现旧模型输出。原48TRAIN阶段头、14阳性观测解码及仅跨不超过7个原缺测连接均不变；不是沿用不适用旧模型冒充新数据公平比较。

{table}

主对原拼接净−37日，对仅去均值及打乱分别净+22及+23日。纯干扰FP41，对原拼接43及仅去均值41；混合形变召回.584低于原拼接.604。七冻结条件{sum(decision['checks'].values())}/7成立，**{decision['status']}**。高于弱消融不足以支持整体识别贡献；不调正则、seed或loss救分，不更名普通树融合或更强数值对照为创新。

72原记录与TRAIN缩放重构，两个无标签表示期望重新拟合逐值相同；11选中模型192720概率、29方法696记录输出及全部活动／阶段／事件／边界评分精确重放。没有重训全部22活动分类模型。两个表示期望准备耗时{projection['seconds']:.3f}秒，没有新视觉编码或API；训练分段费用另存，不冒充端到端部署性能。

表征诊断相对固定TRAIN视觉均值计算每记录R²，仅解释线性可预测量，不表示分类收益或因果解耦。实际看图0386在约第500日的短时变化之后为平台，NCVR新增20个参考负日；0407存在两段明显持续运动，主对原拼接少8FP、多3FN，净+5日。都是事后开发者诊断，不改变既有参考，也不据平台截图否认短快速事件的过程核心。

"""
    text = text.replace("## 6 讨论", result + "## 6 讨论", 1)
    discussion = """### 6.5 去冗余不等于定位或识别收益

NCVR能预测并减去一部分视觉表示，但与原拼接、纯数值及强树比较没有总体收益。可能原因包括可预测分量包含有用运动信息，或者剩余表示仍不能精确表达目标日期；本轮没有独立干预区分这些解释。不能仅凭残差范数减小、R²为正或优于打乱对应就宣布解决平台误报。后续先检验局部日期证据能否改善识别及事件边界，再考虑一个机制，避免继续围绕同一全局向量增添归一化和融合模块。数据规模96条只说明受控覆盖，不能替代未参与选择的新记录与真实可信参考。

"""
    text = text.replace("## 7 结论", discussion + "## 7 结论", 1)
    text = text.replace(
        "## 数据、代码与AI使用说明",
        "条件化视觉残差的完整48／24实验同样未通过：主日F1 .793321低于原拼接.805651，2/7条件成立并停止。论文目前是一份可追溯研究初稿，尚缺稳定的新机制与独立确认，不能据此保证发表。\n\n## 数据、代码与AI使用说明",
        1,
    )
    text = text.replace(
        "Codex参与文献定位",
        "NCVR由run/verify/report_landslide_conditional_visual.py及landslide_conditional_visual.py复现，418配置、22分类模型、期望回归、概率及29方法评分均冻结。ResComEmb／VTBench官方原文、两个固定commit源码文件及实际阅读范围保存在conditional-visual-primary-v1／conditional-visual-code-source-v1与conditional-visual-session-v1，未运行作者模型。\n\nCodex参与文献定位",
        1,
    )
    text += "\n[15] ResComEmb. [arXiv:2609.37225](https://arxiv.org/abs/2609.37225), 2026-09-29。实际读摘要、RHC概览、coarse-anchor set与intra-resolution assessor起始，未读完全部公式，未运行作者模型。\n\n[16] VTBench. [arXiv:2604.27259](https://arxiv.org/abs/2604.27259), 2026。[作者代码](https://github.com/via-cs/vtbench)。实际读摘要／引言相关段及拼接、加权融合起始；完整静态读取fusion.py及one_chart_numerical.py，commit 7e20537261af86ca60b21c963a91ebea385316a0，未运行作者模型。\n"
    target = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v8.md"
    if target.exists():
        raise FileExistsError("Preserve the existing research draft snapshot")
    target.write_text(text, encoding="utf-8")
    print(str(target), flush=True)


if __name__ == "__main__":
    write()
