"""Add the completed pooling pilot to the preserved full research manuscript."""

from __future__ import annotations

from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json
from write_landslide_distribution_manuscript import score_table

from gnss_sim.artifacts import sha


def write():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    run = base / "evidence-pooling-calibration-v1"
    replay = base / "evidence-pooling-replay-v1"
    report = base / "evidence-pooling-report-v1"
    for folder in (run, replay, report):
        verify_manifest(folder, "freeze.json")
    receipt = read_json(replay / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["run_manifest_sha256"] != sha(run / "freeze.json"):
        raise ValueError("A matching successful replay is required")
    previous = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v9.md"
    frozen = read_json(base / "manuscript-v9/evidence-bindings.json")
    if sha(previous) != frozen[str(previous)]:
        raise ValueError("The preceding manuscript must remain unchanged")
    text = previous.read_text(encoding="utf-8")
    text = text.replace("2026-10-13，v9。", "2026-10-13，v10。", 1)
    text = text.replace("deformation-manuscript-20261013-v9", "deformation-manuscript-20261013-v10", 1)
    text = text.replace(
        "本版保留NCVR完整结果，补日期排列证据利用探针的五臂190配置、强对照与普通视觉融合负结果。",
        "本版补连续位移证据引导视觉池化（SCVP）的16／8开发小试、九臂342配置、强数值对照、消融及原图误差分析。",
        1,
    )
    summary = read_json(run / "summary.json")
    rows = {row["method"]: row for row in summary}
    main = rows["continuity_pool_numeric"]
    uniform = rows["uniform_pool_numeric"]
    strongest = rows["pilot_global_mass_xgb"]
    abstract = (
        f"进一步在16训练／8暴露验证小试中，使用相邻同向位移变化对原冻结视觉特征图加权池化，"
        f"主日F1为{main['f1']:.4f}，低于均匀池化{uniform['f1']:.4f}与强数值树{strongest['f1']:.4f}；"
        "对均匀池化净纠错−3日，六条件仅三项成立，停止该固定版本。"
    )
    text = text.replace("本文报告可追溯信息收益及机制负结果", abstract + "本文报告可追溯信息收益及机制负结果", 1)
    text = text.replace(
        "The reserved records remain unopened;",
        f"A separate 16/8 development pilot tests Sustained-Change Visual Pooling (SCVP), "
        f"which weights the original frozen feature map using consecutive same-direction displacement changes. "
        f"Its daily F1 is {main['f1']:.4f}, below uniform pooling at {uniform['f1']:.4f} "
        f"and a strong numerical tree at {strongest['f1']:.4f}. It introduces three net additional "
        "day errors versus uniform pooling and meets three of six predeclared conditions; the fixed version is stopped. "
        "The reserved records remain unopened;",
        1,
    )
    related = (
        "MedTVL[17]已用文本自适应引导时间与视觉双路径，再作实例级专家融合；"
        "本文实际阅读3.3、3.4相关内容及固定作者commit三个完整静态文件，未运行作者模型。"
        "VESTA[18]以统计工具支持视觉模型探索、拟合与修订，本文读摘要、第3节定义与3.1及算法起始，"
        "未读完全部方法、未核查其源码或运行作者系统。它们分别是跨模态引导与工具复核的近邻，"
        "任务并非GNSS形变活动。SCVP的差异是假设相邻持续位移能够指导已有曲线特征图的读取；"
        "条件化、加权池化与多模态融合本身不称首创。该假设未获本轮整体识别结果支持。\n\n"
    )
    text = text.replace("## 3 任务与方法", related + "## 3 任务与方法", 1)
    method = r"""### 3.14 连续位移证据引导视觉池化（SCVP）

SCVP（Sustained-Change Visual Pooling）保持原三轴图、冻结MobileCLIP2-S0骨干和预训练投影不变，只检验最终特征图的读取方式。原13个14日块产生12条噪声归一化三轴变化；块缺测时不跨接求导。对相邻两条同轴变化，连续证据为：

$$
c_{t,k,a}=
\begin{cases}
\dfrac{2|\Delta_{t,k,a}||\Delta_{t,k+1,a}|}
{|\Delta_{t,k,a}|+|\Delta_{t,k+1,a}|},
& \text{两边可用，同号，且分母}>10^{-12},\\
0,&\text{其他情况}.
\end{cases}
$$

理想单次偏移即使净位移相同，也没有相邻持续变化；真实噪声会产生非零证据，不能将该量直接解释为活动标签。将11个连续量按原日期映射到256像素图的8列，并按NEU绘图区域与特征行的重叠比例累积为粗权重。令 $C_{t,a,j}$ 为映射量，$r_{i,a}$ 为特征行与原绘图区域的重叠比例：

$$
w_{t,i,j}=\frac{1+\sum_a r_{i,a}C_{t,a,j}}
{\sum_{i',j'}(1+\sum_a r_{i',a}C_{t,a,j'})},
\qquad
z_t^C=\operatorname{norm}_2\!\left(P\sum_{i,j}w_{t,i,j}F_{t,:,i,j}\right).
$$

$F_t$ 为原1024通道、8×8最终特征图，$P$ 为原1024到512维投影，均匀底权重固定为1。权重为零的连续证据退化到原全局池化；最终特征已混合上下文，不能将粗读出权重解释为精确日期定位、因果注意力或归因证据。原全局512维仍保留，主活动头输入为：

$$
p_t=\sigma\!\left(h_\theta([z_t^V;z_t^C;u_t;m_t])\right).
$$

共同数值 $u_t$ 共87维：原18项、全局运动分布21项、连续guide24项、绝对变化guide24项；$m_t$ 为87个缺值指示。因此四主要融合臂均1198维。均匀控制重复原全局编码，打乱控制每查询打乱64个已归一化权重并保持其多集，跳变控制用绝对变化替代连续证据。去显式数值分支仍通过池化权重使用原位移，不能称纯视觉；仅均匀池化视觉组去除所有数值反馈。再加入同87数值MLP与87／505维强树，后者额外读取完整原图描述符及原块profile。

固定人口为已有八族16TRAIN／8暴露VALIDATION，不是新增或独立评价。七神经与两树各38调参机会，共342新配置，加21保留基线为30统一方法。训练中心、顺序、seed、12epoch、16／32隐藏宽度及阶段头保持共同协议。只有在主日F1胜均匀／打乱／跳变、三者净纠错为正、事件不损、纯干扰FP不增加、混合形变召回不损及相对最强同信息数值有视觉价值六项全部成立后，才扩完整48／24；失败停止固定版本，不搜底权重或位置救分。

"""
    text = text.replace("## 4 数据与实验协议", method + "## 4 数据与实验协议", 1)
    methods = (
        "continuity_pool_numeric", "uniform_pool_numeric", "permuted_pool_numeric",
        "jump_pool_numeric", "continuity_pool_no_branch", "uniform_pool_visual",
        "guide_numeric_mlp", "guide_information_xgb", "all_guide_information_xgb",
        "pilot_global_mass_xgb", "global_numeric", "xgb_raw84", "robust_rule", "pelt_rule",
    )
    table = score_table(summary, methods)
    results = """### 5.16 SCVP：混合活动召回改善未转化为整体收益

下表仅比较同16TRAIN／8暴露VALIDATION小试，不能把其高分与48／24的分数直接横比。完整30方法、八族分项、边界、阶段、缺测与费用见冻结报告。

""" + table + """

主Precision／Recall为0.9482／0.8941，均匀池化为0.9816／0.8676。主多识别13个活动日，却增加16个误报日；因此日F1为0.9203，低于均匀0.9211、打乱0.9222和跳变0.9232。四者事件F1均0.8889；主起止核心MAE为4.25／4.75日，均匀为4.75／3.50日，不能称整体边界改善。强数值分布树日／事件F1为0.9455／1.0000，同87数值MLP为0.9324／1.0000。六预设条件通过3/6，STOP_EVIDENCE_POOLING_PILOT；对均匀／打乱／跳变净纠错−3／−4／−3日，不扩大本版。

纯静稳／偏移／尖峰三条记录各方法形变FP为零，不能据此说干扰识别问题已解决；主在形变叠加尖峰记录case_0389净纠正8日，在再活动case_0390净增加8日错误，在单次形变case_0395增加3日错误。主混合形变召回0.8100高于均匀0.7692，但去显式数值分支日F1为0.8279，还低于纯视觉0.8435。改善集中在既有活动边界附近；本轮没有额外评价独立干扰类型识别器，也没有改写参考或用这些曲线赋确定A/S/D。

实际查看case_0389原三轴图，E向持续变化与孤立尖峰可以区分，主比均匀更早进入活动；查看case_0390，N向两次持续运动与中间平台清晰可辨，主扩大第一段活动边界。具体纠错日及逐族错误以绑定CSV和复核记录为准，目测不单独确定起止日。验证组连续池化对原全局编码的逐记录中位余弦多数约0.997—0.999，有效权重token数约61—63／64；case_0390约0.973及34／64。这说明池化改变通常较小、在再活动上较大，却不证明差异幅度是失败原因，更不支持事后调底权重。

24原记录、连续／跳变guide、三类权重和TRAIN缩放逐值重构；216个辅助编码抽样最大误差1.19×10⁻⁷；9选中模型52560概率、30方法240输出及统一活动／事件／边界／阶段评分精确重放。没有重训全部18分类器或全部重编码。编码准备8526次原图前向、25578个辅助池化向量、430.504秒，包含全部消融准备，不能称主方法部署成本。主推理只需一个原图骨干前向、两池化投影及活动头，无新语言请求。首次业务测试因float32求和误差1.192×10⁻⁷超过默认容差而失败，仅将单位和测试绝对容差改为2×10⁻⁷，算法未变，随后3项业务测试通过。

"""
    text = text.replace("## 6 讨论", results + "## 6 讨论", 1)
    text = text.replace(
        "## 7 结论",
        "### 6.6 连续引导不能替代可纠正的新增视觉证据\n\n"
        "SCVP用一项物理连续性机制读取原特征图，完成匹配容量与同信息强对照，"
        "但主指标未提高，当前不作为论文的有效创新。辅助guide区分理想跳变的业务测试通过，"
        "也不能替代噪声观测上的识别验证。下一步需先证明局部观察能解决强数值仍错的具体活动／边界，"
        "再实施有界视觉复核，避免继续变换同一表征而没有新增纠错证据。\n\n## 7 结论",
        1,
    )
    text = text.replace("## 数据、代码与AI使用说明", abstract + "\n\n## 数据、代码与AI使用说明", 1)
    text = text.replace(
        "Codex参与文献定位",
        "SCVP由landslide_evidence_pooling.py及encode/train/verify/report_landslide_evidence_pooling.py复现；"
        "encoding/calibration/replay/report-v1分别保存协议、缓存、342网格、18训练模型、预测及30方法评分。"
        "MedTVL／VESTA官方原文、MedTVL固定作者源码与实际阅读、事后看图范围保存在"
        "evidence-pooling-primary/code-source/session-v1，未运行两作者模型。没有新增记录或打开24保留内容。\n\n"
        "Codex参与文献定位",
        1,
    )
    text += (
        "\n[17] Ye J, Li J, Tsung F. MedTVL: Harnessing Vision and Language for Medical Time Series Classification. "
        "[arXiv:2608.28605](https://arxiv.org/abs/2608.28605)。[作者代码](https://github.com/start2020/MedTVL)。"
        "按已核查预印本引用；官方元数据列KDD2026，但未另核出版者。固定源码commit "
        "4f05ba8ff55a8387b95dcee38b0e20a6c99aa1d7；未运行。\n\n"
        "[18] Rudman W, et al. VESTA: Visual Exploration with Statistical Tool Agents. "
        "[arXiv:2606.00384v2](https://arxiv.org/abs/2606.00384v2)。"
        "[作者代码链接](https://github.com/wrudman/VESTA)已定位，未读取源码或运行。\n"
    )
    target = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v10.md"
    if target.exists():
        raise FileExistsError("Preserve the preceding complete draft")
    target.write_text(text, encoding="utf-8")
    print(str(target), flush=True)


if __name__ == "__main__":
    write()
