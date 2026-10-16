"""Extend the complete draft with the actual local-pair failure and source limits."""

from __future__ import annotations

from pathlib import Path

from run_landslide_window_optimization import read_json


def write():
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    receipt = read_json(base / "local-persistence-replay-v1/verification.json")
    if receipt["status"] != "VERIFIED":
        raise ValueError("Do not write verified result claims before replay")
    source = root / "docs/gnss-deformation-manuscript-draft-2026-10-12-v4.md"
    target = root / "docs/gnss-deformation-manuscript-draft-2026-10-12-v5.md"
    if target.exists():
        raise FileExistsError(target)
    text = source.read_text(encoding="utf-8")
    text = text.replace("deformation-manuscript-20261012-v4", "deformation-manuscript-20261012-v5")
    text = text.replace(
        "本版补充完整48／24跨记录分歧纠错、同信息堆叠强对照、持续期诊断和实际原图复核。",
        "本版补充完整48／24局部持续性配对、342新配置及22方法比较、近期反事实／幅度保留近邻、实际原图复核；原v4完整快照保留。",
    )
    text = text.replace(
        "本文公开完整负结果、缺测与评分契约",
        "随后保留窗口其他活动的局部持续性配对，主日F1为0.7372，低于同局部增广0.7625与同中心全窗配对0.7657；纯干扰误报53日，对同增广净损98日，停止该版本。本文公开完整负结果、缺测与评分契约",
    )
    text = text.replace(
        "These findings bound the usefulness of pairing and revision",
        "A subsequent context-preserving local-pair intervention obtains daily F1 of 0.7372 versus 0.7625 for identical local augmentation and 0.7657 for a same-center global-pair control, adding 53 nuisance-only false-positive days and failing its development criteria. These findings bound the usefulness of pairing and revision",
    )
    literature = """近期Anlu用同查询、不同正常参考规则及不同标签进行反事实监督，迫使冻结时序基础模型消费参考[11]。本轮仅实际读取其官方摘要，HTML路由返回表格片段而非完整论文，读取限制保留。CAPS和Anlu使通用上下文锚定、反事实和配对不能成为本文首创。PAI比较同表征库的余弦／欧氏分数，并补正常训练校准的median/MAD与局部均值偏离[12]；本轮读摘要、引言及3.3起始说明，源码HEAD已确认但README原文超时，未运行作者模型。PAI把幅度事件也作为异常，而本文设备偏移是另识别的干扰，不能直接把其幅度补偿移作形变优势。这些近邻支持明确任务差异和做匹配对照，不能为本文失败方案提供效果背书。

"""
    text = text.replace("## 3 任务与方法", literature + "## 3 任务与方法", 1)
    method = r"""### 3.8 保留周边运动的局部持续性配对

局部持续性配对（CP-PP，Context-Preserving Local Persistence Pairing）检验全窗负例清除其他运动是否削弱目标日期归属。仅在受控训练侧，给定原潜在形变 $m_t$、中心 $c$、端点 $u=c-21$、$v=c+21$，偏移日 $s=c-14$ 或 $s=c+14$ 交替固定，构造：

$$
r_t=\begin{cases}
m_t,&t<u\ \text{或}\ t>v,\\
m_u,&u\le t<s,\\
m_v,&s\le t\le v.
\end{cases}
\qquad \widetilde x_t=x_t-m_t+r_t.
$$

区间外观测及潜在形变保留原字节，端点相等，原误差与缺测保留。目标14日查询内没有潜在增量，但其他窗口运动可以存在。推理不读取潜在值、参考标签或干扰类型，仍使用原图和数值；反事实不是观测预处理。主复用原平衡分类及配对排序损失，与同局部增广、打乱关系、去视觉／数值比较，不再增加损失或框架。全窗对照使用相同中心和偏移日，但干预跨度造成偏移幅度不同，明确不将其称为仅改变上下文的严格单因素消融。潜在差异只限于此GNSS训练干预；共享上下文及反事实学习已有近邻。

"""
    text = text.replace("## 4 数据与实验协议", method + "## 4 数据与实验协议", 1)
    results = """### 5.10 局部持续性配对：未保住同增广收益

完整48训练／24已暴露验证。按28日原网格及原训练stride3中心选所有合格阳性，不设弱运动或形态配额，得到127对、28训练父记录、五类形变／混合干扰；快速族也参与。每对局部／全窗两反事实共254新编码，原25586中心复用。反事实不是127条新独立记录。七神经和两同增广强树各38机会，共342新配置；13原方法的训练人口、输入、缩放、共同持续性及阶段头逐值一致，455原机会保留，不重新拟合不变的原对照。

| 方法 | P | R | 日F1 | 事件F1 .3/.5 | FP/FN | 起止MAE（日） |
| --- | ---: | ---: | ---: | --- | --- | --- |
| **局部配对视觉数值** | **0.881383** | **0.633540** | **0.737188** | **0.717949/0.512821** | **151/649** | **12.000/16.000** |
| 同局部增广 | 0.951055 | 0.636364 | 0.762517 | 0.823529/0.705882 | 58/644 | 7.714/16.929 |
| 同局部打乱配对 | 0.946237 | 0.546584 | 0.692913 | 0.722222/0.611111 | 55/803 | 13.231/19.231 |
| 局部配对去数值 | 0.578829 | 0.580463 | 0.579645 | 0.578947/0.473684 | 748/743 | 24.273/35.091 |
| 局部配对去视觉 | 0.897035 | 0.614907 | 0.729648 | 0.700000/0.600000 | 125/682 | 9.071/19.286 |
| 纯数值同局部增广 | 0.938843 | 0.641446 | 0.762160 | 0.742857/0.628571 | 74/635 | 7.769/18.000 |
| 同中心全窗配对 | 0.859958 | 0.690006 | 0.765664 | 0.697674/0.511628 | 199/549 | 14.733/25.800 |
| raw84同局部增广 | 0.893983 | 0.704687 | 0.788128 | 0.780488/0.634146 | 148/523 | 10.688/23.000 |
| 图信息树同局部增广 | 0.836888 | 0.753247 | 0.792868 | 0.800000/0.700000 | 260/437 | 14.562/19.000 |
| 普通视觉数值 | 0.826113 | 0.764540 | 0.794135 | 0.761905/0.619048 | 285/417 | 11.375/20.500 |
| 原raw84 | 0.926754 | 0.678713 | 0.783572 | 0.888889/0.722222 | 95/569 | 10.563/23.313 |

主对同增广／全窗净−98／−52日，纯干扰FP53对同增广0，混合召回0.633333对0.593333。六事前性能条件仅2项成立，STOP_LOCAL_PERSISTENCE。主虽优于打乱和纯数值配对，但未优于普通增广，不能据局部有利比较升级为论文贡献；同增广树收益仍是数值对照，不更名。未调整干预跨度、偏移日、loss或seed救分，也未打开保留参考。

实际原图0385两次水平转换附近，主活动[144,163)和[271,288)多36误报日，同增广为空；原N/E转换后主要平台，不支持这些日期全为持续形变。0388过程形变与干扰同现，主较同增广补11阳性、增加6陰性，净+5，但提前扩边。原14日块与图检查只作评分后原因分析，不把潜在过程支持倒推为视觉已确认精确边界，不重写参考，也不赋新增A/S/D。

全部127局部／全窗反事实、原误差／缺测、外侧运动和目标零增量、数值／图／编码重构通过；九选中模型、157680概率、22方法／528组原日历活动／阶段及统一评分精确重放。准备总397.023秒，254编码12.797秒，原预训练视觉权重与transform一致；分段训练计时另存，不当完整部署基准。内部阶段F1主0.452666、同增广0.457209，均来自共同数值头及不同活动门控，不称新视觉阶段能力。完整22方法、八族及图见[本轮报告](../artifacts/gnss-deformation-diversity-2026-10-12/local-persistence-report-v1/report.md)。

"""
    text = text.replace("## 6 讨论", results.replace("陰性", "阴性") + "## 6 讨论", 1)
    text = text.replace(
        "继续围绕事件与平台区分开发一个更有效的小机制",
        "局部上下文持续性配对也未保住普通增广的日F1与事件收益。应继续围绕事件与平台区分开发一个更有效的小机制",
    )
    text = text.replace(
        "保留评价文件只核验哈希，不在本稿工作中加载内容。",
        "局部配对由prepare/train/verify/report_landslide_local_persistence.py复现，127组反事实、全部网格和重放保存；近期官方检索、原文阅读及访问失败另存local-persistence-session-v1。保留评价文件只核验哈希，不在本稿工作中加载内容。",
    )
    text += "\n[11] Anlu: Enabling In-Context Time Series Anomaly Detection in Foundation Models via Counterfactual Supervision. [arXiv:2610.06180](https://arxiv.org/abs/2610.06180), 2026-10-05. 本轮实际读取官方摘要，HTML正文不可用，未运行作者模型。\n\n[12] PAI: Preserving Amplitude Information in Representation-Based Time-Series Anomaly Detection. [arXiv:2606.08935](https://arxiv.org/abs/2606.08935), 2026. [作者源码](https://github.com/pantheon5100/PAI). 实际读取摘要、引言及3.3起始说明，固定源码HEAD已确认，README原文超时，未读实现或运行模型。\n"
    target.write_text(text, encoding="utf-8")
    print(target, flush=True)


if __name__ == "__main__":
    write()
