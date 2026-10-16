"""Write the fixed residual candidate result without replacing stopped decisions."""

from __future__ import annotations

from probe_landslide_endpoint_fit import BASE, ROOT, csv_rows, export
from report_landslide_core_lockbox import full_table
from run_landslide_window_optimization import read_json


def write():
    run = BASE / "visual-residual-development-v1"
    report = BASE / "visual-residual-calibration-report-v2"
    archive = BASE / "manuscript-v46"
    preservation = read_json(archive / "v45-preservation-before.json")
    if not preservation["all_match"]:
        raise ValueError("The previous delivery must be preserved before writing")
    decision = read_json(run / "calibration-decision.json")
    if decision["decision"] != "STOP_VISUAL_RESIDUAL_CALIBRATION":
        raise ValueError("This writer documents only the stopped fixed candidate")
    manuscript_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    if manuscript_path.read_bytes() != (archive / "landslide-adaptive-review-manuscript-draft-v45-original.md").read_bytes():
        raise ValueError("Write exactly once from the archived v45 manuscript")
    manuscript = manuscript_path.read_text(encoding="utf-8")
    mechanism = r"""
保留固定强XGB84数值概率，不让联合视觉网络从头重学数值判断。令 $p_{\mathrm{num},t}$ 为数值初判，先裁到 $[10^{-6},1-10^{-6}]$，将裁剪后的概率记为 $\bar p_{\mathrm{num},t}$，其对数优势为：

$$
\ell_{\mathrm{num},t}=\log\frac{\bar p_{\mathrm{num},t}}{1-\bar p_{\mathrm{num},t}},
\qquad
\ell_t=\ell_{\mathrm{num},t}+\log(4)\tanh r_\theta(I_t,z_t,\ell_{\mathrm{num},t}),
\qquad
p_t=\frac{1}{1+\exp(-\ell_t)}.
$$

残差最后一层零初始化，使初始概率等于裁剪后的数值概率；树权重不参与神经优化。相对于裁剪后的anchor，最终优势比只可乘以 $[1/4,4]$：

$$
\frac{1}{4}\leq
\frac{p_t/(1-p_t)}
{\bar p_{\mathrm{num},t}/(1-\bar p_{\mathrm{num},t})}
\leq4.
$$

这个结构限制修改幅度，不保证准确、置信度校准或未知区域可靠。图像沿用32×61六通道原始三轴图，无反射/删除增量；数值沿用18原始多尺度特征和18不可用标志。新辅助仅为anchor logit，图像和去图像头同样得到它。每训练日使用本记录完全未进入anchor拟合的概率：过程24训练按原顺序每6条一折、四折，每折六家族各一；作者6训练每2条一折、三折。验证使用全部训练记录重新拟合的固定depth4 anchor。不做逐日随机交叉验证，不使用验证标签生成训练anchor。

新四臂为有界视觉OOF主模型、去图像有界数值残差、无界视觉OOF、用in-sample替代OOF。各两宽16/32、固定12epochs、seed20261012、两CPU线程、正类加权BCE；两CNN沿用8/16通道。纯视觉、普通融合及纯数值MLP合法复用相同原训练日/归一化下的两宽冻结模型；纯视觉沿用之前登记的反射一致性训练，明确不是新训普通纯图像臂。数值anchor独立控制depth2/4；主残差anchor固定depth4，不依据验证分数切换。另拟合获得18数值+366原画布列位置/可用性的384维XGB，不反射增广。

所有九臂都得到相同19阈值0.05至0.95，合计每臂38个机会，仍以日F1+事件F1选首个最大值。扩大网格对所有控制同样开放，不修改旧六机会结果；该目标可能选较低日F1但较高事件F1的配置，不按日F1另选“最佳”结果。同训练/验证划分、支撑、14日持续性和冻结阶段头保持。新16神经、15数值拟合（包括7折anchor）及684配置全部保存。
""".strip()
    review = """
## 实际视觉与原始数值复核

当前Codex实际看四过程和两作者原始NEU/概率/逐日输出图，并读取对应14日原始中位块；按评分后相对数值anchor/无图像残差的净纠错极值选图，仅解释机制，不调整门槛或标签。作者已知历史研究，非独立领域专家。过程参考精确潜在机制不等于可观测起止，图上边界不能被解释成精确物理真值。

- 0028：N/E在约130–300日逐渐累积后转平台，U同段下降。N原块中位112–126日−0.773mm、126–140日0.028、140–154日3.545、154–168日6.690，显示渐进起变。主相对anchor漏[125,135)十日，仅恢复一日静稳，净−9；对无图像净−17。局部画布的较保守起点没有证明边界改善。
- 0036：前段较强运动后，后半N仍缓慢下降，U约465日单次偏移，E较平。504–518日至588–602日N中位−90.661→−93.485mm，持续弱运动有原始依据。主对anchor净+17，排[439,464)提前误报，但又漏[671,682)弱尾；对无图像仅净+1，并损[485,496)。不能把排偏移误报全部归功于视觉，因为anchor及无图像本就保持指定阶跃家族零误报。
- 0034：N/E约130–300日持续累积后平台，缺测不插值。原126–140日NE中位1.255/2.442mm，154–168日2.979/9.354；140–154日仅4个完整观测，中位不报告。主对anchor净−6、无图像−7，集中起止，保留缺测未知。
- 0030：前强后弱两段累积，E后段持续下降；原462–476日E中位−86.523、504–518日−89.062、532–546日−90.956mm。主对anchor恢复尾段[646,655)九日，对无图像净+8；单例有益不抵消总体弱召回略下降。
- 0050作者参考：约204日N/E单次偏移，之后平台；210–224日N中位−19.884、224–238日−20.005、238–252日−20.070mm。主相对anchor新增37误报日，包括[189,195)、[215,229)、[531,548)；后者N原块约−19.843→−19.071mm，在噪声背景下未见与主确定活动相称的持续累积。较低阈值把边缘概率变成活动，幅度约束本身不保护最终判断。
- 0057作者参考：N中段缓慢下降，E上升，U较噪。N原块224–238日−2.499、280–294日−4.845、336–350日−6.131mm，有跨块持续变化。主对anchor/无图像恢复81日，连接其原断裂；但两组整体净纠错均0，其他记录新增误报抵消这项改善。

先确认原始持续位移，再解释区间内阶段。参考未知无正确静稳信用，缺测不赋活动或A/S/D；阶段头未创新，指标变化来自最终活动门控。已有作者标签不升级成独立专家参考。
""".strip()
    (report / "visual-review.md").write_text(review + "\n", encoding="utf-8")
    costs = csv_rows(run / "training-cost.csv")
    neural_seconds = sum(float(row["wall_seconds"]) for row in costs
                         if row["arm"] in ("bounded_visual_oof", "bounded_numeric_oof",
                                           "unbounded_visual_oof", "bounded_visual_in_sample"))
    tree_seconds = sum(float(row["wall_seconds"]) for row in costs
                       if row["arm"] in ("numeric_anchor", "anchor_oof", "xgboost_chart_information"))
    summaries = {target: csv_rows(report / target / "summary.csv")
                 for target in ("primary_process", "secondary_developer")}
    rows = [
        "# 固定数值初判与有界视觉修正：校准否定结果", "",
        "## Material Passport", "",
        "- Origin Skill: academic-research-suite / experiment-agent；run/validate inline。",
        "- 日期：2026-10-11；状态：训练、固定校准、冻结权重/OOF重放、视觉数值复核完成。",
        "- 范围：原过程24训练/12验证及作者6训练/6验证；开发暴露材料，非现场独立确认。",
        "- 决定：STOP_VISUAL_RESIDUAL_CALIBRATION；不提升较好控制，不调此候选救分。", "",
        "结论：保留数值初判后的主过程F1 .919160，略高于anchor .917346、对其净+13日；"
        "但低于去图像残差.923920、对其净−15日，弱召回也略低于anchor。"
        "只5/9冻结条件通过，尚未证明视觉独立增量。作者F1 .921933高于anchor .913682，"
        "但90新增正确活动日与90新增静稳误报抵消，净0；不能把F1提高等同净可靠性提高。", "",
        "## 最小机制、交叉拟合与公平控制", "", mechanism, "",
        "## 全部统一校准指标", "",
    ]
    for target, table in summaries.items():
        rows.extend(["### " + target, "", full_table(table), ""])
    rows.extend([
        "过程弱参考839日：主692/839=.824791，anchor694/839=.827175，"
        "去图像710/839=.846246；普通融合.761621，纯图像.917759。"
        "纯图像弱R较高，但FP388，主109、anchor145、去图像143，不能单一指标替换主准入。"
        "指定instrument_step两记录FP主/anchor/去图像均0，普通73、纯图像188；"
        "这不涵盖quiet背景偏移，也不证明视觉提供该零误报。", "",
        "主对普通融合净+239，对数值MLP+202，对anchor+13、去图像−15、in-sample−4、"
        "无界+17、同信息XGB+99。主日F1 .919160略高于无界.914428，"
        "但低于in-sample .919911，交叉拟合作用没有稳定得到支持。"
        "作者对anchor/去图像/in-sample/无界均净0，对同信息XGB−39。"
        "作者主FP90/FN57，anchorFP0/FN147，同信息XGBFP30/FN78。", "",
        "各自调阈值后的纠错包含校准选择，不全部归为视觉表示或OOF的因果效应。"
        "作者主选阈值.05、anchor.35，过程主.60、去图像.55、anchor.35；"
        "更多阈值机会甚至使某些日F1低于旧六机会结果，因为目标含事件F1。"
        "同信息XGB本轮没有反射或替代图增广，与前两轮不同；不得把跨轮分数下降称为新机制收益。"
        "原108/96完整开发规则、PELT、强XGB、直接VLM、固定/Agent对照与成本均保持；"
        "校准人口不同，本轮不拼入完整开发表。固定候选停止，未跑完整开发或新独立确认。", "",
        review, "",
        "## 文献、验证与成本", "",
        "近期定向检索与相关源码阅读见[实际阅读范围]"
        "(../artifacts/landslide-adaptive-review-2026-10-09/visual-residual-literature-v1/reading-notes.md)。"
        "Stacked generalization仅元数据；DeepGBM摘要/README/两短源码；"
        "VETime官方摘要、§3.1和§3.5–3.6、固定SHA主模型前87行及路由相关源码。"
        "未运行这些作者模型，不称残差/OOF/多模态融合首创。"
        "检索发生于冻结以后，未追溯宣称据其预登记。", "",
        "18个选中模型及7折数值权重重放：逐日概率/预测、全部选中聚合分数、"
        "9217个训练anchor日逐值精确一致，无记录交叉；118260日历输出支撑/缺测/阶段门控通过，"
        "有限logit修正契约通过。仅重放冻结权重，未重训或证明全训练逐字节重现。"
        "3新业务测试加3旧输入测试通过；评分器回归检查单独保存。", "",
        "第一次报告在两目标重放完后导出成本CSV失败：第一行缺case_id。"
        "失败v1及异常说明/manifest保留，修正行结构后在v2重放成功；"
        "训练权重、选择、684配置和准入未改。", "",
        f"16神经训练循环墙钟{neural_seconds:.6f}秒；15数值拟合/预测计时段{tree_seconds:.6f}秒。"
        "这些不是完整执行CPU或端到端耗时。预处理另存，推理核验CSV还含契约复算，"
        "不冒充纯推理重复基准；残差需要固定anchor和一次神经前向，纯数值不需图像编码。", "",
        "新增API/记录0，累计仍3825启动/3823回执、至少16694503报告token；"
        "受控192/半合成60不变。原参考不改，不新增伪专家标签。", "",
        "复现：在PYTHONPATH=src;scripts、PYTHONIOENCODING=utf-8下用"
        "artifacts/reproduction-2026-10-07/environment/Scripts/python.exe执行"
        "probe_landslide_visual_residual.py prepare/calibrate --out 新目录；"
        "report_landslide_visual_residual.py --run 冻结训练目录 --out 新报告目录。"
        "历史source-bindings要求原文件原字节；重放读取已冻权重，训练源码存档。", "",
        "goal active：完成一轮实际机制/消融及可追溯否定证据，未形成稳定贡献或独立可信确认。"
        "下一步先检查视觉时间定位和弱运动证据的获取，不继续修改此残差幅度/阈值救分，"
        "也不把更强纯数值包装为Agent/VLM。", "",
    ])
    (ROOT / "docs/landslide-visual-residual-results.md").write_text("\n".join(rows), encoding="utf-8")
    section = "\n### 6.4.10 固定数值初判与有界视觉残差的校准探索\n\n" + mechanism + """

过程主F1 .919160，数值anchor .917346、去图像.923920、无界.914428、in-sample .919911、普通融合.864181、纯图像.887010；对anchor净+13，但对去图像−15、in-sample−4。弱召回主692/839=.824791低于anchor694/839=.827175。指定阶跃家族主/anchor/去图像均零误报，不证明视觉的独立排阶跃作用。作者主.921933、anchor.913682、同信息XGB.940067；主新增90活动正确日却同时新增90静稳误报，净0。5/9准入通过，STOP_VISUAL_RESIDUAL_CALIBRATION；不调幅度/seed/损失/阈值/子集，不升级更好消融为核心贡献。详见[全部对比、成本和复现](landslide-visual-residual-results.md)。

实际复核四过程/两作者原三轴和14日原始块：0028渐进起点损10日、0034起止净−6；0036排提前误报且丢弱尾，净+17；0030弱尾净+9。作者0057恢复81日，0050平台新增37误报。净收益不稳定；有限优势比不等于有限错误风险。旧标签/未知/活动内阶段规则保持，作者已知历史结果，非独立专家。18选中模型、7折anchor精确重放，9217训练anchor日及118260输出日契约通过。报告v1仅成本表字段失败，保留后修复到v2，未重训或改评分。

OOF/stacking[33]及树与NN结合[34]已有。VETime[35]已有时间/视觉对齐与任务融合；本候选保留固定数值logit并限制修正，技术区别未获得独立识别增益。来源阅读范围有限，未运行作者模型；不宣称通用残差或融合首创。此轮止于校准，未进入完整开发/未见确认，旧108/96完整开发对照与停止决定保持。

"""
    manuscript = manuscript.replace("完整中文初稿v45", "完整中文初稿v46", 1)
    old = "更新前v44主稿及三状态文档原字节存档。新增单主导跳变的有限替代图训练、固定/去一致性/去数值消融及同信息XGB、权重重放与六图视觉数值复核；候选停止于校准。"
    new = "更新前v45主稿及三状态文档原字节存档。新增固定数值初判、有界视觉残差、按记录OOF与消融、同等阈值机会及六图视觉数值复核；候选停止于校准。"
    if old not in manuscript or "### 6.4.10" in manuscript:
        raise ValueError("Expected v45 header and unused section identifier")
    manuscript = manuscript.replace(old, new, 1)
    manuscript = manuscript.replace("### 6.5 阶段与计算成本", section + "### 6.5 阶段与计算成本", 1)
    manuscript += """

[33] Wolpert, D. H. (1992). Stacked generalization. Neural Networks. [DOI](https://doi.org/10.1016/S0893-6080(05)80023-1). 本轮仅核对OpenAlex题名/作者/年/DOI；无摘要，未取得或阅读全文。

[34] Ke, G., Xu, Z., Zhang, J., Bian, J., & Liu, T.-Y. (2019). DeepGBM: A Deep Learning Framework Distilled by GBDT for Online Prediction Tasks. KDD, 384–394. [DOI](https://doi.org/10.1145/3292500.3330858). [作者源码](https://github.com/motefly/DeepGBM)。摘要/README与固定SHA6926d35bdcc3249befc127973d313e3c061f2bb2的deepgbm.py、gbdt2nn.py全短文件已读；未阅读全文或运行作者模型。

[35] Yang, Y., Lan, T., Gao, Y., Lu, Y., He, W., Wang, M., Liu, C., & Zhang, C. (2026). VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection. [arXiv:2602.16681v1](https://arxiv.org/abs/2602.16681v1). [作者源码](https://github.com/yyyangcoder/VETime)。官方摘要、§3.1及§3.5–3.6相关部分、README前100行与固定SHA7b81a7efbae4760eafc438ffe32fa19df8de7465的主模型前87行及router/M_moe源码已读；非全文阅读、未运行作者模型。
"""
    manuscript_path.write_text(manuscript, encoding="utf-8")
    status = """
## 2026-10-11：固定数值初判与有界视觉修正停止于校准，初稿v46

[完整结果](landslide-visual-residual-results.md)。新最小候选保留XGB84概率，零初始化CNN残差并限制优势比×1/4至4；按整记录4/3折生成训练OOF，验证使用全训练anchor。去图像/去幅度约束/in-sample消融、普通融合/纯图像/纯数值及同信息XGB共九臂；原24/12与6/6，所有臂同19阈值×2宽或深=38机会，日F1+事件F1选择。新16神经/15数值拟合、684配置保存，未修改旧六机会结果。

过程主F1 .919160，anchor .917346、去图像.923920、无界.914428、in-sample .919911；净纠错对anchor+13、去图像−15、in-sample−4。弱R .824791低于anchor.827175；5/9准入通过，STOP_VISUAL_RESIDUAL_CALIBRATION。作者主.921933高于anchor.913682但新增90正确活动/90静稳误报、净0。同信息XGB作者.940067。更多阈值与事件选择会改变结果，不把跨轮差异全部归视觉。旧108/96完整开发强基线/VLM/固定/Agent指标/成本及停止决定保持。

实际六图及原14日块复核：0028渐进起点损10日，0034起止净−6；0036排提前误报但丢弱尾、净+17；0030尾段净+9；0057恢复81日，0050平台新增37误报。作者非独立专家，参考不改。有限logit修正不是准确/校准保证。Stacking仅元数据；DeepGBM摘要/两短源码、VETime部分方法与固定SHA路由源码已读，未运行作者模型，不称融合/OOF首创。

18选中模型及7折anchor逐日概率/预测/聚合精确重放，9217训练anchor日、118260输出日支撑/缺测/阶段门控通过。报告v1末尾成本CSV字段失败保留，修正v2成功；无重训/参数/评分修改。3新+3旧输入测试通过，评分器回归及Ruff另存。分段训练墙钟/含契约推理核验成本公开，非端到端CPU；不声称全训练字节复现。

新增API/记录0，累计仍3825启动/3823回执、至少16694503报告token，受控192/半合成60不变。v45四文档原字节存档、265文件旧manifest保持并按档案映射核验。v46公式继续LaTeX。goal active：实际方法/消融与否定证据完成，稳定实质贡献和独立可信确认仍未完成。后续研究视觉时间定位与真实弱运动证据，不调此候选救分。
"""
    for filename in ("project-status.md", "landslide-adaptive-review-method.md", "landslide-submission-roadmap.md"):
        path = ROOT / "docs" / filename
        archived = archive / (path.stem + "-v45-original.md")
        if path.read_bytes() != archived.read_bytes():
            raise ValueError("Status documents must still match their archived version")
        heading, remainder = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(heading + "\n\n" + status.strip() + "\n\n" + remainder.lstrip("\n"), encoding="utf-8")
    export(archive / "run-result.json", {
        "status": "progress", "goal_complete": False, "candidate_decision": decision["decision"],
        "new_api_calls": 0, "new_records": 0,
        "completed": ["Method implementation", "Frozen grouped calibration and ablations",
                      "Selected weights and OOF replay", "Raw visual/numerical review",
                      "Targeted literature/source verification", "LaTeX Markdown manuscript update"],
        "failed_attempt_preserved": "visual-residual-calibration-report-v1",
    })


if __name__ == "__main__":
    write()
