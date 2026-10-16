"""Document the frozen date-aligned readout result and preserve earlier decisions."""

from __future__ import annotations

from probe_landslide_endpoint_fit import BASE, ROOT, csv_rows, export
from report_landslide_core_lockbox import full_table
from run_landslide_window_optimization import read_json


def write():
    run = BASE / "temporal-readout-development-v1"
    report = BASE / "temporal-readout-calibration-report-v1"
    archive = BASE / "manuscript-v47"
    preservation = read_json(archive / "v46-preservation-before.json")
    if not preservation["all_match"]:
        raise ValueError("Archive the preceding delivery before writing")
    decision = read_json(run / "calibration-decision.json")
    if decision["decision"] != "STOP_TEMPORAL_READOUT_CALIBRATION":
        raise ValueError("This writer documents only the stopped fixed candidate")
    manuscript_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    if manuscript_path.read_bytes() != (archive / "landslide-adaptive-review-manuscript-draft-v46-original.md").read_bytes():
        raise ValueError("Write only once from the archived v46 source")
    mechanism = r"""
本轮检验“窗口里发生过运动”和“目标日仍在运动”的时间对应关系。沿用原始三轴六通道32×61画布，每列是至少2个完整观测支持的3日中位数。对以原日 $t$ 为中心的图，第 $j$ 列严格对应：

$$
q_{t,j}=t+3(j-30),\qquad j=0,\ldots,60.
$$

两层3×3图像CNN仅沿高度以2步长下采样，时间步长为1；高度池化到2行、仍保留61列。随后两层一维时间卷积，kernel9、dilation1/4，输出每列16维图像特征。每列接其自身原日的18数值特征+18不可用标志及所有臂都有的相对图内位置。共享单头分别输出61列活动概率，推理只读中心第30列；没有重复问答、教师、残差anchor、反射或删除跳变。图像不沿时间插值，也不补原缺测。

训练中心仍是每2日一个原支撑且已知的训练日。辅助列的标签必须是该列原日已有标签，不取位移3日中位数的“多数标签”，也不据导数生成标签。有效监督掩码为：

$$
M_{t,j}=\mathbf1[
0\leq q_{t,j}<T,\ 
\operatorname{support}(q_{t,j})=1,\ 
y_{q_{t,j}}\in\{0,1\}].
$$

令 $\mathcal B$ 为本批合法训练中心，$B_w$ 为使用同一原训练总体正类权重的BCE。中心不重复计入辅助项：

$$
\mathcal L_{\mathrm{center}}
=\frac{1}{|\mathcal B|}
\sum_{t\in\mathcal B}B_w(p_{t,30},y_t),\qquad
\mathcal L_{\mathrm{aux}}
=\frac{\sum_{t\in\mathcal B}\sum_{j\neq30}
M_{t,j}B_w(p_{t,j},y_{q_{t,j}})}
{\sum_{t\in\mathcal B}\sum_{j\neq30}M_{t,j}},
\qquad
\mathcal L=\frac{\mathcal L_{\mathrm{center}}+\mathcal L_{\mathrm{aux}}}{2}.
$$

无有效辅助列的批次退回中心损失。标签来自同一训练记录，外侧、未知与无支撑不监督。新增多列是已知原标签的重复使用，并非新独立样本；过程408743、作者75156个辅助标签出现次数不得充作样本量。

五神经臂为主逐列融合、只监督中心（去辅助监督）、全窗池化后广播相同逐列数值/位置与监督（去逐列图像对齐）、逐列纯图像（去数值）、同逐列监督的纯数值。主与去对齐使用相同图像卷积前端；对照将其压成2×5全图向量，按原日/图内位置答各列。原过程24训练/12验证、作者6/6、两宽16/32、12epochs、seed20261013、两CPU线程、同19阈值0.05–0.95。每臂38机会，仍按日F1+事件F1选首个最大值。

密集监督会接触更多原训练日期，因此重新拟合18维XGB84及获得全部原图列位置/可用性的384维XGB，均使用全部已知原支撑训练日：过程15319、作者3118日。两深2/4、120树、lr.05、seed20261007，同38机会。没有以不适用的旧stride2模型冒充公平强基线。原活动支撑、14日持续性、共同冻结阶段头和缺测/未知契约保持。
""".strip()
    review = """
## 实际视觉与原始数值复核

实际查看四过程及两作者的原始NEU、活动概率、冻结逐日对齐图，并读取原14日数值块；按评分后相对纯数值/强XGB的净纠错极值选图，仅诊断，不修改参考或机制。作者已知历史结果，非独立领域专家。

- 0033：N/E约210–410日缓慢累积，U噪声较大。原350–364日E中位−9.303mm，378–392日−10.739、392–406日−11.402、406–420日−12.161，说明被删尾部仍有持续变化。主对纯数值净−38、强XGB−39，丢起[210,219)及尾[373,402/403)；图像日期对齐没有保住弱首尾。
- 0036：前强后弱，N后半仍缓慢下降，U约465日偏移。主对纯数值净+33，主要排[438,470)提前误报；对强XGB却净−78，在[496,524)、[525,551)、[552,587)等弱运动区域断裂。原N中位504–518日−90.661→588–602日−93.485mm；排偏移与漏持续运动同时存在。原缺测与无特征支撑保持未知，不能把全部断裂称真实停滞。
- 0034：N/E渐进起动后进入平台；126–140日N/E中位1.255/2.442mm，154–168日2.979/9.354，140–154日只有4完整观测，不报告中位。主对纯数值净−28、强XGB−39，丢渐进起点和结束附近；从对齐图看不到稳定边界增益。
- 0035：两段明显累积，中间平台；238–252日NE中位28.616/84.757，252–266日28.521/84.569。主对纯数值净+24、强XGB+13，撤早期弱证据误报和平台尾误报，但丢[238,248)并较强XGB漏第二段起点[471,474)。单例平台修正不支持总体提升。
- 0057作者参考：N中段缓降、E缓升，U较噪。原224–238日N中位−2.499、280–294日−4.845、336–350日−6.131mm。主对强XGB净+21，恢复部分段并撤[474,480)误报；对纯数值却净−6、仍有断裂。该记录的原作者A/S/D全未知，不凭活动恢复追加确定阶段。
- 0050作者参考：约204日单次NE偏移后平台，原210–224日N中位−19.884、224–238日−20.005、238–252日−20.070mm。主对强XGB新增38误报日，包括[189,195)、[215,229)、[536,554)；对纯数值净+11只是撤其[229,240)误报。仍未解决持续位移与单次偏移的区别。

先看原始持续位移再解释区间内阶段；不更改潜在生成或已有作者参考，不把未知赚作正确静稳，不把误报后的阶段头输出赋予新真值意义。
""".strip()
    (report / "visual-review.md").write_text(review + "\n", encoding="utf-8")
    costs = csv_rows(run / "training-cost.csv")
    neural = sum(float(row["wall_seconds"]) for row in costs if not row["arm"].startswith("xgboost"))
    trees = sum(float(row["wall_seconds"]) for row in costs if row["arm"].startswith("xgboost"))
    summaries = {target: csv_rows(report / target / "summary.csv")
                 for target in ("primary_process", "secondary_developer")}
    lines = [
        "# 原日期逐列视觉活动判断：校准否定结果", "",
        "## Material Passport", "",
        "- Origin Skill: academic-research-suite / experiment-agent，run/validate inline。",
        "- 日期：2026-10-11；状态：固定训练/对比/消融与选中权重重放完成。",
        "- 决定：STOP_TEMPORAL_READOUT_CALIBRATION；原24/12及6/6开发校准，不是独立现场确认。", "",
        "结论：逐列原日对齐与辅助监督相对仅监督中心、全窗池化有过程增益；"
        "但主F1 .869210低于去图像.886644和公平重拟合XGB84 .924425，"
        "弱R .647199低于去图像.697259及XGB.843862。对纯数值净−40、XGB−200日。"
        "4/8预登记条件通过，停止固定候选；不调整kernel、监督权重、seed、阈值或子集救分。", "",
        "## 机制、标签契约与公平校准", "", mechanism, "",
        "## 全部对比与消融", "",
    ]
    for target, table in summaries.items():
        lines.extend(["### " + target, "", full_table(table), ""])
    lines.extend([
        "过程主543/839弱日，中心监督410/839、去数值664/839、纯数值585/839、"
        "全窗池化533/839、同信息XGB657/839、强XGB708/839。"
        "主较中心监督净+176、较池化+112、较纯图像+149；"
        "但较纯数值126得/166失、净−40，较强XGB100得/300失、净−200。"
        "主Precision .975754高于XGB .936114，但Recall .783643低于.913025；"
        "事件F1 .785714高于XGB.75，日F1却损.055215，超过新候选预先允许的.02容限。"
        "事件增益保留报告，但不升级为已验证可靠性。", "",
        "作者主F1 .934043，中心监督.936306、纯数值.931998、全窗池化.920401、"
        "强XGB.949461、同信息XGB.953620。对纯数值净+5，中心监督−4，"
        "强XGB−35、同信息XGB−42；六条全部参与，不选正例替代总体。"
        "原阶段头保持，阶段指标变化来自活动门控，不称原生视觉A/S/D创新。", "",
        "20神经/8XGB新拟合及全部532配置、训练历史/原标签曝光/逐日概率/输出保存。"
        "重新校准和模型结构共同改变输出，主与各消融独立选阈值，差异不全部归某一层。"
        "原108/96完整开发规则、PELT、强XGB、直接VLM、固定和Agent实验/成本保持；"
        "本轮校准人口不同，未拼入完整开发或重跑新确认。", "",
        review, "",
        "## 下一步行动依据", "",
        "已读取原阶段特征和拟合调用：共同阶段头是三尺度速度/切向加速度等原12特征，"
        "先活动再阶段门控。阶段机会诊断仅在原参考明确A/S/D及原stage特征支撑日统计，"
        "不为预测提供真值gate：过程2090日中110阶段错、53错置信度>=.9；"
        "作者303日中17错，全部高置信度S→A。作者原活动内部有622未知阶段日，"
        "不能借新视觉把它们当已知。诊断已开发暴露，仅决定下一轮方向，"
        "不证明阶段复核有效，也不自动启动新规则或修正标签。", "",
        "当前证据支持停止继续叠加小CNN融合模块。下一轮应使用已有VLM，"
        "先做少量原始位移和数值证据共同复核，核查强数值剩余错误是否有可修正证据；"
        "再冻结有限机制与同信息数值对照。不能只凭阶段置信度选择复核，"
        "因为本批实际错误包含高置信度；不能把改善某一项就称核心贡献。", "",
        "## 来源、验证、成本与复现", "",
        "VETime已有时间对齐/插值/池化/位置编码和动态融合；本轮§3.3与对齐源码在冻结前实际读，"
        "冻结后补读unfold_image函数，准确SHA/阅读范围见"
        "[文献记录](../artifacts/landslide-adaptive-review-2026-10-09/temporal-readout-literature-v1/reading-notes.md)。"
        "未运行或移植其模型，不称一般时间对齐/密集监督首创。", "",
        "14选中模型冻结逐日概率/预测与全部选中聚合分数精确重放，91980日历输出支撑/缺测/阶段门控通过。"
        "9217中心及483899辅助标签出现次数逐槽核对原日期、值与支撑，"
        "外侧/未知/无支撑训练标签0。未重训全20神经/8数值，不声称全训练字节复现。"
        "3新业务测试加3旧输入测试通过；集中评分器/Ruff检查另存。", "",
        f"20神经训练循环墙钟{neural:.6f}秒，8数值拟合/推理计时段{trees:.6f}秒；"
        "预处理另存。单次本地分段计时非完整CPU或重复端到端基准，"
        "不能因API调用为0就称图像编码无成本。主预测一次NN前向并只读中心，"
        "全部61列不是61次额外观察。", "",
        "复现：使用artifacts/reproduction-2026-10-07/environment/Scripts/python.exe，"
        "PYTHONPATH=src;scripts、PYTHONIOENCODING=utf-8。执行"
        "probe_landslide_temporal_readout.py prepare/calibrate --out 新目录；"
        "report_landslide_temporal_readout.py --run 冻结目录 --out 新报告目录。"
        "源码/协议/权重及源绑定保存。新增API/记录0；"
        "累计3825启动/3823回执、至少16694503报告token、受控192/半合成60不变。", "",
        "goal active：本轮实现并完成有依据的小机制对比与消融，得到否定证据；"
        "稳定贡献、独立可信确认仍未完成。", "",
    ])
    (ROOT / "docs/landslide-temporal-readout-results.md").write_text("\n".join(lines), encoding="utf-8")
    manuscript = manuscript_path.read_text(encoding="utf-8")
    if "完整中文初稿v46" not in manuscript or "### 6.4.11" in manuscript:
        raise ValueError("Expected untouched v46 manuscript")
    old_header = "更新前v45主稿及三状态文档原字节存档。新增固定数值初判、有界视觉残差、按记录OOF与消融、同等阈值机会及六图视觉数值复核；候选停止于校准。"
    new_header = "更新前v46主稿及三状态文档原字节存档。新增原日期逐列视觉判断、密集监督/池化/去数值消融、公平重拟合强XGB及六图视觉数值复核；候选停止于校准。"
    if old_header not in manuscript:
        raise ValueError("The preceding delivery header changed")
    section = """
### 6.4.11 原日期逐列视觉判断与辅助监督

""" + mechanism + """

过程主F1 .869210，中心监督.815176、全窗池化.837738、去图像.886644、强XGB84 .924425、同信息XGB.904084。主较中心/池化净+176/+112，却对纯数值−40、强XGB−200；弱R .647199低于纯数值.697259和XGB.843862。事件F1 .785714高于强XGB .75，但日F1损.055215，超过冻结.02容限。作者主.934043，中心监督.936306、纯数值.931998、强XGB.949461。4/8准入通过，STOP_TEMPORAL_READOUT_CALIBRATION；不调整kernel/辅助权重/seed/阈值/子集，不将有利事件指标代替主条件。全部统一指标、成本与复现见[报告](landslide-temporal-readout-results.md)。

实际看0033/0036/0034/0035及0057/0050原NEU和数值块：0033尾部E仍由−9.303累积到−12.161mm却被截短；0036排提前误报但弱尾断裂；0034渐进起点丢失；0035平台部分改善。0057对强XGB净+21，0050平台新增38误报。作者已知历史研究，非独立专家；未知/缺测及原阶段参考不改。新20神经/8XGB与532配置、14选中权重精确重放保存，91980输出日及483899辅助标签出现次数的原日期/支撑契约通过，未全重训复现。

VETime[35]已包含视觉时间对齐，本轮实际补读§3.3及相关源码、冻结后unfold_image函数；不称一般对齐/监督首创。当前只证明相对本次更弱消融的局部收益，未证明相对强数值可接受的整体识别收益，止于校准。阶段诊断另显示存在高置信度S→A错误，仅用于下一轮研究选向，未新增阶段机制或更改评分。旧108/96完整开发与此前停止决定保持。

"""
    manuscript = manuscript.replace("完整中文初稿v46", "完整中文初稿v47", 1)
    manuscript = manuscript.replace(old_header, new_header, 1)
    manuscript = manuscript.replace("### 6.5 阶段与计算成本", section + "### 6.5 阶段与计算成本", 1)
    manuscript_path.write_text(manuscript, encoding="utf-8")
    status = """
## 2026-10-11：原日期逐列视觉判断停止于校准，初稿v47

[完整结果](landslide-temporal-readout-results.md)。新最小候选保留61原图时间列，逐列绑定原t+3k日数值/标签，推理中心日；同记录原已知支撑辅助监督，不插值/填未知。去辅助监督、去对齐全窗池化、去数值、同密集监督纯数值共五神经臂；强18/384维XGB重新用全部原已知支撑训练日拟合，公平覆盖新增标签曝光。原24/12、6/6，seed20261013、12epochs、38机会/臂；新20神经/8XGB、532配置保存。

过程主F1 .869210，中心.815176、池化.837738、纯数值.886644、XGB84 .924425；净对中心+176/池化+112，但纯数值−40/XGB−200。弱R .647199低于纯数值.697259/XGB.843862。事件F1 .785714高于XGB.75，但日F1代价超过冻结.02容限。作者主.934043、中心.936306、XGB.949461。4/8准入通过，STOP_TEMPORAL_READOUT_CALIBRATION，不调kernel/监督权重/seed/阈值/子集救分。旧108/96完整开发强基线/VLM/固定/Agent指标、成本及停止决定保持。

实际六图/14日原块复核：0033弱首尾损、0036弱尾断裂、0034渐进起点损；0035平台有局部收益。0057对强XGB净+21，0050平台新增38误报。作者非独立专家，不改标签/未知。14选中权重逐日概率/预测/聚合精确重放，91980输出日、9217中心/483899辅助标签出现次数原日/支撑核对通过；辅助标签重复不是新独立样本。3新+3旧输入测试通过，评分器/Ruff集中检查另存，未全训练字节复现。

VETime§3.3/对齐源码在冻结前实际读，冻结后补读unfold_image；不称通用对齐/密集监督首创，未运行作者模型。额外阶段诊断在原已知ASD且原特征支撑日：过程2090日中110错（53高置信），作者303日中17错，全高置信S→A；这只是暴露开发选向，未实现新阶段机制或改变本轮评分。下一步回到已有VLM，少量核查强数值残余错误是否有真正可修正证据，不继续叠加小CNN模块。

新增API/记录0，累计3825启动/3823回执、至少16694503报告token，受控192/半合成60不变。v46四文档原字节存档，321文件旧manifest保持并按档案映射核验；v47公式继续LaTeX。goal active：本轮有机制开发、消融、否定证据与初稿更新，稳定实质贡献/独立可信确认未完成。
"""
    for filename in ("project-status.md", "landslide-adaptive-review-method.md", "landslide-submission-roadmap.md"):
        path = ROOT / "docs" / filename
        if path.read_bytes() != (archive / (path.stem + "-v46-original.md")).read_bytes():
            raise ValueError("Status document differs from the archived source")
        heading, remainder = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(heading + "\n\n" + status.strip() + "\n\n" + remainder.lstrip("\n"), encoding="utf-8")
    export(archive / "run-result.json", {
        "status": "progress", "goal_complete": False, "candidate_decision": decision["decision"],
        "new_api_calls": 0, "new_records": 0,
        "completed": ["Method implementation", "Frozen calibration and ablations", "Strong baseline refits",
                      "Selected weight and dated-label replay", "Raw visual/numerical review",
                      "Nearest-method source reading", "LaTeX manuscript update"],
    })


if __name__ == "__main__":
    write()
