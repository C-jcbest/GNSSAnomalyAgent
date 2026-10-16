"""Write the stopped jump-view result from the frozen calibration and replay."""

from __future__ import annotations

from probe_landslide_endpoint_fit import BASE, ROOT, csv_rows, export
from report_landslide_core_lockbox import full_table
from run_landslide_window_optimization import read_json


def write():
    run = BASE / "jump-view-development-v1"
    report = BASE / "jump-view-calibration-report-v1"
    decision = read_json(run / "calibration-decision.json")
    if decision["decision"] != "STOP_JUMP_VIEW_CALIBRATION":
        raise ValueError("Only document this stopped fixed candidate")
    manuscript_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    manuscript = manuscript_path.read_text(encoding="utf-8")
    if "完整中文初稿v44" not in manuscript or "### 6.4.9" in manuscript:
        raise ValueError("Archive and update v44 exactly once")
    costs = csv_rows(run / "training-cost.csv")
    neural = sum(float(row["training_wall_seconds"]) for row in costs if row["arm"] != "xgboost_paired_information")
    numeric = sum(float(row["training_wall_seconds"]) for row in costs if row["arm"] == "xgboost_paired_information")
    preprocessing = sum(float(row["load_features_charts_and_scaling_wall_seconds"])
                        for row in csv_rows(run / "preprocessing-cost.csv"))
    mechanism = r"""
新候选为受监督图像/数值双输入视觉模型，保留原图，只在单个相邻原始中位增量主导时取得一张替代图。每目标日仍用前后90日61个不交叠3日中位列，每列至少2个完整NEU观测。令 $m_j$ 为第 $j$ 列三轴中位数，$\sigma$ 为原日差噪声代理，仅相邻两列都可用时计算：

$$
d_j=m_{j+1}-m_j,\qquad
u_j=\|d_j/\sigma\|_2,\qquad
j^\star=\arg\max_j u_j.
$$

最大范数至少6且严格超过第二大范数的两倍才触发。缺列不建立跨缺口差分；两次相反大增量不作为单次主导变化。触发后只对后续已有列减去一个观测增量：

$$
\widetilde m_j=m_j-d_{j^\star}\mathbf1[j>j^\star].
$$

时间、缺测和原位移均不改；后续相邻差保持，原图始终保留。此替代表示不识别仪器原因，也可能削弱真实快速运动，不能当作已确认清洗。画布沿用六通道32×61，数值18原始特征及18不可用指示沿用XGB84。两层8/16通道CNN与单头16/32保持。

训练原图和替代图的同一活动监督，使用两图平均正类加权BCE加固定 $\lambda=1$ 概率平方差。推理仅在原支撑且触发的日期取两概率平均，最多两次前向；其余只看原图。固定双图控制去掉触发门槛，对所有支撑日期减最大相邻增量并训练/推理；去一致性控制使用相同配对、$\lambda=0$；视觉单输入去掉神经数值分支。没有坐标反射、RL、循环或大语言模型。
""".strip()
    review = """
## 实际视觉与原始数值复核

当前Codex实际查看两训练输入和四过程/两作者的原始NEU、概率、逐日输出图；六验证图按评分后净纠错极值选取，仅解释机制。原标签不改，作者具有历史结果暴露，非独立专家。

- 0026：原三轴在约230日单次偏移后为平台。主模型相对普通视觉净+45日，对纯数值+5；但对固定双图−14，仍误报[247,261)。该记录180个支撑日期触发替代图，最大增量15.9437噪声单位。原14日中位块再次核对平台，与上一轮同原始数值，不据图像像素推断毫米速率。
- 0031：原图约530日有较小三轴偏移，属于quiet家族背景中的质量偏移，并非新增controlled instrument_step家族。最大相邻增量4.4230，小于冻结门槛6，零额外观察。主模型与原视觉净0，却对纯数值损49个静稳日[515,564)，对固定双图损23。说明高显著幅值门槛漏掉仍能诱发视觉误报的小偏移；本轮不下调门槛救分。
- 0036：N后半仍有缓慢累积、E较平、U约465日有较大偏移。124个支撑日期触发替代图，最大6.8880。主模型相对原视觉净−33，新增[439,470)平台误报；相对纯数值净−23，还漏弱运动的部分起止。对固定双图净−10。此前相同原N中位块由−90.661缓移至−93.485mm，弱尾有依据，但本模型没有保住它的完整识别。
- 0035：两次明显累积之间有平台，零替代观察；主模型相对原视觉净+18、纯数值+11、固定双图+1。重新训练会改变零触发日的原图概率，不存在继承旧视觉输出的保证，不能把所有差异归为本记录的额外观察。
- 0057作者参考：中段N连续变化、E缓慢升、U较噪；零替代观察。主模型相对原视觉净−13、纯数值−19、固定双图−89，活动内有断裂。阶段仍由原目标头门控；不能把活动漏检计成原生阶段识别创新。
- 0055作者参考：三轴以起伏为主，未见清楚持续累积。零替代观察，主模型与原视觉、固定双图、纯数值的确定正确判断完全同值；静稳对照保持，单例静稳正确不能抵消弱活动损失。

从原始位移确认变化再解释活动内阶段。潜在机制已知不等于现场可辨；参考未知不计正确静稳，原缺测保持未知。本复核不新建独立标签或修改生成机制参考。
""".strip()
    (report / "visual-review.md").write_text(review + "\n", encoding="utf-8")
    rows = {target: csv_rows(report / target / "summary.csv")
            for target in ("primary_process", "secondary_developer")}
    text = [
        "# 单主导跳变的有限替代观察：校准否定结果", "",
        "## Material Passport", "",
        "- Origin Skill: academic-research-suite / experiment-agent，run/validate inline。",
        "- 日期：2026-10-11；状态：校准与已冻权重推理重放完成。",
        "- 范围：原过程24训练/12验证，作者6训练/6验证，固定候选；非未见现场确认。",
        "- 决定：STOP_JUMP_VIEW_CALIBRATION；旧坐标反射候选继续停止。", "",
        "结论：主模型减少指定单阶跃家族误报90→42日，过程F1 .863181→.867513，"
        "相对普通视觉净+20日；但弱召回.764005→.755662，对纯数值净−140，对固定双图−68。"
        "去配对一致性F1 .868397更高、净纠错主损7日；关键损失项和自适应观察的独立识别增益未获支持。"
        "停止当前候选，不用较好的控制、记录或指标替换主比较。", "",
        "## 机制与公平校准", "", mechanism, "",
        "训练前固定规则、seed20261011、12 epochs、每2日eligible训练、两CPU线程；"
        "每臂两头宽×三个概率阈值.35/.5/.65，六个正常校准机会，目标为日F1+事件F1。"
        "无早停、seed/lambda/跳变门槛/画布/子集搜索。不同臂独立选阈值，最终差异不全部归因于损失函数。", "",
        "复用已冻普通图像和纯数值MLP：同训练/验证、eligible日期、归一化、seed、epoch与六机会，"
        "重新从冻结权重推理并检查逐日/分数一致，不将不适用旧模型充作新目标基线。"
        "作者使用自己的6/6模型，结果不选择过程模型。新增同信息XGB得到18原数值、"
        "原/替代各366位置/可用性以及触发/最大/第二大三事实，共753维，"
        "同eligible日期、120树、depth2/4、lr.05、三阈值六机会。新16神经/4XGB、60配置均保存。", "",
        "## 完整校准对比与消融", "",
    ]
    for target, table in rows.items():
        text.extend(["### " + target, "", full_table(table), ""])
    text.extend([
        "过程主弱检634/839=.755662，原视觉641/839=.764005、纯数值666/839=.793802、"
        "固定648/839=.772348、去一致性639/839=.761621、同信息XGB743/839=.885578。"
        "作者校准没有独立弱属性，继续不可评价。指定单阶跃家族FP主42、去一致性42、固定28、"
        "视觉单输入14、普通90、纯数值51、同信息XGB0；家族指标不包括quiet背景中的偏移，"
        "不能写为全部偏移误报率。", "",
        "相对原视觉主过程95得/75失、净+20；纯数值108得/248失、净−140；"
        "固定40得/108失、净−68；去一致性15得/22失、净−7；同信息XGB62得/327失、净−265。"
        "作者分别净−21/−50/−110/+12/−94。全部12/6记录参与，未知参考不赚静稳信用。", "",
        "过程仅3/7登记检查通过：较少第二观察、单阶跃FP低于原视觉、配对概率差低于无一致性。"
        "其余日F1对数值、弱召回、日F1对固定/无一致性均失败，STOP_JUMP_VIEW_CALIBRATION。"
        "配对概率差.130316低于无一致性.141479，但没有相应识别增益；作者主.054568"
        "还高于无一致性.052946。不把两个同源图当独立证据或校准置信度。", "",
        "同信息XGB过程.928818/作者.925044为数值分类结果，未包装成Agent/VLM贡献。"
        "它与前轮仅原图XGB的输入及反射增广训练不同，不能把两轮分数差全归于去主导跳变。"
        "本轮只支持丰富原始时间形状可以被强数值学习器利用，尚未证明CNN有效吸收这些信息。"
        "原108/96完整开发强规则/PELT/XGB、视觉/固定/Agent指标与API成本保持；"
        "校准人口不同，本轮不混入完整开发结果或声称完成未见确认。", "",
        review, "",
        "## 来源、验证、成本与复现", "",
        "GNSS-FM已使用设备目录增量掩码和双流，CAPS已有配对模拟、结构一致和反事实重构；"
        "本候选不宣称去偏移、多视图或一致性首创。"
        "读GNSS-FM预处理/双流相关段、CAPS引言/相关工作/部分定义、作者README/两份有限源码，"
        "Anlu只读官方摘要，200 HTML实际缺正文。未运行作者模型。"
        "详见[阅读范围](../artifacts/landslide-adaptive-review-2026-10-09/jump-view-literature-v1/reading-notes.md)。", "",
        "3新增业务测试加3既有输入测试通过，Ruff通过。"
        "14个选中模型（8新神经、4复用神经、2新XGB）加载冻结权重，逐日预测、"
        "全部选中聚合分数及各配对差精确一致；91980日历输出检查原支撑、缺测、阶段门控。"
        "既有sklearn日指标、一对一IoU .3/.5及边界计数复核通过。"
        "未重训16神经/4XGB，不能声称完整训练逐字节复现。", "",
        f"训练16神经循环墙钟{neural:.6f}秒，4XGB拟合/保存{numeric:.6f}秒，"
        f"两目标读入特征、画布与归一化路径{preprocessing:.6f}秒；"
        "这些分段路径未包含全部选择、输出、共同阶段或评分，不相加为完整端到端运行时间。"
        "主选中概率计算为过程12记录.109365秒、作者6记录.054229秒，"
        "固定选中.251668/.126910秒；均共享已准备输入、计时含数组选取，"
        "单次本地测量非重复CPU基准。", "",
        "过程第二视图478/7786支撑日，固定7786/7786；作者161/3818，固定3818/3818。"
        "原图对全部原日历推理，再按原支撑门控；节省第二图前向不等于减少全部计算。"
        "监督标签、渲染与训练不能当零样本免费能力。新增API/记录0，"
        "累计仍3825启动/3823回执、至少16694503报告token，受控192/半合成60不变。", "",
        "准备/训练：probe_landslide_jump_view.py prepare或calibrate --out 新目录；"
        "冻结推理重放：report_landslide_jump_view.py --run 冻结目录 --out 新报告目录。"
        "使用既有artifacts/reproduction-2026-10-07/environment/Scripts/python.exe，"
        "PYTHONPATH设src;scripts，PYTHONIOENCODING=utf-8。全部源码、协议、权重、"
        "60校准配置、第二视图证据、逐日预测、纠错和原图/原块保存并封哈希。", "",
        "goal active：本轮有实际方法开发、消融和否定证据，尚未形成稳定核心贡献或独立可信现场确认。"
        "后续应检验视觉模型获取原始时间形状的能力与可靠增量，不继续调此跳变门槛救分。",
    ])
    (ROOT / "docs/landslide-jump-view-results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    section = """
### 6.4.9 单主导跳变触发替代图的校准探索

""" + mechanism + """

过程12验证主F1 .867513，普通.863181、固定双图.880580、去一致性.868397、视觉单输入.856559、纯数值.900979、同信息XGB.928818。主对普通净+20、固定−68、去一致性−7、纯数值−140、同信息XGB−265。指定单阶跃家族FP90→42，但弱召回.764005→.755662，低于纯数值.793802。作者6验证主.869514，普通.882695、固定.938150、纯数值.899008、同信息XGB.925044，对普通净−21、固定−110。全部结果见[完整报告](landslide-jump-view-results.md)，不同校准人口不混入原108/96完整开发表。

主配对概率差.130316小于去一致性.141479，自适应第二视图仅478/7786支撑日、固定7786/7786；作者161/3818。但日F1对固定/去一致性/纯数值、弱召回条件均失败，STOP_JUMP_VIEW_CALIBRATION；不调门槛、seed、损失、子集或更换控制为主组。局部排阶跃与较少前向没有证明整体可靠性改善。

实际复核0026/0031/0035/0036及0057/0055原始NEU和数值块：0026平台纠错净+45，0031较小背景偏移最大4.423噪声单位、未触发，仍有49过程误报；0036弱尾部分丢失，对普通净−33。零触发记录的新训练概率也可能改变，不声称继承旧视觉。作者已历史暴露，非独立专家，不改参考或阶段。阶段头仍沿用原目标头，指标改变来自活动门控。

GNSS-FM[30]已有设备目录增量处理/双流，CAPS[31]已有配对监督/结构一致；Anlu[32]仅读摘要、HTML正文缺失。未运行这些作者模型，不称通用去偏移/配对训练首创。新16神经/4XGB和全部60配置保存，14选中权重逐日/聚合/配对差精确重放，91980日契约和6输入业务测试/Ruff通过。分段本地成本公开；新增API/记录0。此候选停止于校准，未进入完整开发或未见确认。

"""
    manuscript = manuscript.replace("完整中文初稿v44", "完整中文初稿v45", 1)
    manuscript = manuscript.replace(
        "更新前v43主稿及三状态文档原字节存档。新增坐标反射训练一致性、五臂/同信息XGB校准消融、冻结权重重放及六图视觉数值复核；候选停止于校准。",
        "更新前v44主稿及三状态文档原字节存档。新增单主导跳变的有限替代图训练、固定/去一致性/去数值消融及同信息XGB、权重重放与六图视觉数值复核；候选停止于校准。",
        1,
    )
    manuscript = manuscript.replace("### 6.5 阶段与计算成本", section + "### 6.5 阶段与计算成本", 1)
    manuscript += """

[30] Teutschmann, N., Crocetti, L., Lehmann, F., Trentini, L., & Soja, B. (2026). GNSS-FM: A Self-Supervised Foundation Model for Daily GNSS Displacement Time Series. [arXiv:2606.07725v1](https://arxiv.org/abs/2606.07725v1). 已读官方摘要及III-B/III-C等相关段；未读完整论文、未确认作者模型权重或运行模型。

[31] Gao, Y., Lan, T., Lu, Y., An, X., Wang, M., He, W., Li, Y., & Zhang, C. (2026). Supervision Recovery for Time Series Anomaly Detection via Context-Anchored Pairing. [arXiv:2609.33610v1](https://arxiv.org/abs/2609.33610v1). 已读引言后半/相关工作/部分定义，作者仓库README、loss_new.py全部短文件及loss_v20.py前180行，固定SHAe6b4c4ad6403d5aac569241ad19f30990e71b57a；未阅读全文或运行作者模型。

[32] Lan, T., Gao, Y., Lu, Y., An, X., Wang, M., Pan, Y., He, W., & Zhang, C. (2026). Anlu: Enabling In-Context Time Series Anomaly Detection in Foundation Models via Counterfactual Supervision. [arXiv:2610.06180v1](https://arxiv.org/abs/2610.06180v1). 仅官方摘要；HTML200但正文缺失，不声称读了方法或取得源码。
"""
    manuscript_path.write_text(manuscript, encoding="utf-8")
    status = """
## 2026-10-11：单主导跳变替代图停止于校准，初稿v45

[完整结果](landslide-jump-view-results.md)。新受监督多模态视觉候选保留原图，仅当最大相邻3日中位变化>=6噪声单位且>2倍次大变化时补一张单增量删除图，最多两次前向、不插值/改原位移或标签。固定双图、去一致性、去神经数值输入及获得两图/选图事实的753维XGB共同校准；同原过程24/12、作者6/6，原图和纯数值模型合法复用并从权重重放。训练前冻结seed、12epochs、每臂六机会；新16神经/4XGB、60配置保存。

过程主F1 .867513，普通.863181、固定.880580、去一致性.868397、纯数值.900979、同信息XGB.928818。对普通净+20日、固定−68、去一致性−7、纯数值−140。指定单阶跃家族FP90→42，但弱R .764005→.755662；作者主.869514，普通.882695、固定.938150。仅3/7登记检查通过，STOP_JUMP_VIEW_CALIBRATION；不调阈值/seed/损失/子集救分，不把更好的控制或XGB升级成Agent/VLM贡献。旧108/96完整开发指标/成本及停止决定保持。

实际看两训练输入、四过程及两作者原始NEU/概率/输出图，核对原中位块：0026平台净+45；0031小偏移最大4.423未触发、仍49误报；0036弱尾受损，对原视觉净−33。零触发也会因重训练改变概率，不声称继承旧视觉。作者历史暴露，非独立专家，不改标签。GNSS-FM部分预处理/双流、CAPS部分正文与固定SHA有限源码已读；Anlu仅读摘要、HTML无正文；未运行作者模型。

14选中权重逐日/聚合/配对差精确重放，91980日支撑/缺测/阶段门控、6输入业务测试和Ruff通过。第二视图过程478/7786、作者161/3818，固定全部；分段预处理/训练/推理墙钟保存，不冒充全运行CPU或反复基准，未验证完整训练逐字节复现。

新增API/记录0，累计仍3825启动/3823回执、至少16694503报告token，受控192/半合成60不变。v44四文档原字节存档，276文件旧manifest保持并按档案映射核验。v45公式继续LaTeX。goal active：本轮完成有限新机制、对比/消融、否定证据与初稿更新，稳定贡献和独立可信确认未完成。下方为历史记录。

"""
    for filename in ("project-status.md", "landslide-adaptive-review-method.md", "landslide-submission-roadmap.md"):
        path = ROOT / "docs" / filename
        heading, remainder = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(heading + "\n\n" + status.strip() + "\n\n" + remainder.lstrip("\n"), encoding="utf-8")
    export(BASE / "manuscript-v45/run-result.json", {
        "status": "progress", "goal_complete": False, "candidate_decision": decision["decision"],
        "new_api_calls": 0, "new_records": 0,
        "completed": ["Method implementation", "Fixed calibration and ablation", "Frozen weight replay",
                      "Raw visual/numerical review", "Literature check", "Manuscript update"],
    })


if __name__ == "__main__":
    write()
