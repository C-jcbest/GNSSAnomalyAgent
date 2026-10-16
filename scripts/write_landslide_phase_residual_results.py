"""Record the bounded phase probe, its negative result and the next research evidence."""

from __future__ import annotations

from probe_landslide_endpoint_fit import BASE, ROOT, csv_rows, export
from run_landslide_window_optimization import read_json


def write():
    archive = BASE / "manuscript-v48"
    report = BASE / "phase-residual-analysis-v1"
    decision = read_json(report / "decision.json")
    if decision["decision"] != "STOP_PHASE_RESIDUAL_PILOT":
        raise ValueError("This report documents the fixed stopped phase pilot")
    names = ("landslide-adaptive-review-manuscript-draft", "landslide-adaptive-review-method",
             "landslide-submission-roadmap", "project-status")
    for name in names:
        if (ROOT / "docs" / (name + ".md")).read_bytes() != (archive / (name + "-v47-original.md")).read_bytes():
            raise ValueError("Archive untouched v47 document bytes before writing once")
    if not read_json(archive / "v47-preservation-before.json")["all_match"]:
        raise ValueError("Previous delivery is not preserved")
    mechanism = r"""
旧阶段矛盾选择器只检查明显反向变化或S的明显速率变化，不能据高置信度判定没有复核需要。本轮另检验一种显示机制：先保留强XGB84活动，只在其活动内部取得98/70/42日中最长可用证据；必要时向活动内部平移。原已知阴性不跨越，缺测最多桥接7日，每14日块至少8完整NEU观测。目标为原日历14日，所有观察日期及改写权限从预测活动决定，不用参考活动当部署gate。

令 $m_k$ 为原始三轴块中位数，$\sigma$ 为原日差MAD噪声代理，$K$ 为块数。以原端点净变化确定运动坐标：

$$
z_k=(m_k-m_0)\oslash\sigma,\qquad
u=\frac{z_{K-1}-z_0}{\|z_{K-1}-z_0\|_2},\qquad p_k=u^\mathsf Tz_k.
$$

其中 $\oslash$ 表示逐轴相除；零净变化不构造该坐标。拟合常速直线并显示同一观测的残差：

$$
(\hat\alpha,\hat\beta)=\arg\min_{\alpha,\beta}
\sum_k[p_k-\alpha-\beta(t_k-\bar t)]^2,\qquad
r_k=p_k-\hat\alpha-\hat\beta(t_k-\bar t).
$$

原始逐日曲线保留缺测空位；块中心不创造新观测。噪声单位不是置信区间，残差符号不是A/D，方向变化也会限制投影。通用投影/去趋势/残差不称首创。主组给局部原NEU、运动坐标图和原块数值；普通局部图、加残差图但去数值、只有相同数值无图为消融。提示前缀、目标、数值表、阶段支撑、原活动及输出上限相同。仅改目标有观测、原特征支撑且活动阳性日期；目标外逐日严格保留，失败复核保留原阶段并单列失败。
""".strip()
    review = """
实际查看八个原NEU及八个运动坐标图，先核查原持续位移，再解释阶段，未修改参考。作者已见历史研究，不是独立专家；五个错误目标和A/S/D三个正确控制按评分后固定规则选择，不能作为代表性或未见确认。

- 0029[112,126)：E块中位1.666→3.869→8.889→12.453→17.990→22.517→27.609mm，原始持续运动可见；投影相邻块速率.2486/.5187/.3756/.5893/.4969/.5030波动。所有臂都判S，未恢复原参考A，不能以活动明显替代增速证据或改参考。
- 0034[266,280)：N/E持续累积，但目标附近有原始尖峰；块投影速率.5309/.5275/.4077/.3424呈降速线索。全部臂两遍仍判S，未修原参考D；数值信息提供不等于模型使用正确。
- 0052[182,196)：N块中位11.138→18.649→25.288→32.497→40.536→47.837→55.590mm；投影速率.7117/.6141/.6740/.7382/.6797/.7144近似稳定，残差块点约零附近。所有臂两遍均S，修好原强头7个A误判日；普通局部图已足够，不能归因于新残差图。
- 0028[322,336)：42日原NEU只有小幅继续累积，噪声占比较大；仅3块，投影两速率.1923/.1485。主两遍S，损9个原正确D日；raw第二遍细分D/A/S不稳定。短窗与少块不能当可靠阶段证据。
- 0030[98,112)：N增加、E下降，整体累积变快；投影速率.2586/.5167/.5258/.7022/.5745/.6311。主两遍全S，既没有修该目标原错误，又损原正确A日；数值组A→S分段修2日而没有损原正确日。图像组S→A方向相反，残差图未纠正。
- 正确控制0028[126,140)：相邻速率.1391→.3304→.3402→.4937→.6187→.6516；原始NEU出现渐进离开平台。图像主两遍仍S，损14个正确A日，数值组两遍A保留。
- 正确控制0027[210,224)：原E在较大噪声中缓慢累积；块速率包含−.0060及.1385波动，不足以对每个日期断言精确恒速。纯图两遍未知；主数值反馈与纯数值两遍S，原14个正确日保留。没有新增独立S真值或修改原弱活动。
- 正确控制0029[588,602)：原E趋平台，投影速率.5010→.4288→.3923→.3400→.2479→.1425有降速线索。主第一遍S损14个正确D日、第二遍D保留；纯数值两遍D保留。识别线索存在，但图像增量和重复稳定性未成立。
""".strip()
    (report / "visual-review.md").write_text(review + "\n", encoding="utf-8")
    literature = """
# 阶段残差与下一研究方向：实际阅读范围

2026-10-11。官方arXiv API定向检索8条最新匹配元数据，查询、原响应、SHA及摘要保存；不是系统综述。该检索在阶段pilot冻结/推理后，不冒充其预登记依据。

El Hili、Tiomoko、Ancourt，LLM-Generated Feature Pools for Time Series Anomaly Detection，arXiv:2609.21801v1。官方HTML200，实际读取第3.1—3.5，抽取正文字符13917—21979及两个原文摘录保存；未完整读实验、统计和全文。其方法用目标序列滑窗特征、逐特征median/MAD、特征/时间聚合；VLM看少量有标签tuning窗的同轴图，提出十个numpy特征函数，经探针验证后同下游协议筛选。VLM不看评价集或 detector 分数，不逐日直接判异常；通用VLM生成特征池及专家池合并已有先例，不能直接套用后称首创。

所读官方HTML的GitHub链接仅指向arXiv/LaTeXML基础设施。按feature pools/time series的GitHub仓库定向检索HTTP200、0条；这不是源码不存在的证明。未取得或运行作者源代码。可核查方法仍由所读正文支持，后续若需复现应明确为自行重实现。

本轮原始位移残差帮助作者看清0052近似恒速，但所有臂都能修它；图像输入反而在A/D正确控制上有损害，故停止残差阶段pilot。新论文仅提供一个待验证方向：用有限视觉错误分析提出少量可解释GNSS运动证据特征，在固定强数值模型和同等校准机会下检验增量，再做新独立数据确认。它尚未实现/冻结，不构成论文贡献；需要明确GNSS技术差异，不能把本文特征池生成照搬或把换模型名称算创新。
""".strip()
    (BASE / "phase-residual-literature-v1/reading-notes.md").write_text(literature + "\n", encoding="utf-8")
    table = ["| 方法 | 遍 | 已知ASD日 | 正确 | 错误确定 | 未知 | macro-F1 | 修好/改坏 | 净纠错 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in csv_rows(report / "summary.csv"):
        if row["target"] == "all":
            table.append(f"| {row['method']} | {int(row['repetition']) + 1} | {row['known_stage_days']} | {row['correct']} | {row['wrong_definite']} | {row['unknown']} | {float(row['macro_f1']):.6f} | {row['gained']}/{row['lost']} | {row['net']} |")
    result = """# 活动内运动坐标残差复核：阶段纠错小试否定结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent，run/validate inline；2026-10-11。
- 状态：64真实新请求、四臂两遍、原回答确定重放与同源证据核验完成。
- 决定：STOP_PHASE_RESIDUAL_PILOT，6/14条件通过；不调窗口、seed、提示、目标或选择更好控制救分。
- 范围：已暴露开发人口、评分后选择八个14日目标，112已知支撑ASD日；不是完整记录效果、专家金标准或独立确认。

主残差反馈macro-F1 .151724/.334606低于强阶段头.658201和纯数值证据.608046/.608046。较强头7得/47失、7得/33失，净−40/−26；对普通局部图有收益，却没有对强方法的纠错收益。0052近恒速的7日修正由全部臂获得，没有残差图特有价值。阶段能力仍未建立，停止扩大本机制。

## 机制与消融

""" + mechanism + "\n\n## 全部阶段比较\n\n" + "\n".join(table) + """

主过程98日F1 .108262/.289644、较强头净−47/−33；作者14日F1 .333333/.333333、净+7/+7。作者这14日全S，按固定三类宏平均最高也只是1/3，不把此数值解释为实际仅33%正确。所有配置逐类P/R/F1、混淆、覆盖和逐目标损益见results.json/per-target.csv。A/S/D共同分母37/33/42日，原正确控制42日主两遍损28/14日。不是只选择低分目标展示收益。

所有组的活动逐日相同，所以活动P/R/F1、事件、边界和弱召回均不发生方法差异；本pilot不以截取目标成绩替代已有108/96完整开发的规则、PELT、数值学习、直接视觉及固定Agent比较。没有新训练或校准；使用当前同目标已公平重拟合的强XGB84活动、原目标阶段头。本轮评价的是冻结活动条件下的阶段改写增量，不声称完整异常识别方法已经成立。

""" + review + """

## 成本、失败与验证

64启动/64回执/64成功JSON，0阶段契约失败、0重试，134544报告token，API分段耗时93.796秒、推理循环94.788秒。各臂每遍8调用，16调用/臂，图像数量和实际token不同，不能称同token成本；原强头不需远端调用，CPU没有重测端到端。新增记录0，受控192/半合成60不变；累计3889启动/3887回执、至少16829047报告token，未回usage成本仍未知。

准备v1实际目检发现日期刻度相邻重叠，0API；v2改刻度但写running状态时误用只创建新文件的export，0API且request目录为空。v3复用已有save_json写可变状态。两个原版本及错误原因保留，未重试远端请求；八目标、14份关键输入/证据文件及提示保持一致，v2/v3输入图也相同。源码和新协议在实际调用前冻结。

64保存回答重放一致、46720日历输出活动/阶段/缺测/权限契约通过，八个同源方向/块/残差逐字段精确重算；64完整请求体SHA及24组独立逐类计数F1核对通过。22业务/评分/解析测试与Ruff通过。远程模型未再次推理，不声称API随机输出可逐字节再现。作者实际查看八原图和八残差图，修改的是方法预测，原参考未改。

![阶段小试对比](../artifacts/landslide-adaptive-review-2026-10-09/phase-residual-analysis-v1/comparison.png)

复现使用原environment/Scripts/python.exe，PYTHONPATH=src;scripts、PYTHONIOENCODING=utf-8。probe_landslide_phase_residual.py prepare/infer --out新目录；report_landslide_phase_residual.py --run冻结目录 --out新报告目录。源码、提示、输入、数值、所有回执、逐日预测、PNG/PDF和指标均保存。

## 相关研究与下一步

实际读最新LLM-Generated Feature Pools的3.1—3.5，其窄角色是从调参窗提出统计特征，严格保持下游评价；不是实时VLM复核。未取得作者源码，当前只有官方正文支持。阅读和源码发现限制见[文献笔记](../artifacts/landslide-adaptive-review-2026-10-09/phase-residual-literature-v1/reading-notes.md)。下一轮检验有限视觉分析是否能提出强数值池缺少的持续运动证据；需同预算拟合/消融、明确GNSS技术差异和新独立确认，不能以通用特征生成称首创。当前只是待验证方向，尚未实现或注册候选。所有旧停止决定保持，goal active。
"""
    (ROOT / "docs/landslide-phase-residual-results.md").write_text(result, encoding="utf-8")
    manuscript_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    manuscript = manuscript_path.read_text(encoding="utf-8")
    if "完整中文初稿v47" not in manuscript or "### 6.4.12" in manuscript:
        raise ValueError("Expected archived v47 manuscript")
    manuscript = manuscript.replace("完整中文初稿v47", "完整中文初稿v48", 1)
    manuscript = manuscript.replace("更新前v46主稿及三状态文档原字节存档。新增原日期逐列视觉判断、密集监督/池化/去数值消融、公平重拟合强XGB及六图视觉数值复核；候选停止于校准。",
                                    "更新前v47主稿及三状态文档原字节存档。新增活动内运动坐标残差阶段小试、四臂两遍与强阶段头比较、八图视觉数值复核及相关文献；候选停止。", 1)
    section = "\n### 6.4.12 活动内运动坐标残差阶段复核\n\n" + mechanism + """

在已暴露开发记录上评分后选5个错误目标及A/S/D三个正确控制，每目标14日。四臂两遍64真实请求全部通过契约；共同37/33/42个A/S/D日，不是完整记录效果。主macro-F1 .151724/.334606，普通局部图.105572/.152460、去数值.105572/.090171、纯数值.608046/.608046、强阶段头.658201。主对强头净−40/−26，对作者0052的7日修正由所有臂取得；三个原正确控制主损28/14日。6/14准入通过，STOP_PHASE_RESIDUAL_PILOT，不调窗口、提示、seed或目标。当前既不能证明新图像阶段能力，也不能用较弱消融上的收益替代强对照。原活动和完整活动指标不变；原规则/PELT/XGB/直接视觉/固定Agent对比及停止决定保持。

实际八原NEU/八残差图与块速率复核显示：0052近恒速可纠正，但0028增速控制和0029降速控制仍被图像主组误判S。64请求体SHA、原回答重放、46720日历权限与缺测契约、八证据逐字段核验通过；22相关测试/Ruff通过。134544报告token、API分段93.796秒，非端到端CPU；准备版本0API失败已保存。详见[完整报告](landslide-phase-residual-results.md)。

最新特征池研究[36]已将VLM限定为从调参图提出特征、固定下游检测器再比较；一般残差/特征生成不能称首创。本轮实际读3.1—3.5，未取得作者源码或运行模型；它提供下一轮研究依据，尚未实现新特征方法或证明GNSS贡献。当前作者复核非独立专家，所有原参考/未知保持。

"""
    manuscript = manuscript.replace("### 6.5 阶段与计算成本", section + "### 6.5 阶段与计算成本", 1)
    manuscript += "\n[36] El Hili, Y. A., Tiomoko, M., & Ancourt, C. (2026). LLM-Generated Feature Pools for Time Series Anomaly Detection. [arXiv:2609.21801v1](https://arxiv.org/abs/2609.21801v1). 官方HTML3.1—3.5已读；未完整阅读全文、取得作者源码或运行其模型。\n"
    manuscript_path.write_text(manuscript, encoding="utf-8")
    status = """
## 2026-10-11：活动内残差阶段小试停止，初稿v48

[完整结果](landslide-phase-residual-results.md)。新增活动内最长98/70/42日原位移运动坐标与常速线残差；只改固定14日目标原观测/支撑活动阳性日，活动不改。评分后选5错误目标和A/S/D正确控制，四臂两遍64新请求/64回执全部通过契约，112已知ASD日37/33/42，非独立或完整记录效果。

主F1 .151724/.334606低于强阶段头.658201和同数值无图.608046；对强头7得/47失、7得/33失，净−40/−26。0052修7日全部臂都能做到，正确控制主损28/14日；6/14条件，STOP_PHASE_RESIDUAL_PILOT，不调窗口、提示、seed、目标救分，不扩展该阶段机制。旧完整108/96基线和各停止决定保持。

实际看8原图/8残差图与块速率；64payload SHA、原回答重放、46720日历活动/阶段/缺测/权限和8证据重算通过，22相关测试/Ruff通过，无远端再推理。准备v1刻度重叠、v2状态写入失败均0API，版本保留；v3用原save_json写可变状态，关键目标/数值/提示不改。新API134544报告token，分段93.796秒；累计3889启动/3887回执、至少16829047token，受控192/半合成60不变。

官方arXiv检索8新匹配元数据，实际读最新LLM-Generated Feature Pools正文3.1—3.5；只发现基础设施GitHub链接，定向仓库检索0不证明无源码，未取得/执行作者模型。它支持下一轮在固定强数值模型下检验有限视觉分析提出的持续运动特征，尚未实现/冻结，通用特征池生成已有先例，必须另证GNSS技术差异与效果。v47四文档原字节存档，273旧文件映射核验；v48公式LaTeX。goal active，稳定实质贡献和独立可信确认未完成。
""".strip()
    for name in names[1:]:
        path = ROOT / "docs" / (name + ".md")
        heading, remainder = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(heading + "\n\n" + status + "\n\n" + remainder.lstrip("\n"), encoding="utf-8")
    export(archive / "run-result.json", {"status": "progress", "goal_complete": False,
                                         "candidate_decision": decision["decision"], "new_api_calls": 64,
                                         "new_records": 0, "reported_tokens": 134544,
                                         "completed": ["Motion-coordinate evidence implementation", "Frozen four-arm repeated pilot",
                                                       "Strong-stage correction comparison", "Actual visual/numerical review",
                                                       "Response/payload/evidence replay", "Nearest-method section reading", "LaTeX draft v48"]})


if __name__ == "__main__":
    write()
