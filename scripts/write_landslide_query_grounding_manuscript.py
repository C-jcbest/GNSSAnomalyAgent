"""Bind frozen QSG evidence and actual reading into the complete Chinese draft."""

from __future__ import annotations

from pathlib import Path

from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json
from write_landslide_distribution_manuscript import score_table

from gnss_sim.artifacts import sha, write_json


def write(root):
    base = root / "artifacts/gnss-deformation-diversity-2026-10-12"
    folders = [base / name for name in (
        "query-shape-grounding-v1", "query-shape-grounding-replay-v1", "query-shape-grounding-report-v1",
        "query-grounding-literature-v1", "query-grounding-primary-v1",
    )]
    for folder in folders:
        verify_manifest(folder, "freeze.json")
    run, replay, report = folders[:3]
    if read_json(replay / "verification.json")["status"] != "VERIFIED":
        raise ValueError("Successful replay is required")
    session = base / "query-grounding-session-v1"
    session.mkdir(exist_ok=False)
    (session / "passport.md").write_text(
        "# 局部图形监督完整开发与解释\n\n## Material Passport\n\n"
        "- Origin Skill: academic-research-suite / experiment-agent；延续用户授权自主研究。\n"
        "- Scope: 48TRAIN／24已暴露VALIDATION；总96×730不变，24保留内容未加载。\n"
        "- Outcome: STOP_QUERY_SHAPE_GROUNDING；不是语言VLM／Agent贡献。\n"
        "- Verification Status: VERIFIED（72观测／教师重构、14保留权重／266校准、25方法600输出和评分，未重训）。\n"
        "- Reference Status: 过程参考；本次原图分析是开发者AI辅助解释，未修改参考。\n"
        "- 新API／新记录／新编码均0；研究目标active，独立有效贡献尚未成立。\n",
        encoding="utf-8",
    )
    (session / "actual-visual-review.md").write_text(
        "# 实际原图与数值复核\n\n"
        "实际通过view_image打开report-v1的0403与0395五联PNG。按对plain净纠错最小／最大选图，"
        "属于事后开发者分析，图含过程参考，不作盲标或输入VLM。\n\n"
        "0403：原NEU约100—200日呈较弱持续N下降／E上升／U下降，之后平台；"
        "约500—600日另一次更明显持续变化。plain检出第一段的一部分，QSG整段未检出，"
        "第二段也没有解决早起／迟止。精确追踪对plain撤51TP、补TP与误报变化均0，净−51。"
        "主阈值.80、plain .75，概率在第一段并非全零；选参包含事件目标和解码，"
        "不能把全部差异因果归为辅助损失或据图调阈值。\n\n"
        "0395：约600日后N/E持续增加、U下降，约660—680后逐渐平台；"
        "此前长期围绕旧水平，累计位移高位不延长活动。QSG起点比plain更迟，"
        "精确去15FP、撤1TP，净+14。不是总体正收益，也未修改参考起止。\n\n"
        "主对plain：去25FP、加3FP、补6TP、撤96TP，净−68；对numeric净−200。"
        "形状预测MAE .082160，零.084492、TRAIN均值.084534，平均R² .047938；"
        "六维解释率只有约.0018—.1337，U较强。这是有限图形信息，不是稳定异常贡献。"
        "纯视觉监督MAE更低.081734、R² .060487，活动F1却由.645599降至.607308；"
        "不能用更低重建误差替代检测验证。\n",
        encoding="utf-8",
    )
    (session / "reading-notes.md").write_text(
        "# 文献检索与实际阅读边界\n\n"
        "3项arXiv官方定向检索均200：time series visual learning、time series auxiliary reconstruction、"
        "time series cross modal distillation；ViSTA/UTOPYA abs与html四页面均200。非系统综述。\n\n"
        "ViSTA（2609.31448v1，2026-09-25）：实际读官方摘要与保存HTML文本175—212行，"
        "确认在原视觉token位置通过原数值encoder进行残差修正、零初始化、预训练语言／视觉主干冻结。"
        "本次QSG是小CNN图像分支局部形状监督，未实现其时序数值adapter、语言接口或QA。"
        "未读完其方法／实验，未定位并读取作者源码，未运行模型；不称ViSTA复现。\n\n"
        "UTOPYA（2605.18188v1，2026-05-18）：实际读摘要及HTML文本364—435附近方法概览，"
        "确认八模态FiLM／attention／gating、预测／分类／重建与物理正则。"
        "作者也报告部分常见正则负结果；这些信息不能为本候选有效性背书。"
        "未读完整损失／实验、未定位和读取作者源码、未运行。\n\n"
        "复用原VAN-AD官方保存文本135—160、210—247段落，确认正常／异常都可较好重构时"
        "检测可能失去判别，故不把教师读出改善当异常收益。作者仓库PenyChen/VAN-AD已在此前定位；"
        "本轮未新增作者代码读取或模型运行。通用辅助损失、视觉重建与模态融合已有先例，"
        "不能把QSG称为这些思想首创。\n\n"
        "历史纠正：v13描述本轮TimeRLM API403及未读源码，仅对该次访问成立。"
        "docs/landslide-nearest-method-code-review.md已保存更早一轮TimeRLM源码阅读记录，"
        "commit31fcd847b7cb37a1f1e6859b1dca7973ed0eae74；不应把一次403解释成项目从未读取作者实现。"
        "本轮只核查该历史记录，未复现作者模型。\n",
        encoding="utf-8",
    )
    (session / "statistical-interpretation.md").write_text(
        "# ARS指定统计谬误：11/11检查\n\n"
        "以下为描述性开发解释，没有显著性／因果／现场泛化结论。\n\n"
        "1. Simpson's Paradox：25方法八族全计数保存；总体不利不能隐藏于单个0395正例。\n"
        "2. Ecological Fallacy：记录级八族各3条，不由汇总F1推断每条记录均改善；日和重叠中心非独立单位。\n"
        "3. Berkson's Paradox：合成覆盖分布、反复暴露验证和极值诊断存在选择限制；非现场人群。\n"
        "4. Collider Bias：活动训练不按教师可用性筛掉记录，教师掩码只影响辅助损失；未依据预测挑主评分人口。\n"
        "5. Base Rate Neglect：保存TP/FP/FN、阴性与各族计数，Precision不外推现场发生率。\n"
        "6. Regression to the Mean：极好／极差图仅事后说明，没有按极差例选完再宣布总体修复。\n"
        "7. Survivorship Bias：48训练和24验证全记录保留，缺测／未知使用统一协议，未删除失败或低分例。\n"
        "8. Look-Elsewhere Effect：7臂14模型266配置全部保存，停止主方法，不提升较好打乱控制，不报告挑选p值。\n"
        "9. Garden of Forking Paths：loss/seed/epoch/teacher与继续门在评分前冻结；AP和零／均值读出为事后诊断，未调模型。\n"
        "10. Correlation != Causation：辅助读出低MAE与F1不等同，最终差异也包含各自校准和解码；不推断真实物理成因。\n"
        "11. Reverse Causality：离线居中图包含未来观察，不称实时预测；已知过程不提供检测器潜在标签或独立现场参考。\n",
        encoding="utf-8",
    )
    text = (root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v13.md").read_text(encoding="utf-8")
    start = text.index("> 中文完整研究初稿")
    stop = text.index("\n\n## Material Passport", start)
    text = text[:start] + (
        "> 中文完整研究初稿，2026-10-13，v14。整体形变异常为主，跳变／尖峰另识别且可同现。"
        "保留v13全部历史研究，新增QSG-VN局部图形监督完整强对照与负结果；96×730、"
        "48／24／24不变，保留内容未加载。公式采用LaTeX。稳定Agent/VLM核心贡献尚未成立。"
    ) + text[stop:]
    text = text.replace("Version Label: deformation-manuscript-20261013-v13", "Version Label: deformation-manuscript-20261013-v14", 1)
    text = text.replace("## 摘要", (
        "- QSG Verification: 72原观测栅格及教师、14保留权重、266校准配置、25方法600输出及统一评分精确重放；"
        "未重跑优化器，辅助事实与AP是事后诊断。\n\n## 摘要"
    ), 1)
    text = text.replace("当前证据支持减少错误扩展的开发价值及召回取舍，", (
        "再以查询局部图形变化监督小型视觉数值分类器，主F1为0.7114，"
        "低于同结构去监督0.7443及纯数值0.7930；六输出平均R²虽为0.0479，"
        "仍未转为检测收益，固定版停止。该分类器不属于语言VLM或Agent。"
        "当前证据支持减少错误扩展的开发价值及召回取舍，"
    ), 1)
    text = text.replace("not anomaly-detection performance.", (
        "not anomaly-detection performance. A further query-shape-supervised small visual-numerical classifier "
        "attains daily F1 of 0.7114 versus 0.7443 without auxiliary supervision and 0.7930 for its numerical-only "
        "control. A mean shape R-squared of 0.0479 does not translate into detection gains; this fixed classifier "
        "is stopped and is not a language VLM or Agent."
    ), 1)
    related = (
        "ViSTA[22]在既有图像token位置加入数值残差、冻结预训练主干，并保留语言输出接口；"
        "UTOPYA[23]组合多模态预测／分类／重建和物理正则。二者分别提示目标位置对齐和辅助任务的研究可能性，"
        "但并不能支持本项目效果。QSG-VN只在已有小CNN上检验无数值捷径的局部图形监督，"
        "不是两篇作者模型的复现，也不称辅助监督或多模态融合首创。本轮只读摘要与规定方法概览段落。"
        "VAN-AD[14]关于重构判别性的讨论提醒：形状重建更好不等于异常检出更好。\n\n"
        "TimeRLM[21]本轮403只限制这次源码访问；项目更早已在固定commit阅读其部分源码，"
        "最近邻记录保留。本稿v13的未读源码描述按本轮访问理解，不概括整个项目历史。\n\n"
    )
    text = text.replace("## 3 任务与方法", related + "## 3 任务与方法", 1)
    method = r"""### 3.19 查询局部图形监督：QSG-VN固定候选

QSG-VN（Query-Shape-Grounded Visual-Numeric learning）复用6通道32×61原栅格及小型两卷积分类器。与原数值特征拼接的活动头不变，新增六输出辅助头只读取图像编码，防止数值分支直接回答教师。它是小型视觉数值模型，不是语言VLM或Agent。设图像列可用的纵坐标为 $r_{t,k,a}$，取负还原位移方向，三个查询相邻列组为 $G_-=23{:}28$、$G_0=28{:}33$、$G_+=33{:}38$（Python半开索引）。三日列中位至少两完整NEU观测，每组至少四列可用，否则对应教师未知：

$$
m_{t,j,a}=\operatorname{median}_{k\in G_j}(-r_{t,k,a}),
\qquad
d_t=[m_{t,0}-m_{t,-};\ m_{t,+}-m_{t,0}]\in\mathbb R^6.
$$

教师只是归一化图形变化，不含被画图缩放消除的物理幅度。均值与标准差仅TRAIN估计。辅助掩码 $M_{t,i}$ 不删活动样本，两个任务共享图像编码 $E_\theta$：

$$
\widehat d_t=h_\phi(E_\theta(I_t)),\qquad
p_t=\sigma\!\left(f_\psi([E_\theta(I_t),z_t^N])\right),
\qquad
\mathcal L=\mathcal L_{\mathrm{BCE}}+
\frac{\sum_{t,i}M_{t,i}\operatorname{SmoothL1}(\widehat d_{t,i},\widetilde d_{t,i})}
{\sum_{t,i}M_{t,i}}.
$$

辅助头输出TRAIN标准化教师，导出再还原栅格单位；无有效辅助项的batch只优化活动项。全部10187合法TRAIN中心不变，固定seed20261014、12epoch、Adam .001、weight decay .0001、batch128、系数1。7臂各16／32宽和19活动阈值，总14权重266机会，同日F1+IoU.3事件F1择首个最大值。局部监督、去监督、整窗首中尾同尺寸教师、打乱TRAIN教师、纯图像有／无监督与纯数值均保留；原18同划分重训强基线不变。阶段头、缺测掩码与持续期解码共享。

"""
    text = text.replace("## 4 数据与实验协议", method + "## 4 数据与实验协议", 1)
    summary = read_json(run / "summary.json")
    arms = list(read_json(run / "protocol.json")["arms"])
    result = "### 5.21 QSG-VN：事实读出改善没有形成活动识别收益\n\n" + score_table(summary, arms)
    result += (
        "\n\n主F1 .711368，plain .744262，纯数值 .792971；强query_motion_mass .844167、"
        "global_motion_mass .837166、旧visual_numeric .794135、xgb_chart384 .780330。"
        "完整25方法表与八族各3条的分族计数另存；不能只用优于纯视觉作为足够创新证据。"
        "主相对plain去25FP、加3FP、补6TP、撤96TP，净−68；相对global净−4、"
        "shuffled净+4（但F1更低）、numeric净−200。继续门7项仅4项通过，"
        "核心匹配收益、视觉机制及数值增量失败，STOP_QUERY_SHAPE_GROUNDING。"
        "不得提升较好shuffled、更名或改变本版loss／teacher／threshold救分。\n\n"
        "补零变化与TRAIN均值事实控制：\n\n"
        "| 预测 | MAE（栅格纵坐标单位） | 六输出平均R² |\n| --- | ---: | ---: |\n"
    )
    shape_scores = read_json(report / "query-fidelity-baselines.json")["scores"]
    for name in ("zero", "train_mean", "query_shape_grounded", "query_shape_plain", "query_shape_shuffled", "query_shape_visual"):
        row = shape_scores[name]
        result += f"| {name} | {row['mae']:.6f} | {row['r2_uniform']:.6f} |\n"
    result += (
        "\n主有少量真实图形读出收益，六维R²均正但仅.0018—.1337，不可只比未监督head宣称可靠读取。"
        "纯图像监督R²更高却日F1更低，说明教师与主任务判别不等价。"
        "global head训练不同教师，其对局部教师的误差仅跨目标诊断，不作为匹配重建结果。"
        "此处单位不同于5.20，MAE不得直接跨探针比较。\n\n"
        "事后未解码已知supported日AP：主.765724、plain .770000、global .775760、"
        "shuffled .765027、纯视觉监督.631404／去监督.675962、纯数值.799170。"
        "未重新调权重或阈值，不能替代活动／事件评分。原图极值复核0403撤51TP净−51，"
        "较弱第一活动整段漏掉；0395去15FP、撤1TP净+14，仅说明更保守取舍。"
        "两例实际view_image，参考未改，非盲标／专家判断。\n\n"
        "72观测和教师独立重构、TRAIN均值／尺度、全部14权重和266选择机会、600活动／阶段输出及"
        "评分精确重放，最大概率误差0，未重跑优化器。9业务测试通过，集中Ruff通过；"
        "验证器完成后lambda仅改def，实际运行源码保留，不改训练冻结。"
        "原图／教师准备25.295秒、全14模型训练78.701秒，含消融，不称部署成本。"
        "0新API／记录／编码；24保留内容未加载。\n\n"
    )
    text = text.replace("## 6 讨论", result + "## 6 讨论", 1)
    discussion = (
        "### 6.9 可读图形事实与任务判别的差异\n\n"
        "QSG教师可以从图像读取，但总体解释率有限；形状变化也同时包含测量扰动和形变。"
        "有序局部变化并不自动给出每一日活动归属，辅助损失还可能偏离稀少活动的判别目标。"
        "本轮数据不能区分容量、损失竞争与事实无关性各自贡献，不能据结果猜定唯一原因。"
        "固定版停止，不继续无依据loss／seed／prompt搜索。下一项机制必须先显示可纠正且独特的"
        "活动证据价值，并与已有残差、空间读出、竞争拟合和权限方案区分；当前没有新方法通过此条件。\n\n"
    )
    text = text.replace("## 7 结论", discussion + "## 7 结论", 1)
    text = text.replace("## 数据、代码与AI使用说明", (
        "局部图形自监督小试也未形成整体检测增量。已有对比与消融可以支撑可复核开发结论，"
        "但还不能完成用户要求的有效Agent/VLM核心贡献、独立验证或发表就绪结论。\n\n"
        "## 数据、代码与AI使用说明"
    ), 1)
    text += (
        "\n[22] Gao J, Shi Y, Hu P, Harrison E M. ViSTA: A Simple Bridge Extends Visual Alignment to Clinical "
        "Time-Series Understanding in Multimodal LLMs. [arXiv:2609.31448v1](https://arxiv.org/abs/2609.31448v1), "
        "2026-09-25。实际读摘要、引言设计段，未复现。\n\n"
        "[23] Pessoa R W S, Amblard J, Russo A, Nogueira I B R. UTOPYA: A Multimodal Deep Learning Framework "
        "for Physics-Informed Anomaly Detection and Time-Series Prediction. "
        "[arXiv:2605.18188v1](https://arxiv.org/abs/2605.18188v1), 2026-05-18。实际读摘要与方法概览，未复现。\n\n"
        "QSG复现入口：run/verify/report_landslide_query_shape_grounding.py；"
        "完整结果在query-shape-grounding-report-v1，实际阅读／看图及统计边界在query-grounding-session-v1。"
        "冻结训练和所有旧结果保留，0新记录／API，未打开保留输入／参考。\n"
    )
    target = root / "docs/gnss-deformation-manuscript-draft-2026-10-13-v14.md"
    if target.exists():
        raise FileExistsError("Preserve complete manuscript versions")
    target.write_text(text, encoding="utf-8")
    snapshot = base / "manuscript-v14"
    snapshot.mkdir(exist_ok=False)
    (snapshot / "draft.md").write_bytes(target.read_bytes())
    bindings = read_json(base / "manuscript-v13/evidence-bindings.json")
    for path in [folder / "freeze.json" for folder in folders] + [
        session / "actual-visual-review.md", session / "reading-notes.md", session / "statistical-interpretation.md",
    ]:
        bindings[str(path.resolve())] = sha(path)
    for path, expected in bindings.items():
        if sha(Path(path)) != expected:
            raise ValueError("Bound evidence changed: " + path)
    write_json(snapshot / "evidence-bindings.json", bindings)
    write_json(snapshot / "manifest.json", dict(
        source=str(target.resolve()), source_sha256=sha(target), decision=read_json(run / "decision.json"),
        contribution_validated=False, independent_field_reference=False, reserved_content_loaded=False,
        files={path.name: sha(path) for path in snapshot.iterdir()},
    ))
    print(dict(manuscript=str(target), bindings=len(bindings)), flush=True)


if __name__ == "__main__":
    write(Path(__file__).resolve().parents[1])
