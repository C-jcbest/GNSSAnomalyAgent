"""Export weak-reference scoring and document the stopped fixed visual candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import BASE, ROOT, csv_rows, export
from probe_landslide_visual_invariance import TRAINING, calibration_labels, verify
from report_landslide_core_lockbox import full_table
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_core_lockbox import score_target

REPORT = BASE / "visual-invariance-calibration-report-v1"
RUN = BASE / "visual-invariance-development-v1"
DELIVERY = BASE / "manuscript-v44"


def complete_scoring():
    """Use the actual process weak attribute, preserving the frozen gate."""
    verify(RUN)
    verify_bindings(RUN, "calibration-freeze.json")
    verify_bindings(REPORT, "manifest.json")
    out = BASE / "visual-invariance-validation-scoring-v1"
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    summary_by_target = {}
    for target, source in TRAINING.items():
        ids = read_json(RUN / "protocol.json")["calibration_splits"][target]["validation"]
        pairs = {cid: load_case_features(source, cid) for cid in ids}
        labels = calibration_labels(source, pairs)
        predictions = read_json(REPORT / target / "predictions.json")
        # Process weak motion is a separate attribute, not an A/S/D feature.
        # The author calibration CSV contains no independent weak-day reference.
        if target == "primary_process":
            weak = {cid: np.array([int(row["activity_label"]) == 1 and int(row["weak_signal"]) == 1
                                   for row in rows]) for cid, rows in labels.items()}
        else:
            weak = {cid: np.zeros(len(rows), dtype=bool) for cid, rows in labels.items()}
        results, summary, records = score_target(
            labels, {cid: pair[1].observed for cid, pair in pairs.items()}, weak, predictions, {}, {},
        )
        previous = read_json(REPORT / target / "results.json")
        if results != previous:
            raise ValueError("Supplementary weak scoring changed frozen task metrics")
        old_rows = {row["method"]: row for row in csv_rows(REPORT / target / "summary.csv")}
        for row in summary:
            for field, value in row.items():
                if field in {"weak_days", "weak_detected_days", "weak_recall"}:
                    continue
                old = old_rows[row["method"]][field]
                expected = "" if value is None else str(value)
                if old != expected:
                    raise ValueError(f"Metric changed outside weak reporting: {field}")
        folder = out / target
        export(folder / "results.json", results)
        write_csv(folder / "summary.csv", summary)
        write_csv(folder / "per-record.csv", records)
        summary_by_target[target] = summary
    export(out / "verification.json", {
        "all_frozen_daily_event_boundary_stage_metrics_unchanged": True,
        "weak_reference": {"primary_process": "Known activity intersect original weak_signal",
                           "secondary_developer": "Unavailable in the original calibration CSV; no invented denominator"},
        "reporting_correction": "Initial replay report used slow_displacement feature and omitted process weak denominator; preserve it and supplement original weak_signal",
        "calibration_gate_unchanged": True, "retraining": False, "new_api_calls": 0,
    })
    return summary_by_target


def write_documents(summaries):
    decision = read_json(RUN / "calibration-decision.json")
    if decision["decision"] != "STOP_VISUAL_INVARIANCE_CALIBRATION":
        raise ValueError("This report documents only the registered stopped candidate")
    manuscript_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    manuscript = manuscript_path.read_text(encoding="utf-8")
    if "完整中文初稿v43" not in manuscript or "### 6.4.8" in manuscript:
        raise ValueError("Archive and update v43 exactly once")
    neural_seconds = sum(float(row["training_wall_seconds"]) for row in csv_rows(RUN / "training-cost.csv"))
    mechanism = r"""
这是受监督的小型图像/数值双输入活动模型，不是大语言VLM或Agent。每个目标日取原始NEU前后90日的61个不交叠3日中位列；每列至少2个完整日。三条32行栅格与三条可用性掩码构成六通道输入，不插补不可用列、不反转时间。每轴减局部中位数，纵幅至少为六倍日噪声代理。数值18维与既有XGB84相同，包括原12特征及42/84日原始持续运动速率、噪声比和方向；训练均值/尺度归一化并另给18不可用指示。

令 $I_t$ 为原始栅格、$z_t$ 为数值输入、$\mathcal R_s$ 为逐分量坐标符号反射，模型活动概率为 $p_\theta$。三轴符号独立改变，时间、原始缺测和持续活动目标保持。训练使用正类加权二元交叉熵 $\ell_w$：

$$
\mathcal L=
\frac{\ell_w(p_\theta(I_t,z_t),y_t)+
\ell_w(p_\theta(\mathcal R_s I_t,z_t),y_t)}{2}
+\lambda\left[p_\theta(I_t,z_t)-p_\theta(\mathcal R_s I_t,z_t)\right]^2.
$$

主一致性臂固定 $\lambda=1$，仅增广臂为零；普通臂不反射。其余消融为去数值输入的视觉臂和纯数值MLP。两层3×3卷积为8/16通道，2×5池化后与数值拼接，头宽16或32。每臂固定同seed、12 epochs、两头宽×三个阈值六个选择机会；无早停或seed/lambda搜索。数值支撑、14日持续性和目标各自已冻阶段头保持。训练增广不要求推理时八次投票或硬共识弃判。

同信息XGBoost从每列栅格的第一矩恢复183个位置与183个可用性，再加18个数值特征，共384维；同eligible训练日的八种坐标符号版本都进入训练。120树、深度2/4、同三个阈值六机会。它获得新画布的时间信息，防止把额外观察信息误称视觉网络贡献。
""".strip()
    review = """
## 实际视觉和原始数值辅助复核

当前Codex实际查看两张训练输入栅格、四张过程验证原始NEU/概率/逐日输出图和两张作者验证图。四/两记录按评分后普通训练及纯数值净纠错极值选择，选择规则已导出，仅解释错误，不改标签、全人口或准入。

- 0026：三轴在约230日一次跳变后转入平台。中位块[196,210)为−.051/−.327/2.924mm，[238,252)为−11.783/5.540/−7.036mm；其后N在−11.8附近，E/U继续起伏。主视觉相对纯数值新增43个过程误报日，主要为[201,242)，其中很多在阶跃前。居中图像能看见未来阶跃，但未必能辨别持续活动。
- 0031：约530日有一次三分量偏移，之前局部起伏、之后平台。主臂相对纯数值新增32个误报日[526,558)。[518,532)中位数.196/−.375/.355mm，[532,546)为−1.085/−1.638/−3.608mm，[546,560)为−1.459/−1.654/−3.481mm；不是两次清晰持续累积的证据。反射一致不等于阶跃可靠性。
- 0035：两次明显累积之间保留高位平台。相对普通训练净+33日，主要恢复第一段尾部并减少第二段起点提前误报；对纯数值净+26日。原E中位数[224,238)82.822、[238,252)84.757、[252,266)84.569mm，尾缘应结合局部累积与平台判断，不能仅靠高位像素。
- 0036：主臂相对普通训练净+18、相对纯数值净+28日，恢复部分弱尾；但约465日U偏移前仍新增平台误报。原N中位数[504,518)−90.661、[532,546)−91.924、[588,602)−93.485mm显示较慢持续变化；E较平，U有噪声。因此弱尾恢复有原始依据，尚非完整解决。
- 0050作者参考：偏移后的N/E平台与U起伏；主臂相对普通/纯数值各损20个已确认静稳日，[536,554)为新增误报。参考未知日不计得失；不能把未知范围内预测撤回算作正确静稳。
- 0057作者参考：中段N逐渐累积、E缓慢上升、U噪声较大。主臂相对普通训练净+91、相对纯数值净+85日；但相对仅增广净−3，且恢复与分段切断都存在。作者部分收益不能改变过程准入失败。

这些是作者已知历史研究后的AI视觉/数值复核，不是独立领域专家或现场金标准；没有回填新标签。先核查原始持续位移，再解释活动内阶段；阶段头未创新，内部阶段指标改变主要来自活动门控。
""".strip()
    (REPORT / "visual-review.md").write_text(review + "\n", encoding="utf-8")
    text = ["# 坐标反射训练一致性：校准收益与停止结论", "",
            "## Material Passport", "",
            "- Origin Skill: academic-research-suite / experiment-agent，run/validate inline。",
            "- 日期：2026-10-11；范围：固定候选校准及已冻权重重放。",
            "- 状态：校准完成、重放通过；STOP_VISUAL_INVARIANCE_CALIBRATION；完整开发/未见确认未执行。",
            "- 参考：原过程24训练/12验证、作者6训练/6验证；作者开发暴露，非独立专家。", "",
            "结论：主臂过程日F1 .894128高于普通训练.863181和仅增广.883966，"
            "弱召回也提高；但低于纯数值.900979和同信息XGBoost.912946。"
            "预登记要求对纯数值日F1不下降，因此停止本固定候选。"
            "局部收益真实保留，不把训练概率更一致当总体识别提升，不搜索参数救分。", "",
            "## 实现与冻结", "", mechanism, "",
            "过程与作者分别训练/校准，作者结果不决定过程模型。每臂六机会，目标为日F1+事件F1；"
            "各臂最优阈值不同，因此选中模型间差异同时包含训练与正常阈值校准，不能声称纯损失函数因果效应。"
            "包括两种头宽的全部72校准配置、20神经权重、4数值模型与历史损失保留。", "",
            "## 校准与最少消融", ""]
    for target, rows in summaries.items():
        text.extend(["### " + target, "", full_table(rows), ""])
    text.extend([
        "过程弱分母839日来自原weak_signal∩确定活动；712/839=.848629，"
        "普通641/839=.764005、仅增广698/839=.831943、纯数值和同信息XGB均666/839=.793802。"
        "作者校准CSV没有独立弱属性，弱召回保持不可评价，不从预测或生成机制补分母。"
        "初版权重重放报告误沿用slow_displacement字段，过程弱分母显示为空；"
        "已保留原版，并在visual-invariance-validation-scoring-v1补正确属性。"
        "全部日/事件/边界/阶段指标、选择与准入逐值不变。", "",
        "相对普通训练过程188得/67失、净+121，8条净受益；相对仅增广72得/37失、净+35，7条受益。"
        "相对纯数值112得/151失、净−39，5条受益；同信息XGB92得/196失、净−104。"
        "作者对普通净+102、纯数值+73、同信息XGB+50，但对仅增广−3。"
        "参考未知日不给正确静稳信用，所有原验证记录都参与。", "",
        "选中过程主臂反射概率平均绝对差.028070，低于普通.069907和仅增广.032190，"
        "其余三项登记检查通过，唯一失败是日F1对纯数值不下降。"
        "反射差是相同输入语义稳定性的描述性指标，没有独立扰动标签或现场可靠性证明。"
        "事件F1主.727273、普通.75、纯数值.647059；边界误差只在各自匹配事件上统计，"
        "不同匹配集合不能直接推出共同物理边界改善。", "", review, "",
        "## 来源、验证与成本", "",
        "时序转图/CNN、双视图增强、时序不变性训练已有先例。"
        "核查Wang/Oates官方摘要，TS-TCC官方摘要/README/固定SHA增强短源码，"
        "WinoTS引言/相关工作与部分方法正文；未运行其模型，未声称全文阅读或通用机制首创。"
        "详见[阅读范围](../artifacts/landslide-adaptive-review-2026-10-09/visual-invariance-literature-v1/reading-notes.md)。", "",
        "3个输入业务测试及Ruff通过。冻结20神经权重中的选中10个及选中2个XGB"
        "按原验证日期重新推理，神经逐日预测、全部选中聚合分数以及七反射概率统计与冻结结果精确一致；"
        "78840日历输出检查原缺测、数值支撑及活动内阶段门控。补评分复用原逐日输出，"
        "独立sklearn日指标与一对一事件计数通过，阶段、边界及未知统计不变。"
        "这证明选中权重推理可重放，未重新训练全部20模型，不能声称训练逐字节重现。", "",
        f"20个神经模型训练循环墙钟合计{neural_seconds:.6f}秒，两个CPU线程、无GPU；"
        "未计画布预处理、XGB训练、验证选择或文件读写，不能当总体训练成本。"
        "选中主模型原图+七反射的验证重放为过程12条1.579520秒/作者6条.796941秒，"
        "含加载和输出契约步骤、未含所有共同预处理与评分；不是部署单次原图延迟。"
        "监督训练标签成本不能当免费零样本能力。新增API/记录0，累计仍3825启动/3823回执、"
        "至少16694503报告token，受控192/半合成60不变。", "",
        "原108过程/96作者完整开发方法结果与旧API成本保持；本轮停止于校准，"
        "不把校准12/6结果混入旧完整24/12表，也不事后更换主臂或挑记录进入确认。"
        "原五停止/未确认决定保持，goal active；稳定方法贡献和独立可信确认仍未完成。", "",
        "## 复现", "",
        "运行probe_landslide_visual_invariance.py prepare/calibrate到未存在的新目录可重训；"
        "固定seed与环境但未实际验证整训练逐字节复现。运行report_landslide_visual_invariance_calibration.py"
        " --run 冻结目录 --out 新报告目录可重放已冻权重，不请求远程模型。"
        "源码与数据哈希由freeze/source-bindings/calibration-freeze绑定；本轮准备、"
        "训练、权重、全校准选择、反射、纠错、原图和数值块均保存。",
    ])
    (ROOT / "docs/landslide-visual-invariance-results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    section = """
### 6.4.8 坐标反射训练一致性的校准探索

本轮改为可训练小型图像/数值双输入活动模型，区别于推理时反射硬共识。候选与全部消融在训练前固定，不用新开发记录或提示词搜索。

""" + mechanism + """

过程12条验证的主F1 .894128，普通训练.863181、仅增广.883966、视觉单输入.884963、纯数值MLP.900979、同信息XGBoost.912946。主弱召回712/839=.848629，普通.764005、仅增广.831943、纯数值/同信息XGB.793802。相对普通净+121日，相对仅增广+35，对纯数值−39，同信息XGB−104。主反射概率差.028070小于普通.069907和仅增广.032190；更一致仍不等于更准确。

作者6条验证主F1 .946121，普通.882695、仅增广.947652、纯数值.899008、同信息XGB.911765；对仅增广净−3。作者目标不选择过程模型，弱分母缺失保持不可评价。初版报告误用slow_displacement而漏掉过程弱分母，已用原weak_signal补充；全部日/事件/阶段、选择和准入保持原值。各臂独立同机会选阈值，不能把选中模型差异全归因于一致性损失。

预登记校准唯一未满足项是主日F1对纯数值不下降，决定STOP_VISUAL_INVARIANCE_CALIBRATION；不调seed、lambda、epoch、表示或子集救分，也不进入完整24/12开发或未见确认。该校准人口与旧完整开发不同，不跨表比较F1。完整P/R/F1、IoU .3/.5事件、端点MAE、活动内/端到端阶段与成本见[完整报告](landslide-visual-invariance-results.md)，原108/96完整开发组保持。

实际视觉/数值复核四过程及两作者图：0026/0031的阶跃前后新增过程误报，对纯数值各净−43/−32；0036弱尾部分恢复，对纯数值净+28。弱尾具有原N逐步累积依据，但一致性没有消除阶跃误判。作者复核者有历史暴露，非独立专家，不改参考。

Wang/Oates[27]、TS-TCC[28]、WinoTS[29]已有图像时序及语义保持增强/一致性先例，不主张通用原理首创。选中权重逐日预测及聚合分数精确重放，78840日契约检查，3业务测试/Ruff通过。20神经训练循环墙钟82.4192秒，不含画布、XGB训练、验证与读写，未验证整训练复现；0新增API/记录。阶段头沿用原目标模型，阶段指标变化不构成原生视觉阶段创新。

"""
    manuscript = manuscript.replace("完整中文初稿v43", "完整中文初稿v44", 1)
    manuscript = manuscript.replace(
        "更新前v42主稿、三状态及差图报告原字节存档。新增局部端点拟合、统一控制/消融及四图视觉数值复核；候选未通过登记并停止。",
        "更新前v43主稿及三状态文档原字节存档。新增坐标反射训练一致性、五臂/同信息XGB校准消融、冻结权重重放及六图视觉数值复核；候选停止于校准。",
        1,
    )
    manuscript = manuscript.replace("### 6.5 阶段与计算成本", section + "### 6.5 阶段与计算成本", 1)
    manuscript += """

[27] Wang, Z., & Oates, T. (2015). Imaging Time-Series to Improve Classification and Imputation. [arXiv:1506.00327v1](https://arxiv.org/abs/1506.00327v1). 本轮仅读官方摘要；未阅读全文或运行作者模型。

[28] Eldele, E., Ragab, M., Chen, Z., Wu, M., Kwoh, C. K., Li, X., & Guan, C. (2021). Time-Series Representation Learning via Temporal and Contextual Contrasting. [arXiv:2106.14112v1](https://arxiv.org/abs/2106.14112v1). 本轮读官方摘要、作者README前90行及增强短源码，固定SHA15bdfd6eced6aeb6c638839828effe1b4b2ea391；未阅读全文或运行作者模型。

[29] Major, N., Razmadze, K., & Shavit, Y. (2026). WinoTS: Wavelet-based Self-Distillation for Time Series Models. [arXiv:2609.39337v1](https://arxiv.org/abs/2609.39337v1). 本轮读官方摘要、引言/相关工作与部分方法至Objective开头；未读完整目标/实验或全文，未确认作者源码、未运行模型。各引用的实际范围见本轮reading-notes。
"""
    manuscript_path.write_text(manuscript, encoding="utf-8")
    state = """
## 2026-10-11：坐标反射训练一致性停止于校准，初稿v44

[完整结果](landslide-visual-invariance-results.md)。新候选为受监督图像/数值双输入CNN，不是大语言VLM/Agent；按原过程24/12、作者6/6分别校准。训练前冻结seed、12epochs、两头宽×三阈值六机会。主一致性、仅增广、普通、去数值、纯数值五臂及获得全部画布列信息的384维XGB共72配置、20神经/4XGB模型保留，没有按分数重调。

过程主F1 .894128，普通.863181、仅增广.883966、纯数值.900979、同信息XGB.912946；主弱R .848629，对普通/增广提高，对纯数值净−39日。反射概率差.028070低于普通.069907/增广.032190。作者主.946121、增广.947652，对增广净−3。校准日F1不低于纯数值的登记项失败，STOP_VISUAL_INVARIANCE_CALIBRATION；不进入完整开发或新确认，不调seed/lambda/epoch/画布/子集救分。旧五决定、108/96完整开发指标/API成本保持。

实际查看两训练输入、四过程及两作者原始NEU/概率/输出图，并核对14日原始中位块：0026/0031阶跃被视觉误报，对纯数值各净−43/−32；0036弱尾有部分恢复、净+28。作者已知历史结果，非独立专家或现场金标准，不改标签。TS-TCC固定SHA增强源码/WinoTS部分方法已读，未运行作者模型；不称CNN/增强/一致性首创。

选中10神经/2XGB权重冻结推理、逐日预测与聚合分数精确重放，78840日契约检查，3业务测试/Ruff通过。过程弱分母初版误用feature而缺失，保留原报告，用原weak_signal补839日；其余指标/选择/准入严格不变，作者无弱分母保持不可评价。20神经训练循环82.4192秒不含预处理、XGB训练或完整选择，未验证全训练逐字节重现。

新增API/记录0，累计仍3825启动/3823回执、至少16694503报告token，受控192/半合成60不变。v43四可变文档原字节存档，243文件旧manifest保持并按档案映射核验。v44公式继续LaTeX。goal active：局部和弱召回收益存在，但固定候选未通过准入，稳定贡献与独立可信确认尚未完成。下方为历史记录。

"""
    for name in ("project-status.md", "landslide-adaptive-review-method.md", "landslide-submission-roadmap.md"):
        path = ROOT / "docs" / name
        heading, remainder = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(heading + "\n\n" + state.strip() + "\n\n" + remainder.lstrip("\n"), encoding="utf-8")
    export(DELIVERY / "run-result.json", {
        "status": "progress", "candidate_decision": decision["decision"],
        "goal_complete": False, "scope": "Calibration, ablation, weight replay, raw visual/numerical review and manuscript",
        "new_api_calls": 0, "new_records": 0,
    })


if __name__ == "__main__":
    write_documents(complete_scoring())
