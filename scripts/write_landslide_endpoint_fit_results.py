"""Write frozen endpoint results and update the previously archived manuscript."""

from __future__ import annotations

from probe_landslide_endpoint_fit import BASE, ROOT, TARGETS, csv_rows, method_name
from report_landslide_core_lockbox import full_table
from run_landslide_window_optimization import read_json


def paired_table(rows):
    lines = [
        "| 目标 | 重复 | 规则 | 修订F1 | F1差 | 净正确日 | 弱R前→后 | 共同匹配数 | 共同端点MAE前→后 |",
        "|---|---|---|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        if not row["base"].startswith("direct_visual"):
            continue
        lines.append(
            f"| {row['target']} | {row['base'][-1]} | {row['rule']} | "
            f"{float(row['treatment_f1']):.6f} | {float(row['f1_difference']):+.6f} | "
            f"{row['net_correct']} | {float(row['control_weak_recall']):.6f}→"
            f"{float(row['treatment_weak_recall']):.6f} | {row['common_matches']} | "
            f"{float(row['common_endpoint_mae_before']):.6f}→"
            f"{float(row['common_endpoint_mae_after']):.6f} |"
        )
    return "\n".join(lines)


def report():
    analysis = BASE / "endpoint-fit-analysis-v1"
    comparisons = csv_rows(analysis / "comparisons.csv")
    verification = read_json(BASE / "endpoint-fit-report-v1/verification.json")
    formula = r"""
对每个已有事件的onset或offset，使用端点前后至多98日原始NEU，14日不交叠中位块，每块至少8个完整日、总计至少6块。搜索原端点前后49日整数日期，每侧至少2块、事件至少保留28日。三轴分别拟合截距和斜率：

$$
h_{\mathrm{on}}(t;k)=\max(t-k,0),\qquad
h_{\mathrm{off}}(t;k)=\min(t-k,0),\qquad
\widehat{x}_{a}(t;k)=\alpha_a+\beta_a h(t;k).
$$

平稳、整窗线性漂移及单阶跃作为竞争描述。交替块两折分别在训练块选端点与系数，然后算留出块误差：

$$
E_m^{(q)}=\frac{1}{3|\mathcal H_q|}
\sum_{t\in\mathcal H_q}\sum_{a\in\{N,E,U\}}
\left(\frac{x_a(t)-\widehat{x}_{m,a}^{(q)}(t)}{\sigma_a}\right)^2.
$$

选择性规则要求每折转折误差严格小于三个竞争模型，且全量/两折端点跨度不大于14日；相对误差$10^{-12}$仅用于算术平局。两折来自同一相关观测，不是独立统计检验或校准置信度。平稳不等于地质稳定，转折胜出也不确立原因。

消融为去掉单阶跃竞争、所有可拟合端点无条件应用，以及原未修订输出。全部原未知和缺测保持；不跨大于7日未知；相邻事件共同保留一个原确定静稳分隔日，两端联合修订不足28日不应用。记录截断端点保持，不新建或整段撤销事件。原活动未变日的阶段原值保持，活动变化日使用该目标已冻XGBoost阶段头重门控。
"""
    text = [
        "# 局部原始位移端点拟合：完整开发否定结果", "",
        "结论：STOP_ENDPOINT_FIT。两遍直接视觉的过程日F1、净确定纠错和共同事件边界均退化，"
        "作者日F1也退化。不进入新记录确认，不把较好的单例、作者第二遍自适应或数值组升级为主方案。", "",
        "完整24过程记录与预指定12作者记录均已开发暴露，原标签不改。"
        "新预测先冻结，后读参考评分；这隔离本轮调分顺序，不能消除历史方法选择暴露或建立独立专家金标准。", "",
        "## 机制与评分前登记", "", formula.strip(), "",
        "主比较固定为direct_visual_r0/r1＋selective。登记条件：过程平均F1严格提高、净正确总和为正、"
        "平均弱R不降、每遍至少2记录受益、同一参考事件/预测身份的共同端点MAE下降；"
        "选择性平均F1须优于无条件拟合；作者平均F1和弱R不降且净正确总和为正。"
        "作者确认内核不是精确物理端点，不将其MAE作为准入条件。参数、竞争模型、折与主组不按分数调整。", "",
        "## 两遍主比较与消融", "", paired_table(comparisons), "",
        "直接视觉selective与no_step_rival在两个目标四遍全部逐日输出相同，未证明step竞争在主组的独立作用。"
        "全锚点有过程10/作者9个记录输出差异，完整结果保留。选择性比无条件日F1更好，"
        "但两者都低于未修订主组，不能以相对较少损失主张识别增益。", "",
        "过程第一遍FP601→449、FN1301→1646，第二遍FP805→666、FN1294→1586。共同匹配各19个，"
        "MAE24.6842→29.7895和27.7895→31.9211日，池化退化4.6184日。事件F1仍.8085/.8085，"
        "事件数量保持不意味着边缘正确。作者共同边界MAE下降，但净正确−37/−42且日F1退化，"
        "不能将更靠近保守内部起止解释为总体改善。", "",
        "## 固定、自适应及全部强数值的同机制控制", "",
        "过程19锚点×3、作者16锚点×3；所有适用强数值接受同一规则，使用各自原目标模型输出。"
        "旧51/48组指标和API部署成本严格保持，统一结果扩至108/96组。"
        "新规则未新增输入特征或训练，旧模型仍适用于相同记录/任务；不将过程训练模型套作作者基线。", "",
    ]
    for target in TARGETS:
        rows = csv_rows(analysis / target / "summary.csv")
        numeric = ["pelt_rule", "raw_multiscale_rule", "xgboost", "xgboost_84"]
        visual = [
            f"{stem}_r{rep}{suffix}" for rep in (0, 1) for stem, suffix in (
                ("direct_visual", ""), ("adaptive_no_feedback", "__conflict_scope"),
                ("fixed_image_review", "__whole_target"), ("same_count_fixed_early", "__conflict_scope"))
        ]
        chosen = set(visual + numeric)
        chosen.update(method_name(name, "selective") for name in visual + numeric)
        text.extend(["### " + target, "", full_table([row for row in rows if row["method"] in chosen]), ""])
    text.extend([
        "完整过程所有19锚点的选择性F1均下降。XGB84 .884790→.818612，原始多尺度规则.827595→.790574。"
        "作者部分数值改善，例如XGB.800924→.815155、原始多尺度规则.836711→.842466；"
        "作者自适应第二遍.879151→.881332。这些结果不替代预指定直接视觉主组，不支持Agent独特优势。", "",
        "## 实际视觉与原始数值复核", "",
        "实际查看0169、0171、0176、0188四张原始NEU/局部拟合/两遍对齐图，见"
        "[实际复核记录](../artifacts/landslide-adaptive-review-2026-10-09/endpoint-fit-report-v1/visual-review.md)。"
        "这是评分后解释，不作为新增独立标签。", "",
        "0169第一遍尾451→429使过程净+15，第二遍未修原尾、净−7；作者两遍净−7。"
        "0171第二次起变460→491更靠近明显增速部分，作者两遍净+3，但完整过程−52/−61；"
        "拟合可能忽略此前较弱积累。0176尾339→311后原N/E仍累积，过程各−51，作者各−12。"
        "0188尾300→251确有平台改善，过程净+35/+33，长平台错误事件仍在。"
        "单转折能胜有限竞争且knot稳定，仍可能削掉渐进首尾。后续需要新增持续弱运动的证据，"
        "不换拟合范围、惩罚或阈值救本轮。", "",
        "## 验证、成本、来源与复现", "",
        "五业务测试和Ruff通过：清晰转折、静稳/漂移/阶跃反例、缺测/未知/截断、"
        "坐标反射与平移、多事件分隔。冻结前拓扑修正使按内部reason写的缺口断言两次失败；"
        "改为不可修订/不跨缺口的业务断言后通过，未用开发分数改算法。", "",
        "新增1419120个日历输出通过原未知、缺测、阶段门控和逐事件身份检查；"
        "4个预测/端点证据文件在独立prepare重放逐字节一致，14评分输出亦逐字节一致。"
        "4806端点诊断、5832直接视觉三状态转移已导出；参考未知日不计正确静稳。", "",
        f"本地拟合/缓存计时{verification['local_fit_wall_seconds']:.6f}秒，"
        f"三规则合计应用{verification['local_rule_application_wall_seconds']:.6f}秒；"
        "perf_counter为本地路径墙钟，非纯CPU，不含加载、特征、阶段头预测或评分。"
        "多锚点/规则共享拟合，不能推算单方法总时间。新输出请求/token继承对应锚点；"
        "0新增API/记录，无重试或推理重启。累计仍3825启动/3823回执、至少16694503报告token，"
        "受控192/半合成60不变。", "",
        "CPOP已有连续分段线性、加权残差变点拟合，不主张hinge或边界拟合首创。"
        "仅读两篇官方摘要及CRAN镜像DESCRIPTION/部分R入口与方法；"
        "另读2026 LLM-Augmented Changepoint Detection官方摘要，未运行这些模型或复制GPL源码。"
        "定向检索非穷尽最近邻，见[阅读范围](../artifacts/landslide-adaptive-review-2026-10-09/endpoint-fit-literature-v1/reading-notes.md)。", "",
        "产物为endpoint-fit-development-v1、endpoint-fit-analysis-v1、endpoint-fit-prediction-replay-v1、"
        "endpoint-fit-replay-v1及endpoint-fit-report-v1。完整逐日预测、纠错、共同边界、"
        "全部阶段/事件/日指标与API成本见analysis目录。", "",
        "~~~powershell",
        "$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'",
        "$env:PYTHONIOENCODING='utf-8'",
        "# 选择尚不存在的新目录；均不调用远程模型。",
        "& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/probe_landslide_endpoint_fit.py prepare --out artifacts/landslide-adaptive-review-2026-10-09/endpoint-fit-user-replay",
        "& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/probe_landslide_endpoint_fit.py score --out artifacts/landslide-adaptive-review-2026-10-09/endpoint-fit-user-replay --analysis artifacts/landslide-adaptive-review-2026-10-09/endpoint-fit-user-analysis",
        "~~~", "",
        "goal active。旧NOT_CONFIRMED、STOP_FEEDBACK_PROMOTION、STOP_AXIS_WITNESS_BEFORE_CALLS、"
        "STOP_COMPARISON_DISPLAY保持。新增STOP_ENDPOINT_FIT不掩盖原确认与迁移负结果；"
        "稳定改进和独立现场验证尚未完成。",
    ])
    fence = chr(96) * 3
    (ROOT / "docs/landslide-endpoint-fit-results.md").write_text(
        ("\n".join(text) + "\n").replace("~~~", fence), encoding="utf-8"
    )
    main = [row for row in comparisons if row["base"].startswith("direct_visual") and row["rule"] == "selective"]
    section = """
### 6.4.7 局部连续转折修订端点的否定结果

新增离线候选只修已有活动边缘，保留事件身份和全部未知。完整开发人口两遍、强数值/固定/自适应同机制和两项消融在评分前冻结，不增加API或更换主组。
""" + formula + "\n" + paired_table(main) + """

过程两遍同一参考事件/预测身份各19个，共同端点MAE池化增加4.6184日。第一遍FP减少152而FN增加345，第二遍FP减少139而FN增加292；事件F1不变仍不意味着边缘改善。选择性比无条件日F1更好，但两者都低于未修订。主组去step竞争逐日输出相同，不能主张该竞争项的独立作用。

全部过程19锚点选择性F1均下降，XGB84 .884790→.818612；作者部分数值有收益，但直接视觉F1退化。作者更小边界MAE只是更接近保守确认内核，不能替代日识别效果或称物理边界精度提高。旧51/48组指标和API部署成本保持，新增后108/96完整结果见[端点拟合报告](landslide-endpoint-fit-results.md)。

作者实际读0169/0171/0176/0188原始三轴与局部拟合图：0176停止339→311时N/E仍累积，过程各损51日、作者各损12日；0188停止300→251减少平台误报、过程净+35/+33。渐进起止的单转折折点可落入持续运动内，即使两个描述性折都胜过竞争且折点稳定。依登记STOP_ENDPOINT_FIT，不调范围、阈值或折救分。该候选不加入本稿主方法。

CPOP[24,25]已有连续分段线性、加权残差与独立变点网格；本轮读官方摘要和CRAN镜像部分方法/入口，不称拟合首创或完整复现。最新LLM集成变点方法[26]仅读摘要，未运行发布模型。1419120日契约检查、独立prepare4预测/证据及14评分文件逐字节重放通过。本地拟合/缓存11.6754秒、三规则应用.0719秒为局部路径墙钟，不是单方法总CPU；0新增API，部署请求/token继承原锚点。

"""
    path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    manuscript = path.read_text(encoding="utf-8")
    if "### 6.4.7" in manuscript:
        raise ValueError("Do not duplicate the endpoint section")
    manuscript = manuscript.replace("完整中文初稿v42", "完整中文初稿v43", 1)
    manuscript = manuscript.replace(
        "更新前v41主稿及三状态文档原字节存档。新增同事实位移差图配对及作者视觉/原始数值复核。",
        "更新前v42主稿、三状态及差图报告原字节存档。新增局部端点拟合、统一控制/消融及四图视觉数值复核；候选未通过登记并停止。", 1)
    manuscript = manuscript.replace("### 6.5 阶段与计算成本", section + "### 6.5 阶段与计算成本", 1)
    manuscript = manuscript.replace(
        "单例平台静稳恢复不能替代整体识别收益。",
        "单例平台静稳恢复不能替代整体识别收益。后续局部转折端点拟合使直接视觉F1下降.0378/.0308，共同事件端点MAE增加4.62日，亦停止该候选。", 1)
    manuscript = manuscript.replace(
        "the display candidate is stopped despite a single-record stationary-abstention recovery.",
        "the display candidate is stopped despite a single-record stationary-abstention recovery. Subsequent local transition endpoint fitting reduces direct-vision F1 by .0378/.0308 and increases common-event endpoint MAE by 4.62 days; this candidate is also stopped.", 1)
    manuscript += r"""

[24] Maidstone, R., Fearnhead, P., & Letchford, A. (2017). Detecting changes in slope with an $L_0$ penalty. [arXiv:1701.01672v2](https://arxiv.org/abs/1701.01672v2). 已读官方摘要，未读完整论文；不将连续折线或变点拟合称首创。

[25] Fearnhead, P., & Grose, D. (2022). cpop: Detecting changes in piecewise-linear signals. [arXiv:2208.11009v1](https://arxiv.org/abs/2208.11009v1). 已读官方摘要及[CRAN包镜像](https://github.com/cran/cpop)DESCRIPTION/部分R方法与入口，固定树498992ceaef762ad8f4b7588778a4a7fddc569da；未读底层DP或运行作者模型，未移植GPL代码。

[26] Lukassen, F., Weisser, C., Schlee, M., Kumar, M., Thielmann, A., Saefken, B., Silbersdorff, A., & Kneib, T. (2026). LLM-Augmented Changepoint Detection: A Framework for Ensemble Detection and Automated Explanation. [arXiv:2601.02957v3](https://arxiv.org/abs/2601.02957v3). 官方API题名/作者与摘要已核查；未阅读全文、取得源码或运行模型，不作首创或完整复现结论。
"""
    path.write_text(manuscript, encoding="utf-8")
    state = f"""## 2026-10-11：局部原始转折端点拟合停止，初稿v43

[完整结果](landslide-endpoint-fit-results.md)。局部原始NEU静稳↔线性转折竞争平稳、漂移、阶跃，两折都胜且knot跨度≤14日才修端点。原未知、缺测、事件身份与目标阶段头保持。14日块/98日证据/49日搜索评分前固定，无API、无重训练或标签调参。

完整24过程直接F1 .777231/.760087→.739460/.729262，净正确−193/−153，弱R .501220/.604878→.493293/.600000；共同端点MAE池化增加4.6184日。12作者F1 .778195/.713063→.760229/.691695，净−37/−42。选择性比无条件好但比原组差，去step主组逐日相同；强数值与固定/自适应全部同机制控制。原51/48指标/API成本保持，新增57/48后108/96组。STOP_ENDPOINT_FIT，不调参数/折/主组/子集救分，旧四决定保持。

实际查看0169/0171/0176/0188原始三轴/局部拟合/两遍对齐四图。0176停止339→311时原N/E仍累积，过程各损51日、作者各损12；0188尾300→251平台修正净+35/+33。胜有限竞争且折点稳定仍可能删渐进首尾，不改参考。作者已开发暴露，非独立专家或现场金标准。CPOP两摘要/CRAN镜像部分源码及最新LLM变点摘要已读，不称hinge首创或完整复现。

五业务测试/Ruff通过；1419120日契约/事件身份通过；独立prepare4预测/证据、评分14文件逐字节一致；4806端点诊断/5832三状态转移保存。本地拟合/缓存墙钟{verification['local_fit_wall_seconds']:.4f}秒、规则合计应用{verification['local_rule_application_wall_seconds']:.4f}秒，不含特征、阶段、评分，非单方法总CPU。新增API/记录0；累计仍3825启动/3823回执、至少16694503报告token，受控192/半合成60不变。

v42五文档原字节存档，1258文件旧manifest保持并按档案映射核验。v43公式继续LaTeX。goal active；本轮完成候选开发与否定验证，稳定改进和独立现场验证未完成。下一步需要新增持续运动识别能力的依据，不继续叠同源表示或单转折拟合。下方为历史记录。

"""
    for filename in ("project-status.md", "landslide-adaptive-review-method.md", "landslide-submission-roadmap.md"):
        path = ROOT / "docs" / filename
        heading, remainder = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(heading + "\n\n" + state + remainder.lstrip("\n"), encoding="utf-8")


if __name__ == "__main__":
    report()
