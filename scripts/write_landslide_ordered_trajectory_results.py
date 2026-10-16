"""Publish the mechanism result while preserving v50 and all frozen responses."""

from __future__ import annotations

import argparse

from probe_landslide_endpoint_fit import BASE, ROOT, export
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import read_json
from write_landslide_contrastive_exemplar_results import DOCUMENTS

from gnss_sim.artifacts import sha


def previous_checks(archive, old):
    mapping = {f"docs/{name}.md": archive / (name + "-v50-original.md") for name in DOCUMENTS}
    checks = {name: sha(mapping.get(name, ROOT / name)) == expected for name, expected in old.items()}
    if not all(checks.values()):
        raise ValueError("v50 delivery changed outside archived current documents")
    return {"all_match": True, "count": len(checks), "checks": checks,
            "archived_document_mapping": {key: str(value) for key, value in mapping.items()}}


def write():
    archive = BASE / "manuscript-v51"
    analysis = BASE / "ordered-trajectory-analysis-v1"
    review = BASE / "ordered-trajectory-visual-review-v1"
    previous = BASE / "manuscript-v50/delivery-manifest.json"
    old = read_json(previous)["files"]
    if any(sha(ROOT / name) != expected for name, expected in old.items()):
        raise ValueError("Preserve current v50 before writing")
    decision = read_json(analysis / "decision.json")
    cost = read_json(analysis / "cost.json")
    if decision["decision"] != "STOP_ORDERED_TRAJECTORY_MECHANISM_PILOT" or cost["reported_tokens"] != 71084:
        raise ValueError("Frozen outcome or cost differs")
    original = read_json(BASE / "observability-supervision-preflight-v1/train-only-selection.json")
    replay = read_json(BASE / "observability-supervision-preflight-replay-v1/train-only-selection.json")
    if original != replay:
        raise ValueError("Training conflict coverage did not reproduce")
    archive.mkdir(exist_ok=False)
    for name in DOCUMENTS:
        source = ROOT / "docs" / (name + ".md")
        (archive / (name + "-v50-original.md")).write_bytes(source.read_bytes())
    (archive / "v50-delivery-manifest-original.json").write_bytes(previous.read_bytes())
    export(archive / "v50-preservation-before.json", previous_checks(archive, old))
    stage_rows = {}
    for method, value in read_json(analysis / "results.json").items():
        stage_rows[method] = {
            "support_ASD": value["internal_stages"]["support_ASD"],
            "internal_fixed_three_class_macro": value["internal_stages"]["macro_f1"],
            "end_to_end_defined_class_macro": value["aggregate"]["stage_macro_f1"],
            "per_class_end_to_end": value["aggregate"]["stages"],
        }
    export(review / "stage-metric-contract.json", {
        "reference_support_ASD": [0, 82, 0],
        "explanation": "Registered internal macro averages all3 with undefined classes zero; "
                       "registered end-to-end macro excludes classes with undefinedF1. "
                       "Only S has support here; their different values do not show a new phase effect.",
        "original_protocol_unchanged": True, "rows": stage_rows,
    })
    mechanism = r"""
本轮先检查VLM能否辅助训练监督：按原21/42日两尺度明确反证与训练标签冲突提名98日窗。原24训练中仅四日冲突，均为静稳标签上的孤立运动响应，活动标签明确平台反证为零；不足以支持拟定联合触发监督加权，0API停止前检，不放宽门槛。这不证明所有潜在标签都可从观测辨识。原CSV含其他划分，但只使用训练行选择；24行预检由复现脚本逐值重现。

NRdetector[39]已研究粗段监督与逐点目标间的噪声、可靠负例及PU/时间约束；这些一般机制不是本项目首创，其弱段/未标记设定也不等同于本仓库准确潜在标签。官方检索保存12条题名/摘要节选，实读该文4.1—4.2.2相关段及作者README前2500字符；固定源码头bd5592b457ce110213fd31104cbda73239f46a22，未读实现或运行模型。REFINE仅摘要节选，全文SSL失败保留，不声称完整读过。

另冻结一个观察假设：时间曲线之外，空间投影可否帮助区别持续弱路径与单次偏移后点云？一般时间图像表示[23,27]及散点轨迹/颜色编码不作为首创。与原观察一致，十三个14日中位块定义

$$
\tau_j=-84.5+14j,\quad
z_j=(m_j-m_{j_0})\oslash\sigma,\quad
p_j^{ab}=(z_{j,a},z_{j,b}),\quad j=0,\ldots,12.
$$

每块至少八个完整NEU原日；减首个可用块常数、除原日差噪声代理。NE/NU/EU等坐标跨度投影，仅连接相邻且均可用原槽，不跨缺块。查询 $[t-21,t+21)$ 覆盖槽5/6/7，以黑环和原槽ID标记。颜色只编码原日期，灰图保留相同点、线、查询ID及数值块表；因此它只消融颜色编码，不是去掉全部时间信息。连接中位数不证明两块之间每天运动，也不能给精确日边界。

仅从原24训练提名四窗：两同类弱活动（42日中至少14弱日、按中心较小变化排序）和两同类静稳（远处块变化至少5噪声单位、中心两变化至多2），不同记录，各至少35已观测日、十可用块。标签用于训练窗选择，但不进入VLM提示；没有验证记录或新独立数据。三臂为共同时间图＋原块表、另加彩色轨迹、另加灰轨迹，各两遍24请求。第一张图和提示完全相同，观察内容及查询原日期不改，未知和原缺测保持。
"""
    outcomes = """
四窗有165已知观测日：82弱活动、83静稳。普通时间图与灰轨迹两遍P/R/F1均1；彩色轨迹两遍F1 .993865/1、弱R .987805/1，相对时间图净−1/0确定正确日，没有任何新增修正。第一遍case0018一日由活动变未知，非静稳误判；静稳控制无损失。4/9条件通过，STOP_ORDERED_TRAJECTORY_MECHANISM_PILOT，不改颜色、seed、窗口、提示、样例或换最佳臂救分。

原base18/base_expert在四训练窗的诊断F1均.836879、弱R .719512，漏23日；这些模型已见训练记录，且窗口经潜在标签挑选，不是公平验证，更不能把局部视觉的1.0当作新方法超过强数值的正式结论。局部时间图有恢复价值，但额外空间轨迹未显示必要性，不把普通控制替换成轨迹贡献。

实际看图与原块值：case0021的N弱累计、E小幅下降，中心后一块增量很小，轨迹未提供新增事实；case0018的NE连续弱下降，彩色第一遍多弃判一日；case0020早期E偏移后平台，NU投影的点线也可由噪声漂动形成；case0008显著NEU阶跃在查询之后，不能反传中心活动。四窗作者诊断有历史标签暴露，参考不改、未知不补，不赋视觉A/S/D。

阶段参考仅82个S日，A/D没有支持。保留统一旧协议：内部宏F1固定三类、空类记零，端到端宏F1排除无定义F1类；故普通组内部.289655/端到端.868966的差异含空类处理，不能直接比较二者推导阶段增益。逐类P/R/F1、支持与覆盖完整保存，所有视觉臂只门控共享既有数值阶段头。

24启动/24回执、全部合法且0重试、71084报告token，API58.389秒、推理循环58.461秒；彩色两遍合计25741token，时间图19602，额外图无识别收益。24payload/回复重建、168查询日原支持/缺测/阶段检查及sklearn日/事件/边界核验通过；18相关测试与五新增源码Ruff通过。事件和边界只在42日裁窗，不是完整活动起止。

准备v1在任何响应前发现轨迹编号重叠及等比例面板过窄，原图/源码保留、0API；v2统一方形等跨度、仅分开标记查询5/6/7，顺序/窗口/协议/提示/数值和四时间图八文件字节相同。图像已实际检查。这里不是评分后改图救分。
"""
    table = (analysis / "results.md").read_text(encoding="utf-8")
    table = table[table.index("| 方法"):table.index("\n{\n")]
    report = ("# 训练监督冲突前检与原日期空间轨迹小试\n\n"
              "## Material Passport\n\n"
              "- Origin Skill: academic-research-suite / experiment-agent\n"
              "- Origin Mode: run / validate inline\n"
              "- Origin Date: 2026-10-11\n"
              "- Verification Status: 预检逐值重现、冻结请求预测重放、实际视觉/数值诊断；非独立验证\n"
              "- Version Label: ordered-trajectory-pilot-v2 / analysis-v1 / manuscript-v51\n\n"
              + mechanism.strip() + "\n\n" + table + "\n\n" + outcomes.strip()
              + "\n\n## 复现与研究状态\n\n"
              "原前检/replay为observability-supervision-preflight-v1/"
              "observability-supervision-preflight-replay-v1；新候选在ordered-trajectory-pilot-v2，"
              "评分ordered-trajectory-analysis-v1，实际复核及阶段空类说明在ordered-trajectory-visual-review-v1。\n\n"
              "```powershell\n"
              "$env:PYTHONPATH='src;scripts'\n"
              "$env:PYTHONIOENCODING='utf-8'\n"
              "& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/report_landslide_ordered_trajectory.py --run artifacts/landslide-adaptive-review-2026-10-09/ordered-trajectory-pilot-v2 --out artifacts/landslide-adaptive-review-2026-10-09/ordered-trajectory-user-replay\n"
              "```\n\n"
              "选未存在的新报告目录，无API重放，不能重启终态infer。此次未重训基线或新建记录，"
              "原完整数值规则/PELT/XGB/视觉/固定Agent比较保留，不能由四个训练窗替代。"
              "累计3983启动/3981回执、至少17171895报告token，受控192/半合成60不变。"
              "goal active：新核心贡献及独立可信确认仍未完成。下一轮应从真实识别错误与最强控制之间"
              "存在的可辨空间提出假设，不继续扩大轨迹、示例或提示组合。\n")
    (ROOT / "docs/landslide-ordered-trajectory-results.md").write_text(report, encoding="utf-8")
    path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    manuscript = path.read_text(encoding="utf-8").replace("完整中文初稿v50", "完整中文初稿v51", 1)
    manuscript = manuscript.replace(
        "更新前v49主稿及三状态文档原字节存档。新增平衡中心正反训练图检索、四臂两遍64请求、强数值比较及八图视觉数值复核；检索候选停止，弃判与格式失败单独披露。",
        "更新前v50主稿及三状态文档原字节存档。新增监督冲突训练前检及空间轨迹三臂两遍24请求；轨迹无净增益而停止，潜在标签可辨性与阶段空类口径单独披露。", 1)
    addition = """
### 6.4.15 训练前检与空间轨迹观察机制

按原21/42日两尺度明确反证检查训练监督[39]，24训练仅四日静稳标签孤立运动冲突，零活动标签明确平台反证；拟定联合触发加权覆盖不足、0API不扩展，不等于潜在标签全可辨。随后只在四训练窗检查原日NE/NU/EU中位数空间轨迹：两弱活动、两静稳，普通时间图、追加彩色、追加灰图三臂两遍24合法回复。共有165已知日，时间/灰图两遍F1 1，彩色.993865/1，净−1/0、无新增纠错，4/9条件，STOP_ORDERED_TRAJECTORY_MECHANISM_PILOT。颜色及附加图未显示必要性，训练选择与数字模型已见标签，不称正式效果。

原图、块值、视觉诊断和所有失败/修复原字节保留。阶段只有82个S，A/D不可评；原内部固定三类宏平均与端到端排除无定义类口径保留并说明，不把数值差异当新阶段收益。新71084token/API58.389秒，18相关测试/Ruff及24payload、168查询日/阶段/缺测和评分核验通过。完整公式、表、消融、成本与复现见[机制结果](landslide-ordered-trajectory-results.md)；不扩展无收益轨迹、不替代原完整强基线。

"""
    marker = "### 6.5 阶段与计算成本"
    if manuscript.count(marker) != 1:
        raise ValueError("Manuscript insertion point differs")
    manuscript = manuscript.replace(marker, addition + marker, 1).replace(
        "截至v50累计3959启动/3957回执、至少17100811报告token",
        "截至v51累计3983启动/3981回执、至少17171895报告token", 1)
    manuscript += ("\n[39] Wang, Y., Cheng, H., Xiong, J., Wen, Q., Jia, H., Song, R., Zhang, L., Zhu, Z., & Liu, Y. (2025). "
                   "Noise-Resilient Point-wise Anomaly Detection in Time Series Using Weak Segment Labels. "
                   "[arXiv:2501.11959v1](https://arxiv.org/abs/2501.11959v1). "
                   "[作者源码](https://github.com/UCSC-REAL/NRdetector)。实际读4.1—4.2.2相关段及README前2500字符，"
                   "固定头bd5592b457ce110213fd31104cbda73239f46a22；未读实现或运行作者模型，不将弱段标签噪声直接等同本任务潜在标签。\n")
    path.write_text(manuscript, encoding="utf-8")
    status = ("\n## 2026-10-11：监督训练前检与空间轨迹小试，初稿v51\n\n"
              "[机制结果](landslide-ordered-trajectory-results.md)。24训练两尺度明确反证仅四日静稳冲突、"
              "零活动平台冲突；原联合监督加权触发覆盖不足、0API不扩展，24行逐值复现。"
              "另一空间轨迹机制冻结四训练窗，三臂两遍24合法回复；时间/灰图F1 1，彩色.993865/1，"
              "相对时间净−1/0，无新增纠错；4/9条件STOP_ORDERED_TRAJECTORY_MECHANISM_PILOT，"
              "不救分或扩展。普通局部观察的价值不等于附加轨迹贡献，也不是独立评价。\n\n"
              "实际12图（含准备QA）及原块辅助复核；不追改参考，不赋视觉阶段。"
              "阶段只有82S、A/D无支持，内部固定三类与端到端无定义类排除口径披露。"
              "24payload/预测重放、168查询日原支持/缺测/阶段和评分核验、18测试/Ruff通过。"
              "新71084token/API58.389秒/循环58.461秒；累计3983/3981、至少17171895token，"
              "受控192/半合成60不变。v50四文档原字节存档，536旧文件映射保留，公式LaTeX。"
              "NRdetector选段/README已读，REFINE全文SSL失败保留；无作者模型复现。"
              "goal active：实质核心贡献与独立可信确认尚未完成，不扩大无收益图像或提示组合。\n\n")
    for name in DOCUMENTS[1:]:
        path = ROOT / "docs" / (name + ".md")
        heading, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(heading + "\n" + status + rest, encoding="utf-8")
    export(archive / "v50-preservation-after.json", previous_checks(archive, old))
    export(archive / "progress.json", {
        "status": "goal_active", "classification_previous_turn": "progress",
        "new_calls": 24, "new_reported_tokens": 71084, "new_records": 0,
        "trajectory_decision": decision["decision"], "supervision_preflight": "Do not start sparse joint trigger",
        "training_only": True, "independent_confirmation": False,
    })
    print("Wrote v51 and preserved", len(old), "historical delivered files")


def seal():
    archive = BASE / "manuscript-v51"
    if (archive / "delivery-manifest.json").exists():
        raise ValueError("Seal only a new delivery")
    run = BASE / "ordered-trajectory-pilot-v2"
    verify_bindings(run, "pre-inference-freeze.json")
    verify_bindings(run, "inference-freeze.json")
    for name, expected in read_json(run / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Original source changed")
    for name, folder in (
        ("landslide-adaptive-review-manuscript-draft", "formula-verification"),
        ("landslide-ordered-trajectory-results", "result-formula-verification"),
    ):
        audit = read_json(archive / folder / "formula-audit.json")
        if audit["status"] != "passed" or audit["source_sha256"] != sha(ROOT / "docs" / (name + ".md")):
            raise ValueError("Math audit differs")
    old = read_json(archive / "v50-delivery-manifest-original.json")["files"]
    previous_checks(archive, old)
    folders = (
        "ordered-trajectory-pilot-v1", "ordered-trajectory-pilot-v2",
        "ordered-trajectory-analysis-v1", "ordered-trajectory-visual-review-v1",
        "observability-supervision-literature-v1", "observability-supervision-preflight-v1",
        "observability-supervision-preflight-replay-v1", "manuscript-v51",
    )
    paths = [path for name in folders for path in (BASE / name).rglob("*") if path.is_file()]
    paths += [ROOT / "docs" / (name + ".md") for name in (*DOCUMENTS, "landslide-ordered-trajectory-results")]
    paths += [ROOT / path for path in (
        "src/gnss_sim/landslide_ordered_trajectory.py",
        "scripts/probe_landslide_ordered_trajectory.py", "scripts/report_landslide_ordered_trajectory.py",
        "scripts/probe_landslide_supervision_conflicts.py", "scripts/write_landslide_ordered_trajectory_results.py",
        "tests/test_landslide_ordered_trajectory.py", "scripts/verify_landslide_manuscript_math.cjs",
        "artifacts/landslide-adaptive-review-2026-10-09/manuscript-v50/delivery-manifest.json",
    )]
    files = {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(paths))}
    export(archive / "delivery-manifest.json", {"files": files})
    for name, expected in read_json(archive / "delivery-manifest.json")["files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Delivered byte hash differs")
    print("Sealed and verified", len(files), "files; goal remains active")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.seal:
        seal()
    else:
        write()
