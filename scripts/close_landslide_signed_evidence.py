"""Write the stopped evidence experiment into the manuscript without changing scores."""

from __future__ import annotations

import re
from pathlib import Path

from prepare_landslide_ai_reference_holdout import verify_manifest
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
RUN = BASE / "signed-evidence-development-v1"
ANALYSIS = BASE / "signed-evidence-development-analysis-v1"
REPORT = BASE / "signed-evidence-development-report-v1"
LITERATURE = BASE / "signed-evidence-literature-v1"
OUT = BASE / "manuscript-v36"


def close():
    verify_manifest(ANALYSIS)
    verify_manifest(REPORT)
    decision = read_json(ANALYSIS / "decision.json")
    cost = read_json(ANALYSIS / "cost.json")
    if decision["candidate"] != "STOPPED" or cost["new_receipts"] != 106:
        raise ValueError("This narrative is only for the completed frozen stopped candidate")
    OUT.mkdir(exist_ok=False)
    manuscript = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    original = manuscript.read_bytes()
    (OUT / "manuscript-v35-original.md").write_bytes(original)
    old = original.decode("utf-8").replace("\r\n", "\n")
    table = (REPORT / "compact-table.md").read_text(encoding="utf-8")
    formulation = r"""本轮仅改变局部证据表示：保留原图，提供非重叠14日原始块的逐轴中位数、有效观测数及相邻块的带符号变化。设 $B_j$ 为第j个块，$O_t$ 为完整NEU观测掩码，$n_j$ 为块内有效日数，逐轴原始位移为 $\mathbf{x}_t$，则

$$
\mathbf{m}_j=\begin{cases}
\operatorname{median}_{t\in B_j,\ O_t=1}\mathbf{x}_t,&n_j\ge8,\\
\varnothing,&n_j<8,
\end{cases}
\qquad \Delta\mathbf{m}_j=\mathbf{m}_j-\mathbf{m}_{j-1}.
$$

最后一式仅在相邻两个块均有效时计算，否则为null；不跨缺测块，不插值。与原21/42间距三块的模摘要不同，新卡保留N/E/U符号及块位置；与已停止块线图不同，本轮不更改像素，也不使用其显示选择阈值。符号不是活动答案，块端点不是事件边界，噪声代理不等于置信区间。没有弃判回填、数值活动投票或新的阶段头。
"""
    findings = """完整使用已暴露0157—0168的12条记录，原初判、两次预算、冲突选窗和49日修订权限保持。同期同窗原图及原模摘要复核用于检查历史输出波动；带符号固定、同次数早/晚、仅第一窗、整目标权限和原46方法均保留。两个目标各68方法，共136方法/目标评分。原六数值基线继续使用各自目标匹配的6训练/6验证冻结模型；本轮没有新数据分布或训练机会，不在当前开发标签上调参。

冻结条件要求主方案每遍F1均高于历史原方法、同期原图及同期模摘要，对带符号固定及便宜第一窗平均F1和净确定纠错均提高，对原弱召回均值不降。主方案对历史原方法平均F1提高.029482、净纠错+318，对同期原图平均提高.033172、净+180；但相对同期模摘要，第一遍+.009110、第二遍−.025667，平均−.008278、净−75，因此STOPPED。过程目标相对同期模摘要平均−.004764，同样失败。不是所有数字反馈都失败，而是更详细带符号卡尚无超越已有模摘要的稳定增量；不根据两个较弱对手的改善将本候选升格。

第一遍相对模摘要TP相同、FP减少30；第二遍TP少6、FP多89。实际视觉与逐日对齐说明变化有场景差异：0159第一遍去掉63个早期静稳误报、0161第一遍去掉39个误报；0163则第一/第二遍分别多72/54个FP，第二遍另少4个TP。0164第二遍多30FP、少2TP，0166第二遍多5FP。0163首窗带符号回答声称0—179日全部活动、各轴块变化始终为负；实际0—55日接近初始平台，42—55块相对前块N/E分别为+.2526/+.4867mm，U在28—41块也回升1.9623mm。精确数值暴露没有阻止错误跨越平稳前段，不能以叙述自称持续运动替代源事实核对。未知参考日期不重写、不计静稳收益。

0160原始全局/局部图中N/E低幅变化值得复核，但原选窗两遍均0触发，主方案仍漏掉133个确认活動观测日/遍；改变已选窗内的表示不能修复观察遗漏。固定带符号另取得证据，不据此改选窗或挑子集。主活动事件F1 .7692/.7857，起点内部MAE34.7/61.2日、终点23.0/35.5日，内部阶段F1均.8492来自共同冻结数值头，不能归功于视觉阶段创新。强原始多尺度规则F1 .8552、弱R .3556、0远端调用；XGBoost .8044、原始持久性XGBoost .7880、鲁棒规则 .7163、PELT .5688同表报告，不声称全部优于强数值方法。

全部106新请求/回执，533654报告token、303.391秒、0失败/重试，累计3224启动/3223回执/至少14144287报告token，受控168/半合成60不变。主方法按单次部署计22/23请求、83510/88900token，含复用的全局初判；同期模摘要同为22/23请求但79477/84361token，因此带符号卡增加4033/4539token。实验总成本和单方法部署成本分别列出，历史共享请求不重复计入新增总数。

106回执与冻结计划、全部逐日回答及事实卡逐一重建；92旧基线/目标行所有评分和成本字段精确相同。统一评分用sklearn独立核验逐日P/R/F1，离线评分13文件字节精确重放，12条结果图、全部改变日和逐记录纠错保存。13相关业务测试/Ruff通过。这些是实现与复现证据，不将开发作者复核升级为独立专家或现场准确率。
"""
    section = (
        "### 6.18 带符号原始块证据：相对纯图有收益，但未超越原数值摘要\n\n"
        + formulation
        + "\n"
        + table
        + "\n\n"
        + findings
    )
    section += "\n[全部136组指标、消融、逐日变化与实际视觉复核](landslide-signed-evidence-development-results.md)。\n\n"
    text = old.replace("## 7 讨论", section + "## 7 讨论", 1)
    header = "> 完整中文初稿v36，2026-10-11。新增带符号原始块证据的真实106请求、68方法×2目标比较及视觉/数值错误复核；相对纯图改善，但未稳定超越原模摘要，按冻结条件停止。全部旧表与42处LaTeX公式保留，新增证据表示公式。作者AI参考不属于独立专家或现场金标准。本稿不是投稿定稿。"
    text = re.sub(r"^> 完整中文初稿v35.*$", header, text, count=1, flags=re.MULTILINE)
    note = "\nv36精确v35字节保留于manuscript-v36/manuscript-v35-original.md。全12已暴露记录的带符号证据真实开发与68方法双参考统一比较、实际视觉图、106回执重建、92基线行核验和13文件精确离线重放分别位于signed-evidence-development-v1、signed-evidence-development-analysis-v1、signed-evidence-development-report-v1及signed-evidence-development-replay-v1。候选STOPPED，不接默认。新增533654token，累计3224启动/3223回执/至少14144287报告token；没有新增记录、训练或标签。全部旧LaTeX与表保留。\n\n"
    text = text.replace("## 9 结论", note + "## 9 结论", 1)
    conclusion = "\n进一步的带符号逐轴块事实卡在完整开发记录上改善纯图活动F1和弱召回，但第二遍相对更强的现有模摘要损失6TP、增加89FP，未建立表示机制的稳定独立增量。候选按冻结条件停止；输入包含精确数字也不保证输出叙述符合原始块变化，未来方法必须同时检验弱运动证据消费、平台边界和未触发记录，不能无限增加复核或用事后规则救分。\n\n"
    text = text.replace("## 参考文献", conclusion + "## 参考文献", 1)
    literature_note = "\n近期ViSTA[16]把不规则数值观测编码后以可训练残差校正已有视觉token，显式表达缺测，预训练VLM组件保持冻结。已读取方法第3节，任务为临床预测及时间问答，与无训练的GNSS有界局部事实卡不同；它证明数值/图像结合已有技术先例，不证明本候选首创或有效。未确认作者源码，也未运行其模型。\n\n"
    text = text.replace("## 3 任务定义", literature_note + "## 3 任务定义", 1)
    reference = "[16] Gao, J., Shi, Y., Hu, P., & Harrison, E. M. (2026). ViSTA: A Simple Bridge Extends Visual Alignment to Clinical Time-Series Understanding in Multimodal LLMs. [arXiv:2609.31448v1](https://arxiv.org/abs/2609.31448v1). 官方API书目及HTML第3节方法已核查；不是滑坡现场或GNSS异常基线，未确认公开源码，未复现发布效果。\n\n"
    text = text.replace("## 补充结果与完整数据", reference + "## 补充结果与完整数据", 1)
    old_formulas = re.findall(r"\$\$[\s\S]*?\$\$|(?<![\\$])\$(?!\$)[^$\n]+\$(?!\$)", old)
    old_tables = re.findall(r"(?:^\|.*\n)+", old, re.MULTILINE)
    if not all(value in text for value in old_formulas + old_tables):
        raise ValueError("Preserve all previous formulas and tables")
    manuscript.write_text(text, encoding="utf-8", newline="\n")
    report = (ANALYSIS / "report.md").read_text(encoding="utf-8")
    report += "\n## 固定机制与实际错误复核\n\n" + formulation + "\n" + table + "\n\n" + findings
    report += "\n本轮实际查看0160原始全局/180—359局部、0159/0163全记录结果对齐以及0163首窗/尾窗，共6图；导出12图不等于全部实际阅读。原始块数字和逐日掩码辅助检查，参考及预测均未回写。\n\n"
    for cid in ("case_0159", "case_0163"):
        report += f"![{cid}原始曲线与两遍输出对齐](../{(REPORT / (cid + '.png')).relative_to(ROOT).as_posix()})\n\n"
    report += "## 复现\n\n完成目录拒绝再次执行推理。只重放已冻结回答与评分，可使用下列命令，输出目录须不存在；不会请求模型。\n\n```powershell\n$env:PYTHONPATH='src;scripts'\n& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/probe_landslide_signed_evidence.py score --out artifacts/landslide-adaptive-review-2026-10-09/signed-evidence-development-v1 --analysis artifacts/landslide-adaptive-review-2026-10-09/signed-evidence-extra-offline-replay\n```\n\n独立新推理入口为prepare/run；会产生真实调用，不应把完成目录重新运行。冻结协议、图片、输入、模型、源码和计划在开发目录，全部对照和成本在分析目录，12图与逐日变化在报告目录。\n"
    (ROOT / "docs/landslide-signed-evidence-development-results.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    status = """## 2026-10-11：带符号原始证据开发完成，未超越模摘要，初稿v36

[完整结果](landslide-signed-evidence-development-results.md)：全12已暴露0157—0168、106新请求/回执、533654token、303.391秒、0失败重试。固定14日逐轴中位数/符号变化与原图共同复核，不改原选窗/权限/预算。保留46旧方法及同期原图/模摘要、固定/同次数位置、第一窗/整目标消融，68方法×2目标共136结果。主AI F1 .8801/.8292，对同期原图平均+.033172、弱召回提高；对同期模摘要平均−.008278、净纠错−75，第二遍多89FP少6TP，STOPPED，不接默认，不救门槛。过程目标同样未通过更强摘要比较。

实际看6图并用块数字/逐日变化复核：0163把平稳前段扩为活动，0160原策略两遍0触发仍漏133活动日/遍。106回执/事实卡重建、92旧评分/成本行精确保持、13评分文件字节精确重放、12图和13相关测试/Ruff保留。没有新增训练/记录/标签；累计3224启动/3223回执/至少14144287报告token，受控168/半合成60不变。

[初稿v36](landslide-adaptive-review-manuscript-draft.md)精确备份v35、保留42旧LaTeX公式与全部旧表，新增证据公式。检索8近期官方元数据，实际读4摘要及临床ViSTA第3节数值→视觉残差方法，未确认源码/未运行其模型，不把数值图像结合本身称首创。用户授权Codex视觉＋数值复核继续有效；作者AI参考不升级为独立专家。研究goal active。

下一步从两处明确失败找小机制：已提供数字却错误扩大活动边界，以及未触发观察的低幅运动。先确认相关方法和可复核证据，再冻结一个方案；不复活停止候选、不做提示/阈值搜索或用本批消融救分。

下方为历史记录。

"""
    for name in (
        "project-status.md",
        "landslide-adaptive-review-method.md",
        "landslide-submission-roadmap.md",
    ):
        path = ROOT / "docs" / name
        contents = path.read_text(encoding="utf-8")
        title, remainder = contents.split("\n", 1)
        path.write_text(
            title + "\n\n" + status + remainder.lstrip("\n"), encoding="utf-8", newline="\n"
        )
    actual_images = [
        RUN / "images/case_0160-raw.png",
        RUN / "images/case_0160-window-02.png",
        RUN / "images/case_0163-window-00.png",
        RUN / "images/case_0163-window-05.png",
        REPORT / "case_0159.png",
        REPORT / "case_0163.png",
    ]
    write_json(
        OUT / "visual-inspection.json",
        {
            "reviewer": "Current Codex, method developer; prior outputs seen",
            "actually_viewed_images": {
                path.relative_to(ROOT).as_posix(): sha(path) for path in actual_images
            },
            "all_exported_images_actually_read": False,
            "observations": findings,
            "new_labels": False,
            "independent_expert": False,
            "predictions_or_references_changed": False,
        },
    )
    write_json(
        LITERATURE / "reading-scope.json",
        {
            "metadata_records_inspected": 8,
            "abstracts_actually_read": [
                "2610.05240v1",
                "2609.38789v1",
                "2609.31448v1",
                "2609.24156v1",
            ],
            "fulltext_reading": {
                "2609.31448v1": {
                    "derived_zero_based_half_open_lines": [[152, 186]],
                    "section": "3 method, including missingness/event encoding, axis-aware retrieval, trained residual visual-token correction",
                    "html_sha256": sha(LITERATURE / "2609.31448v1.html"),
                    "full_paper_read": False,
                }
            },
            "author_code_confirmed": False,
            "author_model_executed": False,
            "limits": "Clinical supervised adaptation is not a GNSS nearest anomaly method; original TAMA/AnomaMind/TimeEvo reading scope unchanged. No first-ever numeric-visual fusion claim.",
        },
    )
    write_json(
        OUT / "content-check.json",
        {
            "old_formulas_preserved": len(old_formulas),
            "old_tables_preserved": len(old_tables),
            "old_manuscript_sha256": sha(OUT / "manuscript-v35-original.md"),
            "new_manuscript_sha256": sha(manuscript),
            "candidate": "STOPPED",
            "default_changed": False,
        },
    )
    write_json(
        OUT / "run-ledger-end.json",
        {
            "goal_status": "active",
            "turn_classification": "progress",
            "cumulative_launches": 3224,
            "cumulative_receipts": 3223,
            "at_least_reported_tokens": 14144287,
            "new_remote_calls": 106,
            "new_records": 0,
            "new_training": False,
            "controlled_records": 168,
            "semisynthetic_records": 60,
            "candidate": "STOPPED",
        },
    )
    print(read_json(OUT / "content-check.json"))


if __name__ == "__main__":
    close()
