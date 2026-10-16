"""Record the completed partition experiment and preserve Markdown LaTeX."""

from __future__ import annotations

import re
from pathlib import Path

from prepare_landslide_ai_reference_holdout import verify_manifest
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
RUN = BASE / "partition-review-development-v2"
ANALYSIS = BASE / "partition-review-development-analysis-v2"
REPORT = BASE / "partition-review-development-report-v2"
LITERATURE = BASE / "partition-review-literature-v1"
OUT = BASE / "manuscript-v37"

FORMULATION = r"""本候选改变活动输出的单位：已有PELT基线的恒定拟合速率及缺支撑跨度只提出时间边界，不直接赋活动状态。对长度超过30日的跨度，从其起点按原数值摘要已有30日尺度切分；相邻相同速率合并，故不是所有native PELT端点。设候选段为 $C_j=[b_j,b_{j+1})$，视觉模型对每段返回 $z_j\in\{-1,0,1\}$，分别表示未知、静稳和持续位移。若段内混有活动/静稳转换，要求未知；不能重新输出自由端点。以 $O_t$ 表示原始完整观测掩码，逐日展开为

$$
\widetilde y_t=\begin{cases}
z_j,&t\in C_j\ \text{且}\ O_t=1,\\
-1,&O_t=0.
\end{cases}
$$

沿用原最多两窗、严格多数合并及49日冲突修订权限，权限外保留原全局判断；阶段仍来自共同冻结数值头。PELT penalty160由旧AI验证选择，未在0157—0168调参。居中拟合属于离线；拟合缺支撑不等于原始位移缺测、静稳或活动。35个独特复核窗共有266个PELT候选段，其中35段不足8日、最短1日；没有根据本批结果合并短段或更改长度。

同时比较同期自由区间模摘要、同段数均匀、局部30日均匀、去数值反馈、固定整窗及更便宜第一窗。全记录均匀数值控制是将日历等分为向上取整的段数，长度约29—30日，并非固定30日绝对网格。新同段数值控制复用各参考目标原6训练/6验证的已训练日XGBoost，以段内已观测日概率均值分类；每几何仅在原验证集比较.35/.5/.65三阈值，合计12试次，无当前开发标签调参或新拟合。原68方法及强数值对照全部保留。
"""


def close():
    for folder in (ANALYSIS, REPORT, LITERATURE):
        verify_manifest(folder)
    if read_json(RUN / "state.json")["status"] != "predictions_complete":
        raise ValueError("Complete the full frozen inference before manuscript writing")
    decision = read_json(ANALYSIS / "decision.json")
    cost = read_json(ANALYSIS / "cost.json")
    aborted = read_json(BASE / "partition-review-development-v1/aborted-transport.json")
    inspection = read_json(BASE / "partition-review-developer-inspection-v2.json")
    if inspection["independent_expert"] or inspection["predictions_or_references_changed"]:
        raise ValueError("Developer inspection must retain its disclosed reference status")
    OUT.mkdir(exist_ok=False)
    manuscript = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    original = manuscript.read_bytes()
    (OUT / "manuscript-v36-original.md").write_bytes(original)
    old = original.decode("utf-8").replace("\r\n", "\n")
    if "> 完整中文初稿v36" not in old:
        raise ValueError("Expected exact previous manuscript version")
    table = (REPORT / "compact-table.md").read_text(encoding="utf-8")
    findings = inspection["findings"]
    cumulative = {
        "launches": 3224 + aborted["started"] + cost["new_launches"],
        "receipts": 3223 + aborted["receipts"] + cost["new_receipts"],
        "at_least_reported_tokens": 14144287 + aborted["reported_tokens"] + cost["reported_tokens"],
    }
    verification = read_json(REPORT / "verification.json")
    execution = (
        f"本轮计划148问题，其中{cost['carried_controls']}个精确自由区间回答来自中断批，未重复推理；"
        f"新版新增{cost['new_launches']}启动/{cost['new_receipts']}回执、{cost['reported_tokens']}报告token，"
        f"{cost['seconds']:.3f}秒，{cost['failures']}任务结构失败、0重试。"
        f"中断批另52启动/51回执、44个HTTP400、7个合法对照、34773报告token；"
        "分段提示遗漏JSON导致接口拒绝，1个在途结果未知，已保留冻结源码和失败，不作为机制评分。"
        "新版仅修复JSON契约，不看标签调参，不重发成功对照。"
        f"累计{cumulative['launches']}启动/{cumulative['receipts']}回执/至少{cumulative['at_least_reported_tokens']}报告token；"
        "受控168/半合成60不变。\n\n"
        f"全部{verification['actual_requests_reconstructed']}回执、逐段日历及原始拟合候选重建；"
        f"{verification['unchanged_baseline_target_rows']}旧评分/成本行精确保持；"
        f"{verification['exact_replay_files']}评分文件字节精确离线重放，0新增模型调用。"
        "两目标各102方法，共204评分；19相关业务测试/Ruff通过。"
        "12图已导出，实际阅读范围按复核清单列出，不以导出数量冒称全部读图。\n\n"
    )
    section = (
        "### 6.19 PELT候选段的视觉状态分类\n\n"
        + FORMULATION
        + "\n"
        + table
        + "\n\n"
        + findings
        + "\n\n"
        + execution
        + "[完整比较、逐日变化及实际视觉/数值复核](landslide-partition-review-results.md)。\n\n"
    )
    text = old.replace("## 7 讨论", section + "## 7 讨论", 1)
    header = (
        f"> 完整中文初稿v37，2026-10-11。新增PELT候选段视觉分类、102方法×2目标真实开发比较及视觉/数值复核，"
        f"冻结决定{decision['candidate']}。保留全部旧LaTeX公式与结果表；作者AI参考不属于独立专家或现场金标准。本稿不是投稿定稿。"
    )
    text = re.sub(r"^> 完整中文初稿v36.*$", header, text, count=1, flags=re.MULTILINE)
    nearest = (
        "\n分段视觉判断也有直接先例：LaSTR[17]以TV2数值分段后让VLM按段返回描述，再训练检索模型。"
        "本文不以数值分段或逐段视觉本身主张首创，只检验GNSS活动/静稳/未知的输出约束及既有修订权限是否有实际增量。"
        "已读其IV-A1—3及作者README，当前作者树只有README，未复现其模型。\n\n"
    )
    text = text.replace("## 3 任务定义", nearest + "## 3 任务定义", 1)
    reference = (
        "[17] Dohi, K., Purohit, H., Nishida, T., Endo, T., Ohtsubo, Y., Yawata, K., Takeshita, K., Sasaki, T., & Kawaguchi, Y. (2026). "
        "LaSTR: Language-Driven Time-Series Segment Retrieval. [arXiv:2603.00725v1](https://arxiv.org/abs/2603.00725v1). "
        "方法IV-A1—3与[作者项目页](https://github.com/Kota-Dohi/LaSTR)已核查；当前项目页只有README，未执行作者训练或检索模型。\n\n"
    )
    text = text.replace("## 补充结果与完整数据", reference + "## 补充结果与完整数据", 1)
    formulas = re.findall(r"\$\$[\s\S]*?\$\$|(?<![\\$])\$(?!\$)[^$\n]+\$(?!\$)", old)
    tables = re.findall(r"(?:^\|.*\n)+", old, re.MULTILINE)
    if not all(value in text for value in formulas + tables):
        raise ValueError("All previous formulas and tables must remain exact")
    manuscript.write_text(text, encoding="utf-8", newline="\n")
    result = (ANALYSIS / "report.md").read_text(encoding="utf-8")
    result += (
        "\n## 方法与实际复核\n\n"
        + FORMULATION
        + "\n"
        + table
        + "\n\n"
        + findings
        + "\n\n"
        + execution
    )
    for name in inspection["actually_viewed_images"]:
        result += f"实际查看：`{name}`。\n\n"
    result += "## 离线复现\n\n完成推理目录拒绝重启；只重放冻结回答/预测，不产生远端调用。输出目录须不存在。\n\n```powershell\n$env:PYTHONPATH='src;scripts'\n& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/probe_landslide_partition_review.py score --out artifacts/landslide-adaptive-review-2026-10-09/partition-review-development-v2 --analysis artifacts/landslide-adaptive-review-2026-10-09/partition-review-extra-offline-replay\n```\n"
    (ROOT / "docs/landslide-partition-review-results.md").write_text(
        result, encoding="utf-8", newline="\n"
    )
    status = (
        "## 2026-10-11：分段视觉分类开发完成，初稿v37\n\n"
        + "[完整结果](landslide-partition-review-results.md)。\n\n"
        + findings
        + "\n\n"
        + execution
        + "初稿v37精确保留v36字节、全部47旧LaTeX公式及旧表，新增分段展开公式。"
        "当前Codex实际视觉＋数值复核，不升级为独立专家。最近邻LaSTR数值分段/逐段描述已有先例；源码树仅README。研究goal active。\n\n下方为历史记录。\n\n"
    )
    for name in (
        "project-status.md",
        "landslide-adaptive-review-method.md",
        "landslide-submission-roadmap.md",
    ):
        path = ROOT / "docs" / name
        title, remainder = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(
            title + "\n\n" + status + remainder.lstrip("\n"), encoding="utf-8", newline="\n"
        )
    write_json(OUT / "visual-inspection.json", inspection)
    write_json(
        OUT / "content-check.json",
        {
            "old_formulas_preserved": len(formulas),
            "old_tables_preserved": len(tables),
            "old_manuscript_sha256": sha(OUT / "manuscript-v36-original.md"),
            "new_manuscript_sha256": sha(manuscript),
            "candidate": decision["candidate"],
            "default_changed": False,
        },
    )
    write_json(
        OUT / "run-ledger-end.json",
        {
            "goal_status": "active",
            "turn_classification": "progress",
            "cumulative": cumulative,
            "new_launches_including_interrupted": aborted["started"] + cost["new_launches"],
            "new_records": 0,
            "new_training": False,
            "candidate": decision["candidate"],
        },
    )
    print(read_json(OUT / "content-check.json"))


if __name__ == "__main__":
    close()
