"""Publish the scale-admission development result and preserve the previous draft."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
RUN = BASE / "scale-review-development-v1"


def number(value):
    return f"{float(value):.4f}" if value not in (None, "") else "—"


def table(rows):
    lines = ["| 配置 | P | R | F1 | 弱R | FP | 事件F1 .3/.5 | 起/止MAE日 | 活动内/端到端阶段宏F1 | 请求/token |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        fields = [row["method"], *[number(row[key]) for key in ("precision", "recall", "f1", "weak_recall")],
                  row["fp"], number(row["event_f1_iou03"]) + "/" + number(row["event_f1_iou05"]),
                  number(row["onset_mae_days"]) + "/" + number(row["offset_mae_days"]),
                  number(row["internal_stage_macro_f1"]) + "/" + number(row["stage_macro_f1"]),
                  row["requests"] + "/" + row["reported_tokens"]]
        lines.append("| " + " | ".join(fields) + " |")
    return "\n".join(lines)


def close():
    out = BASE / "manuscript-v14"
    out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    with (RUN / "analysis-v1/summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {row["method"]: row for row in rows}
    with (RUN / "analysis-v1/groups.csv").open(encoding="utf-8", newline="") as handle:
        groups = list(csv.DictReader(handle))
    selected = [indexed[f"{stem}_r{rep}"] for rep in (0, 1) for stem in
                ("direct_visual", "old_adaptive_image", "fixed_image", "scale_gate_image", "matched_budget_image")]
    selected += [row for row in rows if not row["method"].endswith(("_r0", "_r1"))]
    primary_table = table(selected)
    ablation_table = table([indexed[f"{stem}_r{rep}"] for rep in (0, 1) for stem in
                           ("scale_gate_image", "missed_only_image", "scale_without_missed_gate",
                            "scale_gate_numeric_text", "fixed_scale_gate_image")])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    stems = ["direct_visual", "old_adaptive_image", "scale_gate_image", "matched_budget_image"]
    titles = ["Direct", "Old review", "Scale gate", "Matched budget"]
    colors = ["#28475e", "#ae5966", "#24887b", "#bc842d"]
    for axis, metric, title in zip(axes, ("f1", "weak_recall", "requests"),
                                    ("Activity F1", "Weak recall", "Cached complete-flow calls")):
        for position, (stem, color) in enumerate(zip(stems, colors)):
            values = [float(indexed[f"{stem}_r{rep}"][metric]) for rep in (0, 1)]
            axis.plot([position - .1, position + .1], values, color=color, marker="o", linewidth=1)
        axis.set_xticks(range(4), titles, rotation=15)
        axis.set_ylim(0, 80 if metric == "requests" else 1)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Exposed cached development | two repeats | no new VLM calls")
    fig.tight_layout()
    fig.savefig(out / "scale-review-comparison.png", dpi=160)
    plt.close(fig)
    section = f"""### 6.18 历史尺度压缩对有限观察的准入：召回与调用取舍

第6.17节未建立逐日修订收益，故另检验观察机会的价值。原全局图强制至少60mm基础纵轴跨度，这种显示下限也会放大静稳残差；不能将任何局部放大都解释成历史强运动造成的压缩。本轮尺度代理为每轴全记录原始峰峰值除以局部原图已有基础跨度max(局部峰峰值,6倍既有日差MAD噪声代理,10mm)，取三轴最大值。它不含全局60mm显示下限，也不含任意位移参考偏移，不是测量不确定性或精确像素倍率。

仅当原候选至少14个漏运动证据日且尺度代理≥4时，保留原最多两窗的一项复核；四倍代表两个尺度octave的固定工程选择，不是来自文献的最优感知界限。仅筛原已选两窗，不重新搜索所有窗、不改图或提示，不进行反馈循环。局部回答仍通过原49日冲突范围修订，不再按逐日阳性交集产生新碎片。复用全部24条已见背景记录、两遍、原回答与阶段头，开发准入条件在评分前保存：两遍日F1和弱召回不低于直接（F1严格提高）、FP至多为原自适应80%、事件IoU.3 F1不低于直接、至少两条记录有新增TP、调用减少。

{primary_table}

主门控日F1 .6726／.6510对直接.6722／.6330，增量.000485／.017974；弱召回.8083／.6407对.6178／.3910。相对直接新增378／555TP，损失5／2原TP，新增783／862FP、撤回55／0FP。完整调用数从原自适应72／72降为40／42，分别少44.4%／41.7%；已报告token从254639／255929降为124514／132856，缓存只描述若执行该流程的已有回执成本，不能报告本轮重新执行了这些请求。

第一遍通过登记条件，第二遍事件F1 .6667低于直接.6923，完整继续条件未通过，不升级为新确认方法。主事件F1第一遍.6897对直接.6792；匹配边界与阶段全量同时保存，不只依据日F1或调用减量宣布成功。同每条记录复核个数的确定哈希预算对照日F1 .7003／.6600，均高于主组.6726／.6510；它使用相同原两窗回答，不能当独立随机试验或事后替换主方法。该对照显示本批尺度条件尚不能选出更有效的窗位置，部分收益来自拒绝多余复核。

{ablation_table}

去历史尺度条件的missed-only为.5814／.5562，去漏运动门槛的scale-only为.6746／.6622；数值文字为.6674／.6383。保留主图、数值文字和去条件全部结果，不择最高消融救分。固定窗同尺度门控为.6659／.6310，39／39请求。所有门控对照仍受原数值候选池约束，不能将scale-only称完全纯视觉选择器；直接视觉和固定图流程提供普通VLM／固定工作流控制。

公平比较新增同尺度信息XGBoost：在共同train24／validation12按原2深度×3概率门槛的六试次校准，原21／42／84矩阵外给予各轴全局跨度及覆盖每个日期的全部180日候选窗最大尺度代理，同原84观测支撑及阶段头。未读开发标签调模型；特征设计仍发生在已见开发研究中，不使本批重新独立。其F1 .7017、弱召回.0445、FP150，原XGBoost84 .7261／.1290／294仍更高，不能声称尺度特征必然改善学习方法。

强后弱家族主F1 .7753／.7777高于直接.7099／.6205；quiet、instrument_step、weak_constant没有新增门控调用，输出与直接完全一致。ASD家族F1却由.8665／.9092降为.7028／.6921，FP从112／30增为531／569。case2028原活动结束后的背景变化被局部扩成活动，说明历史压缩条件同样允许高位平台附近的残差起伏，不能证明实际持续运动。原数值运动条件不是独立感知证据，背景注入阴性也不是现场无运动真值。

新定向最近邻核查定位ARTIST[19]及作者公开库commit2bbc28a86f346efb7311ddc227f8952c3aa8b21e，实际阅读引言／3.1—3.3、README、段记忆和工具声明源码；其主动选段、controller-reasoner及RL训练已有公开先例，未复现作者模型。多尺度金融视觉基准[20]也比较数值XGBoost，但不能支持GNSS四倍门槛。本文不以局部缩放、有限观察或调用节省为首创。

28配置（18视觉流程行＋10数值控制）、384改写权限核验、240数值预测重算、315旧payload回执、280分组行、490560逐日行和24全记录图已保存；4项新业务测试及最终源码Ruff通过。新增VLM请求0、记录0，累计仍1809启动／1808回执／至少7977130已报告token，108纯受控和60半合成分别计数。基线训练与特征提取CPU成本未隔离计时，不作整体加速或能源测量结论。独立领域复核者暂缺，继续保留观测效果未知；停止本版，不在同批扫四倍门槛或按预算对照换主组。

![尺度准入与同预算对照](../artifacts/landslide-adaptive-review-2026-10-09/manuscript-v14/scale-review-comparison.png)

"""
    group_table = ["| 家族 | 直接F1/1,2 | 主F1/1,2 | 直接FP/1,2 | 主FP/1,2 |", "|---|---:|---:|---:|---:|"]
    group_index = {(row["method"], row["group"]): row for row in groups if row["field"] == "family"}
    for family in ("quiet", "instrument_step", "weak_constant", "asd", "reactivation", "weak_after_strong"):
        values = [family]
        for stem, metric in (("direct_visual", "f1"), ("scale_gate_image", "f1"),
                             ("direct_visual", "fp"), ("scale_gate_image", "fp")):
            pair = [group_index[f"{stem}_r{rep}", family][metric] for rep in (0, 1)]
            values.append("/".join(number(value) if metric == "f1" else value for value in pair))
        group_table.append("| " + " | ".join(values) + " |")
    report = f"""# 历史尺度压缩门控：观察价值存在，窗口选择尚未确认

## Material Passport

- 日期：2026-10-09；流程：ARS实验执行／验证；状态：完整已见开发完成，登记继续条件未全部通过。
- 数据：原24已见半合成记录、4相关背景，非新独立评价，参考只代表注入恢复。
- 本轮新增VLM请求0；新增同信息学习控制共同train24／validation12的六校准试次。
- 原始图／提示／旧回答／失败和冻结源码不改；主门控不升级，不扫描门槛或选择最高消融替代。

{section}

## 家族与错误定位

{chr(10).join(group_table)}

![强后弱恢复与平台混淆](../artifacts/landslide-adaptive-review-2026-10-09/scale-review-development-v1/analysis-v1/case_2030.png)

![结束后残差被扩成活动](../artifacts/landslide-adaptive-review-2026-10-09/scale-review-development-v1/analysis-v1/case_2028.png)

## 结果、源码与复现

[全部28配置及成本](../artifacts/landslide-adaptive-review-2026-10-09/scale-review-development-v1/analysis-v1/summary.csv)、[逐日数据](../artifacts/landslide-adaptive-review-2026-10-09/scale-review-development-v1/analysis-v1/daily-predictions.csv)、[最近邻阅读边界](../artifacts/landslide-adaptive-review-2026-10-09/literature-v1/scale-review-search-v1/reading-notes.md)。代码为src/gnss_sim/landslide_scale_review.py及probe_landslide_scale_review.py、fit_landslide_scale_controls.py、analyze_landslide_scale_review.py；输入、选择、权限、模型、预测和执行源码哈希均保存。

```powershell
$env:PYTHONPATH = "$PWD/src;$PWD/scripts"
$experimentPython = 'artifacts/reproduction-2026-10-07/environment/Scripts/python.exe'
$artifactRoot = 'artifacts/landslide-adaptive-review-2026-10-09'
# 新输出目录必须不存在；只读重算评分，无模型请求或重新拟合。
& $experimentPython scripts/analyze_landslide_scale_review.py --run "$artifactRoot/scale-review-development-v1" --controls "$artifactRoot/scale-information-controls-v1" --out "$artifactRoot/scale-review-analysis-replay"
& $experimentPython -m pytest tests/test_landslide_scale_review.py -q
```

ARTIST[19]和金融多尺度论文[20]书目信息及阅读范围见本轮literature目录；全文正式引用已在[完整稿v14](landslide-adaptive-review-manuscript-draft.md)列出。本轮还不能区分一个记录需要局部观察与其中哪一段值得改写；后续方法需要验证活动结束后残差与持续弱尾的可区分证据，不能只再调整本门槛或继续扩大框架。
"""
    draft_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    previous = draft_path.read_bytes()
    if previous != (BASE / "manuscript-v13/manuscript-v13.md").read_bytes():
        raise ValueError("Current manuscript differs from frozen v13")
    (out / "manuscript-v13-original.md").write_bytes(previous)
    draft = previous.decode("utf-8")
    draft = draft.replace(
        "> 完整中文初稿v13，2026-10-09。新增单向漏运动修订的逐日、校准和连续性三项开发实验：129评分行均保留，三版未有稳定收益，停止扩展；新增API请求0。此前阶段确认0触发、全部失败和强基线继续保留。独立领域观测参考仍缺，本稿不是投稿定稿。",
        "> 完整中文初稿v14，2026-10-09。新增历史尺度压缩的观察准入开发：28配置、同信息学习控制、全部消融与成本；保留弱尾收益并减少调用，但登记继续条件未全部通过，窗口位置优越性未建立。本轮新增VLM请求0；独立领域观测参考仍缺，本稿不是投稿定稿。",
    )
    draft = draft.replace("## 7 讨论", section + "## 7 讨论", 1)
    chinese = "进一步的历史尺度压缩准入减少了开发复核调用并保留弱尾收益，但第二遍事件F1下降，同预算对照的日F1更高，窗口选择优势未建立。完整准入条件未通过，同信息学习控制和全部取舍均公开。"
    english = "A subsequent history-compression admission gate reduces cached development review calls while preserving weak-tail gains. The second repetition loses event F1, and a matched-budget control attains higher daily F1; better window selection is not established. The registered continuation condition fails, and same-information learning controls and all trade-offs are disclosed."
    draft = draft.replace("**关键词：**", chinese + "\n\n**关键词：**", 1)
    draft = draft.replace("## 1 引言", english + "\n\n## 1 引言", 1)
    references = """[19] Shvat Messica, Jiawen Zhang, Kevin Li, Theodoros Tsiligkaridis, Marinka Zitnik. (2026). *Adaptive Time Series Reasoning via Segment Selection*. [arXiv:2602.18645v3](https://arxiv.org/abs/2602.18645v3). 作者公开库[ARTIST](https://github.com/mims-harvard/ARTIST)，阅读范围见本轮笔记；未执行作者模型。

[20] Kaiqi Hu, Linda Xiao, Shiyue Xu, Ziyi Tang, Mingwen Liu. (2026). *Do VLMs Truly “Read” Candlesticks? A Multi-Scale Benchmark for Visual Stock Price Forecasting*. [arXiv:2604.12659v1](https://arxiv.org/abs/2604.12659v1). 已读摘要与3.2视觉数据处理，未运行作者实验。

"""
    draft = draft.replace("## 附录A", references + "## 附录A", 1)
    draft = draft.replace("| 当前全文与v12原字节快照、单向修订图 | manuscript-v13 |",
                          "| 历史全文v13与v12原字节快照、单向修订图 | manuscript-v13 |\n"
                          "| 历史尺度准入28配置、同信息XGB与全部停止条件 | scale-review-development-v1/analysis-v1；scale-information-controls-v1 |\n"
                          "| 当前全文与v13原字节快照、尺度准入图 | manuscript-v14 |")
    draft_path.write_text(draft, encoding="utf-8")
    (out / "manuscript-v14.md").write_text(draft, encoding="utf-8")
    (ROOT / "docs/landslide-scale-review-results.md").write_text(report, encoding="utf-8")
    ledger = read_json(BASE / "manuscript-v13/run-ledger-end.json")
    save_json(out / "run-ledger-end.json", {**ledger, "new_requests_started": 0,
                                            "new_completed_local_receipts": 0, "new_reported_tokens": 0,
                                            "new_records": 0, "round_status": "progress", "goal_status": "active"})
    save_json(out / "checks.json", {"analysis": read_json(RUN / "analysis-v1/state.json"),
                                    "decision": read_json(RUN / "decision.json"), "new_tests_passed": 4,
                                    "v13_snapshot_exact": (out / "manuscript-v13-original.md").read_bytes() == previous,
                                    "files": {path.name: sha(path) for path in out.iterdir() if path.is_file()}})
    print({"manuscript": str(out), "configs": len(rows), "new_VLM_requests": 0}, flush=True)


if __name__ == "__main__":
    close()
