"""Retain the stopped context probe, its costs and the exact preceding draft."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from close_landslide_utility_round import comparison_table
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_context_observation import BASE, verify
from run_landslide_window_optimization import read_json


def close():
    run = BASE / "context-observation-development-v1"
    analysis = run / "analysis-v1"
    verify(run)
    decision = read_json(analysis / "decision.json")
    if decision["allows_full_prototype"]:
        raise ValueError("This closure is for the stopped context experiment")
    draft = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    previous = draft.read_bytes()
    if previous != (BASE / "manuscript-v17/manuscript-v17.md").read_bytes():
        raise ValueError("The current draft is not the delivered v17")
    out = BASE / "manuscript-v18"
    out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    (out / "manuscript-v17-original.md").write_bytes(previous)
    with (analysis / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summaries = list(csv.DictReader(handle))
    indexed = {row["method"]: row for row in summaries}
    costs, checks = [read_json(analysis / name) for name in ("actual-costs.json", "checks.json")]
    changes = read_json(analysis / "changes.json")
    chosen = [indexed[f"{stem}_r{rep}"] for rep in (0, 1) for stem in
              ("direct_visual", "adaptive_local", "adaptive_context", "adaptive_duplicate", "fixed_local", "fixed_context")]
    chosen += [indexed[name] for name in ("robust_rule", "pelt_rule", "xgboost_84", "xgboost_direction_information")]
    section = f"""### 6.25 新增360日视觉上下文：优于重复图，但未稳定超原局部观察

停止收益校准后，本轮直接检验新增视觉证据。按既有六家族顺序交替取两个已见背景，选择case2025静稳、2032仪器阶跃、2027弱匀速、2034A/S/D、2029再活动及2036强后弱，六条均来自已用于机制选择的开发记录，不作独立确认；不按当前结果再挑样本。原21／42日raw冲突选窗、14日触发、最多两个不重叠180日目标、49日修订权限及共同数值阶段头不变。主组在原全局＋180日图之外增加360日原始图，两倍范围以目标中心定位，记录边缘平移保持360日，不插值、去趋势、缩放位移或使用生成标签。输出仍只许原180日目标内，再由原49权限限制最终修订。无学习接受器或数值投票。

重复180图组使用相同第三图说明、相同三张图及同一目标，第三图与180图字节相同，用于区分新增尺度与重复观察。固定360及固定180组控制选窗；三图主组每条仍只有至多两次局部请求，不将额外图像当调用节省。两遍新请求均执行，原180回答缓存复用；过去与当前API回答的时间差仍是限制，重复图控制不能证明无时间漂移。纯视觉指反馈不含数值趋势文字，主组保留数值选窗与范围约束。

{comparison_table(chosen)}

主F1 .5220／.5503，原局部.5288／.5080，重复图.5011／.5119，直接.6553／.6617；相对原局部净确定日纠错−187／+44，帮助记录1／3。第一遍新增5TP、丢29TP、新增197FP、撤214FP；撤回中176个误报变未知，仅38个改为正确阴性，同时4个正确阴性变未知，因此少误报不能等同正确确定判断增加。第二遍新增159TP、无TP损失，但新增203FP、撤88FP。相对重复图两遍净纠错+30／+96，表明新增尺度与单纯复制图的结果不同；它没有建立超过原两图复核的稳定净收益。两遍对直接净纠错−905／−990，XGB84 F1 .7553、方向XGB .7414均更高，强数值控制不省略。主弱R .7307／1.0000，原局部两遍.7307；事件F1 .5000／.6000高于原局部.4444／.4762，但其边界、覆盖和阶段权衡完整列出，不以单项更高替代登记主指标。

针对旧混合提案case2036，window4两遍相对原局部逐日状态完全不变，未分离该窗弱活动与背景。第一遍损失来自另一个window2的29个真活动日未保留，扣除11个修好阴性日，净−18；第二遍window6新增115个正确活动日，同时65个原正确阴性日误报，净+50。新增上下文对部分弱尾有观察价值，尚未可靠地区分残差季节变化与注入活动。背景观测未被独立复核，以上误报仅按已知注入任务定义，不能写成实际坡体假活动。模型简短evidence说明记录可见N/E/U趋势，不能作为这些趋势必然对应注入运动的证明。

执行前登记两遍主F1高于原局部、净确定纠错>0、各至少两记录受益，且不低于重复图、至少一遍严格超重复图。第一遍前三项失败，第二遍通过；整体不允许扩展完整原型。停止此版本，不在该六条搜上下文长度、改权限或择第二遍保留主方法，也不把较高固定组改登记主组。

实际新增68启动／68回执／68有效JSON、377846已报告token，其中360图44、重复180图24，无重试、传输或解析失败；复用56历史回执。主部署每遍18请求／81523或81548token，原局部18／63724或63837，增加约1.78万图像token，没有调用节省；图片渲染与特征CPU未单独计时，不填零CPU。累计2330启动／2329回执／至少10083814已报告token，旧中断费用未知仍保留。新增记录0，108受控与60观测残差半合成分别计数。24配置、60完整状态与权限重放、124实际payload、54共同数值预测和72额外方向／尺度预测重算、192分组、105120逐日行、24目标变化及六图保存。独立逐日与阶段评分交叉核验通过，14相关业务测试及源码Ruff通过。

本轮提供尺度信息与重复图的成对机制证据，没有得到可升级的新稳定贡献。原已冻结受控范围方法及其适用局限保留，收益校准与本上下文版本均停止。后续候选必须先解决目标内运动／背景可分辨性或修订损害，避免继续仅扩图、搜提示或增加整窗接受参数；任何新方法先小规模取证、再冻结未参与选择的验证。独立领域复核者仍缺，正式观测效果未知，不造标签、不重复索要人员。

![成对上下文观察与确定日纠错](../artifacts/landslide-adaptive-review-2026-10-09/manuscript-v18/context-comparison.png)

"""
    report = "# 成对上下文观察：新增尺度优于重复图，未稳定优于原局部复核\n\n" + section
    report += "\n## 全部24项统一比较\n\n" + comparison_table(summaries)
    report += """

## 复现

原68请求已完成，禁止重跑旧推理。只重算时输出目录必须不存在。

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$env:PYTHONIOENCODING='utf-8'
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/analyze_landslide_context_observation.py --run artifacts/landslide-adaptive-review-2026-10-09/context-observation-development-v1 --out artifacts/landslide-adaptive-review-2026-10-09/context-observation-score-replay
```

源、输入、图、协议、请求包、选窗及初判在freeze.json；输出在prediction-freeze.json。analysis-v1含完整结果／事件／阶段／逐日数据、逐目标变化与3×3状态转移。当前六记录及两背景不构成六个独立观测站点，不作显著性结论。
"""
    (ROOT / "docs/landslide-context-observation-results.md").write_text(report, encoding="utf-8")
    (out / "context-observation-report.md").write_text(report, encoding="utf-8")
    text = previous.decode("utf-8").replace("\r\n", "\n")
    header = text.splitlines()[2]
    text = text.replace(header,
        "> 完整中文初稿v18，2026-10-10。新增360日上下文与重复180图成对取证、24项强基线／消融和真实68请求成本；新上下文未稳定超原局部复核，停止扩展。独立领域参考仍缺，本稿不是投稿定稿。", 1)
    text = text.replace("## 7 讨论", section + "## 7 讨论", 1)
    text += "\n\n2026-10-10 v18更新：新增上下文成对机制实验；主F1 .5220/.5503，原局部.5288/.5080，净确定纠错−187/+44，停止此版本。新增68请求／377846token，累计2330启动／2329回执／至少10083814token。goal仍active，独立观测效果未知。\n"
    draft.write_text(text, encoding="utf-8")
    (out / "manuscript-v18.md").write_text(text, encoding="utf-8")
    fig, panels = plt.subplots(1, 2, figsize=(12, 4.7))
    stems = ["direct_visual", "adaptive_local", "adaptive_context", "adaptive_duplicate", "fixed_context", "xgboost_84"]
    for rep in (0, 1):
        scores = [float(indexed[f"{stem}_r{rep}" if stem != "xgboost_84" else stem]["f1"]) for stem in stems]
        panels[0].bar(np.arange(len(stems)) + (rep - .5) * .36, scores, width=.36, label=f"Repeat {rep+1}")
        net = [sum(row["net_correct"] for row in changes[f"adaptive_context_r{rep}"][control].values())
               for control in ("adaptive_local", "adaptive_duplicate")]
        panels[1].bar(np.arange(2) + (rep - .5) * .36, net, width=.36, label=f"Repeat {rep+1}")
    panels[0].set_xticks(range(len(stems)), [name.replace("_", "\n") for name in stems], fontsize=8)
    panels[0].set_ylim(0, .85)
    panels[0].set_title("Six exposed records | activity F1")
    panels[1].set_xticks(range(2), ["vs original 180-day review", "vs duplicate180 control"], fontsize=8)
    panels[1].axhline(0, color="black", linewidth=.7)
    panels[1].set_title("Contextual view | net definite corrected days")
    for panel in panels:
        panel.legend(fontsize=8)
        panel.grid(axis="y", alpha=.2)
    fig.suptitle("Paired context probe | first repeat fails registered continuation | no independent claim")
    fig.tight_layout()
    fig.savefig(out / "context-comparison.png", dpi=150)
    plt.close(fig)
    old = read_json(BASE / "manuscript-v17/run-ledger-end.json")
    ledger = {
        "new_requests_started": costs["new_started"], "new_completed_local_receipts": costs["new_local_receipts"],
        "new_valid_model_json": costs["new_returned_json"], "new_reported_tokens": costs["new_reported_tokens"],
        "cumulative_requests_started": old["cumulative_requests_started"] + costs["new_started"],
        "cumulative_local_receipts": old["cumulative_local_receipts"] + costs["new_local_receipts"],
        "cumulative_reported_tokens_lower_bound": old["cumulative_reported_tokens_lower_bound"] + costs["new_reported_tokens"],
        "context360_collection_requests": 44, "duplicate180_collection_requests": 24,
        "pure_controlled_records": 108, "observed_residual_semisynthetic_records": 60, "new_records": 0,
        "training_requests": 0, "inference_retries": 0, "failures": 0, "cached_receipts_verified": 56,
        "billing": "Lower bound of reported usage only; historical missing receipt and unreported failures retained",
        "goal_status": "active", "round_status": "progress", "context_branch": "stopped",
    }
    save_json(out / "run-ledger-end.json", ledger)
    save_json(run / "run-ledger-end.json", ledger)
    entry = """## 2026-10-10：成对新增上下文实验完成，未通过扩展，全文v18

[新增上下文结果](landslide-context-observation-results.md)：六已见记录／两共享背景，各六家族一条。原raw选窗／49权限不变，全局＋180之外加入360原始图，相同三图提示的重复180控制，以及固定180／360对照。68唯一请求全部成功／377846token，无重试；主F1 .5220/.5503对原局部.5288/.5080，净确定纠错−187/+44，仅1/3记录受益。两遍都优于重复图.5011/.5119（净+30/+96），但第一遍未超原局部，整体预登记门槛失败，停止该版本，不搜长度／提示／权限、不择第二遍或固定救分。旧case2036 window4两遍完全无新增状态修正；第一遍另窗丢29TP，第二遍末尾新115TP也增65FP，不能声称已分离背景。

24配置含12强数值控制，XGB84 F1 .7553更高；60完整状态与权限重放、124实际payload、54共同＋72额外数值重算、192分组、105120逐日行、24目标变化、3×3状态转移及6图保存，14相关业务测试／Ruff通过。每遍部署18调用，约81523/81548token对原局部63724/63837，没有调用节省。全文v18、精确v17原字节、成本算术和图表保留。累计2330启动／2329回执／至少10083814token，新记录0，108纯受控／60半合成分开；历史缺口不填零费。

所有本轮推理及评分已结束，不重跑旧run、不改冻结源／预测。收益校准与此新增上下文版本均停止。后续需提出有依据的运动／背景可辨别观察或修订机制，先小规模取证，再原强基线／关键消融／新冻结确认；不要继续仅扩图或学整窗投票。原范围方法的受控确认仍保留，当前无新稳定背景贡献或正式现场结论。用户暂无独立复核者，继续自主研究，不再索要或造标签；goal active，本轮progress。

"""
    for name in ("project-status.md", "landslide-research-continuation-prompt.md", "landslide-submission-roadmap.md"):
        path = ROOT / "docs" / name
        first, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(first + "\n\n" + entry + rest.lstrip("\n"), encoding="utf-8")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("- [四背景部署提案校准与完整迁移]",
        "- [新增360日上下文与重复图成对取证](docs/landslide-context-observation-results.md)：68真实请求、24项强基线与消融，未稳定超原局部复核，停止该版本，论文v18。\n- [四背景部署提案校准与完整迁移]", 1)
    readme.write_text(text, encoding="utf-8")
    save_json(out / "checks.json", {"exact_v17_preserved": sha(out / "manuscript-v17-original.md") == sha(BASE / "manuscript-v17/manuscript-v17.md"),
        "current_matches_v18": sha(draft) == sha(out / "manuscript-v18.md"), "analysis_checks": checks,
        "gate_allows_expansion": False, "actual_new_costs": costs, "goal_status": "active"})
    print({"version": 18, "ledger": ledger, "goal": "active"}, flush=True)


if __name__ == "__main__":
    close()
