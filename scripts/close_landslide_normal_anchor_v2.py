"""Archive two stopped probes without rewriting their frozen experiments."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from close_landslide_utility_round import comparison_table, read_rows
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_normal_contrast import BASE, verify
from run_landslide_window_optimization import read_json


def close():
    normal = BASE / "normal-contrast-development-v1"
    anchor = BASE / "numerical-anchor-development-v1/cache-screen-v1"
    verify(normal)
    for name, expected in read_json(normal / "prediction-freeze.json").items():
        if sha(normal / name) != expected:
            raise ValueError("Normal-reference predictions changed")
    for name, expected in read_json(anchor / "prediction-freeze.json").items():
        if sha(anchor / name) != expected:
            raise ValueError("Numerical-anchor screening changed")
    normal_decision = read_json(normal / "analysis-v1/decision.json")
    anchor_decision = read_json(anchor / "decision.json")
    if normal_decision["allows_full_prototype"] or anchor_decision["allows_three_missing_repeat1_calls"]:
        raise ValueError("This closure requires both registered probes to stop")
    draft = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    previous = draft.read_bytes()
    if previous != (BASE / "manuscript-v18/manuscript-v18.md").read_bytes():
        raise ValueError("Current draft is not delivered v18")
    out = BASE / "manuscript-v19"
    out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    (out / "manuscript-v18-original.md").write_bytes(previous)
    rows = read_rows(normal / "analysis-v1/summary.csv")
    anchor_rows = read_rows(anchor / "summary.csv")
    indexed = {row["method"]: row for row in rows}
    anchor_changes = read_json(anchor / "changes.json")
    costs = read_json(normal / "analysis-v1/actual-costs.json")
    normal_short = [indexed[f"{stem}_r{rep}"] for rep in (0, 1) for stem in
                    ("direct_visual", "normal_retrieved", "normal_fixed", "raw_repeated")]
    normal_short += [indexed["xgboost_84"], indexed["xgboost_raw_persistence"]]
    anchor_short = [row for row in anchor_rows if row["method"] in
                    ("xgboost_84", "numerical_anchor_visual_supplement_r0",
                     "numerical_anchor_fixed_r0", "numerical_anchor_raw_supplement_r0")]
    anchor_net = sum(row["net_correct"] for row in
                     anchor_changes["numerical_anchor_visual_supplement_r0"].values())
    section = f"""### 6.26 实际正常实例检索：新增参照没有带来稳定纠错

本轮检验实际背景实例，而非继续增加观察长度。四个共同train零注入记录case2001/2007/2037/2043提供32个180日窗口。每窗取六个30日原始位移中位数块，后五块减首块组成15维三轴增量；按查询日差MAD噪声代理加权距离，检索三个不同训练背景。参照图仅减首30日中位数常数，不缩幅度、不去趋势、不补缺测。正常仅指无注入，未由专家确认现场静稳。六已见开发记录、两个共享背景、原raw选窗、14日触发、49日权限、最多两局部请求及共同数值阶段头不变，查询标签不参与检索。

检索参照、固定参照和重复查询三组使用相同提示、三图和共同坐标范围。固定参照取前三训练记录的相同窗口索引，控制检索内容；它不是固定查询选窗，原固定复核另作为legacy保留。TAMA已有正常图示例，Anlu已有参照条件化训练[22]；本项目具体增量检索本身不足以成为贡献。MORA用同端点短/长上下文重建同一目标[23]，其单向分数范围关系也不保证GNSS二值识别无损。

{comparison_table(normal_short)}

检索F1 .5511/.5683，重复原图.5691/.6015，固定参照.5562/.6101；两遍相对重复原图净确定日纠错−350/−179，新增正确与损失正确分别0/350、251/430。第一遍新增346FP、撤236FP，没有新增TP；第二遍新71TP也增430FP、撤60FP。误报改未知不算正确阴性，不能只凭撤FP宣称可靠性收益。对直接净纠错−682/−752，对固定参照+21/−243。两遍所有登记条件均失败，停止此版本，不按固定参照或第二遍救分，也不追加检索派生数值控制来挽救无收益主组。

原12强数值控制使用共同train24/validation12，已包含四条零注入记录；同可评价日期、事件、边界、活动内/端到端ASD及覆盖完整保留。XGB84 .7553、原始持续性XGB .7659高于本轮视觉配置。弱召回提高并不足以抵消背景FP，活动内阶段仍由共同数值头产生，不称原生VLM阶段创新。六记录只来自两个背景，不作独立确认或显著性结论。

实际72启动/72回执/72成功JSON、453576已报告token，无重试或失败；复用56历史回执。主部署每遍18请求、90432/90382token，重复原图90365/90441、固定参照90490/90611；无调用节省。24配置、36完整状态/权限重放、32训练窗重建、24检索/固定排名重算、128payload、54共同及72额外数值预测重算、105120逐日行、192分组、六全图保存；3新增加14相关业务测试已通过，不重复运行旧推理。

### 6.27 数值初判加视觉漏检补充：首遍必要门槛失败，未补第二遍

基于局部VLM误报累积，另登记以共同XGB84代替全局视觉初判。仅用raw21/42的missed_motion_days排名，不用总冲突；仍14日触发、两个不重叠180日目标与49日邻域。保留数值已有阳性，只允许非阳性日期由局部VLM改写。匹配固定查询窗及相同权限纯数值raw补充作为消融。历史纯图提示不含初判状态，故只复用相同目标、原图与提示的真实回答，不能为未执行回答补结果。

{comparison_table(anchor_short)}

全部第一遍所需回答已有缓存，先冻结完整预测再读标签，新增API为0。主F1 .5400低于数值初判.7553，也未超raw补充.5403；弱召回从.2974提高到.5761。新增121TP、1173FP，无原TP损失，净确定纠错{anchor_net}日，六条均为负。保留强初判阳性不等于总识别可靠性改善。主事件F1 .4348/.4348，数值初判.6250/.5000；完整边界和阶段取舍列出。12强数值加六视觉/补充配置，共18配置、78840逐日行；24补充payload重放、原阳性保留和权限外不变验证通过。

登记要求两遍主F1高于数值初判及raw补充、弱R提高、净纠错为正且至少两记录受益，并至少一遍严格超固定查询补充。第一遍必要条件已失败，两遍条件不可能通过；不执行三条缺失第二遍请求，不生成或评分第二遍预测。此结果是单遍提前停止，不称两遍已验证。部署首遍12局部请求/48847token属于历史回答成本，缓存筛选不是免费推理；纯数值补充无VLM调用，其计算CPU未单独测量。

两项机制均停止，未得到新的稳定背景贡献。保留原冲突范围方法的受控确认及场景限制，不将投票、检索函数、额外图像或最佳消融当论文创新。正式观测效果仍缺独立领域参考；用户暂无复核者，继续可自主研究，不造专家标签。累计2402启动/2401回执/至少10537390已报告token，历史费用缺口保留，新记录0；108纯受控与60观测残差半合成分别计数。

![正常实例及数值初判筛选](../artifacts/landslide-adaptive-review-2026-10-09/manuscript-v19/normal-anchor-comparison.png)

"""
    reproduction = """

## 复现与范围

原正常参照72请求已结束，不重跑推理、不覆盖冻结源码/预测。仅重新评分时使用不存在的输出目录：

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$env:PYTHONIOENCODING='utf-8'
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/analyze_landslide_normal_contrast.py --run artifacts/landslide-adaptive-review-2026-10-09/normal-contrast-development-v1 --out artifacts/landslide-adaptive-review-2026-10-09/normal-contrast-score-replay
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/screen_landslide_numerical_anchor.py --prepared artifacts/landslide-adaptive-review-2026-10-09/numerical-anchor-development-v1 --out artifacts/landslide-adaptive-review-2026-10-09/numerical-anchor-score-replay
```

前者freeze.json、execution-bindings.json、prediction-freeze.json绑定输入与已执行代码；后者bindings.json及cache-screen-v1/prediction-freeze.json绑定预备与单遍预测。分析导出全部逐日、事件、阶段、修订与统一评分，正式现场参考未知。文献定向查询12条检索记录有重复，不是12篇纳入；Anlu/MORA仅读笔记中登记段落，未定位可信作者源码或运行其模型。
"""
    report = "# 正常实例检索与数值初判补充：两项机制停止\n\n" + section
    report += "\n## 正常参照全部24配置\n\n" + comparison_table(rows)
    report += "\n\n## 数值初判首遍全部18配置\n\n" + comparison_table(anchor_rows)
    report += reproduction
    (ROOT / "docs/landslide-normal-anchor-results.md").write_text(report, encoding="utf-8")
    (out / "normal-anchor-report.md").write_text(report, encoding="utf-8")
    text = previous.decode("utf-8").replace("\r\n", "\n")
    text = text.replace(text.splitlines()[2],
        "> 完整中文初稿v19，2026-10-10。新增正常实例检索两遍成对实验和数值初判单遍提前停止；全部强基线、消融及成本保留。未建立新的稳定背景收益，独立领域参考仍缺，本稿不是投稿定稿。", 1)
    text = text.replace("## 7 讨论", section + "## 7 讨论", 1)
    text += "\n\n[22] Lan T, Gao Y, Lu Y, et al. Anlu: Enabling In-Context Time Series Anomaly Detection in Foundation Models via Counterfactual Supervision. arXiv:2610.06180v1, 2026. https://arxiv.org/abs/2610.06180v1\n\n[23] Mou X, Wang T, Wang R, et al. MORA: Modeling Observed Changes for Drift-Robust Time-Series Anomaly Detection. arXiv:2610.09473v1, 2026. https://arxiv.org/abs/2610.09473v1\n"
    draft.write_text(text, encoding="utf-8")
    (out / "manuscript-v19.md").write_text(text, encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    stems = ("normal_retrieved", "normal_fixed", "raw_repeated", "direct_visual", "xgboost_84")
    for rep in (0, 1):
        values = [float(indexed[stem if stem == "xgboost_84" else f"{stem}_r{rep}"]["f1"]) for stem in stems]
        axes[0].bar(np.arange(len(stems)) + (rep - .5) * .36, values, width=.36, label=f"Repeat {rep+1}")
    selected_anchor = [row for name in ("xgboost_84", "numerical_anchor_visual_supplement_r0",
                                       "numerical_anchor_fixed_r0", "numerical_anchor_raw_supplement_r0")
                       for row in anchor_short if row["method"] == name]
    axes[1].bar(range(4), [float(row["f1"]) for row in selected_anchor], color="#668da8")
    axes[0].set_xticks(range(5), [stem.replace("_", "\n") for stem in stems], fontsize=8)
    axes[1].set_xticks(range(4), ["XGB84", "Visual\nsupplement", "Fixed\nsupplement", "Raw\nsupplement"], fontsize=8)
    axes[0].set_title("Normal-reference contrast | both repeats fail")
    axes[1].set_title("Numerical anchor | repeat 1 only, early stop")
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.set_ylim(0, .85)
        axis.set_ylabel("Activity F1")
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Six exposed records / two shared backgrounds | no independent field claim")
    fig.tight_layout()
    fig.savefig(out / "normal-anchor-comparison.png", dpi=150)
    plt.close(fig)
    old = read_json(BASE / "manuscript-v18/run-ledger-end.json")
    ledger = {
        "new_requests_started": costs["new_started"],
        "new_completed_local_receipts": costs["new_local_receipts"],
        "new_valid_model_json": costs["new_returned_json"],
        "new_reported_tokens": costs["new_reported_tokens"],
        "cumulative_requests_started": old["cumulative_requests_started"] + costs["new_started"],
        "cumulative_local_receipts": old["cumulative_local_receipts"] + costs["new_local_receipts"],
        "cumulative_reported_tokens_lower_bound": old["cumulative_reported_tokens_lower_bound"] + costs["new_reported_tokens"],
        "new_records": 0, "pure_controlled_records": 108, "observed_residual_semisynthetic_records": 60,
        "inference_retries": 0, "failures": 0, "normal_contrast_branch": "stopped",
        "numerical_anchor_branch": "stopped_after_repeat0", "numerical_anchor_new_api": 0,
        "unexecuted_missing_repeat1_requests": 3, "repeat1_anchor_scored": False,
        "billing": "Reported usage lower bound; historic unknown costs retained, shared caches counted once",
        "goal_status": "active", "round_status": "progress",
    }
    for destination in (out, normal):
        save_json(destination / "run-ledger-end.json", ledger)
    entry = """## 2026-10-10：正常实例检索与数值初判补充停止，全文v19

[本轮结果](landslide-normal-anchor-results.md)：实际train零注入实例的原始30日块增量检索，检索/固定参照/重复原图三组同提示、图数及尺度。六已见记录、两共享背景、两遍、原49权限不变。检索F1 .5511/.5683对重复原图.5691/.6015，净确定纠错−350/−179，两遍全部登记条件失败，停止，不换固定或搜检索参数。24统一配置含12强数值；原始持续性XGB .7659、XGB84 .7553均更高。72真实请求全部成功、453576token，无重试；36状态/权限、32训练窗、24检索选择、128payload、105120逐日行、192分组及六图核验保存；17业务测试已通过。

另登记XGB84数值初判＋VLM漏检补充，所有初判阳性保留，raw missed_motion排名／14日触发／两窗／49权限不变，固定窗与同权限raw补充对照。第一遍完整缓存API0，主F1 .5400对初判.7553及raw .5403；新121TP也增1173FP，净确定纠错−1036，六条均为负。必要条件失败，不收三缺失第二遍请求，不生成/评分第二遍。18配置、78840逐日行、24补充payload及阳性保留/权限外不变验证通过；首遍部署12局部/48847token是历史成本，缓存复用不称零推理。

论文v19、精确v18原字节、全部活动/事件/边界/阶段/未知/成本保留；累计2402启动/2401回执/至少10537390token，新记录0，108纯受控/60半合成分开。Anlu/MORA官方PDF、真实阅读段落和未找到可信作者源码的限制保存；TAMA正常图示例先例不主张首创。两项新机制停止，旧冻结源/模型/预测及失败版本不修改，不重跑旧推理。原冲突范围方法的受控确认仍保留，暂无稳定新背景贡献或正式现场结论。用户暂无领域复核者，继续自主研究，不再询问或造专家标签；goal active。

"""
    for name in ("project-status.md", "landslide-research-continuation-prompt.md", "landslide-submission-roadmap.md"):
        path = ROOT / "docs" / name
        first, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(first + "\n\n" + entry + rest.lstrip("\n"), encoding="utf-8")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8").replace("- [新增360日上下文",
        "- [正常实例检索与数值初判补充](docs/landslide-normal-anchor-results.md)：检索两遍失败，数值初判首遍提前停止；72新请求、42评分配置行，论文v19。\n- [新增360日上下文", 1)
    readme.write_text(text, encoding="utf-8")
    save_json(out / "checks.json", {
        "exact_v18_preserved": sha(out / "manuscript-v18-original.md") == sha(BASE / "manuscript-v18/manuscript-v18.md"),
        "current_matches_v19": sha(draft) == sha(out / "manuscript-v19.md"),
        "normal_checks": read_json(normal / "analysis-v1/checks.json"),
        "anchor_checks": read_json(anchor / "checks.json"), "anchor_net_definite_correction": anchor_net,
        "registered_branches_stopped": True, "goal_status": "active",
    })
    print({"version": 19, "ledger": ledger}, flush=True)


if __name__ == "__main__":
    close()
