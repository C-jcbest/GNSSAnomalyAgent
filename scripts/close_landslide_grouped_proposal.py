"""Record grouped calibration transfer failure and preserve the delivered v16."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from close_landslide_utility_round import comparison_table
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close():
    calibration = BASE / "grouped-proposal-utility-development-v1"
    complete = BASE / "grouped-proposal-prototype-development-v1"
    if not read_json(calibration / "analysis-v1/decision.json")["allows_full_prototype"]:
        raise ValueError("The grouped pilot did not pass the registered continuation gate")
    if read_json(complete / "decision.json")["allows_independent_confirmation"]:
        raise ValueError("This closure is for the stopped grouped branch")
    out = BASE / "manuscript-v17"
    out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    draft = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    previous = draft.read_bytes()
    if previous != (BASE / "manuscript-v16/manuscript-v16.md").read_bytes():
        raise ValueError("Current manuscript is not the delivered v16")
    (out / "manuscript-v16-original.md").write_bytes(previous)
    val_rows, full_rows = rows(calibration / "analysis-v1/summary.csv"), rows(complete / "summary.csv")
    val, full = [{row["method"]: row for row in values} for values in (val_rows, full_rows)]
    val_short = [val[f"{stem}_r{rep}"] for rep in (0, 1) for stem in
                 ("direct_visual", "grouped_proposal_guard", "numeric_vote", "visual_guard", "fixed_grouped_guard")]
    val_short.append(val["xgboost_84"])
    full_short = [full[f"{stem}_r{rep}"] for rep in (0, 1) for stem in
                  ("direct_visual", "grouped_proposal_guard", "numeric_vote", "visual_guard", "fixed_grouped_guard")]
    full_short += [full[name] for name in ("robust_rule", "pelt_rule", "xgboost_84", "xgboost_direction_information")]
    section = f"""### 6.23 四背景部署提案校准：开发验证通过，实际增益需区别未知

单背景校准停止后，另登记共同train24四背景、validation12两背景的部署提案校准。五项实际提案特征、Ridge(alpha=1)、零收益准入、原raw选窗及49日权限均不变；每训练记录每遍只采部署实际选中的至多两窗，而非采全部8窗。四个背景各以另外18条共同训练记录，按原raw84的固定120树／深度4／学习率.05重拟合背景外数值先验；验证使用原共同train24先验。没有搜正则、阈值或用验证标签拟合接受器。96训练提案由24记录×两遍×两窗构成，依然只有四个相关背景，不是96独立样本。训练模型在新增验证请求前冻结，但前六验证记录及其回答已参与开发，整个验证划分不作独立效果确认。

新登记继续条件以两遍活动F1提高、弱召回非损、相对纯视觉接受器及简单数值投票不低、至少一遍严格超过投票、每遍至少两记录净确定日纠错为正为依据。事件、边界、阶段和成本完整报告，不要求所有指标普遍非损；这是在本版本新回答前登记的目标，不用于复活6.20—6.21失败版本。它直接检验接受机制是否有超过数值投票的价值，符合不追求全部指标优胜的研究范围。

{comparison_table(val_short)}

两遍主F1 .6805／.7878对直接.6353／.7482，弱R .6802／.7174对.6802／.6593，登记条件通过，允许完整开发重放。第一遍相对数值投票.6604的F1增益来自179个阳性日变未知，**新增正确确定判断及损失正确确定判断均为0**；覆盖由投票1降至.9796，事件F1 .6667低于投票.72。第二遍与数值投票完全相同。故该验证尚未建立学习收益校准的真实额外纠错，不能把少FP或条件通过单独当可靠性贡献。XGB84 F1 .7774，第一遍明显更高、第二遍略低；其弱R .1988、事件／边界及API代价共同报告。

### 6.24 完整24条重放：相对数值投票没有稳定额外纠错

保持四背景接受模型逐字节原样，复用24条已见完整开发记录的原全局／局部回答和原raw两窗，不再请求模型。完整继续条件保留活动F1、弱召回、相对数值投票及多个记录实际纠错，并在重放评分前另登记两遍相对数值投票的净正确确定判断必须大于0，防止未知变动支撑“真实纠错”声明。事件、边界及阶段不作普遍优胜要求，但必须完整报告。未扩大训练、改变提示或准入。

{comparison_table(full_short)}

主组第一遍F1 .6645低于直接.6722，弱R同为.6178；事件F1 .6207低于.6792。第二遍F1 .6782高于直接.6330，弱R .5733高于.3910，事件F1 .7143高于.6923。相对简单数值投票，第一遍F1 .6645低于.6707，TP3744对3839、FP2764对2849，净正确确定判断−95，全部来自case2036；第二遍两组逐日预测完全相同，净增0。case2036第一遍window4数值预计正确率增量为+.00795，学习器预计净效用−.00552，因此拒绝整个混合提案；少85误报日同时少95真活动日，不能通过整窗拒绝分离弱尾与背景。它延续了单窗内部收益与损失混合的问题。完整两遍继续条件均失败，停止该版本，**不再在同批扩展收益校准分支或调Ridge／零门槛**。固定接受器第一遍日F1 .6754及事件.7586更高，也不按此替换登记主方法。

原规则、PELT、XGBoost及长尺度／方向／尺度学习控制均保留，完整12强控制共享原共同train24／validation12，数值预测全部重算一致；XGB84 F1 .7261、方向XGB .7350仍高于本方法两遍。VLM仍定义局部提案，学习器只决定是否接受，但当前效果主要复现数值投票，未证明其为独立Agent/VLM贡献。活动内与端到端A/S/D仍为共同数值阶段头重门控，不声称原生视觉阶段改进。

本版本实际新165启动／165回执／165成功JSON、595385token，无传输／解析失败、无重试，其中训练108、验证57；复用pilot216份回执，不重复计账。完整重放新API0，每遍部署流程72请求／254639或254820token，所有拒绝提案仍计复核成本，相对直接24／24没有调用节省。额外四数值先验拟合及特征CPU未隔离测量，不填零CPU。累计2262启动／2261回执／至少9705968已报告token；历史未知usage缺口仍保留，新增记录0，108纯受控与60半合成分开。

验证23配置、96实际训练提案、108数值验证预测重算、216缓存＋165新payload、184分组、201480逐日行和12全图；完整28配置、384提案／权限、288数值预测重算、315旧＋72此前补充payload、280分组、490560逐日行和24全图保存。补充重建96真实训练效用、精确重拟合2接受模型、复算36背景外／完整先验概率、检查4背景排除人口、72个投票相对记录差及模型原字节。文件元数据及执行逻辑佐证模型在新增验证前冻结，不作为外部时间戳证明。两新＋四相关业务测试和最终Ruff通过，无需为缓存评分重跑旧推理。

新检索官方arXiv两查询各6条，共12条检索记录，不等同纳入12篇研究。补读TSHarness[17]官方PDF摘要、3.1—3.5：单源冻结工具／记忆、按答案正确率与工具成本学习选择、带预算证据反馈均已有先例。FreshCast[21]3.5／3.6及4节相关段落根据预测错误与修正对齐闭式校准权重，非收益时保留原预测；该思想也不作为本文首创，其MSE和独立／同分布假设不能移用为本文窗口和背景漂移的无损保证。两HTML TLS EOF失败后官方PDF取得成功，原字节／hash／读取范围保留；未定位两篇作者源码、未运行作者模型。不把下载或全文抽取写成全文阅读。

该分支已完成训练、真实推理、完整强基线及消融，但没有得到稳定额外视觉纠错，停止，不升级独立确认。后续优先直接验证是否有新增视觉证据能分离混合提案内的持续弱运动与背景，先比较同一目标的局部与更长上下文，限制总复核次数；不继续把数值投票重命名为Agent或增加校准参数。原冲突范围方法的已冻结受控确认仍保留，正式现场总活动／阶段结论继续依赖独立可信参考，现状未知。

![四背景校准的验证和完整开发取舍](../artifacts/landslide-adaptive-review-2026-10-09/manuscript-v17/grouped-proposal-comparison.png)

"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    stems = ["direct_visual", "grouped_proposal_guard", "numeric_vote", "fixed_grouped_guard"]
    for axis, indexed, metric, title in ((axes[0], val, "f1", "12 validation | activity F1"),
                                          (axes[1], full, "f1", "24 exposed records | activity F1"),
                                          (axes[2], full, "event_f1_iou03", "24 exposed records | event F1")):
        for position, stem in enumerate(stems):
            axis.plot([position - .1, position + .1], [float(indexed[f"{stem}_r{rep}"][metric]) for rep in (0, 1)],
                      marker="o", linewidth=1)
        axis.set_xticks(range(len(stems)), [stem.replace("_", "\n") for stem in stems], fontsize=8)
        axis.set_ylim(.55, .9)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Frozen four-background calibration | paired repeats | no stable gain over numerical vote")
    fig.tight_layout()
    fig.savefig(out / "grouped-proposal-comparison.png", dpi=150)
    plt.close(fig)
    report = "# 四背景部署提案校准：验证通过，完整开发无稳定额外纠错\n\n" + section
    report += "\n## 全部登记比较\n\n### 完整验证划分\n\n" + comparison_table(val_rows)
    report += "\n\n### 24条完整开发\n\n" + comparison_table(full_rows)
    report += """

## 复现

原165请求已执行完成，不重跑旧推理。仅重算时输出目录必须不存在。

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$env:PYTHONIOENCODING='utf-8'
$experimentPython='artifacts/reproduction-2026-10-07/environment/Scripts/python.exe'
& $experimentPython scripts/analyze_landslide_grouped_proposal.py --run artifacts/landslide-adaptive-review-2026-10-09/grouped-proposal-utility-development-v1 --out artifacts/landslide-adaptive-review-2026-10-09/grouped-calibration-score-replay
& $experimentPython scripts/replay_landslide_grouped_proposal.py --calibration artifacts/landslide-adaptive-review-2026-10-09/grouped-proposal-utility-development-v1 --complete artifacts/landslide-adaptive-review-2026-10-09/utility-prototype-development-v1 --out artifacts/landslide-adaptive-review-2026-10-09/grouped-prototype-score-replay
```

冻结源／模型／请求位于grouped-proposal-utility-development-v1，评分analysis-v1；完整重放grouped-proposal-prototype-development-v1，实际投票相对确定日差与独立核验在supplement-v1。字段、未知与原始缺测掩码未改。
"""
    (ROOT / "docs/landslide-grouped-proposal-results.md").write_text(report, encoding="utf-8")
    (out / "grouped-proposal-report.md").write_text(report, encoding="utf-8")
    text = previous.decode("utf-8").replace("\r\n", "\n")
    text = text.replace("> 完整中文初稿v16，2026-10-09。新增实际视觉纠错收益训练、完整24条迁移及提案接受检验；两个新机制均未通过继续条件，完整成本、消融和失败保留。独立领域参考仍缺，本稿不是投稿定稿。",
                        "> 完整中文初稿v17，2026-10-10。新增四背景部署提案校准、12验证及24完整开发。验证通过，完整失败且没有稳定超过简单数值投票；强基线、真实成本和损失全部保留。独立领域参考仍缺，本稿不是投稿定稿。")
    if text.count("## 7 讨论") != 1:
        raise ValueError("Manuscript insertion location is ambiguous")
    text = text.replace("## 7 讨论", section + "## 7 讨论", 1)
    note = "进一步扩展至四训练背景并排除训练内数值先验后，完整开发仍未取得相对简单数值投票的额外确定日纠错，第一遍还损失95个正确日。问题不能仅归于单背景样本数；整窗接受把弱尾与背景损益混合，未知与精度必须分别核算。停止收益校准分支，把后续重点转回可直接检验的视觉新增证据。"
    text = text.replace("## 8 可复现性、完整性与AI使用说明", note + "\n\n## 8 可复现性、完整性与AI使用说明", 1)
    reference = "[21] He, C., Xu, J., Guo, X., Liu, R., Ding, H., He, R., Xie, Y., & Song, D. (2026). *Retrieval Is Not Enough: Refreshing Memory for Frozen Time-Series Forecasters*. [arXiv:2610.07834v2](https://arxiv.org/abs/2610.07834v2). 实际读取3.5／3.6及4节相关设定／命题段落，未读完整证明、未运行作者方法。\n\n"
    text = text.replace("## 附录A：产物定位", reference + "## 附录A：产物定位", 1)
    text = text.replace("| 当前全文v16与精确v15原字节 | manuscript-v16 |",
                        "| 历史全文v16与精确v15原字节 | manuscript-v16 |\n| 四背景部署校准及完整重放 | grouped-proposal-utility-development-v1；grouped-proposal-prototype-development-v1 |\n| 当前全文v17、精确v16原字节及比较图 | manuscript-v17 |")
    text += "\n最新执行状态：6.23—6.24已完整纳入四背景校准、投票对照和真实165请求成本；累计2262启动／2261回执／至少9705968token。校准分支停止，goal仍active，独立观测结论未取得。\n"
    draft.write_text(text, encoding="utf-8")
    (out / "manuscript-v17.md").write_text(text, encoding="utf-8")
    ledger = read_json(BASE / "manuscript-v16/run-ledger-end.json")
    ledger.update({"new_requests_started": 165, "new_completed_local_receipts": 165, "new_valid_model_json": 165,
                   "new_reported_tokens": 595385, "cumulative_requests_started": 2262,
                   "cumulative_local_receipts": 2261, "cumulative_reported_tokens_lower_bound": 9705968,
                   "pilot_collection_requests": 165, "pilot_reported_tokens": 595385,
                   "training_requests": 108, "validation_requests": 57, "complete_missing_view_requests": 0,
                   "complete_reported_tokens": 0, "complete_replay_new_requests": 0,
                   "goal_status": "active", "round_status": "progress"})
    save_json(out / "run-ledger-end.json", ledger)
    entry = """## 2026-10-10：四背景部署提案校准完成并停止，全文v17

[四背景校准结果](landslide-grouped-proposal-results.md)：共同train24／validation12，四背景各以其他18训练记录拟合原raw84先验，原五特征、Ridge(alpha1)、零准入及49权限不变。96实际部署训练提案，165真实新请求／595385token（108训练＋57验证），无失败／重试。验证F1 .6805/.7878对直接.6353/.7482，登记条件通过；第一遍相对数值投票的179阳性改未知没有新增正确确定判断，第二遍等同投票，不单独当可靠性增益。

精确保留模型扩展24已见完整记录、384提案重放、新API0。主F1 .6645/.6782对直接.6722/.6330；相对简单数值投票第一遍净确定纠错−95，全部case2036混合窗，第二遍完全相同、增0。两遍新确认条件均失败，停止校准分支，不扫Ridge／准入、不按固定较高替换。12强数值控制不删，XGB84 .7261／方向XGB .7350保持。验证23配置／184分组／201480逐日行／12图；完整28配置／288数值重算／280分组／490560逐日行／24图；实际训练96效用／2模型／36概率／4背景排除及72记录投票差重放。2新＋4相关测试及最终Ruff通过。

最新官方arXiv两查询12记录，补读TSHarness3.1—3.5和FreshCast3.5/3.6及4节相关段落：收益驱动取证、校准修正与拒绝非收益已有先例。两HTML TLS EOF失败后官方PDF成功，未找到作者源码／未运行作者模型，不套用其独立同分布MSE保证。全文v17、精确v16原字节和全部成本保留，当前累计2262启动／2261回执／至少9705968token，新记录0，108受控与60半合成分别计数。当前所有实验进程已结束，旧冻结源／预测不改，不重跑旧请求。

后续研究重点回到实际视觉新增证据：先在已见目标比较局部与更长上下文是否能分离同窗弱尾和背景，有限两次复核、完整强基线及简单对照；不继续扩展数值先验校准或将投票包装为Agent。暂未生成该新候选或独立确认。用户暂无独立复核者，继续自主研究、不重复询问，不造标签。goal active，本轮progress，尚无稳定新贡献或正式现场结论。

"""
    for name in ("project-status.md", "landslide-research-continuation-prompt.md", "landslide-submission-roadmap.md"):
        path = ROOT / "docs" / name
        first, rest = path.read_text(encoding="utf-8").split("\n", 1)
        path.write_text(first + "\n\n" + entry + rest.lstrip("\n"), encoding="utf-8")
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("- [实际视觉纠错收益训练与完整迁移]",
                        "- [四背景部署提案校准与完整迁移](docs/landslide-grouped-proposal-results.md)：真实165请求、23/28配置；验证通过但完整无稳定超数值投票增益，停止分支，论文v17。\n- [实际视觉纠错收益训练与完整迁移]", 1)
    readme.write_text(text, encoding="utf-8")
    nearest = ROOT / "docs/landslide-nearest-method-code-review.md"
    with nearest.open("a", encoding="utf-8") as handle:
        handle.write("\n\n## 2026-10-10：TSHarness及FreshCast补读\n\n" +
                     "TSHarness官方PDF3.1—3.5已实读，源冻结的工具选择、答案收益－调用代价及有预算重新感知已有先例。FreshCast3.5／3.6及4节相关命题把修正与原错误对齐来校准权重，非收益保留原预测；其MSE／同分布独立假设不能移用为本项目可靠性保证。[原文、失败获取与读取范围](../artifacts/landslide-adaptive-review-2026-10-09/literature-v1/grouped-proposal-search-v1/reading-notes.md)保存，未定位作者源码、未复现。四背景校准也未建立稳定超简单数值投票的增益，不把学习框架本身当贡献。\n")
    save_json(out / "checks.json", {"v16_original_exact": sha(out / "manuscript-v16-original.md") == sha(BASE / "manuscript-v16/manuscript-v16.md"),
                                    "current_draft_matches_v17": sha(draft) == sha(out / "manuscript-v17.md"),
                                    "grouped_validation_gate_passed": True, "complete_confirmation_gate_passed": False,
                                    "actual_costs": read_json(calibration / "analysis-v1/actual-costs.json"),
                                    "independent_replay_checks": read_json(complete / "supplement-v1/checks.json"),
                                    "cumulative_token_arithmetic": 9110583 + 595385})
    print({"version": 17, "cumulative_started": 2262, "cumulative_receipts": 2261,
           "cumulative_reported_tokens_lower_bound": 9705968, "goal": "active"}, flush=True)


if __name__ == "__main__":
    close()
