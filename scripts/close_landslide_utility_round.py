"""Publish actual utility results, exact previous manuscript bytes and round costs."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    return f"{float(value):.4f}" if value not in (None, "") else "—"


def comparison_table(rows):
    lines = ["| 配置 | P | R | F1 | 弱R | FP | 覆盖 | 事件F1 .3/.5 | 起/止MAE日 | 活动内/端到端ASD宏F1 | 回执请求/token |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["method"], *[number(row[key]) for key in ("precision", "recall", "f1", "weak_recall")],
                  row["fp"], number(row["coverage"]),
                  number(row["event_f1_iou03"]) + "/" + number(row["event_f1_iou05"]),
                  number(row["onset_mae_days"]) + "/" + number(row["offset_mae_days"]),
                  number(row["internal_stage_macro_f1"]) + "/" + number(row["stage_macro_f1"]),
                  row["requests"] + "/" + row["reported_tokens"]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def close():
    pilot = BASE / "review-utility-development-v1/analysis-v2"
    full = BASE / "utility-prototype-development-v1/analysis-v1"
    proposal = BASE / "proposal-utility-development-v2"
    if not read_json(pilot / "decision.json")["allows_full_prototype"]:
        raise ValueError("The pilot did not authorize the completed prototype")
    if read_json(full / "decision.json")["allows_independent_confirmation"]:
        raise ValueError("This report is for a stopped complete-prototype branch")
    if read_json(proposal / "decision.json")["allows_full_prototype"]:
        raise ValueError("This report is for a stopped proposal-acceptance probe")
    out = BASE / "manuscript-v16"
    out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    draft = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    previous_bytes = draft.read_bytes()
    if previous_bytes != (BASE / "manuscript-v15/manuscript-v15.md").read_bytes():
        raise ValueError("Current manuscript is not the delivered corrected v15")
    (out / "manuscript-v15-original.md").write_bytes(previous_bytes)
    pilot_rows, full_rows, proposal_rows = [read_rows(path) for path in
        (pilot / "summary.csv", full / "summary.csv", proposal / "supplement-v2/summary-with-events.csv")]
    indexed = [{row["method"]: row for row in rows} for rows in (pilot_rows, full_rows, proposal_rows)]
    p, f, a = indexed
    pilot_short = [p[f"{stem}_r{rep}"] for rep in (0, 1) for stem in
                   ("direct_visual", "calibrated_utility", "same_count_fixed")] + [p["xgboost_84"]]
    full_short = [f[f"{stem}_r{rep}"] for rep in (0, 1) for stem in
                  ("direct_visual", "calibrated_utility", "visual_only_utility", "same_count_fixed")]
    full_short += [f[name] for name in ("pelt_rule", "raw_multiscale_rule", "xgboost_84", "xgboost_direction_information")]
    proposal_short = [a[f"{stem}_r{rep}"] for rep in (0, 1) for stem in
                      ("direct_visual", "proposal_guard", "visual_proposal_guard", "numeric_vote_guard", "fixed_proposal_guard")]
    proposal_short += [a["xgboost_84"]]
    section = f"""### 6.20 学习实际视觉纠错收益：小规模正结果没有稳定跨背景迁移

方向约束没有新增主组活动后，研究改为检验原始冲突或数值错误先验是否真的对应可修正的视觉回答。保持原始图、image-only提示和49日权限，从全局初判及共同train24的raw84 XGBoost提取五项选窗特征：初判活动比例、初判未知比例、遗漏原始运动比例、缺原始运动支撑的初判活动比例，以及数值先验预计初判错误比例。训练目标是权限内新增正确确定二值判断减去损失正确确定判断，再除以可评价日期数；输出未知也算错误，阴性改未知不能赚取TN信用。采用StandardScaler＋Ridge(alpha=1)，不搜索正则或阈值。预测收益比例乘可改写日期数，仅选预计净收益大于零的至多两个不相交180日窗口；零表示无预计收益，不是扫出的最优门槛。视觉仍供应活动和边界，数值先验不能自行赋新活动。

共同划分的第一6条train（2001—2006）和第一6条validation（2013—2018）分别来自一个共享背景。训练记录先采全局及8个局部候选、各两遍，共108请求，96候选窗拟合后冻结模型；验证初判、全部选窗分别在验证初判请求前和局部请求前冻结，再取108验证回答。96窗包含重叠和重复，不当96独立样本。仅这6条训练标签用于纠错校准，原数值控制仍按共同train24／validation12拟合。训练数值先验包含该训练背景，是训练内预测，可能过于乐观；后续提案试验改用排除同背景的预测。初判空缺与局部失败保留未知，本轮216新请求均取得成功且满足区间契约。

{comparison_table(pilot_short)}

小规模主组日F1由直接.6887／.7293升至.7368／.7542，弱召回.5116／.4791对.5000／.4791，覆盖均.9589而直接为1。第一遍新增42TP、损失23TP、新增169FP、撤回316FP；第二遍26／23／128／202。两遍正净确定判断纠正均集中2条记录，事件IoU.3 F1为.7692／.8333对.5455／.8333。全部预登记条件通过，因此执行完整原型；XGB84日F1 .7765仍较高，不称全面超过强数值。

完整原型把上述6条训练模型逐字节原样用于24条已见半合成开发记录、4个共享背景。全部新增选窗在补充回答前保存，复用48旧全局初判及171已有局部回答，仅补72个缺失局部回答；每个方法共享同窗同遍回答。未重新训练、改门槛或按最佳消融换主方法。这24条及其标签已参与过方法分析，不能再次称未见测试或独立现场验证。

{comparison_table(full_short)}

完整主组第一遍F1 .6349低于直接.6722、同记录同个数固定.6592；弱召回.6061也低于.6178，事件F1 .6557低于.6792。新增234TP、损失183TP、新增1300FP、撤回535FP。第二遍F1 .6773高于直接.6330，弱召回.6055高于.3910，但事件F1 .6349低于.6923；新增698TP、损失77TP、新增1034FP、撤回494FP。覆盖提高至.9902／.9942并不消除误报。完整流程57／57请求对直接24／24、原冲突72／72，选择确实减少了调用，却没有稳定改善识别。两遍登记条件均失败，停止该冻结选窗版本，不进入新确认。

逐窗事后诊断进一步区分预计收益与实际纠错：第一遍背景4／5预计净纠正946.40／789.92日，实际为−116／−128；第二遍538.07／601.87对应−88／−138。背景8两遍实际+186／+243，背景9为−14／+253。这些是66个已选窗的描述性结果，非独立因果证据或显著性结论。所见偏差支持“单背景选窗效用不能直接迁移”的诊断，但不能证明只增加训练背景便必然有效，也不能把残差真实背景认作无活动现场金标准。纯视觉特征消融完整F1 .6855／.7079较高，第二遍事件F1仍下降；不以最好消融代替登记主方法。

### 6.21 接受实际局部提案的收益校准：没有优于简单数值投票的增益

另冻结一个机制差异明确的开发检验：原raw冲突仍决定至多两窗，但局部回答不是无条件覆盖初判。每窗以原49日权限内实际变化构成整项提案，提取变化比例、变化中初判未知比例、局部未知比例、原始运动一致性增量、数值预计正确率增量。未知预计正确率为0。用同96训练回答学习净确定日纠错比例，仍固定Ridge(alpha=1)、零准入；预计收益不为正时完整保留初判，不把拒绝变成未知。接受的日期必须来自视觉提案，不能由分类器直接创造活动或边界。训练先验用原raw84参数（120树、深度4、学习率.05）在排除背景0的18条共同训练记录上重拟合，验证仍用共同train24模型；没有新参数搜索。阶段仍由共同数值阶段头按最终活动重门控。

{comparison_table(proposal_short)}

主组两遍F1 .7015／.7562高于直接.6887／.7293，弱召回同直接.5000／.4791，覆盖均为1；第一遍撤回89FP也损失21TP，第二遍新增39TP、撤回21FP。但第一遍事件TP仍3、FP由2增3、FN仍3，case2017的删除将一个未匹配长事件拆成两个未匹配事件，IoU.3／.5 F1由.5455降至.5000。第二遍IoU.3 F1同为.8333，IoU.5由.6667升至.8333。主组第一遍预登记事件非损条件失败，整体停止，不进入完整扩展或独立确认。纯视觉特征接受器两遍均无改动；固定观察接受器第一遍F1 .7099及事件.8333更高。尤其是**主组两遍完整预测与不学习纠错收益、只检查数值预计正确率增量为正的投票消融完全相同**，没有建立学习接受机制相对这一简单控制的增益，不能把数值投票收益包装为新Agent贡献。每遍18个已有回执、63551／63626token，所有被拒绝的复核也计成本，无调用节省。

### 6.22 成本、复现及方法选择边界

本轮实际新增288启动／288回执／288成功模型JSON、1133453已报告token，其中小规模216／840599（108训练、108验证，后者96为验证反事实取证）和完整补充72／292854。主组15／13及57／57是共享回执计算的流程成本，不是本轮实际总请求，不能跨消融相加。提案开发及诊断新增API为0，但复用216采集成本和额外数值拟合CPU不能称零成本。累计2097启动／2096回执／至少9110583已报告token；历史一条中断缺口和其他无usage失败不填零费用。新记录0，108纯受控与60观测残差半合成分别计数。

小规模23配置、72权限核验、96训练效用／96验证候选、216payload及100740逐日行；完整28配置、288权限与288数值预测重算、315旧payload／72新payload、280家族／背景分组、490560逐日行和24全图；提案25配置、96审批重放、12数值概率重算、2模型精确重拟合、50事件计数和109500逐日行、6全图均保存。效用评分v1因误读局部request_id失败，v2只修回执索引；提案v1预测保存后因未筛选标签CSV失败，v2只筛选共同验证记录，数值模型、概率、纠错模型、训练效用、预测、提案与权限7项与失败v1逐字节相同。首次pytest遗漏PYTHONPATH导致导入失败，补齐项目环境后2新＋4相关业务测试通过，最终Ruff检查结果另存。没有重跑旧API、覆盖模型或修改冻结结果。

ARTIST同commit2bbc28a86f346efb7311ddc227f8952c3aa8b21e源码controller_reasoner_rewards.py前188行已实际核查：ControllerReward将reasoner consistency_score乘权重，扣轮次代价，加入片段探索／格式奖励；因此按下游效果学习有限片段选择已有先例。配置rl_tsqa.yaml因TLS EOF没有取得，不能声称读取运行权重，也没有运行作者模型。本文可能的具体差异只剩GNSS活动／阶段两层语义下的未知计损、实际确定日纠错及原始权限约束，尚不足以凭当前开发结果证明新校准机制的贡献。最近邻阅读原字节、URL、hash与读取范围保留。

当前两项校准尝试都停止，不在这批重扫Ridge、收益阈值或改用最高消融。后续若检验跨背景校准，必须另登记共同train24的背景外预测、仅采部署实际选中的有限提案，先证明相对简单数值投票的增益；不把新增样本数本身当创新。原冲突范围方法的受控确认与强基线继续保留，正式现场总活动／阶段效果仍因缺独立领域参考而未知。用户已授权继续自主研究，无需重复索要复核者。

![小规模与完整开发的纠错收益](../artifacts/landslide-adaptive-review-2026-10-09/manuscript-v16/review-utility-comparison.png)

"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for axis, rows, stems, title, metric in (
        (axes[0], p, ["direct_visual", "calibrated_utility", "same_count_fixed"], "Six-record validation | F1", "f1"),
        (axes[1], f, ["direct_visual", "calibrated_utility", "visual_only_utility", "same_count_fixed"], "24 exposed records | F1", "f1"),
        (axes[2], a, ["direct_visual", "proposal_guard", "numeric_vote_guard", "fixed_proposal_guard"], "Proposal probe | event F1", "event_f1_iou03"),
    ):
        for position, stem in enumerate(stems):
            axis.plot([position - .1, position + .1], [float(rows[f"{stem}_r{rep}"][metric]) for rep in (0, 1)],
                      marker="o", linewidth=1)
        axis.set_xticks(range(len(stems)), [stem.replace("_", "\n") for stem in stems], fontsize=8)
        axis.set_ylim(.45, .9)
        axis.set_title(title, fontsize=11)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Development only | paired repeats shown separately | both calibrated mechanisms stopped")
    fig.tight_layout()
    fig.savefig(out / "review-utility-comparison.png", dpi=150)
    plt.close(fig)
    report = "# 实际视觉纠错收益研究：小规模有效，完整迁移和机制增益未成立\n\n" + section
    report += "\n## 全部登记比较\n\n### 六条验证记录\n\n" + comparison_table(pilot_rows)
    report += "\n\n### 完整24条开发记录\n\n" + comparison_table(full_rows)
    report += "\n\n### 提案接受开发\n\n" + comparison_table(proposal_rows)
    report += """

## 复现

原推理已完成，不重跑请求。以下仅重算，输出目录必须不存在；完整数据、冻结模型与实际回执保留在对应版本目录。

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$env:PYTHONIOENCODING='utf-8'
$experimentPython='artifacts/reproduction-2026-10-07/environment/Scripts/python.exe'
& $experimentPython scripts/analyze_landslide_review_utility_v2.py --run artifacts/landslide-adaptive-review-2026-10-09/review-utility-development-v1 --out artifacts/landslide-adaptive-review-2026-10-09/review-utility-score-replay
& $experimentPython scripts/analyze_landslide_utility_prototype.py --run artifacts/landslide-adaptive-review-2026-10-09/utility-prototype-development-v1 --out artifacts/landslide-adaptive-review-2026-10-09/utility-prototype-score-replay
& $experimentPython -m pytest tests/test_landslide_review_utility.py tests/test_landslide_proposal_utility.py -q
```

提案有效版本proposal-utility-development-v2；supplement-v2给出完整IoU.5、逐日数据及独立重放。diagnostics-v1保存66个选窗预计／实际效用。失败analysis-v1及proposal-utility-development-v1保留，不覆盖。
"""
    (ROOT / "docs/landslide-review-utility-results.md").write_text(report, encoding="utf-8")
    (out / "utility-report.md").write_text(report, encoding="utf-8")
    text = previous_bytes.decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"> 完整中文初稿v15[^\n]*", "> 完整中文初稿v16，2026-10-09。新增实际视觉纠错收益训练、完整24条迁移及提案接受检验；两个新机制均未通过继续条件，完整成本、消融和失败保留。独立领域参考仍缺，本稿不是投稿定稿。", text, count=1)
    if text.count("## 7 讨论") != 1:
        raise ValueError("Manuscript section insertion point is ambiguous")
    text = text.replace("## 7 讨论", section + "## 7 讨论", 1)
    discussion = "最新校准选窗在六条验证记录上两遍提高日F1，完整跨背景却不稳定；实际提案接受开发日F1提高但第一遍事件下降，且两遍完全等同简单数值投票。证据不支持把局部收益、覆盖增加或单背景监督包装为新的稳定Agent贡献。停止这两个冻结版本，保留成本与损失，不按最高消融替换主方法。"
    text = text.replace("## 8 可复现性、完整性与AI使用说明", discussion + "\n\n## 8 可复现性、完整性与AI使用说明", 1)
    text = text.replace("| 当前全文v15、精确v14原字节与方向比较图 | manuscript-v15 |", "| 历史全文v15、精确v14原字节与方向比较图 | manuscript-v15 |\n| 实际纠错收益训练与完整开发 | review-utility-development-v1/analysis-v2；utility-prototype-development-v1/analysis-v1 |\n| 提案接受有效评分及逐日／事件重放 | proposal-utility-development-v2；supplement-v2 |\n| 当前全文v16与精确v15原字节 | manuscript-v16 |")
    text += "\n最新稿件状态：6.20—6.22已纳入小规模通过、完整失败、提案与数值投票等价及真实新增288请求成本。当前累计2097启动／2096回执／至少9110583token，目标仍active；两个校准机制均停止，不升级、不声称现场准确率。\n"
    draft.write_text(text, encoding="utf-8")
    (out / "manuscript-v16.md").write_text(text, encoding="utf-8")
    ledger = read_json(BASE / "manuscript-v15/run-ledger-end.json")
    ledger.update({"new_requests_started": 288, "new_completed_local_receipts": 288,
                   "new_valid_model_json": 288, "new_reported_tokens": 1133453,
                   "cumulative_requests_started": 2097, "cumulative_local_receipts": 2096,
                   "cumulative_reported_tokens_lower_bound": 9110583, "new_records": 0,
                   "pilot_collection_requests": 216, "pilot_reported_tokens": 840599,
                   "complete_missing_view_requests": 72, "complete_reported_tokens": 292854,
                   "proposal_probe_new_requests": 0, "goal_status": "active", "round_status": "progress"})
    save_json(out / "run-ledger-end.json", ledger)
    note = """# ARTIST奖励源码补读：收益驱动片段选择已有先例

2026-10-09，固定作者commit `2bbc28a86f346efb7311ddc227f8952c3aa8b21e`。已取得并实际读取`artist/controller_reasoner_rewards.py`第1—188行，包含ControllerReward参数、调用、`_reward_one`及片段探索奖励的开始。原字节为本目录`artist__controller_reasoner_rewards.py`，hash `fd49095cb37726adf1bbdf20034459c6223f14828879f21050e2899eb2e4a8fd`；URL／状态见retrieval.json。

源码将metadata的consistency_score乘权重作为奖励，减去轮次惩罚，加入片段探索／格式奖励。以reasoner表现学习有限片段选择已有直接源码先例，不能宣称收益学习、有限复核或controller选择是本文首创。本文候选的区别仅在GNSS两层活动语义、未知计损、实际确定日净纠错和原始49日权限，必须有超过简单控制的识别效果支撑。

配置`configs/rl_tsqa.yaml`请求因SSLEOFError失败，没有取得配置，不声称实际运行权重。未读完整奖励实现、训练器或运行作者模型。构造参数的默认值不能当作作者全部实验实际设置。当前两项校准开发均停止，提案主组与简单数值投票完全相同，没有建立其独立机制增益。
"""
    (BASE / "literature-v1/review-utility-code-reading-v1/reading-notes.md").write_text(note, encoding="utf-8")
    status = """## 2026-10-09：实际视觉纠错收益研究完成，两个校准机制停止，全文v16

[纠错收益结果](landslide-review-utility-results.md)：6条共同train、6条validation各一个背景，216真实请求／840599token；固定Ridge(alpha1)、零收益界限、原49权限及至多两窗。主组小规模F1 .7368/.7542对直接.6887/.7293，全部登记条件通过。保持六条校准模型原字节扩展完整24已见背景记录两遍，复用171局部回执、仅补72请求／292854token；主F1 .6349/.6773对直接.6722/.6330，两遍事件均下降，第一遍新增1300FP、撤回535FP，完整条件失败，停止。原共同数值／学习强控制12项保留，XGB84 .7261、方向XGB .7350，不按纯视觉消融较高替换主组。

另冻结实际提案接受开发：仍原raw选窗，未知计损、拒绝保留初判；训练先验排除背景0、用另18共同train重拟合原raw84参数。复用96训练局部回答，无新API；主F1 .7015/.7562、弱R同直接、覆盖1，但第一遍case2017事件拆分使事件F1 .5455→.5，停止。两遍全部预测均与不学习收益的数值投票消融完全相同，不能宣称学习接受机制有增益。有效proposal-utility-development-v2，v1评分标签未筛选失败保留，7项模型／预测与v1字节一致。

小规模23配置／216payload／100740逐日行；完整28配置／288权限与288数值重算／315旧＋72新payload／280分组／490560逐日行／24全图；提案25配置、96审批／12概率／2模型重放、50事件和109500逐日行、6图保存。2新＋4相关测试通过，首次遗漏PYTHONPATH的导入失败保留；最终Ruff通过。ARTIST同commit奖励前188行实读，收益驱动选段已有先例；配置TLS EOF未读、未执行作者模型。v16及精确更正v15原字节保存。

本轮288启动／288回执／1133453token，累计2097启动／2096回执／至少9110583已报告token，历史缺口不填零费用；新增记录0，108纯受控与60半合成分开。选窗15/13及57/57是共享回执流程成本，不是实际总请求，训练／反事实采集单列。全部进程已结束，旧冻结源、模型、预测和失败版本不改、不重跑旧请求。用户暂无独立复核者，继续自主研究、不重复询问或造标签；goal active，本轮progress。

后续若检验跨背景校准，另登记共同train24的背景外预测、仅采部署实际选中提案，先验证相对简单数值投票的增益；不在当前批扫Ridge／收益阈值，不把增加训练样本当创新。若无额外可分辨信息，停止学习校准分支，把研究重点回到视觉观察本身。原范围方法的受控确认与全部强基线继续保留，现场总运动效果仍未知。

"""
    for name in ("project-status.md", "landslide-submission-roadmap.md", "landslide-research-continuation-prompt.md"):
        path = ROOT / "docs" / name
        content = path.read_text(encoding="utf-8")
        first_line, rest = content.split("\n", 1)
        path.write_text(first_line + "\n\n" + status + rest.lstrip("\n"), encoding="utf-8")
    readme = ROOT / "README.md"
    content = readme.read_text(encoding="utf-8")
    needle = "- [历史方向约束与前缀锚点]"
    addition = "- [实际视觉纠错收益训练与完整迁移](docs/landslide-review-utility-results.md)：小规模通过、完整24条未稳定提高；提案接受与简单数值投票等价，均停止。288新请求、完整消融和论文v16。\n"
    content = content.replace(needle, addition + needle, 1)
    readme.write_text(content, encoding="utf-8")
    nearest = ROOT / "docs/landslide-nearest-method-code-review.md"
    with nearest.open("a", encoding="utf-8") as handle:
        handle.write("\n\n## 2026-10-09：ARTIST奖励实现与实际纠错收益校准\n\n" +
                     "同commit补读controller_reasoner_rewards.py前188行：consistency_score奖励、轮次惩罚、片段探索及格式奖励已实现，按下游效果学习片段选择已有先例；配置TLS EOF未读，未执行作者模型。[读取范围](../artifacts/landslide-adaptive-review-2026-10-09/literature-v1/review-utility-code-reading-v1/reading-notes.md)与原字节保留。新的[纠错收益开发](landslide-review-utility-results.md)小规模选窗通过但完整迁移失败；提案接受两遍与简单数值投票完全相同，不能以收益学习或数值投票包装Agent贡献。\n")
    save_json(out / "checks.json", {"previous_v15_snapshot_exact": sha(out / "manuscript-v15-original.md") == sha(BASE / "manuscript-v15/manuscript-v15.md"),
                                    "current_draft_equals_v16": sha(draft) == sha(out / "manuscript-v16.md"),
                                    "utility_complete_gate": False, "proposal_complete_gate": False,
                                    "round_costs_from_receipts": read_json(full / "actual-costs.json"),
                                    "cumulative_token_arithmetic": 7977130 + 840599 + 292854})
    print({"manuscript": str(draft), "version": 16, "ledger": ledger}, flush=True)


if __name__ == "__main__":
    close()
