"""Bind the completed matched-count trial to a focused manuscript and durable status."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from close_landslide_coordinate_review import table
from finalize_landslide_adaptive_evidence import verify_manifest
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(run, out):
    base = run.parent
    for name in ("analysis-v1", "supplement-v1", "failure-sensitivity-v1", "figures-v1"):
        verify_manifest(run / name)
    development = base / "budget-matched-selection-development-v2"
    verify_manifest(development)
    decision = read_json(run / "analysis-v1/decision.json")
    if not decision["selection_confirmed"] or decision["scope_additional_benefit_retested"]:
        raise ValueError("This closure text describes a passed placement confirmation, not a new scope test")
    rows = read_rows(run / "analysis-v1/summary.csv")
    indexed = {row["method"]: row for row in rows}
    development_rows = read_rows(development / "summary.csv")
    costs = read_json(run / "supplement-v1/actual-costs.json")
    previous = read_json(base / "manuscript-v21/run-ledger-end.json")
    corrections = read_rows(run / "supplement-v1/corrections.csv")
    manuscript_path = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    original = manuscript_path.read_bytes()
    if sha(manuscript_path) != sha(base / "manuscript-v21/manuscript-v21.md"):
        raise ValueError("Current draft no longer matches the preserved v21")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    (out / "manuscript-v21-original.md").write_bytes(original)
    ledger = {
        "new_requests_started": costs["started"], "new_completed_local_receipts": costs["receipts"],
        "new_returned_json": costs["returned_json"], "new_task_contract_failures": len(costs["failures"]),
        "new_reported_tokens": costs["reported_tokens"],
        "cumulative_requests_started": previous["cumulative_requests_started"] + costs["started"],
        "cumulative_local_receipts": previous["cumulative_local_receipts"] + costs["receipts"],
        "cumulative_reported_tokens_lower_bound": previous["cumulative_reported_tokens_lower_bound"] + costs["reported_tokens"],
        "new_records": 12, "pure_controlled_records": 132, "observed_residual_semisynthetic_records": 60,
        "inference_retries": 0, "failures": costs["failures"], "goal_status": "active", "round_status": "progress",
        "placement_confirmation_passed": True, "scope_additional_benefit_retested": False,
        "field_accuracy_confirmed": False, "billing": "Reported usage lower bound; all historical gaps retained"}
    save_json(out / "run-ledger-end.json", ledger)
    save_json(run / "run-ledger-end.json", ledger)
    brief = """原raw21/42冲突、14日触发、最多两非重叠180日窗、49日权限、纯图提示与共同阶段头全部保持。先在已见case0109—0120上用完整缓存进行同次数消融，再在生成前冻结seed2026101075、流索引120—131的新case0121—0132，实际六族各两条、730日。K按每条记录/每遍原自适应选择的0/1/2次数确定，固定主对照取既有等间距位置中最早K，最晚K完整敏感性；两者均不读标签。同次数对照隔离观察位置，并非新的独立部署算法。整目标与冲突范围同时评价，可改写有观测日数不另强制相同；局部缺测可以不同。九数值模型沿用适用的同分布train24/validation12校准，全部重新计算预测，本批不训练或调参。

两遍×两范围四比较均满足生成前登记的F1更高、弱召回不降低、正净确定纠错。主冲突范围F1 .8474/.8794对K固定最早.7775/.7842、最晚.7443/.7913、全固定同范围.7695/.7794、直接.7200/.7708。弱R .6906/.7600对最早.6419/.6054，分母821弱日。相对最早净确定纠错+318/+297，受益4/5记录；分别增155/468TP、丢46/101TP、增5/83FP、撤219/57FP。相对直接净+625/+446，增304/471TP、丢33/96TP、增24/101FP、撤301/133FP。净确定纠错还包括正确阴性与未知转换，不等于TP减FP。

XGB84日F1 .9155、弱R .7674仍较高；raw42规则.8858、同支撑XGB42 .9025也高于主日F1。主事件F1(.3) .6667/.8000对XGB84 .7273，不能宣称两遍事件全面领先。主起/止MAE 36.82/26.55与50.58/18.00日；匹配条件变化，不能把不同事件集合的MAE单独当更好边界。主真活动内ASD .7686/.8240、端到端.7565/.8029，对XGB84 .8259/.8221；共同阶段头的输出变化来自活动门控，不称原生VLM阶段内容创新。

整目标与冲突范围本轮主TP/FP/F1相同，第一遍范围覆盖.9323→.9269，第二遍不变；旧严格范围额外条件仍未确认，本轮没有给它再过关机会。等日数居中范围F1 .8406/.8849，第二遍更高，主组不切换。数值文字.8975/.9170亦更高，但历史反馈增量不稳定，完整报告而不切换。每族仅两条，家族是描述性：再活动主F1 .8471/.9256对直接.7306/.5564；强后弱.7573/.8368对.7032/.7234；弱恒速第一遍提高(.9499对.8460)、第二遍下降(.8282对.9103)。quiet两遍FP0；仪器阶跃FP6/0对直接16/0。不能把汇总弱R提高解释为所有弱运动均改善，P227未见背景的负结果仍保留。

实际114启动/114回执/114传输JSON、448486报告token、417秒推理墙钟；两次全局字段契约失败（0124第一遍、0125第二遍）保留未知和成本，无重试、无裁剪。每遍主24/25部署调用及78678/82809token，对全固定36/36、127459/127470；与K固定调用数相同，token不要求完全相同。九数值无VLM调用，CPU/渲染未分别计时，不报零计算成本。事后逐遍去掉对应失败记录，同次数早位主F1差仍+.05838/+.05391、弱R差+.04872/+.15469，冲突范围净纠错+251/+177；该11记录敏感性不替换完整12条主分数或预登记条件。
"""
    correction_table = ["| 主组 | 对照 | 净确定纠错 | 增/丢TP | 增/撤FP | 受益记录 |",
                        "|---|---|---:|---:|---:|---:|"]
    for repeat in (0, 1):
        main = f"adaptive_no_feedback_r{repeat}__conflict_scope"
        for stem in ("direct_visual", "fixed_image_review", "same_count_fixed_early", "same_count_fixed_late"):
            control = f"{stem}_r{repeat}" + ("" if stem == "direct_visual" else "__conflict_scope")
            selected = [row for row in corrections if row["method"] == main and row["control"] == control]
            sums = {key: sum(int(row[key]) for row in selected) for key in
                    ("net_correct", "gained_tp", "lost_tp", "added_fp", "removed_fp")}
            correction_table.append(f"| r{repeat} | {stem} | {sums['net_correct']} | "
                f"{sums['gained_tp']}/{sums['lost_tp']} | {sums['added_fp']}/{sums['removed_fp']} | "
                f"{sum(int(row['net_correct']) > 0 for row in selected)}/12 |")
    development_selected = [row for row in development_rows if row["method"].startswith(
        ("adaptive_no_feedback", "same_count_fixed_early", "same_count_fixed_late"))
        and row["method"].endswith("__conflict_scope")]
    report = """# 同次数观察位置消融与生成前冻结确认

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Mode：run + descriptive validation
- Date：2026-10-10
- Status：completed；原方法位置机制确认通过，未新增模块；现场效果未确认

""" + brief + "\n## 先完成的已见开发消融\n\n"
    report += "case0109—0120原主F1 .8163/.7581，最早K固定.7238/.7204、最晚.7120/.7166；调用均26/26，净确定纠错对最早+272/+108。四登记开发比较通过，但这是看过原确认后追加的机制诊断，不能冒称独立确认。0新API，117旧payload复验、108数值记录重算、37配置/324120逐日行。\n\n" + table(development_selected)
    report += "\n\n## 新确认全部37配置\n\n" + table(rows)
    report += "\n\n## 完整逐记录损益汇总\n\n" + "\n".join(correction_table)
    report += """

![同次数观察位置对比](../artifacts/landslide-adaptive-review-2026-10-09/budget-selection-confirmation-v1/analysis-v1/comparison.png)

## 可追溯性与复现

analysis-v1核查全部生成前源快照、数值模型与冻结预测，重建原自适应及新增K控制，统一活动/事件/边界/内部及端到端ASD；supplement-v1另外检查324120逐日预测及参考导出、222家族、888事件行、24条记录/遍同次数、活动/缺测/阶段掩码和114实际payload。24完整活动/阶段图已生成，比较图、0126活动、0124阶段已目检。阶段核验器internal-stages.json仍使用历史固定scope文字“selected targets”，实际覆盖全部730日参考活动，ASD支撑462/1357/453；文字限制披露，不修改冻结核验器。

预检失败保留：budget-matched-selection-precheck-v1错误假设原分析有manifest、且未用csv；在输出分析创建前停止，API0，另存v2。budget-selection-confirmation-data-v1的count132超过schema120，在任何观测生成前失败，API0；v2将真实样本数置12，显式seed/流索引和所有门槛不变，未改生成器schema。失败版本、源码及记录不覆盖。

仅离线重放，不重发114请求；输出目录必须不存在：

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$env:PYTHONIOENCODING='utf-8'
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/analyze_landslide_budget_confirmation.py --run artifacts/landslide-adaptive-review-2026-10-09/budget-selection-confirmation-v1 --out artifacts/landslide-adaptive-review-2026-10-09/budget-selection-score-replay
```

新数据protocol/pre-generation-freeze/manifest、推理freeze/prediction-freeze、scope-v1/manifest和analysis-v1/manifest绑定实际输入、源、模型、回答与评分。原v21精确字节在manuscript-v22/manuscript-v21-original.md。未增加提示、阈值、框架或失败模块；不把原始块趋势、Agent或自适应观察本身宣称首创，技术差异以GNSS持续证据与视觉冲突的有限观察位置选择及其可检验增量为限。最近邻实际阅读/运行范围沿用正文和既有官方源码笔记；本轮未新增文献获取或作者模型推理。

该确认只支持独立随机实现的已知潜在运动恢复。每族两条、逐日相关、两遍非独立样本，不作统计显著性或真实场地结论。用户暂无领域复核者，继续自主，不再索要人员或造专家标签。goal active，正式现场可信参考仍缺。
"""
    report_path = ROOT / "docs/landslide-budget-selection-results.md"
    report_path.write_text(report, encoding="utf-8")
    (out / "budget-selection-report.md").write_text(report, encoding="utf-8")
    manuscript = original.decode("utf-8").replace("\r\n", "\n")
    manuscript = manuscript.replace("完整中文初稿v21", "完整中文初稿v22", 1)
    manuscript = manuscript.replace("新随机种子整体视觉收益复制，范围严格条件未全部复制。",
        "新随机种子整体视觉收益复制；新增同次数位置消融与生成前冻结确认支持观察位置机制，范围严格条件未全部复制。", 1)
    abstract_note = "另在生成前冻结的12条新受控记录上，同次数固定位置消融两遍F1为.7775/.7842，原方法为.8474/.8794，弱召回未下降、净确定纠错+318/+297，支持观察位置选择的独立增量；XGBoost84仍达.9155。"
    manuscript = manuscript.replace("结果支持一个场景受限的视觉识别改进", abstract_note + "结果支持一个场景受限的视觉识别改进", 1)
    manuscript = manuscript.replace("\n## 1 引言", "\nA further pre-frozen 12-record controlled trial isolates observation placement with the same review count per record and repeat: activity F1 is .8474/.8794 versus .7775/.7842 for fixed early locations, with non-decreasing weak recall. XGBoost84 remains higher at .9155. This supports controlled placement gains, not field accuracy or a unique scope benefit.\n\n## 1 引言", 1)
    design = "\n另为隔离次数与位置，先在已见0109—0120以完整缓存比较原自适应与每记录/遍同K固定位置，随后生成前冻结seed2026101075及case0121—0132六族各二。原主方法、九模型、评分器及最早K主对照/最晚K敏感性均在观测存在前冻结；新增消融不改方法、提示或门槛。这12条未用于新拟合、调参或选择主组。确认只针对位置选择，不是旧范围条件换seed救分。\n"
    manuscript = manuscript.replace("\n### 5.2 公平对比与最少消融", design + "\n### 5.2 公平对比与最少消融", 1)
    ablation = "\n同次数控制复用原固定两位置，但只取K个，K由相同全局初判和原运动证据计算。最早K为预登记主消融，最晚K为完整敏感性；不根据标签、局部回答或最后得分选顺序。整目标/冲突范围均保留，共37配置，匹配每记录/遍请求次数，非强制每图token或缺测后可修订日数一致。预登记要求两遍两范围相对最早K均F1提高、弱R不降低、净确定纠错为正；全部四比较共同判定。\n"
    manuscript = manuscript.replace("\n共同九数值控制", ablation + "\n共同九数值控制", 1)
    selected = [indexed[f"{stem}_r{repeat}" + ("" if stem == "direct_visual" else "__conflict_scope")]
                for repeat in (0, 1) for stem in
                ("direct_visual", "fixed_image_review", "same_count_fixed_early", "same_count_fixed_late", "adaptive_no_feedback")]
    selected += [indexed[name] for name in ("raw_multiscale_rule", "xgboost_42_matched_support", "xgboost_84")]
    section = "\n### 6.6 同次数观察位置机制：新冻结确认\n\n" + brief + "\n\n" + table(selected)
    section += "\n\n![同次数位置消融](../artifacts/landslide-adaptive-review-2026-10-09/budget-selection-confirmation-v1/analysis-v1/comparison.png)\n\n全部37配置、九强数值、222家族、逐记录损益、两次契约失败及重放命令见[位置机制报告](landslide-budget-selection-results.md)。已见开发消融不独立，新12只支持受控恢复；不作显著性、范围最优或现场效果声明。\n"
    manuscript = manuscript.replace("\n## 7 讨论", section + "\n## 7 讨论", 1)
    manuscript = manuscript.replace("调用少于固定观察。权限约束", "调用少于固定观察。同记录同次数固定位置消融与新冻结确认均支持位置选择的增量，因此收益不只来自复核次数变化；但第二遍弱恒速仍退化，优势有家族边界。权限约束", 1)
    old_cost = "整个研究累计2579启动/2578回执/至少11205039已报告token，其中新增坐标开发60请求/204210token；历史缺失费用保留。"
    manuscript = manuscript.replace(old_cost, "整个研究累计2693启动/2692回执/至少11653525已报告token，其中本轮新增114请求/448486token；132纯受控与60观测残差半合成分开计数，历史缺失费用保留。", 1)
    manuscript = manuscript.replace("本轮前v20精确字节另存manuscript-v21/manuscript-v20-original.md，", "本轮前v20精确字节另存manuscript-v21/manuscript-v20-original.md，v21精确字节另存manuscript-v22/manuscript-v21-original.md，", 1)
    manuscript = manuscript.replace("新随机种子两遍支持其相对直接和固定视觉的受控活动收益及调用取舍；", "新随机种子两遍支持其相对直接和固定视觉的受控活动收益及调用取舍，另一个生成前冻结批次的同次数消融确认了观察位置机制；", 1)
    manuscript = manuscript.replace("- [新种子全部29配置", "- [同次数位置消融及新冻结确认37配置](landslide-budget-selection-results.md)\n- [新种子全部29配置", 1)
    manuscript_path.write_text(manuscript, encoding="utf-8")
    (out / "manuscript-v22.md").write_text(manuscript, encoding="utf-8")
    entry = """## 2026-10-10：同次数观察位置机制新确认通过，全文v22

[位置机制结果](landslide-budget-selection-results.md)：原方法不改。已见0109—0120完整缓存同K消融API0，原主.8163/.7581对最早.7238/.7204、最晚.7120/.7166；仅开发归因。随后生成前冻结seed2026101075、显式流120—131/0121—0132、六族各二，原模型/提示/门槛及最早K主、最晚K敏感性不改。四比较(两遍×整目标/冲突范围)全通过；主F1 .8474/.8794对最早.7775/.7842、最晚.7443/.7913，弱R .6906/.7600对最早.6419/.6054，净确定纠错+318/+297。主对直接净+625/+446，但仍丢33/96TP、增24/101FP。XGB84 F1 .9155更高；位置确认不救旧范围条件，本轮范围TP/FP/F1与整目标相同。居中第二遍更高，数值文字两遍更高，均不事后换主组。第二遍弱恒速低于直接，半合成迁移失败保留。

114新启动/回执，448486token，417秒；2次全局字段契约失败保留未知，0重试。主24/25部署调用对固定36/36，与K固定相同。37配置/324120逐日行、108数值记录重算、114payload、222家族、888事件和24活动/阶段图完成；3图目检。去掉对应失败记录的事后敏感性仍有正位置差，不替换主分数。Ruff通过。v21精确原字节在manuscript-v22；累计2693启动/2692回执/至少11653525token，132受控/60半合成分开。当前全部推理与评分已结束，不重跑或改冻结源；goal active，正式可信现场参考仍缺，用户暂无人员，继续自主、不再询问或造标签。

本轮预检两个失败保留并另存v2：旧analysis没有本地manifest；count132在观测生成前超过schema120。修复只按实际绑定/真实count12读取，seed与显式流索引及条件不变。当前方法贡献聚焦原始运动冲突的有限观察位置选择；不添加已停止模块、不挑下一seed救范围、不把框架包装称创新。后续优先形成观测定义下可信参考和场景验证，或有真实新依据的小机制；无需重复本批审计/推理或低价值提示搜索。

"""
    for name in ("docs/project-status.md", "docs/landslide-research-continuation-prompt.md", "docs/landslide-submission-roadmap.md"):
        path = ROOT / name
        existing = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        first, remaining = existing.split("\n", 1)
        path.write_text(first + "\n\n" + entry + remaining.lstrip("\n"), encoding="utf-8")
    method_path = ROOT / "docs/landslide-adaptive-review-method.md"
    method_text = method_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    first, remaining = method_text.split("\n", 1)
    method_path.write_text(first + "\n\n当前[同次数位置机制新确认](landslide-budget-selection-results.md)支持原raw21/42冲突选择观察位置：同每记录/遍K主F1 .8474/.8794对固定最早.7775/.7842，四登记比较通过。保持原方法与49权限，不引入新模块；XGB84 .9155更高，范围额外条件未确认，第二遍弱恒速仍退化。完整稿v22，正式现场参考未知。\n\n" + remaining.lstrip("\n"), encoding="utf-8")
    save_json(out / "checks.json", {"selection_confirmed": True, "scope_claim_not_retested": True,
        "v21_exact_bytes": sha(out / "manuscript-v21-original.md") == sha(base / "manuscript-v21/manuscript-v21.md"),
        "new_manuscript_matches": sha(manuscript_path) == sha(out / "manuscript-v22.md"),
        "actual_costs": costs, "goal_status": "active"})
    save_json(out / "manifest.json", {"files": {str(path.relative_to(out)): sha(path)
                                               for path in out.rglob("*") if path.is_file()}})
    print(ledger, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    close(arguments.run, arguments.out)
