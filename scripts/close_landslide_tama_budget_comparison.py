"""Report nearest-design tradeoffs and preserve a complete manuscript with unchanged primary evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from close_landslide_budget_confirmation import read_rows
from close_landslide_coordinate_review import table
from finalize_landslide_adaptive_evidence import verify_manifest
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import read_json


def close(run, out):
    base = run.parent
    analysis = run / "analysis-v3"
    for folder in (analysis, run / "analysis-v2", run / "figures-v1",
                   base / "tama-budget-comparison-registration-v2", base / "tama-budget-comparison-registration-v3"):
        verify_manifest(folder)
    for name in ("predictions.json", "results.json", "summary.csv", "daily.csv", "family-results.csv", "corrections.csv"):
        if (analysis / name).read_bytes() != (run / "analysis-v2" / name).read_bytes():
            raise ValueError("Documentation-binding correction changed actual predictions or scores")
    costs = read_json(analysis / "actual-costs.json")
    rows = read_rows(analysis / "summary.csv")
    indexed = {row["method"]: row for row in rows}
    doc = ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"
    original = doc.read_bytes()
    if sha(doc) != sha(base / "manuscript-v22/manuscript-v22.md"):
        raise ValueError("Current complete manuscript must match the exact previous version")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    (out / "manuscript-v22-original.md").write_bytes(original)
    previous = read_json(base / "manuscript-v22/run-ledger-end.json")
    ledger = {"new_requests_started": costs["started"], "new_completed_local_receipts": costs["receipts"],
        "new_returned_json": costs["returned_json"], "new_reported_tokens": costs["reported_tokens"],
        "cumulative_requests_started": previous["cumulative_requests_started"] + costs["started"],
        "cumulative_local_receipts": previous["cumulative_local_receipts"] + costs["receipts"],
        "cumulative_reported_tokens_lower_bound": previous["cumulative_reported_tokens_lower_bound"] + costs["reported_tokens"],
        "failures": costs["failures"], "inference_retries": 0, "new_records": 0,
        "pure_controlled_records": 132, "observed_residual_semisynthetic_records": 60,
        "round_status": "progress", "goal_status": "active", "field_accuracy_confirmed": False,
        "comparison": "Same-data unchanged TAMA-inspired GNSS adaptation, appended after primary scores",
        "billing": "Reported usage lower bound; historical missing fees retained"}
    save_json(out / "run-ledger-end.json", ledger)
    save_json(run / "run-ledger-end.json", ledger)
    context = """本轮不新增方法或记录，原raw21/42选窗、49权限、数值文字消融及九数值基线不改。在已完成位置确认的case0121—0132、seed2026101075、六族各二的同12条记录上，执行以前已使用的TAMA GNSS任务适配，源码与0061—0072适配完全相同，未按当前标签调整提示、窗口、阶段头或控制器。当前人口已见，比较在原37配置评分后追加；不冒称另一项独立方法确认或原论文发布效果。

共同训练集0001/0007/0013三静稳参考，每遍先提正常描述一次，完整检测时保留三参考图与全局测试图；仅成功非空活动初判触发一次首尾各2/3视窗复核，并提供原答案。两遍都保留无复核与阳性复核；共同Qwen参数、严格GNSS活动/未知、原始三分量画法、共同学习阶段头、无重试/修复/point adjustment与按同分布共享正常描述均为公开适配差异。正常参考不含测试记录，原数据/缺测/辅助特征和阶段头逐字节一致。

TAMA适配复核F1 .7741/.7708、弱R均.3873；主方法.8474/.8794、弱R .6906/.7600。其复核相对自己的正常参照初判F1 .6866/.7165提高，主要撤809/408FP，但也丢145/78TP，仅增7/10TP。主相对TAMA复核增360/609TP、丢36/99TP，同时增166/184FP、撤59/1FP，不能称无损。主FP201/233对TAMA94/50，Precision .9004/.8940低于.9408/.9668，Recall .8002/.8653高于.6576/.6408。

主正确确定决策相对TAMA复核净−200/+359（各6/12记录净受益），覆盖.9269/.9493低于.9981/.9549；F1和弱召回增益不等于更高整体确定可靠性。主第一遍有全局契约失败，TAMA全44传输/契约成功，失败不从主人口删掉，不将两方法不同初判视为共享初判的单因素试验。位置机制仍由原同K纯视觉消融支持；这个最近邻比较评价完整工作流，正常条件化、观察跨度与控制器同时不同，不能将差异全部归因于阳性触发。

TAMA复核事件F1(.3) .7407/.8000对主.6667/.8000，IoU.5 .5926/.6400对主.5455/.7333；第一遍事件较高、第二遍.3相同。TAMA起/止MAE46.4/20.9与50.4/20.9日，主36.82/26.55与50.58/18.00，边界各有取舍且均为匹配条件指标。TAMA内部ASD .6815/.6753、端到端.6748/.6696，主.7686/.8240及.7565/.8029，来自不同活动门控，非VLM阶段内容创新。XGB84日F1 .9155、弱R .7674仍较高，原九强数值全部保留并未新训练或调参。

家族只作描述：TAMA强后弱F1 .5918/.5247、弱尾召回两遍0，主.7573/.8368、弱尾召回.4173/.7136；再活动TAMA .8491/.8591，主.8471/.9256，第一遍主略低。TAMA弱恒速两遍F1 .8185、弱R .7644；主.9499/.8282及.9567/.8053。但主第二遍弱恒速仍低于直接.9103，不能借较弱最近邻掩盖此退化。quiet两方法均FP0；阶跃TAMA16/0、主6/0，未证明普遍背景区分。

每遍主24/25调用、78678/82809报告token；TAMA复核23/21调用、236718/212790token，含正常描述各一次和10/8复核。主调用数更多，报告token少约66.8%/61.1%；不把token等同账单金额或能耗。TAMA无复核13/13调用、116097/116209token。新实际44启动/44回执/44合法任务JSON，449508token，115.094秒推理墙钟；不同批次网络时延不可用来宣称CPU/延迟优势。无失败/重试，新记录0，累计2737启动/2736回执/至少12103033报告token；132受控/60半合成分别计数，历史缺口保留。
"""
    report = """# 同记录TAMA设计适配：弱活动收益与可靠性取舍

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent
- Mode：run + descriptive validation
- Date：2026-10-10
- Status：completed；最近邻任务适配补充，原主方法不改；正式现场效果未确认

""" + context
    report += "\n## 同任务关键比较\n\n"
    selected = [indexed[name] for repeat in (0, 1) for name in
        (f"direct_visual_r{repeat}", f"tama_gnss_no_review_r{repeat}", f"tama_gnss_positive_review_r{repeat}",
         f"same_count_fixed_early_r{repeat}__conflict_scope", f"adaptive_no_feedback_r{repeat}__conflict_scope")]
    selected += [indexed[name] for name in ("raw_multiscale_rule", "xgboost_42_matched_support", "xgboost_84")]
    report += table(selected)
    report += "\n\n## 全部41配置\n\n" + table(rows)
    report += """

![同记录最近邻比较](../artifacts/landslide-adaptive-review-2026-10-09/tama-budget-comparison-v1/analysis-v3/comparison.png)

## 源码、复现及限制

本轮重新核查官方默认分支最新commit：TAMA db99a75a83d0ef05caf7ff0d199b75b594e98584（2024-11-06）、AnomaMind bf2c0cb3aaeb5a47aa7acabd1714ce123ddcdaa1（2026-06-16），与此前读取源码一致。这仅是代码版本查询，不能声称系统综述更新或作者模型复现。TAMA首次httpx TLS EOF失败、一次urllib文献获取重试成功；失败metadata和原响应保留在literature-v1/tama-budget-codecheck-v1，模型请求无重试。

registration-v1保留评分器I001导入排序预检失败，输入准备实际完成、推理尚未开始；v2只拆分导入并在第一请求前冻结，行为未变。推理完成且v2评分成功后，发现注册将可编辑当前稿当成永久评分依赖，会使后续写稿破坏重放；v3仅把该绑定指向已存在的精确manuscript-v22/manuscript-v22.md，原v2注册/源码/评分不改。相同v2评分器在v3注册下重放，所有41预测、分数、逐日、家族和纠错表与v2精确字节相同；该文档归属修复不是方法/评价变更，也不是新的独立证据。

analysis-v3保留原37预测与评分完全不变，并从实际44payload重建四适配组/48记录预测与24阳性触发决定，独立sklearn核验活动/内部及端到端阶段；统一41配置、359160逐日行、246家族及全部事件/边界保存。24活动/阶段图已生成；原始训练参考、比较图和强后弱活动图已目检。阶段核验器scope文字仍为历史“selected targets”，实际为全部730日参考活动；ASD支撑462/1357/453，文字限制披露，不改冻结源。

仅离线重放评分；指定新输出目录，不重发44请求：

```powershell
$env:PYTHONPATH='D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/src;D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/scripts'
$env:PYTHONIOENCODING='utf-8'
& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/analyze_landslide_tama_budget_comparison_v2.py --run artifacts/landslide-adaptive-review-2026-10-09/budget-selection-confirmation-v1 --tama artifacts/landslide-adaptive-review-2026-10-09/tama-budget-comparison-v1 --registration artifacts/landslide-adaptive-review-2026-10-09/tama-budget-comparison-registration-v3 --out artifacts/landslide-adaptive-review-2026-10-09/tama-budget-score-replay
```

无权重版本或作者原始模型效果复现承诺，适配成绩不填写为原论文完整TAMA。新增比较在主成绩已见后追加，方法沿用之前固定源码；每家族两条、逐日相关，两遍非独立，不作显著性或普遍可靠性保证。旧范围严格条件仍未确认，P227半合成迁移负结果和主第二遍弱恒速退化仍保留。领域复核者暂缺，正式现场可信参考仍未知；用户授权继续自主，不再索要人员或造专家标签，goal active。
"""
    report_path = ROOT / "docs/landslide-tama-budget-comparison-results.md"
    report_path.write_text(report, encoding="utf-8")
    (out / "tama-budget-report.md").write_text(report, encoding="utf-8")
    manuscript = original.decode("utf-8").replace("\r\n", "\n")
    manuscript = manuscript.replace("完整中文初稿v22", "完整中文初稿v23", 1)
    manuscript = manuscript.replace("受控机制参考与现场可观测参考分开，", "同记录补齐TAMA设计适配比较，报告误报/未知与弱运动取舍；受控机制参考与现场可观测参考分开，", 1)
    abstract = "同记录固定TAMA设计的GNSS适配复核F1 .7741/.7708、弱召回.3873，低于主方法但误报更少；主调用更多而报告token较少，正确确定决策第一遍更少。"
    manuscript = manuscript.replace("结果支持一个场景受限的视觉识别改进", abstract + "结果支持一个场景受限的视觉识别改进", 1)
    manuscript = manuscript.replace("This supports controlled placement gains, not field accuracy or a unique scope benefit.",
        "On the same records, the unchanged TAMA-inspired GNSS adaptation reaches .7741/.7708 with review and weak recall .3873. It produces fewer false positives and higher coverage; the proposed method uses more calls but fewer reported tokens. Its net definite-correction difference is negative in the first repetition. These comparisons support controlled placement gains and disclose reliability tradeoffs, without establishing field accuracy or a unique scope benefit.", 1)
    manuscript = manuscript.replace("本新种子批次未追加TAMA请求。", "在位置确认批0121—0132完成原评分后，本稿另追加相同固定适配、同数据两遍比较，不把它称另一独立方法确认。", 1)
    section = "\n### 6.7 同记录最近邻设计适配\n\n" + context
    section += "\n\n" + table([indexed[f"tama_gnss_{stem}_r{repeat}"] for repeat in (0, 1) for stem in
        ("no_review", "positive_review")])
    section += "\n\n完整41配置、全部家族/损益/成本及[任务适配与可复现说明](landslide-tama-budget-comparison-results.md)保留。该比较不是共享初判的单因素机制试验，位置作用仍以同K消融解释；不作显著性或现场结论。\n"
    manuscript = manuscript.replace("\n## 7 讨论", section + "\n## 7 讨论", 1)
    manuscript = manuscript.replace("整个研究累计2693启动/2692回执/至少11653525已报告token，其中本轮新增114请求/448486token；",
        "整个研究累计2737启动/2736回执/至少12103033已报告token，新增最近邻适配44请求/449508token；", 1)
    manuscript = manuscript.replace("v21精确字节另存manuscript-v22/manuscript-v21-original.md，", "v21精确字节另存manuscript-v22/manuscript-v21-original.md，v22精确字节另存manuscript-v23/manuscript-v22-original.md，", 1)
    manuscript = manuscript.replace("强数值日F1仍更高，范围严格条件未全部复制，", "强数值日F1仍更高，最近邻TAMA适配误报/覆盖具有优势、主确定可靠性并非两遍更高，范围严格条件未全部复制，", 1)
    manuscript = manuscript.replace("- [同次数位置消融", "- [同记录最近邻任务适配与41配置](landslide-tama-budget-comparison-results.md)\n- [同次数位置消融", 1)
    doc.write_text(manuscript, encoding="utf-8")
    (out / "manuscript-v23.md").write_text(manuscript, encoding="utf-8")
    literature = base / "literature-v1/tama-budget-codecheck-v1"
    notes = """# 2026-10-10 最近邻当前代码版本核查

官方GitHub默认分支最新提交：TAMA db99a75a83d0ef05caf7ff0d199b75b594e98584（2024-11-06），AnomaMind bf2c0cb3aaeb5a47aa7acabd1714ce123ddcdaa1（2026-06-16），均与此前实际读取源码相同。本次仅查询版本元数据，不新增论文全文阅读或作者模型执行，不称系统综述更新。

TAMA首个httpx请求TLS EOF失败，另一次urllib获取成功，原失败metadata、成功响应/sha/date均保留；模型推理无重试。当前固定TAMA GNSS设计适配沿用0061—0072研究完整原源码，实际在0121—0132追加两遍44新请求/449508token，不是作者完整发布模型。F1/弱召回低于主方法，但误报/覆盖更好、事件第一遍更高，主净正确确定决策−200/+359。细节及适配边界见docs/landslide-tama-budget-comparison-results.md。
"""
    (literature / "reading-notes.md").write_text(notes, encoding="utf-8")
    save_json(literature / "manifest.json", {"files": {path.name: sha(path)
                                                      for path in literature.iterdir() if path.is_file()}})
    entry = """## 2026-10-10：同记录TAMA设计适配完成，全文v23

[最近邻统一比较](landslide-tama-budget-comparison-results.md)：在已评分0121—0132追加以前冻结的TAMA GNSS适配原源码，两遍44真实请求/449508token，正常train0001/0007/0013，阳性触发首尾2/3一次复核，无失败/重试。不新增人口/改方法/选参；本批已见，不称新独立确认或原完整TAMA。复核F1 .7741/.7708、弱R均.3873对主.8474/.8794、.6906/.7600，但FP94/50少于主201/233、覆盖更高、事件.3第一遍更高第二遍相同。主相对复核净正确确定决策−200/+359，不能将F1优势写成可靠性全面提高。主24/25调用多于适配23/21，报告token78678/82809少于236718/212790，不等同能耗或金额。

统一41配置，原37预测和分数完全不变；四适配48记录预测/24触发重建、44payload、359160逐日行、246家族与完整阶段/事件/边界保留，24活动/阶段图生成。原v22精确字节在manuscript-v23。累计2737启动/2736回执/至少12103033token，132受控/60半合成不变。Ruff通过；原I001评分器保留v1，第一请求前仅修导入为v2。评分后登记文档归属修复v3，将可编辑当前稿改绑已经存在的精确v22快照，同v2评分器重放所有关键导出原字节一致，非方法或参考修改。官方当前TAMA/AnomaMindcommit与已读源码一致，TAMA文献TLS失败/重试留存，未跑AnomaMind作者权重。

本轮全部推理/评分结束，不重发44请求、不改冻结源/预测、不换方法或救旧范围条件。位置贡献由原同K新冻结确认支持；最近邻完整工作流包含多因素差异。领域参考仍缺，用户暂无人员，继续自主、不再询问或造专家参考；goal active，本轮progress。

"""
    for name in ("docs/project-status.md", "docs/landslide-research-continuation-prompt.md", "docs/landslide-submission-roadmap.md"):
        path = ROOT / name
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        first, remaining = text.split("\n", 1)
        path.write_text(first + "\n\n" + entry + remaining.lstrip("\n"), encoding="utf-8")
    nearest = ROOT / "docs/landslide-nearest-method-code-review.md"
    with nearest.open("a", encoding="utf-8") as handle:
        handle.write("\n\n## 2026-10-10：同记录最近邻适配补齐\n\n官方最新TAMA/AnomaMindcommit与已读版本一致。原GNSS适配同源码在0121—0132追加44请求两遍，已统一41配置；复核F1 .7741/.7708低于主但FP更少、覆盖更高，主净确定纠错−200/+359。主调用更多、报告token更少，不能称整体可靠性或能耗优势。具体适配、追加时间、版本修复及[完整结果](landslide-tama-budget-comparison-results.md)保留，未复现作者发布模型。\n")
    method = ROOT / "docs/landslide-adaptive-review-method.md"
    with method.open("a", encoding="utf-8") as handle:
        handle.write("\n\n当前同记录TAMA任务适配已补齐，见[41配置结果](landslide-tama-budget-comparison-results.md)。主F1/弱R较高、FP和弃判更多，净确定决策第一遍较少；原方法不变，不将完整工作流差异当单因素触发证明。完整稿v23，范围严格条件和现场结论仍未确认。\n")
    save_json(out / "checks.json", {"v22_exact_bytes": sha(out / "manuscript-v22-original.md") == sha(base / "manuscript-v22/manuscript-v22.md"),
        "current_manuscript_matches": sha(doc) == sha(out / "manuscript-v23.md"),
        "v2_v3_all_scores_and_predictions_identical": True, "original_37_scores_unchanged": True,
        "new_records": 0, "goal_status": "active"})
    save_json(out / "manifest.json", {"files": {path.name: sha(path) for path in out.iterdir() if path.is_file()}})
    print(ledger, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    close(arguments.run, arguments.out)
