"""Save complete local-observation results and update the full manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_diversity_local_review import NUMERIC
from run_landslide_window_optimization import load_case_features, read_json
from write_landslide_distribution_manuscript import score_table

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import runs


def report(run, verified, out):
    verify_manifest(run, "prediction-freeze.json")
    verify_manifest(run / "analysis-v1", "freeze.json")
    verify_manifest(verified, "freeze.json")
    receipt = read_json(verified / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["analysis_manifest_sha256"] != sha(run / "analysis-v1/freeze.json"):
        raise ValueError("A matching completed replay is required")
    out.mkdir(parents=True, exist_ok=False)
    protocol = read_json(run / "protocol.json")
    summary = read_json(run / "analysis-v1/summary.json")
    predictions = read_json(run / "predictions.json")
    prior = read_json(run / "numeric-initial.json")
    cases, detailed, traces = [], {}, {}
    for rep in range(2):
        method = f"adaptive_numeric_r{rep}"
        for cid in protocol["ids"]:
            with np.load(Path(protocol["encoded"]) / "validation" / (cid + ".npz")) as bank:
                actual = bank["activity"]
            before = np.array(prior[cid]["activity"])
            after = np.array(predictions[method][cid]["activity"])
            corrected = (before == 1) & (after != 1) & (actual == 0)
            lost = (before == 1) & (after != 1) & (actual == 1)
            regained = (before != 1) & (after == 1) & (actual == 1)
            added = (before != 1) & (after == 1) & (actual == 0)
            row = dict(method=method, case_id=cid, removed_fp=int(corrected.sum()), lost_tp=int(lost.sum()),
                       gained_tp=int(regained.sum()), added_fp=int(added.sum()),
                       net_correct=int(corrected.sum() + regained.sum() - lost.sum() - added.sum()))
            cases.append(row)
            traces[method + ":" + cid] = dict(
                removed_fp=runs(corrected), lost_tp=runs(lost), gained_tp=runs(regained), added_fp=runs(added),
                lost_to_unknown=int((lost & (after == -1)).sum()), lost_to_negative=int((lost & (after == 0)).sum()),
            )
        detailed[method] = {field: sum(row[field] for row in cases if row["method"] == method)
                            for field in ("removed_fp", "lost_tp", "gained_tp", "added_fp", "net_correct")}
    write_csv(out / "numeric-paired-errors.csv", cases)
    write_json(out / "exact-error-traces.json", dict(interval_convention="Original0-based half-open", traces=traces))
    write_json(out / "correction-summary.json", detailed)
    methods = (
        NUMERIC, "direct_visual_r0", "fixed_visual_r0", "adaptive_visual_r0",
        "fixed_numeric_r0", "adaptive_numeric_r0", "direct_visual_r1", "fixed_visual_r1",
        "adaptive_visual_r1", "fixed_numeric_r1", "adaptive_numeric_r1",
        "continuity_pool_numeric", "global_numeric", "xgb_raw84", "robust_rule", "pelt_rule",
    )
    table = score_table(summary, methods)
    rows = {row["method"]: row for row in summary}
    fig, panels = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
    for panel, metric in zip(panels, ("f1", "event_f1_iou03")):
        panel.barh(methods, [rows[name][metric] for name in methods],
                   color=["#aa5145" if name.startswith("adaptive_") else "#557e99" for name in methods])
        panel.set_xlim(0, 1)
        panel.set_title(metric)
    panels[0].invert_yaxis()
    fig.suptitle("Existing bounded VLM review:16TRAIN/8exposedVALIDATION;two repeats retained")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=140)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    cid = "case_0389"
    case, _ = load_case_features(Path(protocol["data"]), cid)
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    fig, panels = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for axis in range(3):
        panels[axis].plot(values[:, axis], linewidth=.8)
        panels[axis].set_ylabel(("N", "E", "U")[axis] + "(mm)")
    plot_methods = (NUMERIC, "adaptive_numeric_r0", "adaptive_numeric_r1", "direct_visual_r0", "adaptive_visual_r0")
    for index, name in enumerate(plot_methods):
        activity = np.array((prior if name == NUMERIC else predictions[name])[cid]["activity"], dtype=float)
        activity[activity < 0] = np.nan
        panels[3].plot(np.arange(len(activity)), index + activity * .7)
    panels[3].set_yticks(np.arange(len(plot_methods)))
    panels[3].set_yticklabels(plot_methods, fontsize=8)
    panels[3].set_xlabel("Original day;post-score inspection;missing/unknown remain gaps")
    fig.suptitle("case_0389:the prior loses supported activity after enlarged raw review")
    fig.tight_layout()
    fig.savefig(out / "case_0389-raw-review.png", dpi=140)
    fig.savefig(out / "case_0389-raw-review.pdf")
    plt.close(fig)
    decision = read_json(run / "analysis-v1/decision.json")
    text = """# 多形态记录上的有界局部观察：纠错价值检验

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；继续用户授权研究。
- Scope: 原八族16TRAIN／8暴露VALIDATION，现有Qwen3.8-flash、旧原位移冲突选窗策略与提示，非新增创新、非独立确认。
- Verification: 全部请求、图片与prompt/payload哈希，强数值初判、选窗、共同阶段及40方法320输出／评分精确重放。
- Reference: 受控过程参考；事后Codex看图是开发诊断，不修改标签、不称独立领域复核。

五流程均两遍：直接视觉、固定两窗视觉初判复核、自适应两窗视觉初判复核、强数值初判固定两窗复核、强数值初判冲突选窗复核。相同本地回答在各流程共享；各流程最多两次复核，合计112请求硬上限，实际57，206254token，0传输／解析失败。每条强数值自适应选窗只用其原位移与初判，未读取标签；局部请求只看原全局图和180日原局部图，不看数值预测或参考。所有方法使用共同14合法阳性观测／只跨≤7原缺测解码和16TRAIN阶段头。

""" + table + """

数值初判日／事件F1 .945529/1；自适应视觉复核后.919094/1和.913232/1，固定复核.853147/.909091和.875000/1。自适应确实少花请求且优于固定，但相对原强数值没有净收益；两遍分别撤回36／42TP、重新获得2／3TP，正确活动日检出净少34／39日；各减少12FP，净纠错−22／−27日。主要变化来自形变叠加尖峰case_0389：将局部窗口90—269日中的早期持续趋势判为非活动／未知，随后窗口270—449日从270开始确认活动。没有把未知当正确阴性；具体损伤区间与撤回至未知／阴性数量另存exact-error-traces.json。

视觉初判直接F1 .808470/.869565，固定.805970/.849558，自适应均.830769；第一遍相对两者更高，第二遍低于两者。第一遍事件.909091/.833333→1、第二遍均1，边界整体也未稳定改善。自适应视觉请求13／12，固定24／24；强数值自适应各3，固定各16。各方法token由实际共享请求归属求和，合计57唯一请求206254token单列，不重复累计。没有测全部署延迟或重新训练数值模型。

冻结判断STOP_DIVERSITY_LOCAL_REVIEW，numerical_support=False、visual_support=False，两遍均保留，不能选择第一遍宣称稳定收益。此结果只否定在这批多形态记录上的固定旧策略，不证明所有局部视觉无用。先前窄受控分布的局部观察收益仍作为历史适用范围保留。当前同一开发者看到方法／生成假设，标签不是独立现场金标准。

首版执行器使用独占新建write_json更新已有state，执行前失败，无API请求；保存原源码，改用已有save_json，仅修状态／进度持久化，在新v2目录按相同科学协议执行。v1没有结果，不把它作为一次有效模型重复。2项新失败／支持契约与5项已有缺测持续期测试通过（14个上游弃用warning），集中Ruff通过。没有改提示、阈值、重复或评分救分。

报告v1曾将TP净少34／39日误写为总撤回；其逐日追踪JSON及纠错汇总已给出真实总撤回36／42与新增2／3。v2只修正文稿统计用语，原推理与评分不变，v1报告及未封稿文字保留。
"""
    (out / "report.md").write_text(text, encoding="utf-8")
    write_json(out / "evidence-bindings.json", {
        str(p): sha(p) for p in (run / "prepared-freeze.json", run / "prediction-freeze.json",
                                run / "analysis-v1/freeze.json", verified / "freeze.json")
    })
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    previous = ROOT / "docs/gnss-deformation-manuscript-draft-2026-10-13-v10.md"
    previous_manifest = read_json(Path(protocol["data"]).parent / "manuscript-v10/manifest.json")
    if sha(previous) != previous_manifest["source_sha256"]:
        raise ValueError("Preserve the preceding complete manuscript")
    draft = previous.read_text(encoding="utf-8")
    draft = draft.replace("2026-10-13，v10。", "2026-10-13，v11。", 1)
    draft = draft.replace("deformation-manuscript-20261013-v10", "deformation-manuscript-20261013-v11", 1)
    draft = draft.replace(
        "本版补连续位移证据引导视觉池化（SCVP）的16／8开发小试、九臂342配置、强数值对照、消融及原图误差分析。",
        "本版保留SCVP小试，补旧有界VLM局部复核在多形态记录上的五流程两遍推理、强数值初判比较和活动撤回分析。",
        1,
    )
    abstract = (
        "已有有界局部视觉复核在同16／8开发人口上完成两遍五流程比较；"
        "强数值初判后的自适应复核F1为0.9191／0.9132，低于初判0.9455，"
        "各减少12个误报日，但正确活动检出净少34／39日。视觉初判后的自适应收益亦未跨两遍成立，停止固定版本。"
    )
    draft = draft.replace("本文报告可追溯信息收益及机制负结果", abstract + "本文报告可追溯信息收益及机制负结果", 1)
    draft = draft.replace(
        "The reserved records remain unopened;",
        "The existing bounded local-review workflow is also tested in five arms over two repeats. "
        "Adaptive review of a strong numerical prior attains F1 of 0.9191/0.9132, below the prior at 0.9455, "
        "removing 12 false alarms but yielding a net loss of 34/39 true activity detections. "
        "Adaptive review of a visual prior does not improve both repeats; this fixed workflow is stopped. "
        "The reserved records remain unopened;",
        1,
    )
    method = r"""### 3.15 多形态局部观察的纠错价值检验

在继续增加机制前，将已有原位移冲突选窗复核迁移到同八族16／8开发人口。分别从直接VLM初判或最强全局运动分布树初判开始；对原180日／90日步长窗口，按原21／42日间隔三块持续运动与初判的冲突排序，至少14冲突日，最多选择两不相交窗口。固定控制预设两个等间隔窗口；所有局部判断仅看同全局及原局部图，无新提示或数值文字。选窗保存后才请求局部图，所有推理冻结后才读取参考评分。

$$
\widetilde a_t=
\begin{cases}
a_t^{\mathrm{local}},&t\in W_1\cup W_2,\\
a_t^{\mathrm{prior}},&\text{其他原日期},
\end{cases}
\qquad
\widehat a=\operatorname{Decode}_{14,7}(\widetilde a).
$$

局部未知或失败保持未知，不恢复确定初判；共同解码需要14个合法阳性观测，只跨不超过7日原缺测，未知不补成阴性。原活动支撑与活动内阶段支撑分别使用共同缓存掩码，阶段头仍来自16TRAIN。五流程均两遍保留；请求按窗口共享，最大112，禁止同请求重试。该工作检验旧方法的观察价值，不称新的Agent框架、组合流程或提示为创新。

"""
    draft = draft.replace("## 4 数据与实验协议", method + "## 4 数据与实验协议", 1)
    result_section = (
        "### 5.17 有界VLM复核：误报减少但正确活动撤回更多\n\n"
        + "五流程均两遍：" + text.split("五流程均两遍：", 1)[1] + "\n\n"
    )
    draft = draft.replace("## 6 讨论", result_section + "## 6 讨论", 1)
    draft = draft.replace("## 数据、代码与AI使用说明", abstract + "\n\n## 数据、代码与AI使用说明", 1)
    draft = draft.replace(
        "Codex参与文献定位",
        "多形态局部复核由run/verify/report_landslide_diversity_local_review.py复现；"
        "diversity-local-review-v2保存57请求及旧提示/图片绑定，replay/report-v1保存40方法320输出重放、"
        "确切错误区间与图。没有新增受控记录，24保留输入及参考内容仍未加载。\n\nCodex参与文献定位",
        1,
    )
    target = ROOT / "docs/gnss-deformation-manuscript-draft-2026-10-13-v11.md"
    if target.exists():
        sealed = Path(protocol["data"]).parent / "manuscript-v11"
        if sealed.exists():
            raise FileExistsError("Preserve the sealed complete draft")
        (out / "draft-before-wording-correction.md").write_bytes(target.read_bytes())
    target.write_text(draft, encoding="utf-8")
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir()})
    print(dict(status=decision["status"], exact_corrections=detailed, manuscript=str(target)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "verified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.verified.resolve(), args.out.resolve())
