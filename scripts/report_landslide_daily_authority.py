"""Report the frozen RCRA pilot and full development expansion without relabeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json
from write_landslide_distribution_manuscript import score_table

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import runs


def report(run, replay, out):
    verify_manifest(run, "prediction-freeze.json")
    verify_manifest(run / "analysis-v1", "freeze.json")
    verify_manifest(replay, "freeze.json")
    receipt = read_json(replay / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["analysis_manifest_sha256"] != sha(run / "analysis-v1/freeze.json"):
        raise ValueError("A matching full expansion replay is required")
    out.mkdir(parents=True, exist_ok=False)
    protocol = read_json(run / "protocol.json")
    summary = read_json(run / "analysis-v1/summary.json")
    decision = read_json(run / "analysis-v1/decision.json")
    predictions = read_json(run / "predictions.json")
    rows = {row["method"]: row for row in summary}
    audits = read_json(run / "authority-audit.json")
    audit_summary = {}
    for method in sorted({row["method"] for row in audits}):
        audit_summary[method] = {
            name: sum(row[name] for row in audits if row["method"] == method)
            for name in ("editable_days", "predecoder_changes", "propagated_outside_permission", "retained_successful_abstentions")
        }
    write_json(out / "audit-summary.json", audit_summary)
    traces, paired = {}, []
    controls = ("direct_visual", "fixed_visual", "adaptive_visual", "visual_whole_partial",
                "visual_neighborhood_partial", "visual_matched_partial", "visual_no_local_visual", "visual_daily_strict")
    for rep in range(2):
        main = f"visual_daily_partial_r{rep}"
        for cid in protocol["ids"]:
            with np.load(Path(protocol["encoded"]) / "validation" / (cid + ".npz")) as bank:
                actual = bank["activity"]
            after = np.array(predictions[main][cid]["activity"])
            for control in controls:
                method = f"{control}_r{rep}"
                before = np.array(predictions[method][cid]["activity"])
                removed_fp = (actual == 0) & (before == 1) & (after != 1)
                added_fp = (actual == 0) & (before != 1) & (after == 1)
                gained_tp = (actual == 1) & (before != 1) & (after == 1)
                lost_tp = (actual == 1) & (before == 1) & (after != 1)
                value = dict(method=main, comparator=method, case_id=cid,
                             removed_fp=int(removed_fp.sum()), added_fp=int(added_fp.sum()),
                             gained_tp=int(gained_tp.sum()), lost_tp=int(lost_tp.sum()),
                             net_correct=int(removed_fp.sum() + gained_tp.sum() - added_fp.sum() - lost_tp.sum()))
                paired.append(value)
                traces[main + ":" + method + ":" + cid] = {
                    name: runs(mask) for name, mask in (("removed_fp", removed_fp), ("added_fp", added_fp),
                                                       ("gained_tp", gained_tp), ("lost_tp", lost_tp))
                }
    write_csv(out / "paired-errors.csv", paired)
    write_json(out / "exact-error-traces.json", dict(interval_convention="Original0based half-open", traces=traces))
    totals = []
    for rep in range(2):
        for control in controls:
            comparator = f"{control}_r{rep}"
            subset = [row for row in paired if row["method"] == f"visual_daily_partial_r{rep}" and row["comparator"] == comparator]
            totals.append(dict(method=f"visual_daily_partial_r{rep}", comparator=comparator,
                               **{name: sum(row[name] for row in subset) for name in
                                  ("removed_fp", "added_fp", "gained_tp", "lost_tp", "net_correct")}))
    write_csv(out / "correction-summary.csv", totals)
    for row in totals:
        expected = rows[row["comparator"]]["fp"] + rows[row["comparator"]]["fn"]
        expected -= rows[row["method"]]["fp"] + rows[row["method"]]["fn"]
        if expected != row["net_correct"]:
            raise ValueError("Paired trace disagrees with unified scoring")
    selected = [row for row in paired if row["method"] == "visual_daily_partial_r0" and row["comparator"] == "direct_visual_r0"]
    ordered = sorted(selected, key=lambda row: (row["net_correct"], row["case_id"]))
    review_ids = list(dict.fromkeys((ordered[0]["case_id"], ordered[-1]["case_id"])))
    write_json(out / "analysis-plan.json", dict(review_ids=review_ids, selection="First repeat net correction minimum/maximum versus direct;posthoc developer diagnosis;no label changes"))
    for cid in review_ids:
        case, _ = load_case_features(Path(protocol["data"]), cid)
        raw = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        fig, panels = plt.subplots(4, 1, figsize=(13, 9), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1, 1.6]))
        for axis, label in enumerate(("N", "E", "U")):
            panels[axis].plot(raw[:, axis], linewidth=.8)
            panels[axis].set_ylabel(label + " (mm)")
        names = ("direct_visual_r0", "visual_daily_partial_r0", "adaptive_visual_r0", "direct_visual_r1", "visual_daily_partial_r1")
        values = np.array([predictions[name][cid]["activity"] for name in names])
        panels[3].imshow(values, aspect="auto", interpolation="nearest", extent=(-.5, len(raw) - .5, len(names) - .5, -.5),
                         cmap=plt.get_cmap("coolwarm", 3), vmin=-1, vmax=1)
        panels[3].set_yticks(range(len(names)), names, fontsize=8)
        panels[3].set_xlabel("Original calendar day; blue unknown, white inactive, red active")
        fig.suptitle(cid + ": development diagnosis; observed raw NEU; predictions are not truth")
        fig.tight_layout()
        fig.savefig(out / (cid + "-raw-revisions.png"), dpi=150)
        fig.savefig(out / (cid + "-raw-revisions.pdf"))
        plt.close(fig)
    methods = ("visual_daily_partial_r0", "visual_daily_partial_r1", "direct_visual_r0", "direct_visual_r1",
               "fixed_visual_r0", "fixed_visual_r1", "adaptive_visual_r0", "adaptive_visual_r1",
               "visual_neighborhood_partial_r0", "visual_neighborhood_partial_r1",
               "visual_matched_partial_r0", "visual_matched_partial_r1", "visual_daily_strict_r0", "visual_daily_strict_r1",
               "visual_no_local_visual_r0", "visual_no_local_visual_r1", "query_motion_mass", "global_motion_mass",
               "xgb_raw84", "xgb_raw12", "robust_rule", "pelt_rule", "visual_numeric", "pure_visual")
    table = score_table(summary, methods)
    fig, panels = plt.subplots(1, 2, figsize=(14, 11), sharey=True)
    for panel, metric in zip(panels, ("f1", "event_f1_iou03")):
        panel.barh(methods, [rows[name][metric] for name in methods],
                   color=["#b85c43" if name.startswith("visual_daily_partial") else "#477b94" for name in methods])
        panel.set_xlim(0, 1)
        panel.set_title(metric)
    panels[0].invert_yaxis()
    fig.suptitle("RCRA:48TRAIN/24exposed VALIDATION;two repeats retained;no independent claim")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=140)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    main0, main1 = rows["visual_daily_partial_r0"], rows["visual_daily_partial_r1"]
    correction_table = ["| 主方法 | 对照 | 去FP | 新FP | 新TP | 撤TP | 净纠错日 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    correction_table.extend(f"| {row['method']} | {row['comparator']} | {row['removed_fp']} | {row['added_fp']} | {row['gained_tp']} | {row['lost_tp']} | {row['net_correct']} |" for row in totals)
    text = f"""# RCRA逐日冲突授权：完整开发扩展

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent。
- Verification Status: VERIFIED，完整请求绑定、权限与评分重放；不是独立效果验证。
- Population: 48TRAIN／24已暴露VALIDATION，保留24内容未加载；新独立记录0。
- Decision: {decision['status']}；主方法固定visual_daily_partial，没有替换为更好的对照。
- Requests: 新{decision['new_api_calls']}次／{decision['new_tokens']}token；原8条视觉回答复用；{len(decision['failures'])}个失败均保留。

## 方法与已知边界

RCRA（Raw-Conflict Revision Authority，逐日冲突授权复核）组合已有全局VLM、21／42日原始持续运动证据和至多两次局部视觉观察。唯一新检验的变化是将修订权限从复核整窗／±49日邻域收缩到实际支持的原日冲突；数值证据仅授权，明确局部回答决定修订。成功弃权保留初判是仓库已有机制，不称首创；其去除消融也同时保留。所有局部请求都不注入数值摘要；这里的数值反馈是选窗与权限。

解码前权限严格受限。公共14阳性观测、只跨≤7原缺测的解码会删除被切碎的短段；主方法两遍权限外传播分别{audit_summary['visual_daily_partial_r0']['propagated_outside_permission']}/{audit_summary['visual_daily_partial_r1']['propagated_outside_permission']}日。whole_partial的editable_days定义是外部许可集合的全部观测日，实际修改仍受局部复核覆盖限制；不能当实际观察日数。no_local_visual仍使用全局VLM初判；其retained_successful_abstentions字段是可用诊断数量，未执行局部弃权保留，不能称整个方法纯数值或声称实际应用了该保留。

## 全部主要对比

{table}

主日F1 {main0['f1']:.6f}/{main1['f1']:.6f}；事件F1 .3 {main0['event_f1_iou03']:.6f}/{main1['event_f1_iou03']:.6f}；强数值query_motion_mass {rows['query_motion_mass']['f1']:.6f}，global_motion_mass {rows['global_motion_mass']['f1']:.6f}。18原强基线按相同48训练记录重训校准，各自合理调参机会保留；局部流程无本轮阈值搜索，全部36方法／864输出统一评分。16／8小试分数不作为本24记录强基线成绩。

全部条件与净纠错见decision.json。两遍均需同时满足，不能以只一遍成立、平均F1或边界改善覆盖失败项。八族结果见analysis-v1/family-summary.csv；整体形变为主、跳变／尖峰干扰可以同现。A/S/D使用统一48TRAIN阶段头，且只能在最终活动内，日F1收益不能当独立阶段模型提升。

## 确切纠错与损伤

{chr(10).join(correction_table)}

确切半开区间见exact-error-traces.json。事后选择{','.join(review_ids)}分别按第一遍相对直接视觉净纠错最小／最大供原位移复核，所画预测不是参考；AI事后诊断不改标签、不称独立专家。

## 消融与发表判断

固定位置／整窗partial／旧49日邻域／同可修改日数居中对照隔离观察选择和修改数量；去局部视觉对照检验原始冲突翻转是否已足够；去成功弃权保留隔离旧机制。若主方法和strict活动指标相同，只能说本数据未显示该保留的增益，不能为所有模块宣称有效。

本结果仍是已用于方法选择的开发验证。即使通过本轮条件，也只支持冻结候选并开展新记录确认，不能保证论文发表、显著性或现场泛化。若失败，停止这版完整扩展，不调整阈值／提示／权限邻域救分；开发小试正结果及完整扩展的全部失败同时报告。

复现：run_landslide_daily_authority_expansion.py prepare/execute/score；verify_landslide_daily_authority_expansion.py独立重构权限／投票；report_landslide_daily_authority.py保存比较图PDF/PNG、逐日错误追踪和统一表。实验、请求、图片、源码和证据哈希冻结；新API与重放API明确分开。
"""
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "evidence-bindings.json", {str(p.resolve()): sha(p) for p in
                                                (run / "prediction-freeze.json", run / "analysis-v1/freeze.json", replay / "freeze.json")})
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir()})
    print(dict(status=decision["status"], review_ids=review_ids, daily_f1=[main0["f1"], main1["f1"]]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "replay", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.replay.resolve(), args.out.resolve())
