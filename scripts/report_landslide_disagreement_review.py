"""Analyze why held-record numerical revision utility failed to generalize."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_disagreement_review import MAIN, load_banks
from run_landslide_window_optimization import load_case_features, read_json
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_disagreement_review import (
    review_benefit,
    review_features,
    revise_disagreements,
)
from gnss_sim.landslide_visual_invariance import numeric_input


def report(data, encoded, run, verified, out):
    verify_manifest(run, "freeze.json")
    receipt = read_json(verified / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["calibration_manifest_sha256"] != sha(
        run / "freeze.json"
    ):
        raise ValueError("Revision replay does not bind this run")
    protocol = read_json(run / "protocol.json")
    split = protocol["split"]
    banks = load_banks(encoded, split)
    choices = read_json(run / "selection.json")
    predictions = read_json(run / "predictions.json")
    base = read_json(run / "fixed-source-probabilities.json")
    scaling = read_json(run / "numeric-scaling.json")
    choice = choices[MAIN]
    threshold = choice["threshold"]
    model = read_json(run / "review-models.json")[MAIN][f"{choice['parameter']}:{threshold}"]
    results = read_json(run / "results.json")
    summary = read_json(run / "summary.json")
    decision = read_json(run / "decision.json")
    stage = XGBClassifier()
    stage.load_model(run / "stage-xgboost.json")
    per_case = []
    for cid in split["validation"]:
        bank = banks[cid]
        visual = np.array(base["visual"][cid])
        numeric = np.array(base["numeric"][cid])
        features = review_features(
            visual, numeric, numeric_input(bank["numbers"], scaling), threshold
        )
        benefit = review_benefit(model, features)
        value, switched = revise_disagreements(
            visual, numeric, bank["supported"], benefit, threshold
        )
        actual = bank["activity"]
        known = bank["supported"] & (actual >= 0)
        before_correct = (visual >= threshold) == actual
        after_correct = (value >= threshold) == actual
        codes = stage.predict(bank["stage_matrix"]).astype(int) + 1
        before_activity = decode(visual, threshold, bank, codes).activity
        after_activity = np.array(predictions[MAIN][cid]["activity"])
        effects = bank["observed"] & (before_activity != after_activity) & ~switched
        left = results[MAIN]["per_case"][cid]["daily"]
        right = results["fixed_blend"]["per_case"][cid]["daily"]
        per_case.append(
            dict(
                case_id=cid,
                disagreements=int(
                    (known & ((visual >= threshold) != (numeric >= threshold))).sum()
                ),
                switched=int((known & switched).sum()),
                corrected_queries=int((known & switched & ~before_correct & after_correct).sum()),
                harmed_queries=int((known & switched & before_correct & ~after_correct).sum()),
                predicted_benefit_outside_unit_range=int((known & (np.abs(benefit) > 1)).sum()),
                predicted_benefit_outside_on_switch=int(
                    (known & switched & (np.abs(benefit) > 1)).sum()
                ),
                duration_effects_outside_switched_queries=int(effects.sum()),
                activity_net_vs_fixed=right["fp"] + right["fn"] - left["fp"] - left["fn"],
                min_predicted_benefit=float(benefit[known].min()),
                max_predicted_benefit=float(benefit[known].max()),
            )
        )
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "utility-errors.csv", per_case)
    totals = {
        key: sum(row[key] for row in per_case)
        for key in (
            "disagreements",
            "switched",
            "corrected_queries",
            "harmed_queries",
            "predicted_benefit_outside_unit_range",
            "predicted_benefit_outside_on_switch",
            "duration_effects_outside_switched_queries",
            "activity_net_vs_fixed",
        )
    }
    ids = list(
        dict.fromkeys(
            [
                min(per_case, key=lambda row: row["activity_net_vs_fixed"])["case_id"],
                max(per_case, key=lambda row: row["activity_net_vs_fixed"])["case_id"],
            ]
        )
    )
    write_json(
        out / "analysis-plan.json",
        dict(
            review_ids=ids,
            selection="least/greatest net correct vs fixed blend, first tie",
            totals=totals,
            training_disagreements=model["samples"],
            training_benefit_sum=model.get("benefit_sum"),
            labels_modified=False,
            test_truth_opened=False,
        ),
    )
    for cid in ids:
        case, _ = load_case_features(data, cid)
        raw = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        figure, panels = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        for axis in range(3):
            panels[axis].plot(raw[:, axis], linewidth=0.7)
            panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
        lines = [("Process reference", banks[cid]["activity"])] + [
            (name, np.array(predictions[name][cid]["activity"]))
            for name in (MAIN, "fixed_blend", "scores_only_review", "visual_numeric", "xgb_raw84")
        ]
        for index, (name, value) in enumerate(lines):
            curve = value.astype(float)
            curve[curve < 0] = np.nan
            panels[3].plot(np.arange(len(curve)), curve * 0.7 + index, linewidth=1)
        panels[3].set_yticks(np.arange(len(lines)), [name for name, _ in lines])
        panels[3].set_xlabel("Original day; unknown decisions remain gaps")
        figure.suptitle(cid + " | post-score raw error analysis")
        figure.tight_layout()
        figure.savefig(out / (cid + "-raw-review.png"), dpi=140)
        plt.close(figure)
    figure, panels = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    ordered = sorted(summary, key=lambda row: row["f1"], reverse=True)
    for panel, metric, title in zip(
        panels, ("f1", "event_f1_iou03"), ("Daily deformation F1", "Event F1 (IoU0.3)")
    ):
        colors = ["#b85940" if row["method"] == MAIN else "#4d8193" for row in ordered]
        panel.barh(
            [row["method"] for row in ordered], [row[metric] for row in ordered], color=colors
        )
        panel.set_xlim(0, 1)
        panel.set_title(title)
        panel.grid(axis="x", alpha=0.2)
    panels[0].invert_yaxis()
    figure.suptitle("Cross-fitted disagreement review: 48 training /24 exposed validation")
    figure.tight_layout()
    figure.savefig(out / "unified-comparison.png", dpi=140)
    figure.savefig(out / "unified-comparison.pdf")
    plt.close(figure)
    table = [
        "| Method | P | R | Daily F1 | Event F1 .3/.5 | FP/FN | Boundary MAE d |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary:
        table.append(
            f"| {row['method']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['event_f1_iou03']:.6f}/{row['event_f1_iou05']:.6f} | {row['fp']}/{row['fn']} | {row['onset_core_mae_days']:.3f}/{row['offset_core_mae_days']:.3f} |"
        )
    text = [
        "# 跨记录学习分歧纠错：原数值反馈未显示增益",
        "",
        "Material Passport: academic-research-suite / experiment-agent；disagreement-v1；VERIFIED模型/OOF/权限/统一评分；不是未见评价。",
        "",
        "## 结论",
        "",
        f"**{decision['decision']}**，{sum(decision['checks'].values())}/7条件。新增原证据门控日F1 .746338低于固定混合.757966和去原证据.759184；普通同证据堆叠.765206也更高。纯干扰误报66对固定20，净纠错−48/−54日，不扩大或换alpha/seed/阈值救分。",
        "",
        "## 方法及差异",
        "",
        "四折按TRAIN原ID顺序每4记录一折，每折12完整记录从视觉数值头、raw84树及数值缩放中排除，余36拟合。源头固定width16/depth2，不以验证表现挑源；OOF共10187个stride3训练查询。以每阈值下数值正确性减视觉数值正确性为−1/+1收益，只在分歧查询拟合StandardScaler＋Ridge(alpha1/10)。主6概率/边距特征＋36原数值/缺失标志，消融仅6分数。验证支撑且两源分歧且预测收益>0才换为原数值分数一次。",
        "",
        "一致分数的保护发生在逐日预判层；共同14阳性观测/≤7原缺测解码仍可能影响邻接日，不能声称最终活动区间中一致日绝不改变。缺测保持未知，已知未检仍计FN；干扰同现不否决形变。",
        "",
        "收益学习、跨记录预测、动态融合和堆叠均有已有工作及本仓库先例：旧视觉残差已有OOF，旧窗口效用学习已经失败，VETime已有动态融合，通用复核见VLM4TS/Refining。这里只检验分歧权限与原数值证据的增量，不把这些通用操作称首创或新Agent框架。没有新观察、语言生成、API或成本优化声明。",
        "",
        "## 公平强对照及消融",
        "",
        "相同48/24和既有原图缓存，新重拟合纯视觉/纯数值/视觉数值及raw12/raw84/chart384强树；三个神经及三个树臂各38机会。主/去原证据及两种普通OOF堆叠各38，固定0.25/0.5/0.75混合57，规则/PELT各9，合计455配置。所有方法采用同新缺测持续性和同train48阶段头；神经10187训练中心，树全部合法日（raw84为30659），差异明示。普通堆叠拥有相同OOF来源和全部42特征、直接学活动，避免把多训练一个分类器错当机制创新。",
        "",
        *table,
        "",
        "## 纠错迁移诊断",
        "",
        "以下是选中模型在暴露验证上的评分后诊断，不作为新阈值/模块依据。原幅度数据及预测收益不是显著性或置信概率。",
        "",
        "```json",
        json.dumps(totals, ensure_ascii=False, indent=2),
        "```",
        "",
        "corrected/harmed_queries比较相同0.4阈值的固定源、发生在持续性解码前；activity_net_vs_fixed比较各自公平校准后的最终输出。因此二者不能混为同一净效果。收益目标本身仅−1/+1，线性模型可能外推到范围外；只观察到外推，不能未经隔离实验认定它是唯一失败原因。源OOF使用36记录而部署源使用48记录也可能改变校准，尚未单因素验证。",
        "",
        "在原图上按主对固定混合净纠错最差/最好选例，不重标过程参考，不称独立领域复核。",
    ]
    for cid in ids:
        text += ["", f"![{cid}原三轴和活动]({cid}-raw-review.png)"]
    text += [
        "",
        "![统一日/事件对比](unified-comparison.png)",
        "",
        "## 复现及代价",
        "",
        "run_landslide_disagreement_review.py→verify_landslide_disagreement_review.py→本脚本。既有25586原编码缓存全部复用，无重编码；四折及六神经/六树共20源模型拟合，源训练秒数见training-costs.csv，未单独测完所有门控拟合/网格的端到端成本。OOF来源与选中模型、原日输出/阶段/评分均重放，普通stacker恢复StandardScaler的float32逐步运算，全部最终输出精确。保留24参考未打开。",
        "",
    ]
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(dict(review_ids=ids, totals=totals), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "run", "verified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(
        args.data.resolve(),
        args.encoded.resolve(),
        args.run.resolve(),
        args.verified.resolve(),
        args.out.resolve(),
    )
