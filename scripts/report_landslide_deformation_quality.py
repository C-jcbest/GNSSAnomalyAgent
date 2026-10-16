"""Report the complete multi-shape validation, including failed gates and costs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import average_precision_score

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores


def render_record(case, labels, predictions, path):
    raw = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    figure, panels = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for axis in range(3):
        panels[axis].plot(raw[:, axis], linewidth=0.6, color="#487784")
        panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
        panels[axis].grid(alpha=0.2)
    rows = [("Process reference", np.array([int(row["activity_label"]) for row in labels]))]
    rows.extend(
        (name, np.array(predictions[name][case.case_id]["activity"]))
        for name in ("visual_numeric_single", "visual_numeric_dual", "numeric_dual", "xgb_raw84")
    )
    for index, (name, state) in enumerate(rows):
        values = state.astype(float)
        values[state == -1] = np.nan
        panels[-1].plot(values + index * 1.5, linewidth=1, label=name)
    panels[-1].set_yticks(np.arange(len(rows)) * 1.5, [name for name, _ in rows], fontsize=8)
    panels[-1].set_xlabel("Original day index; activity rises by1; unknown decisions remain gaps")
    figure.suptitle(
        case.case_id + " | Post-score validation error inspection, not independent reference"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def report(data, encoding, run, out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "analysis-plan.json",
        {
            "scope": "Complete24 validation; model selection/development, not unseen confirmation",
            "bootstrap": "2000 paired whole-record draws, seed2026101250; descriptive intervals only",
            "error_examples": "Largest nuisance-only false positives and largest net lost correctness vs single; selected after scoring, no reference modification",
            "test_truth_opened": False,
            "thresholds_changed": False,
        },
    )
    summaries = read_json(run / "summary.json")
    table = {row["method"]: row for row in summaries}
    results = read_json(run / "results.json")
    decision = read_json(run / "decision.json")
    predictions = read_json(run / "predictions.json")
    probability = read_json(run / "probabilities.json")
    quality = read_json(run / "quality-query-summary.json")
    split = read_json(encoding / "protocol.json")["split"]
    ids = split["validation"]
    labels = {cid: [] for cid in ids}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    ledger = read_json(data / "training-generation-ledger.json")
    families = {
        row["case_id"]: row["truth"]["family"] for row in ledger if row["group"] == "validation"
    }
    family_rows = []
    for method in predictions:
        for family in sorted(set(families.values())):
            selected = [cid for cid in ids if families[cid] == family]
            score = aggregate_scores([results[method]["per_case"][cid] for cid in selected])
            family_rows.append(
                {
                    "method": method,
                    "family": family,
                    **score["daily"],
                    "event_f1_iou03": score["events"]["f1"],
                    "event_fp": score["events"]["fp"],
                }
            )
    write_csv(out / "family-summary.csv", family_rows)
    ap_rows = []
    for method, cases in probability.items():
        actual_all = []
        scores_all = []
        for cid in ids:
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            with np.load(encoding / "validation" / (cid + ".npz")) as bank:
                common = bank["supported"] & (actual >= 0)
            actual_all.extend(actual[common].tolist())
            scores_all.extend(np.array(cases[cid]["activity"])[common].tolist())
        ap_rows.append(
            {
                "method": method,
                "auprc_common_supported_dates": float(
                    average_precision_score(actual_all, scores_all)
                ),
            }
        )
    write_csv(out / "activity-auprc.csv", ap_rows)
    draws = np.random.default_rng(2026101250).integers(0, len(ids), size=(2000, len(ids)))
    main = "visual_numeric_dual"
    comparisons = []
    for baseline in ("visual_numeric_single", "numeric_dual", "xgb_raw84", "xgb_chart384"):
        for metric in ("daily", "events"):
            counts = {
                name: np.array(
                    [
                        [results[name]["per_case"][cid][metric][key] for key in ("tp", "fp", "fn")]
                        for cid in ids
                    ]
                )
                for name in (main, baseline)
            }
            values = {}
            for name, rows in counts.items():
                total = rows[draws].sum(axis=1)
                denominator = 2 * total[:, 0] + total[:, 1] + total[:, 2]
                values[name] = np.divide(
                    2 * total[:, 0], denominator, out=np.zeros(len(draws)), where=denominator > 0
                )
            difference = values[main] - values[baseline]
            comparisons.append(
                {
                    "baseline": baseline,
                    "metric": metric + "_f1",
                    "low": float(np.quantile(difference, 0.025)),
                    "high": float(np.quantile(difference, 0.975)),
                }
            )
    write_csv(out / "paired-bootstrap.csv", comparisons)
    negative = [cid for cid in ids if families[cid] in {"quiet", "step_only", "spike_only"}]
    worst_fp = max(negative, key=lambda cid: results[main]["per_case"][cid]["daily"]["fp"])
    worst_change = max(
        ids,
        key=lambda cid: sum(
            results[main]["per_case"][cid]["daily"][key]
            - results["visual_numeric_single"]["per_case"][cid]["daily"][key]
            for key in ("fp", "fn")
        ),
    )
    examples = list(dict.fromkeys([worst_fp, worst_change]))
    for cid in examples:
        case, _ = load_case_features(data, cid)
        render_record(case, labels[cid], predictions, out / (cid + "-error.png"))
    write_json(
        out / "figure-selection.json",
        {"case_ids": examples, "post_score_selection": True, "labels_modified": False},
    )
    encoding_cost = read_json(encoding / "encoding-cost.json")
    with (run / "training-costs.csv").open(encoding="utf-8-sig", newline="") as stream:
        costs = list(csv.DictReader(stream))
    trained_seconds = sum(float(row["training_seconds"]) for row in costs)
    delta = table[main]["f1"] - table["visual_numeric_single"]["f1"]
    lines = [
        "# DQ-VN：形变／干扰双输出机制的多形态验证",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite；用户授权的方法开发与实验",
        "- Origin Date: 2026-10-12",
        "- Version Label: DQ-VN calibration v1",
        "- Verification Status: 完整验证集选择与开发结果；重放状态另见实际核验记录；不是保留评价或独立领域参考",
        "",
        "## 主要结果",
        "",
        f"决定：**{decision['decision']}**；预设条件{sum(decision['checks'].values())}/{len(decision['checks'])}。主方法相对同数据单输出的日F1差{delta:+.6f}，净纠错{decision['net_correct']:+d}日。以下报告全部12项方法，不删除表现更好的数值方法或失败的消融。",
        "",
        "| 方法 | Precision | Recall | 日F1 | 事件F1(.3) | 事件F1(.5) | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        values = [
            "未定义" if row[key] is None else f"{row[key]:.6f}"
            for key in ("precision", "recall", "f1", "event_f1_iou03", "event_f1_iou05")
        ]
        lines.append(
            "| " + row["method"] + " | " + " | ".join(values) + f" | {row['fp']} | {row['fn']} |"
        )
    lines.extend(
        [
            "",
            "纯干扰记录误报和混合记录召回：",
            "",
            "```json",
            json.dumps(
                {
                    "negative_fp": decision["negative_fp"],
                    "mixed_recall": decision["mixed_recall"],
                    "checks": decision["checks"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "## 核心机制及消融",
            "",
            "冻结MobileCLIP2-S0图文预训练模型的视觉分支与原位移/诊断数值输入共享一个小表示层；增加一个独立的质量查询输出，与形变输出可同时阳性。只比较辅助监督开关，不增加训练图或负例。",
            "",
            "$$",
            r"h_t=f_\theta([z_t^V,z_t^N]),\quad p_t^{\mathrm{def}}=\sigma(w_d^\top h_t),\quad p_t^{\mathrm{quality}}=\sigma(w_q^\top h_t).",
            "$$",
            "",
            "$$",
            r"\mathcal L=\mathcal L_{\mathrm{def}}^{\mathrm{balanced}}+\mathcal L_{\mathrm{quality}}^{\mathrm{balanced}}.",
            "$$",
            "",
            "单输出消融关闭第二项损失；纯数值双/单输出与纯视觉单输出检验模态作用。双输出不是新的多任务框架，本轮只检验这项GNSS任务干预；没有新的语言生成或完整VETime复现，不能把图文模型的冻结视觉分支称为新增Agent推理。",
            "",
            "新增96条独立随机流记录分48训练/24验证/24保留评价。8类覆盖纯噪声/偏移/尖峰、普通形变、快速连续起变、再活动和形变叠加干扰，5类内部速度形态不强制A/S/D顺序。弱运动只为子组，阶段只在活动内部使用共同的新数值头，不能当视觉阶段创新。",
            "",
            "## 独立质量任务是否学会",
            "",
            "质量标签表示中央14日内已观测的偏移转换或尖峰；支持不足为未知，不把永久平台标为每天的新事件。固定概率.5，已知阳性上的不可用预测仍计漏检，AUPRC只对共同确定支持查询计算，不称精确故障日定位。",
            "",
            "| 方法 | 查询Precision | 查询Recall | 查询F1 | 覆盖率 | AUPRC |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in quality.items():
        values = [
            "未定义" if row[key] is None else f"{row[key]:.6f}"
            for key in ("precision", "recall", "f1", "coverage", "auprc_known_supported_queries")
        ]
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 公平性、稳定性与边界",
            "",
            "所有模型使用相同48/24划分重训/校准；新神经臂各2头宽×19阈值，与raw84/chart-info树各2深度×19阈值同为38机会。规则/PELT复用原合理网格而重校准，不拿旧人口模型冒充适用基线。神经训练每3日一个中心；强树使用完整支持日期，输入模态变化也改变小头参数量。这些差异不隐去，不把去视觉差异自动解释为视觉信息的纯因果作用。",
            "",
            "各新学习臂使用相同原短/42/84支持和14日最短段后处理；8—13日快速事件仍在参考与总分中，可能漏检。缺测保留未知，不压缩日历。AUPRC与族表另存；日F1用完整共同参考分母。匹配事件边界均值受匹配集合变化影响，不能直接当同事件配对改善。",
            "",
            "整记录2000次配对bootstrap描述本24条的波动，已用于模型选择的数据不是独立测试；区间不消除调参、方法选择或多重比较偏差。",
            "",
            "| 主方法相对对照 | 指标 | 95%描述性区间 |",
            "| --- | --- | --- |",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['baseline']} | {row['metric']} | [{row['low']:.6f}, {row['high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## 成本、决定及下一步",
            "",
            f"训练/验证共享编码{encoding_cost['encoded_centers']}中心，实测{encoding_cost['seconds']:.3f}秒；导出的新头/树训练及树验证循环累计{trained_seconds:.3f}秒，原规则/PELT校准与完整评分没有逐项计时，保留未知。这些是当前CPU实测，不是部署端到端速度或免费视觉成本。无新API请求。",
            "",
            "预设门槛失败则停止当前版本，不改种子、损失、阈值或挑子组挽救；保留24条评价标签继续封存。门槛通过才冻结模型，在预指定8条原始观测Codex复核后推理/评分全部24条。Codex熟悉生成设计与历史结果，属于开发者观测敏感性，不是独立领域专家；已知合成过程参考也不能证明现场准确率。",
            "",
            "所有失败条件、历史NCDA及PP-VN取舍保留。本报告是实际机制研究结果，不自动意味着形成可发表贡献；贡献必须由视觉/辅助监督消融及未参与选择的评价支持。",
            "",
            "产物：本目录family-summary.csv、activity-auprc.csv、paired-bootstrap.csv与原图错误检查；原模型、选择网格、概率、活动/阶段与质量评分位于兄弟quality-calibration-v1目录。原公式继续使用Markdown LaTeX，旧v54主稿没有被覆盖。",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out / "manifest.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print("Complete validation report", decision["decision"], "examples", examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoding", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.data.resolve(), args.encoding.resolve(), args.run.resolve(), args.out.resolve())
