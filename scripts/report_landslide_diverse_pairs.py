"""Report broad-population mechanism results and every retained strong control."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores


def report(data, encoding, pairs, axis_run, run, out):
    out.mkdir(parents=True, exist_ok=False)
    split = read_json(encoding / "protocol.json")["split"]
    summary = read_json(run / "summary.json")
    results = read_json(run / "results.json")
    decision = read_json(run / "decision.json")
    axis_decision = read_json(axis_run / "decision.json")
    predictions = read_json(run / "predictions.json")
    ledger = read_json(data / "training-generation-ledger.json")
    families = {
        row["case_id"]: row["truth"]["family"] for row in ledger if row["group"] == "validation"
    }
    rows = []
    for name in predictions:
        for family in sorted(set(families.values())):
            score = aggregate_scores(
                [
                    results[name]["per_case"][cid]
                    for cid in split["validation"]
                    if families[cid] == family
                ]
            )
            rows.append(
                {
                    "method": name,
                    "family": family,
                    **score["daily"],
                    "event_f1_iou03": score["events"]["f1"],
                }
            )
    write_csv(out / "family-summary.csv", rows)
    main = "visual_numeric_pair"
    draws = np.random.default_rng(2026101251).integers(0, 24, (2000, 24))
    intervals = []
    for control in (
        "visual_numeric_augmentation",
        "visual_numeric_plain",
        "numeric_pair",
        "xgb_raw84_augmentation",
        "dq_xgb_raw84",
    ):
        for metric in ("daily", "events"):
            values = {}
            for name in (main, control):
                counts = np.array(
                    [
                        [results[name]["per_case"][cid][metric][key] for key in ("tp", "fp", "fn")]
                        for cid in split["validation"]
                    ]
                )
                total = counts[draws].sum(axis=1)
                denominator = 2 * total[:, 0] + total[:, 1] + total[:, 2]
                values[name] = np.divide(
                    2 * total[:, 0], denominator, out=np.zeros(2000), where=denominator > 0
                )
            difference = values[main] - values[control]
            intervals.append(
                {
                    "control": control,
                    "metric": metric + "_f1",
                    "low": float(np.quantile(difference, 0.025)),
                    "high": float(np.quantile(difference, 0.975)),
                }
            )
    write_csv(out / "paired-bootstrap.csv", intervals)
    negative = [
        cid for cid, family in families.items() if family in {"quiet", "step_only", "spike_only"}
    ]
    worst = max(negative, key=lambda cid: results[main]["per_case"][cid]["daily"]["fp"])
    labels = []
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] == worst:
                labels.append(row)
    case, _ = load_case_features(data, worst)
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    figure, panels = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for axis in range(3):
        panels[axis].plot(values[:, axis], linewidth=0.6)
        panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
        panels[axis].grid(alpha=0.2)
    states = [("Process reference", np.array([int(row["activity_label"]) for row in labels]))]
    states.extend(
        (name, np.array(predictions[name][worst]["activity"]))
        for name in (
            "visual_numeric_plain",
            "visual_numeric_augmentation",
            main,
            "xgb_raw84_augmentation",
        )
    )
    for index, (name, state) in enumerate(states):
        display = state.astype(float)
        display[state == -1] = np.nan
        panels[-1].plot(display + index * 1.5, linewidth=1)
    panels[-1].set_yticks(np.arange(len(states)) * 1.5, [name for name, _ in states], fontsize=8)
    panels[-1].set_xlabel("Original day; unknown decisions are gaps")
    figure.suptitle(worst + " | Post-score error analysis; references not modified")
    figure.tight_layout()
    figure.savefig(out / (worst + "-nuisance-error.png"), dpi=130)
    plt.close(figure)
    write_json(
        out / "analysis-plan.json",
        {
            "bootstrap_seed": 2026101251,
            "draws": 2000,
            "unit": "whole paired record",
            "selection": "Worst nuisance-only FP after scoring",
            "test_truth_opened": False,
            "labels_modified": False,
            "interpretation": "Descriptive development intervals, not selection-adjusted tests",
        },
    )
    cost = read_json(pairs / "preparation-cost.json")
    lines = [
        "# 多形态GNSS形变识别：PP-VN扩展、负结果与强基线",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite；用户授权的方法、数据及实验开发",
        "- Origin Date: 2026-10-12",
        "- Version Label: diverse-PPVN-calibration-v1",
        "- Verification Status: 48训练/24暴露验证开发结果；重放状态见兄弟paired-axis-replay-v1.json；24保留参考未打开",
        "",
        "## 结论",
        "",
        f"**{decision['decision']}**，{sum(decision['checks'].values())}/8条件。主日F1 .773036仅比同增广.771318高.001719，却新增82FP、补检56日，净−26；事件F1 .666667低于同增广.736842，纯干扰FP49→95。停止本次扩展，不改参数或打开保留标签救分。",
        "",
        "旧窄分布PP-VN受控确认和120条结果原样保留；此次多形态负结果限制其推广，不能将不同人口日F1直接相减解释为模型退化比例。",
        "",
        "## 实验与消融",
        "",
        f"复用全部{cost['reused_original_centers']}原视觉编码；25个TRAIN parent共118对，覆盖五种形变族。只去除连续形变分量并替换同净位移偏移，原测量误差、已有偏移/尖峰及缺测保留。不是118条新独立记录。",
        "",
        "$$",
        r"\widetilde{x}_t=x_t-g_t+g_u+\mathbf{1}_{t\geq k}(g_v-g_u),\qquad (u,v)=(c-91,c+91).",
        "$$",
        "",
        "$$",
        r"\mathcal L=\mathcal L_{\mathrm{BCE}}+\frac{1}{|\mathcal P|}\sum_{(c^+,c^-)\in\mathcal P}\log(1+\exp(1-z_{c^+}+z_{c^-})).",
        "$$",
        "",
        "采用既有冻结MobileCLIP2-S0编码器、小ReLU头、seed20261020、12epoch与固定配对系数1；每臂2宽/深×19阈值=38机会。七神经臂隔离配对、增广、配对关系和模态；两同增广强树获取完全相同负中心，且其原训练日期覆盖优于神经stride3。共同48训练数值阶段头不变。",
        "",
        "旧12项DQ同人口对照全部列出；名称前缀dq仅标来源，不意味双输出。它们与新PP普通头有不同固定seed及小头结构，核心配对效应只用新同架构对照判断。直接VLM/固定工作流本批尚未运行，不拿旧人口成绩填此表；历史完整比较见旧v54及120条报告。",
        "",
        "| 方法 | Precision | Recall | 日F1 | 事件F1(.3) | 事件F1(.5) | FP | FN | 起止MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        metrics = [
            "未定义" if row[key] is None else f"{row[key]:.6f}"
            for key in ("precision", "recall", "f1", "event_f1_iou03", "event_f1_iou05")
        ]
        bounds = [
            "未定义" if row[key] is None else f"{row[key]:.3f}"
            for key in ("onset_core_mae_days", "offset_core_mae_days")
        ]
        lines.append(
            "| "
            + row["method"]
            + " | "
            + " | ".join(metrics)
            + f" | {row['fp']} | {row['fn']} | "
            + "/".join(bounds)
            + " |"
        )
    lines.extend(
        [
            "",
            "完整预设条件及分组：",
            "",
            "```json",
            json.dumps(decision, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 逐轴证据小试",
            "",
            "只在原18输入加入9个原始三轴/21、42、84日的带符号持续性证据；池化消融看完全相同原块。原raw84预测精确复现。逐轴日F1 .736512、事件F1 .764706、净−81；池化日F1 .786714但事件F1 .756757，不能将更好的控制事后升级为贡献。",
            "",
            "```json",
            json.dumps(axis_decision, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 统计、代价与限制",
            "",
            "24条记录用于模型选择和机制反馈，2000次整记录配对bootstrap只描述本组波动，不消除选择偏差。逐日存在强相关；730日不等于730个独立样本。匹配边界均值来自不同匹配集合，不能称同事件配对定位改善。",
            "",
            f"新增配对准备{cost['total_seconds']:.3f}秒，其中仅118负图编码{cost['encoder_seconds']:.3f}秒。原编码1375.669秒需要计入首次部署/准备成本，不因缓存称视觉免费。0新API。训练与验证循环实测另存各training-costs.csv；这些不是部署端到端计时。",
            "",
            "主评活动Precision/Recall/F1、事件IoU与边界；弱运动与活动内阶段留在原summary.csv。最短14日后处理、缺测与原支持规则保留，8—13日快变事件仍计分。当前参考是设计运动过程，无法保证微弱起止都可从观测确定；Codex原图检查是开发者错误分析，不是独立领域参考。",
            "",
            f"评分后最大的纯干扰误报记录：{worst}。图保留原位移、缺测及各预测；实际看图结论在项目文档另记，生成图本身不等于完成复核。",
            "",
            "本轮停止三项无稳定收益版本，保留24条参考用于后续通过机制的冻结确认。下一核心改动应解释并解决事件碎裂与平台误报，不能再只放大弱召回，或叠加框架/提示数量。当前没有充分证据将新广分布方法称为可发表的算法贡献。",
        ]
    )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out / "manifest.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print("Complete report", decision["decision"], "error figure", worst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoding", "pairs", "axis-run", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(
        args.data.resolve(),
        args.encoding.resolve(),
        args.pairs.resolve(),
        args.axis_run.resolve(),
        args.run.resolve(),
        args.out.resolve(),
    )
