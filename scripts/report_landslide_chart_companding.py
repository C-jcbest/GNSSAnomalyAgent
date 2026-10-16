"""Report target-anchored chart evidence, complete ablations and raw errors."""

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

MAIN = "target_companded_numeric"
CONTROLS = [
    "global_numeric",
    "global_companded_numeric",
    "numeric",
    "target_companded_visual",
    "xgb_raw84",
    "xgb_target_chart384",
]


def report(data, encoded, run, verified, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = read_json(encoded / "protocol.json")
    ids = protocol["split"]["validation"]
    results = read_json(run / "results.json")
    summary = read_json(run / "summary.json")
    predictions = read_json(run / "predictions.json")
    decision = read_json(run / "decision.json")
    ledger = read_json(data / "training-generation-ledger.json")
    families = {row["case_id"]: row["truth"]["family"] for row in ledger if row["case_id"] in ids}
    family_rows = []
    for name in predictions:
        for family in sorted(set(families.values())):
            score = aggregate_scores(
                [results[name]["per_case"][cid] for cid in ids if families[cid] == family]
            )
            family_rows.append(
                {
                    "method": name,
                    "family": family,
                    **score["daily"],
                    "event_f1_iou03": score["events"]["f1"],
                }
            )
    write_csv(out / "family-summary.csv", family_rows)
    draws = np.random.default_rng(2026101253).integers(0, len(ids), (2000, len(ids)))
    intervals = []
    for control in CONTROLS:
        for metric in ("daily", "events"):
            values = {}
            valid = np.ones(2000, dtype=bool)
            for name in (MAIN, control):
                counts = np.array(
                    [
                        [results[name]["per_case"][cid][metric][key] for key in ("tp", "fp", "fn")]
                        for cid in ids
                    ]
                )
                total = counts[draws].sum(axis=1)
                denominator = 2 * total[:, 0] + total[:, 1] + total[:, 2]
                valid &= denominator > 0
                values[name] = np.divide(
                    2 * total[:, 0], denominator, out=np.full(2000, np.nan), where=denominator > 0
                )
            difference = (values[MAIN] - values[control])[valid]
            intervals.append(
                {
                    "control": control,
                    "metric": metric + "_f1",
                    "defined_draws": int(valid.sum()),
                    "low": float(np.quantile(difference, 0.025)),
                    "high": float(np.quantile(difference, 0.975)),
                }
            )
    write_csv(out / "paired-bootstrap.csv", intervals)
    names = [
        "global_numeric",
        MAIN,
        "global_companded_numeric",
        "original_visual",
        "target_companded_visual",
        "numeric",
        "xgb_raw12",
        "xgb_raw84",
        "xgb_chart384",
        "xgb_target_chart384",
        "xgb_global_companded_chart384",
        "robust_rule",
        "pelt_rule",
    ]
    table = {row["method"]: row for row in summary}
    figure, panels = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    for panel, key, title in zip(
        panels,
        ("f1", "event_f1_iou03"),
        ("Daily deformation F1", "Event F1 (IoU 0.3)"),
        strict=True,
    ):
        values = [table[name][key] for name in names]
        panel.barh(
            np.arange(len(names)),
            values,
            color=["#b95b45" if name == MAIN else "#52869a" for name in names],
        )
        panel.set_xlim(0, 1.06)
        panel.set_title(title)
        panel.grid(axis="x", alpha=0.2)
        panel.set_axisbelow(True)
        for index, value in enumerate(values):
            panel.text(value + 0.01, index, f"{value:.4f}", va="center", fontsize=9)
    panels[0].set_yticks(np.arange(len(names)), names, fontsize=9)
    panels[0].invert_yaxis()
    figure.suptitle("Target-anchored companding pilot: 16 training /8 exposed validation")
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    figure.savefig(out / "daily-event-comparison.png", dpi=160)
    figure.savefig(out / "daily-event-comparison.pdf")
    plt.close(figure)
    net_by_case = {}
    for cid in ids:
        main = results[MAIN]["per_case"][cid]["daily"]
        control = results["global_numeric"]["per_case"][cid]["daily"]
        net_by_case[cid] = control["fp"] + control["fn"] - main["fp"] - main["fn"]
    error_ids = list(
        dict.fromkeys(
            [min(ids, key=lambda cid: net_by_case[cid]), max(ids, key=lambda cid: net_by_case[cid])]
        )
    )
    labels = {cid: [] for cid in error_ids}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    raw_evidence = {}
    for cid in error_ids:
        case, _ = load_case_features(data, cid)
        values = np.array(
            [row if row is not None else [np.nan] * 3 for row in case.displacement_mm]
        )
        figure, panels = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        for axis in range(3):
            panels[axis].plot(values[:, axis], linewidth=0.6)
            panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
            panels[axis].grid(alpha=0.2)
        states = [
            ("Process reference", np.array([int(row["activity_label"]) for row in labels[cid]]))
        ]
        states.extend(
            (name, np.array(predictions[name][cid]["activity"]))
            for name in ["global_numeric", "global_companded_numeric", MAIN, "xgb_raw84"]
        )
        for index, (name, state) in enumerate(states):
            display = state.astype(float)
            display[state == -1] = np.nan
            panels[-1].plot(display + index * 1.5, linewidth=0.9)
        panels[-1].set_yticks(
            np.arange(len(states)) * 1.5, [name for name, _ in states], fontsize=8
        )
        panels[-1].set_xlabel("Original day; unknown decisions remain gaps")
        figure.suptitle(
            cid + f" | post-score error analysis, net vs linear global {net_by_case[cid]:+d}"
        )
        figure.tight_layout()
        figure.savefig(out / (cid + "-error.png"), dpi=130)
        plt.close(figure)
        # These regular blocks are selected by the original calendar, not by
        # errors or derivative peaks. They assist the actual raw-plot review.
        blocks = []
        for first in range(0, len(values), 30):
            sample = values[first : first + 14]
            available = np.isfinite(sample).all(axis=1)
            blocks.append(
                {
                    "start": first,
                    "stop_exclusive": min(first + 14, len(values)),
                    "observations": int(available.sum()),
                    "raw_neu_median_mm": np.median(sample[available], axis=0).tolist()
                    if available.any()
                    else None,
                }
            )
        raw_evidence[cid] = blocks
    write_json(out / "raw-review-blocks.json", raw_evidence)
    cost = read_json(encoded / "encoding-cost.json")
    lines = [
        "# 目标锚定噪声尺度图像编码：TAC-VN小试",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent；用户授权自主研究",
        "- Origin Date: 2026-10-12",
        "- Version Label: chart-companding-pilot-v1",
        "- Verification Status: "
        + read_json(verified / "verification.json")["status"]
        + "；全部保存分数、活动、阶段及统一评分重放；不是独立效果确认",
        "",
        "## 实际决定",
        "",
        f"**{decision['decision']}**，{sum(decision['checks'].values())}/7。主对原全局/全窗锚点asinh净纠错分别{decision['net_correct']['global_numeric']:+d}/{decision['net_correct']['global_companded_numeric']:+d}日。完整决定与分组见下，不用最好消融替换主方法。",
        "",
        "## 最小机制及最近邻",
        "",
        "TAC-VN（Target-Anchored Companding Visual-Numeric Detection）沿用冻结MobileCLIP2-S0与原数值证据，只更改图像坐标。原日期不压缩、阶跃不扣除、缺测不补值。目标锚点为中心14日可用原观测中位数，只决定坐标原点，不赋活动；尺度为原始日差MAD噪声代理的3倍，不是置信区间。",
        "",
        "$$",
        r"a_{t,j}=\operatorname{median}_{s\in[t-7,t+7)\cap\mathcal O}x_{s,j},\qquad u_{t,k,j}=\operatorname{asinh}\!\left(\frac{m_{t,k,j}-a_{t,j}}{3\sigma_j}\right).",
        "$$",
        "",
        "$$",
        r"b_{t,j}=\max\!\left(\max_k|u_{t,k,j}|,\operatorname{asinh}(1)\right),\qquad y_{t,k,j}=31\left(\frac12-\frac{u_{t,k,j}}{2b_{t,j}}\right).",
        "$$",
        "",
        "全窗锚点消融只替换锚点为61可用位移槽的中位数，变换、噪声、图、编码器和头容量完全一致。原线性图对照提供同原三轴、180日前后覆盖与原数值。所有视觉数值头均548维，同seed、12epoch、16/32宽及38机会；去数值视觉512维明确披露。缺测14阳性观测共同判定与阶段头完全相同。",
        "",
        "GNSS-FM（arXiv:2606.07725，III-C）已经使用median/MAD及asinh位移/增量预处理；ITF-TAD（arXiv:2408.14756）已有图像归一化与冻结模型检测。asinh和视觉数值组合本身不是首创。新候选仅检验目标局部锚点及噪声差分尺度用于形变图的额外价值；不是完整GNSS-FM复现，不引入其补缺测策略。非线性坐标改变曲线导数，不能据此赋A/S/D；所有阶段依然来自原数值头。",
        "",
        "读取范围与HTTP失败见兄弟chart-scaling-literature-v1/reading-notes.md及chart-scaling-primary-v1。VLM4TS preprocessing原始来源超时/网页503未被写成已读。",
        "",
        "## 同任务完整比较",
        "",
        "七原对照确实在完全相同16/8、支撑、标签、数值缩放和共同缺测规则下训练/校准，全部输入逐值核验后复用；不是不适用旧模型。六新臂各38机会，共228配置；两新图表数值树还获取对应366个位置/可用列描述、同18数值，共384维，提供相同图像信息的强非视觉对照。树使用全部已知支撑TRAIN日，神经所有3382中心stride3；不隐藏覆盖差异。",
        "",
        "| 方法 | P | R | 日F1 | 事件F1(.3) | 事件F1(.5) | FP | FN | 起止MAE（日） | 阶段宏F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary:

        def fmt(key):
            return "未定义" if row[key] is None else f"{row[key]:.6f}"

        lines.append(
            "| "
            + row["method"]
            + " | "
            + " | ".join(
                fmt(key)
                for key in ["precision", "recall", "f1", "event_f1_iou03", "event_f1_iou05"]
            )
            + f" | {row['fp']} | {row['fn']} | "
            + fmt("onset_core_mae_days")
            + "/"
            + fmt("offset_core_mae_days")
            + " | "
            + fmt("stage_end_to_end_macro_f1")
            + " |"
        )
    lines.extend(
        [
            "",
            "```json",
            json.dumps(decision, ensure_ascii=False, indent=2),
            "```",
            "",
            "## 代价、统计与原图",
            "",
            f"复用{cost['original_centers_reused']}原编码，新增两图共{cost['new_view_centers']}中心、CPU{cost['seconds']:.3f}秒；训练/校准另列training-costs.csv。主推断仍每中心1图，实验两图成本不冒充部署两次复核或API工具次数。没有新增API或保留参考访问。",
            "",
            "此前TRAIN显示诊断阳性像素变化中位数10.924→17.014，阴性13.011→14.275；也放大噪声，不等于识别效果。所有八条验证记录已参与方法选择，两类数据与完整24验证不能混合比较。2000次整记录配对bootstrap仅描述开发波动，有效抽样次数逐项留存，非选择校正显著性；每族只有1条验证，不作族总体推断。边界均值可能有不同匹配事件集合，事件F1为1不保证边界精准。",
            "",
            "![全部方法日与事件对比](daily-event-comparison.png)",
            "",
            "以下两原图按主对原线性图的净纠错最小/最大选择；普通原位移30日间隔的14日块辅助值在raw-review-blocks.json。仅评分后错误分析，不改过程真值，不称独立专家：",
            "",
        ]
    )
    for cid in error_ids:
        lines.extend([f"![{cid}原三轴与活动]({cid}-error.png)", ""])
    lines.extend(
        [
            "## 复现",
            "",
            "按encode_landslide_chart_companding.py、train_landslide_chart_companding.py、verify_landslide_chart_companding.py和本脚本执行；各步骤绑定父数据、源文件、权重、选择配置、完整日历概率/活动/阶段。旧停止方向不被改写。",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        out / "analysis-plan.json",
        {
            "bootstrap_seed": 2026101253,
            "draws": 2000,
            "unit": "paired original record",
            "selection": "Minimum/maximum net correct vs linear-global, post-score only",
            "review_ids": error_ids,
            "net_by_case": net_by_case,
            "test_truth_opened": False,
            "labels_modified": False,
        },
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(out / "report.md", flush=True)


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
