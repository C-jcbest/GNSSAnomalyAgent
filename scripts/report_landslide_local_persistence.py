"""Report a stopped local intervention without promoting its better controls."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json
from train_landslide_local_persistence import MAIN

from gnss_sim.artifacts import sha, write_json


def report(data, pairs, run, verified, out):
    verify_manifest(run, "freeze.json")
    receipt = read_json(verified / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["calibration_manifest_sha256"] != sha(
        run / "freeze.json"
    ):
        raise ValueError("Replay is not bound to this calibration")
    protocol = read_json(run / "protocol.json")
    results = read_json(run / "results.json")
    summary = read_json(run / "summary.json")
    predictions = read_json(run / "predictions.json")
    decision = read_json(run / "decision.json")
    costs = read_json(pairs / "preparation-cost.json")
    reference = {cid: [] for cid in protocol["split"]["validation"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in reference:
                reference[row["case_id"]].append(int(row["activity_label"]))
    out.mkdir(parents=True, exist_ok=False)
    augmentation = "local_visual_numeric_augmentation"
    case_rows = []
    for cid in reference:
        main = results[MAIN]["per_case"][cid]["daily"]
        control = results[augmentation]["per_case"][cid]["daily"]
        case_rows.append(
            dict(
                case_id=cid,
                main_fp=main["fp"],
                main_fn=main["fn"],
                augmentation_fp=control["fp"],
                augmentation_fn=control["fn"],
                net_vs_augmentation=control["fp"] + control["fn"] - main["fp"] - main["fn"],
            )
        )
    write_csv(out / "paired-case-errors.csv", case_rows)
    review_ids = list(
        dict.fromkeys(
            [
                min(case_rows, key=lambda row: row["net_vs_augmentation"])["case_id"],
                max(case_rows, key=lambda row: row["net_vs_augmentation"])["case_id"],
            ]
        )
    )
    write_json(
        out / "analysis-plan.json",
        dict(
            review_ids=review_ids,
            selection="First least/greatest net correct original record versus local augmentation, post-score explanation only",
            reference_modified=False,
            test_truth_opened=False,
        ),
    )
    for cid in review_ids:
        case = load_case_features(data, cid)[0]
        raw = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        figure, panels = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        for axis in range(3):
            panels[axis].plot(raw[:, axis], linewidth=0.7)
            panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
        lines = [("Process reference", np.array(reference[cid]))] + [
            (name, np.array(predictions[name][cid]["activity"]))
            for name in (
                MAIN,
                augmentation,
                "global_visual_numeric_pair",
                "visual_numeric",
                "xgb_raw84",
            )
        ]
        for index, (_, value) in enumerate(lines):
            curve = value.astype(float)
            curve[curve < 0] = np.nan
            panels[3].plot(np.arange(len(curve)), curve * 0.7 + index, linewidth=1)
        panels[3].set_yticks(np.arange(len(lines)), [name for name, _ in lines])
        panels[3].set_xlabel("Original day; missing and unsupported decisions stay unknown")
        figure.suptitle(cid + " | post-score raw observation review")
        figure.tight_layout()
        figure.savefig(out / (cid + "-raw-review.png"), dpi=140)
        figure.savefig(out / (cid + "-raw-review.pdf"))
        plt.close(figure)
    ordered = sorted(summary, key=lambda row: row["f1"], reverse=True)
    figure, panels = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
    for panel, metric, title in zip(
        panels, ("f1", "event_f1_iou03"), ("Daily deformation F1", "Event F1 (IoU0.3)")
    ):
        panel.barh(
            [row["method"] for row in ordered],
            [row[metric] for row in ordered],
            color=["#ae5344" if row["method"] == MAIN else "#527f98" for row in ordered],
        )
        panel.set_xlim(0, 1)
        panel.set_title(title)
        panel.grid(axis="x", alpha=0.2)
    panels[0].invert_yaxis()
    figure.suptitle("Local persistence pairing:48train/24exposed validation")
    figure.tight_layout()
    figure.savefig(out / "comparison.png", dpi=150)
    figure.savefig(out / "comparison.pdf")
    plt.close(figure)
    rows = [
        "| 方法 | P | R | 日F1 | 事件F1 .3/.5 | FP/FN | 起止MAE（日） | 内部阶段F1 |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for row in ordered:
        rows.append(
            f"| {row['method']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['event_f1_iou03']:.6f}/{row['event_f1_iou05']:.6f} | {row['fp']}/{row['fn']} | {row['onset_core_mae_days']:.3f}/{row['offset_core_mae_days']:.3f} | {row['stage_internal_macro_f1']:.6f} |"
        )
    text = f"""# 保留上下文的局部持续性配对：完整开发负结果

CP-PP（Context-Preserving Local Persistence Pairing）仅在训练侧使用受控潜在形变，把目标中心两侧21日的运动替换为同净位移偏移，偏移日为中心±14，保留区间以外的原形变、全部原误差与缺测。目标14日查询的潜在运动归零，但其他窗口活动保留。推理仍只使用原观测图和原数值，不处理评价潜在真值，不增加语言或工具调用。

完整48训练／24已暴露验证；{costs["pairs"]}组配对来自{costs["parents"]}条训练记录，五形态全部参与，所有合格中心保留。它们不是新独立记录。全窗配对对照使用相同父记录、中心和偏移日，偏移幅度因干预跨度不同而不同，不能称严格隔离上下文的单因素对照。CAPS已有共享上下文配对，通用配对、背景保留与softplus排序不称首创。

七神经、两同增广树各38配置，共342新配置；13原对照使用相同48／24、原编码和数值缩放、共同新缺测持续性及阶段头，455原校准机会保留。树同增广使用30659原已知支撑训练日＋127负例，神经使用10187原stride3中心＋127负例，区别明确。主与消融都独立按共同日F1＋事件F1选首个最大值，不选择最好消融改称主方法。

{chr(10).join(rows)}

预设六性能条件仅{sum(decision["performance_checks"].values())}/6成立：混合召回与相对纯数值配对的有限视觉价值；同局部增广净{decision["net_vs_augmentation"]}日、同全窗净{decision["net_vs_global"]}日。纯干扰FP主{decision["nuisance_fp"][MAIN]}，同增广{decision["nuisance_fp"][augmentation]}；混合召回主{decision["mixed_recall"][MAIN]:.6f}，同增广{decision["mixed_recall"][augmentation]:.6f}。**STOP_LOCAL_PERSISTENCE**。不调整半宽、offset位置、loss系数、seed或阈值救此版本，不打开保留参考，不用同增广强树的改善冒充VLM创新。

127局部／全窗反事实观测、潜在值、数值、图、编码已重构；局部外侧运动及观测字节、原误差／缺测和目标零增量核查通过。九选中模型、157680概率、22方法／528输出及统一活动／阶段／评分精确重放。验证没有重训全部18模型；原13方法通过原绑定验证与输出逐值保持，不冒称重新执行了455原网格。

配对预处理总计{costs["total_seconds"]:.3f}秒，其中254新增视觉编码{costs["encoder_seconds"]:.3f}秒；训练及校准分段计时在training-costs.csv。不是重复端到端部署测量，0API不表示图像编码免费。支持与缺测未知不计作正确阴性；A/S/D仍由共同数值头在活动内输出，不是新视觉阶段贡献。

原图检查对象为{", ".join(review_ids)}，按对同增广净纠错的最差／最佳记录事后选择，只用于错误解释。实际观察另存session，不修改过程标签，不称独立盲标。本轮支持停止该干预，不能据主优于打乱或纯数值局部配对推导其优于普通增广或强数值。

复现使用prepare_landslide_local_persistence.py、train_landslide_local_persistence.py、verify_landslide_local_persistence.py及report_landslide_local_persistence.py。全部输出版本化保留，24保留参考仍未打开。
"""
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "evidence-bindings.json",
        {
            str(path.resolve()): sha(path)
            for path in (
                run / "freeze.json",
                pairs / "encoding-freeze.json",
                verified / "freeze.json",
            )
        },
    )
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(
        dict(
            status=decision["status"],
            performance_conditions=f"{sum(decision['performance_checks'].values())}/6",
            review_ids=review_ids,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "pairs", "run", "verified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(
        args.data.resolve(),
        args.pairs.resolve(),
        args.run.resolve(),
        args.verified.resolve(),
        args.out.resolve(),
    )
