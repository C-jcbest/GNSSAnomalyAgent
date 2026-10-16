"""Report shared-coordinate rendering and matched recognition results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json
from train_landslide_shared_frame import MAIN

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
    augmentation = "shared_visual_numeric_augmentation"
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
            selection="First least/greatest net correct original record versus same-supervision augmentation, post-score explanation only",
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
                "independent_visual_numeric_pair",
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
    figure.suptitle("Shared-coordinate pairing:48train/24exposed validation")
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
    text = f"""# 共享坐标持续性配对：机制隔离与完整开发负结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent。
- Origin Date: 2026-10-12；Version Label: SC-PP-v1。
- Verification Status: 训练侧绘图及选中输出已重放；效果为暴露开发集负结果。

候选名SC-PP（Shared-Coordinate Persistence Pairing，共享坐标持续性配对）。基于已有冻结视觉分支＋数值活动头及训练侧局部运动／偏移反事实，新增的唯一机制是配对图共享参考、纵轴跨度和原噪声下限，让局部替换以外的观测在像素上也严格相同。原CP-PP仍停止；通用配对、共享坐标与背景锚定不称首创。本轮不是新Agent运行时，没有语言调用或新增工具链。

依据来自127组训练侧配对诊断：原观测外围逐值不变，但独立归一化时127组全部发生外围像素变化，中位外围差异占比0.699743，中位外围平均位移0.985536像素。这证明输入的非局部变化，**不证明上一版失败的原因**。新跨度取双方相对正图中位参考的最大绝对值×2及原六噪声单位的较大者，避免负图被裁切；共享正图也需重新编码。

完整48训练／24暴露验证，24保留参考未读取。127组来自28训练父记录，所有原合格中心、42日物理干预及±14日偏移安排不变；不是新独立记录。七神经方法获得**完全相同监督池及样本次序**：10187原中心＋127共享正图＋127共享负图＋127独立坐标负图＝10568。共享配对和独立坐标配对只改变辅助排序使用的正负图；独立正图是已在原监督池中的中心。去配对、打乱、去视觉、去数值与纯数值增广分别隔离机制。两强树使用30659原合法日＋同381增广＝31040。数值重复曝光显式保留，所有神经同原TRAIN缩放，不能把新增正图曝光归于配对收益。

七神经、两树各38配置，共342新配置；13原方法同48／24、原日历与原观测信息、同缺测持续性及共同阶段头，455历史校准配置不改。固定12epoch、seed20261020、BCE及排序weight1／margin1，不改变跨度、seed、损失权重或网格救分。

{chr(10).join(rows)}

预设六性能条件仅{sum(decision["performance_checks"].values())}/6成立。对同监督增广净{decision["net_vs_augmentation"]}日；对独立坐标配对净{decision["net_vs_independent"]}日，日F1几乎相同且略低，不能称可靠收益。纯干扰误报主{decision["nuisance_fp"][MAIN]}日、同增广{decision["nuisance_fp"][augmentation]}日；混合召回主{decision["mixed_recall"][MAIN]:.6f}、同增广{decision["mixed_recall"][augmentation]:.6f}。**STOP_SHARED_FRAME**。只得到部分混合召回增加，没有总体活动和事件改进，不打开保留集。

127组共享图外侧像素全部精确相同，缺测mask保留且所有有效槽trace质量为1，无裁切。物理反事实重新由训练侧潜在值构造；数值重构沿用并绑定原完整重放。254共享编码、9选中模型157680概率、22方法528记录输出及统一活动／阶段评分精确重放；未重训全部18模型，不冒称全部训练字节复现。冻结模型初始化后严格加载视觉权重，初始随机网络warning不表示随机视觉编码。

新增254视觉编码用时{costs["encoder_seconds"]:.3f}秒，预处理总计{costs["total_seconds"]:.3f}秒；18模型训练／校准分段计时另存training-costs.csv。0API不表示编码无成本，也不代表真实部署吞吐；活动内部A/S/D仍是同数值头及不同活动门控，不是新增视觉阶段贡献。

事后按对增广净纠错最差／最好选择原图{", ".join(review_ids)}，只用于机制失败解释，实际看图另存session，不修改标签、不称独立专家。原过程标签能检查受控注入支持，不能证明每个弱尾段在观测中可辨。

复现入口：prepare_landslide_shared_frame.py、train_landslide_shared_frame.py、verify_landslide_shared_frame.py、report_landslide_shared_frame.py。原数据、物理配对和所有历史停止结果保持冻结；本轮结论是像素上下文一致性得到实现验证，但对当前冻结视觉活动头未产生有效识别增益。
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
