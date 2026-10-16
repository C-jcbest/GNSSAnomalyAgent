"""Report the stopped numerical probe and pre-score distribution-view examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_spatial_readout import pilot_split, verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_motion_distribution_view import (
    distribution_view_inputs,
    render_distribution_view,
)


def report(data, run, verified, out):
    verify_manifest(run, "freeze.json")
    receipt = read_json(verified / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["run_manifest_sha256"] != sha(
        run / "freeze.json"
    ):
        raise ValueError("A matching successful replay is required")
    out.mkdir(parents=True, exist_ok=False)
    summary = read_json(run / "summary.json")
    results = read_json(run / "results.json")
    decision = read_json(run / "decision.json")
    rows = []
    for cid in results["query_motion_mass"]["per_case"]:
        main = results["query_motion_mass"]["per_case"][cid]["daily"]
        control = results["global_motion_mass"]["per_case"][cid]["daily"]
        rows.append(
            dict(
                case_id=cid,
                main_fp=main["fp"],
                main_fn=main["fn"],
                control_fp=control["fp"],
                control_fn=control["fn"],
                net_correct=control["fp"] + control["fn"] - main["fp"] - main["fn"],
            )
        )
    write_csv(out / "paired-case-errors.csv", rows)
    ids = list(
        dict.fromkeys(
            [
                min(rows, key=lambda row: row["net_correct"])["case_id"],
                max(rows, key=lambda row: row["net_correct"])["case_id"],
            ]
        )
    )
    predictions = read_json(run / "predictions.json")
    for cid in ids:
        case = load_case_features(data, cid)[0]
        raw = np.array([v if v is not None else [np.nan] * 3 for v in case.displacement_mm])
        figure, panels = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        for axis in range(3):
            panels[axis].plot(raw[:, axis], linewidth=0.7)
            panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
        methods = ("mass_raw84", "global_motion_mass", "query_motion_mass", "visual_numeric")
        for index, name in enumerate(methods):
            activity = np.array(predictions[name][cid]["activity"], dtype=float)
            activity[activity < 0] = np.nan
            panels[3].plot(activity * 0.7 + index, label=name, linewidth=1)
        panels[3].set_yticks(np.arange(len(methods)), methods)
        panels[3].set_xlabel("Original day; post-score review, reference not rewritten")
        figure.suptitle(cid)
        figure.tight_layout()
        figure.savefig(out / (cid + "-raw-review.png"), dpi=140)
        plt.close(figure)
    write_json(
        out / "analysis-plan.json",
        dict(
            review_ids=ids,
            selection="Worst/best net known-day correction vs global mass, post-score explanation",
            reference_modified=False,
            test_truth_opened=False,
        ),
    )
    # These examples use fixed exposed records and day365 before any new view
    # experiment. They are not selected by the proposed view's future score.
    example_rows = []
    ids = pilot_split(data)["validation"]
    figure, panels = plt.subplots(4, 4, figsize=(12, 12))
    for index, cid in enumerate(ids):
        case, features = load_case_features(data, cid)
        raw = np.array([v if v is not None else [np.nan] * 3 for v in case.displacement_mm])
        view = distribution_view_inputs(raw, features.noise)
        for offset, permuted in enumerate((False, True)):
            image = render_distribution_view(
                view["magnitude"][365], view["agreement"][365], permuted
            )
            image.save(out / f"{cid}-day365-{'permuted' if permuted else 'ordered'}.png")
            panel = panels.flat[2 * index + offset]
            panel.imshow(image)
            panel.set_title(cid + (" rank" if permuted else " day365"), fontsize=9)
            panel.axis("off")
        example_rows.append(
            dict(
                case_id=cid,
                day=365,
                magnitude=[float(v) if np.isfinite(v) else None for v in view["magnitude"][365]],
                agreement=[float(v) if np.isfinite(v) else None for v in view["agreement"][365]],
            )
        )
    figure.tight_layout()
    figure.savefig(out / "distribution-view-prescore-examples.png", dpi=120)
    plt.close(figure)
    write_json(out / "prescore-view-values.json", example_rows)
    table = [
        "| 方法 | P | R | 日F1 | 事件F1 .3/.5 | FP/FN | 起止MAE（日） | 活动内阶段F1 |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for row in summary:
        table.append(
            f"| {row['method']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['event_f1_iou03']:.6f}/{row['event_f1_iou05']:.6f} | {row['fp']}/{row['fn']} | {row['onset_core_mae_days']:.3f}/{row['offset_core_mae_days']:.3f} | {row['stage_internal_macro_f1']:.6f} |"
        )
    text = (
        """# 运动变化分布数值探针：信息收益与停止结论

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；Origin Mode: run/validate。
- Version Label: motion-mass-v1；Verification Status: VERIFIED（原证据、选中模型概率、活动／阶段及评分精确重放）；暴露开发验证，不是独立领域参考。

48TRAIN／24VALIDATION、完整记录划分，保留24输入及参考未读。使用原13个14日块，三尺度5／7／13块，只比较相邻可用块；净变化、总路径、coherence、最大变化占比、effective_edges、去最大变化净量及支持比例，共21全局描述符；目标附近占比与运动质量位置另6项。当前缺测NaN，不插值，不增加活动支持，不设硬运动阈值。52维原块profile给强对照。

五树各深2／4×19阈值，190新配置；全部30659合法训练日、120树、lr.05及原seed，沿用共同14阳性观测／最多7原缺测连接解码与同48TRAIN阶段头。raw84重新拟合精确复现原方法。13原方法／455历史配置继续报告。

"""
        + "\n".join(table)
        + f"""

预设五条件{sum(decision["checks"].values())}/5成立，{decision["status"]}。查询分布比全局多75TP也多76FP，净−1日；F1增加不能代替净纠错判据。与raw84比较存在明显开发信息收益：日F1 .783572→.844167，纯干扰FP15→0；混合召回.634667→.738667。但这个数值探针不能证明Agent/VLM贡献，不能把全局对照更名为创新，也不修改停止版本的尺度／seed／阈值。

72原记录证据、5选中模型87600概率、18方法432输出精确重放。第一次校验因JSON方法呈现次序失败，保留失败源码／回执；修正校验为按method比较完整结果集合，未改变模型和指标。未重训全部10模型，不冒称训练字节复现。

后续视觉小试单独冻结：辅助视图直接画12变化量和11方向一致性。八族固定开发例日365已输出有序／rank图用于可读性检查；不是新独立记录。其数值对照获得同视图值和完整块profile，避免比较不公平。
"""
    )
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "evidence-bindings.json",
        {str(p.resolve()): sha(p) for p in (run / "freeze.json", verified / "freeze.json")},
    )
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(
        dict(
            status=decision["status"],
            review_ids=read_json(out / "analysis-plan.json")["review_ids"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "verified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(args.data.resolve(), args.run.resolve(), args.verified.resolve(), args.out.resolve())
