"""Save unified pilot metrics, family scores and post-score raw-record figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_motion_distribution import MAIN
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores


def report(data, encoded, run, verified, out):
    verify_manifest(run, "freeze.json")
    receipt = read_json(verified / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["run_manifest_sha256"] != sha(
        run / "freeze.json"
    ):
        raise ValueError("Matching selected-output replay is required")
    out.mkdir(parents=True, exist_ok=False)
    protocol = read_json(run / "protocol.json")
    summary = read_json(run / "summary.json")
    results = read_json(run / "results.json")
    predictions = read_json(run / "predictions.json")
    decision = read_json(run / "decision.json")
    families = {
        row["case_id"]: row["truth"]["family"]
        for row in read_json(data / "training-generation-ledger.json")
        if row["case_id"] in protocol["split"]["validation"]
    }
    rows = []
    for name in predictions:
        for family in sorted(set(families.values())):
            score = aggregate_scores(
                [results[name]["per_case"][cid] for cid in families if families[cid] == family]
            )
            rows.append(
                dict(method=name, family=family, **score["daily"], event_f1=score["events"]["f1"])
            )
    write_csv(out / "family-summary.csv", rows)
    case_rows = []
    for cid in families:
        main = results[MAIN]["per_case"][cid]["daily"]
        control = results["duplicate_motion_global"]["per_case"][cid]["daily"]
        case_rows.append(
            dict(
                case_id=cid,
                family=families[cid],
                main_fp=main["fp"],
                main_fn=main["fn"],
                duplicate_fp=control["fp"],
                duplicate_fn=control["fn"],
                net_correct=control["fp"] + control["fn"] - main["fp"] - main["fn"],
            )
        )
    write_csv(out / "paired-case-errors.csv", case_rows)
    ids = list(
        dict.fromkeys(
            [
                min(case_rows, key=lambda row: row["net_correct"])["case_id"],
                max(case_rows, key=lambda row: row["net_correct"])["case_id"],
            ]
        )
    )
    reference = {cid: [] for cid in ids}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in reference:
                reference[row["case_id"]].append(int(row["activity_label"]))
    review_methods = (
        MAIN,
        "duplicate_motion_global",
        "permuted_distribution_numeric",
        "global_numeric",
        decision["best_numeric"],
    )
    for cid in ids:
        case = load_case_features(data, cid)[0]
        raw = np.array([v if v is not None else [np.nan] * 3 for v in case.displacement_mm])
        figure, panels = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        for axis in range(3):
            panels[axis].plot(raw[:, axis], linewidth=0.7)
            panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
        lines = [("Process reference", reference[cid])] + [
            (name, predictions[name][cid]["activity"]) for name in review_methods
        ]
        for index, (_, value) in enumerate(lines):
            activity = np.array(value, dtype=float)
            activity[activity < 0] = np.nan
            panels[3].plot(activity * 0.7 + index, linewidth=1)
        panels[3].set_yticks(np.arange(len(lines)), [name for name, _ in lines])
        panels[3].set_xlabel("Original day; post-score explanation only")
        figure.suptitle(cid)
        figure.tight_layout()
        figure.savefig(out / (cid + "-raw-review.png"), dpi=140)
        figure.savefig(out / (cid + "-raw-review.pdf"))
        plt.close(figure)
    write_json(
        out / "analysis-plan.json",
        dict(
            review_ids=ids,
            selection="Worst/best net known-day correction vs identical duplicate, post-score explanation",
            reference_modified=False,
            test_truth_opened=False,
        ),
    )
    ordered = sorted(summary, key=lambda row: row["f1"], reverse=True)
    figure, panels = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
    for panel, metric in zip(panels, ("f1", "event_f1_iou03")):
        panel.barh(
            [row["method"] for row in ordered],
            [row[metric] for row in ordered],
            color=["#b55946" if row["method"] == MAIN else "#527f98" for row in ordered],
        )
        panel.set_xlim(0, 1)
        panel.set_title(metric)
        panel.grid(axis="x", alpha=0.2)
    panels[0].invert_yaxis()
    figure.suptitle("Motion-distribution visual pilot:16TRAIN/8exposedVALIDATION")
    figure.tight_layout()
    figure.savefig(out / "comparison.png", dpi=140)
    figure.savefig(out / "comparison.pdf")
    plt.close(figure)
    table = [
        "| 方法 | P | R | 日F1 | 事件F1 .3/.5 | FP/FN | 起止MAE（日） | 活动内阶段F1 |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for row in ordered:
        table.append(
            f"| {row['method']} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['event_f1_iou03']:.6f}/{row['event_f1_iou05']:.6f} | {row['fp']}/{row['fn']} | {row['onset_core_mae_days']:.3f}/{row['offset_core_mae_days']:.3f} | {row['stage_internal_macro_f1']:.6f} |"
        )
    checks = "\n".join(
        f"- {name}: {'成立' if value else '不成立'}。" for name, value in decision["checks"].items()
    )
    cost = read_json(encoded / "encoding-cost.json")
    text = f"""# 运动变化分布辅助视图：固定开发小试

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；Version Label: MD-VN-pilot-v1。
- Verification Status: 原数值／mask／标签逐值不变、24记录辅助信息与TRAIN缩放重构、96辅助编码抽样重算、10选中模型58400概率及21方法168输出精确重放。
- Reference: 非独立受控过程参考；保留24输入／参考未打开，开发者原图复核单独记录。

候选MD-VN（Motion-Distribution Visual-Numeric Recognition，运动分布视觉数值识别）。在原NEU图和18原数值的基础上，增加原13块导出的12变化量与11邻接方向一致性图。只改变提供给冻结视觉分支的物理证据组织，不增加Agent框架、语言调用或阶段机制。三轴变化用原noise units，缺测槽保留，零向量的方向未知；无硬运动阈值、不插值。通用视觉数值拼接不是创新，是否有本项目收益由对照决定。

固定八族16／8小试：原始stride3训练中心、同标签次序及TRAIN缩放；六新神经臂、四树各38机会，共380新配置。旧11方法原360机会保持。所有阶段用同16TRAIN阶段头，共同14阳性观测解码。树拥有原18＋46辅助＋52完整块profile，另一强树再获得366原图描述符；新增全局／查询分布树重新在16TRAIN拟合。未把48TRAIN的.844数值结果冒充小试成绩。

主／重复／打乱均1060维；去数值1024，辅助单图＋数值548，同信息数值MLP232，树116／482／39／45。打乱保留完整值／缺测multiset，但同时去掉两栏原时序及关系、改变为rank布局，解释不能缩成纯时间消融。重复组重新拟合精确复现原duplicate_global_numeric，控制维度和原模型实现。

{chr(10).join(table)}

七事前条件{sum(decision["checks"].values())}/7成立，**{decision["status"]}**。对重复／打乱净纠错{decision["net_correct"]["duplicate"]}/{decision["net_correct"]["permuted"]}日；最强数值{decision["best_numeric"]}。逐族结果与逐记录净纠错另存，8条验证不支持现场总体推断。

{checks}

新增视觉编码{cost["new_visual_encodings"]}个，编码及准备{cost["seconds"]:.3f}秒；0新独立记录／API，不能把窗口当独立样本或0API当零成本。20模型训练分段时间及参数规模见training-costs.csv。仅选中模型推理重放，未重训所有20模型；96/全部辅助编码抽样数明确，不称全部新编码重新计算。

第一次编码因CSV进度文件x创建方式失败，源代码和回执保留。修正逐记录进度文件后在v2完整重启，图像定义、模型、seed和标签没有改动；不是根据评分救分。训练前协议及完整网格保存。

原图事后按对重复净纠错最差／最好选{", ".join(ids)}，只解释机制，实际看图记录另存session，不修改参考。若停止不重选seed／尺度／阈值；若继续只扩48／24开发验证，仍需未见新记录确认才形成正式效果结论。
"""
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "evidence-bindings.json",
        {
            str(p.resolve()): sha(p)
            for p in (
                encoded / "encoding-freeze.json",
                run / "freeze.json",
                verified / "freeze.json",
            )
        },
    )
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(
        dict(
            status=decision["status"],
            checks=f"{sum(decision['checks'].values())}/7",
            review_ids=ids,
        ),
        flush=True,
    )


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
