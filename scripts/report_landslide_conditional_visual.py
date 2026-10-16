"""Report conditional-visual errors and how much embedding variation was predicted."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_conditional_visual import MAIN, motion_numbers
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import load_case_features, read_json
from threadpoolctl import threadpool_limits

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores
from gnss_sim.landslide_visual_invariance import numeric_input


def report(data, encoded, parent, run, verified, out):
    verify_manifest(run, "freeze.json")
    receipt = read_json(verified / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["run_manifest_sha256"] != sha(
        run / "freeze.json"
    ):
        raise ValueError("A matching complete selected-output replay is required")
    out.mkdir(parents=True, exist_ok=False)
    protocol = read_json(run / "protocol.json")
    results = read_json(run / "results.json")
    summary = read_json(run / "summary.json")
    predictions = read_json(run / "predictions.json")
    decision = read_json(run / "decision.json")
    banks = load_banks(encoded, protocol["split"])
    scaling = read_json(run / "numeric-scaling.json")
    expectation = read_json(run / "visual-expectation.json")
    shuffled = read_json(run / "shuffled-expectation.json")
    diagnostics = []
    started = time.perf_counter()
    for group, ids in protocol["split"].items():
        for cid in ids:
            bank = banks[cid]
            with np.load(parent / "evidence" / (cid + ".npz")) as archive:
                evidence = {key: archive[key] for key in archive.files}
            selected = bank["selected"]
            numbers = numeric_input(motion_numbers(bank, evidence), scaling)[selected]
            visual = bank["embeddings"][selected].astype(float)
            centered = visual - np.array(expectation["visual_mean"])
            center_energy = np.sum(centered**2)
            for name, fitted in (("paired", expectation), ("shuffled", shuffled)):
                residual = (
                    visual
                    - numbers @ np.array(fitted["coefficient"]).T
                    - np.array(fitted["intercept"])
                )
                remaining = np.sum(residual**2)
                diagnostics.append(
                    dict(
                        group=group,
                        case_id=cid,
                        mapping=name,
                        queries=int(selected.sum()),
                        relative_to_fixed_train_mean_r2=float(1 - remaining / center_energy)
                        if center_energy > 0
                        else None,
                        median_residual_norm=float(np.median(np.linalg.norm(residual, axis=1))),
                        median_centered_norm=float(np.median(np.linalg.norm(centered, axis=1))),
                    )
                )
    write_csv(out / "embedding-diagnostics.csv", diagnostics)
    write_json(
        out / "diagnostic-cost.json",
        dict(
            seconds=time.perf_counter() - started,
            scope="Post-score diagnostic only, not inference benchmark",
            training_mean_reference=True,
        ),
    )
    families = {
        row["case_id"]: row["truth"]["family"]
        for row in read_json(data / "training-generation-ledger.json")
        if row["case_id"] in protocol["split"]["validation"]
    }
    family_rows = []
    for name in predictions:
        for family in sorted(set(families.values())):
            score = aggregate_scores(
                [results[name]["per_case"][cid] for cid in families if families[cid] == family]
            )
            family_rows.append(
                dict(method=name, family=family, **score["daily"], event_f1=score["events"]["f1"])
            )
    write_csv(out / "family-summary.csv", family_rows)
    case_rows = []
    for cid in families:
        main = results[MAIN]["per_case"][cid]["daily"]
        control = results["raw_visual_motion"]["per_case"][cid]["daily"]
        case_rows.append(
            dict(
                case_id=cid,
                family=families[cid],
                main_fp=main["fp"],
                main_fn=main["fn"],
                raw_fp=control["fp"],
                raw_fn=control["fn"],
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
    write_json(
        out / "analysis-plan.json",
        dict(
            review_ids=ids,
            selection="First worst/best net known-day correction vs original same-information raw visual fusion; post-score explanation",
            test_truth_opened=False,
            reference_modified=False,
        ),
    )
    methods = (
        MAIN,
        "raw_visual_motion",
        "centered_visual_motion",
        "motion_numeric_mlp",
        "refit_global_motion_mass",
        "query_motion_mass",
    )
    for cid in ids:
        case = load_case_features(data, cid)[0]
        raw = np.array([v if v is not None else [np.nan] * 3 for v in case.displacement_mm])
        figure, panels = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        for axis in range(3):
            panels[axis].plot(raw[:, axis], linewidth=0.7)
            panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
        for index, name in enumerate(methods):
            activity = np.array(predictions[name][cid]["activity"], dtype=float)
            activity[activity < 0] = np.nan
            panels[3].plot(activity * 0.7 + index, linewidth=1)
        panels[3].set_yticks(np.arange(len(methods)), methods)
        panels[3].set_xlabel(
            "Original day; post-score review, unknown/missing decisions not filled"
        )
        figure.suptitle(cid)
        figure.tight_layout()
        figure.savefig(out / (cid + "-raw-review.png"), dpi=140)
        figure.savefig(out / (cid + "-raw-review.pdf"))
        plt.close(figure)
    ordered = sorted(summary, key=lambda row: row["f1"], reverse=True)
    figure, panels = plt.subplots(1, 2, figsize=(14, 11), sharey=True)
    for panel, metric in zip(panels, ("f1", "event_f1_iou03")):
        panel.barh(
            [row["method"] for row in ordered],
            [row[metric] for row in ordered],
            color=["#ae5344" if row["method"] == MAIN else "#527f98" for row in ordered],
        )
        panel.set_xlim(0, 1)
        panel.set_title(metric)
    panels[0].invert_yaxis()
    figure.suptitle("Numerical-conditioned visual residuals:48TRAIN/24exposedVALIDATION")
    figure.tight_layout()
    figure.savefig(out / "comparison.png", dpi=140)
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
    checks = "\n".join(
        f"- {name}: {'成立' if value else '不成立'}。" for name, value in decision["checks"].items()
    )
    text = f"""# NCVR：运动证据条件化视觉残差的完整开发负结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；Version Label: NCVR-v1。
- Verification Status: TRAIN缩放、两表示期望重新拟合精确一致；11选中模型192720概率、29方法696输出及全部活动／阶段／边界评分精确重放。
- Reference: 暴露开发过程参考；保留24内容未加载，实际AI看图是开发者分析，非独立领域标签。

NCVR（Numerical-Conditioned Visual Residuals，数值证据条件化视觉残差）从原冻结512视觉向量中减去由39运动数值及其缺测flags可线性预测的分量，并单位归一化。活动头同时读取该512残差与78数值，590维。期望回归用同10187合法TRAIN中心和固定Ridge alpha1，不读活动／干扰标签，不对VALIDATION拟合，不生成新图。不能把可预测分量全部称为仪器背景，也不能把回归、残差或去冗余称为通用首创；这是GNSS具体证据利用假设。

原向量直接拼接、仅TRAIN均值中心化并归一化、打乱对应条件回归、去归一化、去数值与纯数值MLP进行消融。五主要融合臂均590维、相同宽16／32、seed、12epoch、样本次序与class balancing。七神经／四树各38机会，418新配置；18原方法保持，其中13历史455及5数值探针190机会。数值树获得30659合法TRAIN日；带冻结视觉向量的两树只使用实际编码10187中心，避免未编码的零向量被当真视觉。原图数值／完整块profile树457维，带嵌入树551／969维；不故意削弱数值信息。

{chr(10).join(rows)}

七预设条件{sum(decision["checks"].values())}/7成立，**{decision["status"]}**。对原拼接／仅去均值／打乱净纠错{decision["net_correct"]["raw_visual_motion"]}/{decision["net_correct"]["centered_visual_motion"]}/{decision["net_correct"]["shuffled_visual_motion"]}日。虽然高于去均值与打乱，但主日.793321／事件.823529低于原拼接.805651／.848485；数值MLP.810182、全局分布树.837166、既有查询树.844167也更高。纯干扰FP41对原拼接43和仅去均值41，混合召回.584低于原拼接.604。部分对照改善不足以成立整体视觉收益，不更名最强数值或普通树融合为创新，不调alpha／seed／loss救分。

{checks}

原全局分布数值树重新拟合精确复现其先前输出；共同14阳性观测、<=7原缺测连接及48TRAIN阶段头不变。重放重新拟合无标签表示回归，不重训22活动分类模型、不新增视觉编码，协议／418网格／模型／概率／费用完整冻结。

表征诊断是相对固定TRAIN视觉均值的R²和残差范数，逐原记录保存，允许VALIDATION R²为负；不是因果背景分解、跨模态独立性证明或分类收益。事后选择原图{", ".join(ids)}解释错误，实际看图另存session。不能仅凭表示可预测量降低就宣布异常识别改善。

官方新增检索与ResComEmb／VTBench原文阅读边界单独保存。ResComEmb压缩多粒度视觉token，VTBench研究图／原序列融合，两者均说明去冗余及普通融合已有先例；未运行作者模型，不能将作者成绩列为共同GNSS基线。
"""
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "evidence-bindings.json",
        {
            str(p.resolve()): sha(p)
            for p in (
                run / "freeze.json",
                verified / "freeze.json",
                encoded / "encoding-freeze.json",
                parent / "freeze.json",
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
    for name in ("data", "encoded", "parent", "run", "verified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        report(
            args.data.resolve(),
            args.encoded.resolve(),
            args.parent.resolve(),
            args.run.resolve(),
            args.verified.resolve(),
            args.out.resolve(),
        )
