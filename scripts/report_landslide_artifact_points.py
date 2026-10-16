"""Report artifact localization errors and frozen visual/numeric complementarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_artifact_points import load_banks, scores
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json


def report(data, run, replay, visual, out):
    verify_manifest(run, "freeze.json")
    receipt = read_json(replay / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["run_manifest_sha256"] != sha(
        run / "freeze.json"
    ):
        raise ValueError("Artifact replay receipt differs")
    protocol = read_json(run / "protocol.json")
    all_banks = load_banks(data, protocol["split"])
    banks = {cid: all_banks[cid] for cid in protocol["split"]["validation"]}
    predictions = {
        cid: np.array(value) for cid, value in read_json(run / "predictions.json").items()
    }
    probability = read_json(run / "probabilities.json")
    rows, errors = [], {}
    for cid, bank in banks.items():
        score = scores({cid: bank}, {cid: predictions[cid]}, {cid: probability[cid]})
        errors[cid] = (
            score["all"]["instrument_offset"]["fp"] + score["all"]["instrument_offset"]["fn"]
        )
        for group, tasks in score.items():
            for task, value in tasks.items():
                rows.append(dict(case_id=cid, group=group, task=task, **value))
    mixed_ids = [
        cid for cid, bank in banks.items() if np.any((bank["actual"] > 0) & (bank["activity"] == 1))
    ]
    selected = list(dict.fromkeys([max(errors, key=errors.get), max(mixed_ids, key=errors.get)]))
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "review-plan.json",
        dict(
            selection="most offset day FP+FN overall and among coexisting artifact/deformation records; first tie",
            review_ids=selected,
            labels_modified=False,
            test_truth_opened=False,
        ),
    )
    write_csv(out / "per-case.csv", rows)
    for cid in selected:
        case, _ = load_case_features(data, cid)
        values = np.array(
            [row if row is not None else [np.nan] * 3 for row in case.displacement_mm]
        )
        bank = banks[cid]
        figure, panels = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        for axis in range(3):
            panels[axis].plot(values[:, axis], linewidth=0.7)
            panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
        activity = bank["activity"].astype(float)
        activity[activity < 0] = np.nan
        panels[3].plot(activity, color="steelblue")
        panels[3].set_ylabel("Deformation\nprocess reference")
        for actual, marker, shift, label in (
            (bank["actual"], "o", 0, "Artifact reference"),
            (predictions[cid], "x", 0.1, "Artifact prediction"),
        ):
            for code, color in ((1, "darkorange"), (2, "purple")):
                days = np.flatnonzero(actual == code)
                panels[4].scatter(
                    days,
                    np.full(len(days), code + shift),
                    marker=marker,
                    color=color,
                    label=f"{label}: {('offset', 'spike')[code - 1]}",
                )
        panels[4].set_yticks([1, 2], ["offset", "spike"])
        panels[4].legend(fontsize=7, ncol=2)
        panels[4].set_xlabel("Original day; artifact presence does not veto deformation")
        figure.suptitle(cid + " | post-score developer error review; raw observations")
        figure.tight_layout()
        figure.savefig(out / (cid + "-raw-artifacts.png"), dpi=140)
        plt.close(figure)
    # An oracle diagnoses available correction, without turning truth into a router.
    outputs = read_json(visual / "predictions.json")
    available = outputs["global_numeric"].keys()
    complementarity = []
    for cid in available:
        truth = banks[cid]["activity"]
        left = np.array(outputs["global_numeric"][cid]["activity"])
        right = np.array(outputs["xgb_raw84"][cid]["activity"])
        known = truth >= 0
        jointly_covered = known & (left >= 0) & (right >= 0)
        left_error = left != truth
        right_error = right != truth
        complementarity.append(
            dict(
                case_id=cid,
                known_days=int(known.sum()),
                global_errors=int((known & left_error).sum()),
                numeric_errors=int((known & right_error).sum()),
                numeric_can_correct_global=int((known & left_error & ~right_error).sum()),
                global_can_correct_numeric=int((known & ~left_error & right_error).sum()),
                both_wrong=int((known & left_error & right_error).sum()),
                common_unknown=int((known & (left < 0) & (right < 0)).sum()),
                jointly_covered_days=int(jointly_covered.sum()),
                global_covered_errors=int((jointly_covered & left_error).sum()),
                numeric_covered_errors=int((jointly_covered & right_error).sum()),
                both_covered_wrong=int((jointly_covered & left_error & right_error).sum()),
                both_known_disagree=int(
                    (known & (left >= 0) & (right >= 0) & (left != right)).sum()
                ),
            )
        )
    write_csv(out / "complementarity.csv", complementarity)
    totals = {
        key: sum(row[key] for row in complementarity)
        for key in complementarity[0]
        if key != "case_id"
    }
    write_json(
        out / "complementarity-diagnostic.json",
        dict(
            totals=totals,
            scope="8 exposed pilot records; truth oracle upper bound only, no learned router or gain claim",
            labels_modified=False,
            test_truth_opened=False,
        ),
    )
    table = read_json(run / "scores.json")
    lines = [
        "# 原日尖峰与仪器偏移：独立数值基线",
        "",
        "Material Passport: academic-research-suite / experiment-agent；artifact-point-v1；模型/原始特征/52560概率值/17520逐日输出/全部评分重放VERIFIED。",
        "",
        "形变是主任务。这里补齐原日干扰存在与类型识别，两个任务可同时阳性；干扰输出不作为形变否决。中央14日DQ查询与原日定位不是相同终点，不将两者F1相减声称提升。",
        "",
        "## 方法及公平性",
        "",
        "同96批次中的48训练/24暴露验证，保留24未打开。原NEU日历取左右7日各至少4观测，21维数值证据区分孤立尖峰和永久转换；原日差MAD噪声仅缩放，不扣偏移、不补缺测。支持不足输出未知，已知阳性仍计FN。两深度2/4、120树、学习率0.05、seed20261007，各19阈值，共38机会；TRAIN逆类频权重均值1。按合并干扰日F1选首个最大值，选中depth4/threshold0.95；不按类型另救阈值。",
        "",
        "## 原日验证结果",
        "",
        "| 组 | 任务 | P | R | F1 | TP/FP/FN |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for group, tasks in table.items():
        for task, value in tasks.items():
            lines.append(
                f"| {group} | {task} | {value['precision']:.6f} | {value['recall']:.6f} | {value['f1']:.6f} | {value['tp']}/{value['fp']}/{value['fn']} |"
            )
    lines += [
        "",
        "全部验证有16822已知观测日、38干扰阳性（10偏移、28尖峰）；539个已知日因支撑不足保留未知，覆盖0.967959。TRAIN支持类计数32293正常/17偏移/46尖峰，偏移样本很少。全部原位移特征、拟合和网格约16.057秒，两树拟合和验证分别0.477/0.443秒；不是部署端到端延迟。",
        "",
        "合并干扰31TP而类型合计30TP，说明存在检到干扰但类型判错。偏移精确日F1仅0.4，不宣称质量识别已完善。该终点惩罚相邻日偏移定位，不将其未计分的近邻候选称正确，也不事后调容差救分。",
        "",
        "## 实际原图复核包",
        "",
        "以下按类型误差规则选取，只作评分后原因分析，不能改生成参考或称独立盲标。",
    ]
    for cid in selected:
        lines += ["", f"![{cid}原位移与干扰]({cid}-raw-artifacts.png)"]
    lines += [
        "",
        "## 视觉/数值互补预检",
        "",
        "同16/8原线性视觉数值与raw84输出在8条暴露验证的错误交集见complementarity.csv。",
        "",
        "```json",
        json.dumps(totals, ensure_ascii=False, indent=2),
        "```",
        "",
        "该诊断使用真值判断可纠错上限，不能充当可执行门控、训练证据或真实收益。只有训练侧跨记录预测能学出路由且优于固定融合/同信息数值时，才值得实现自适应机制。既有空间差分、asinh和硬干扰否决停止决定保持。",
        "",
        "## 复现",
        "",
        "run_landslide_artifact_points.py run/verify；report_landslide_artifact_points.py。输入与源、训练计划、网格、权重、模型、概率、输出、费用、评分均保留哈希。",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(totals, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "replay", "visual", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(
        args.data.resolve(),
        args.run.resolve(),
        args.replay.resolve(),
        args.visual.resolve(),
        args.out.resolve(),
    )
