"""Diagnose frozen query-shape learning without changing models or references."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import average_precision_score
from write_landslide_distribution_manuscript import score_table

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores, runs


def shape_metrics(actual, predicted, available):
    errors = predicted - actual
    per_axis = []
    for axis in range(actual.shape[1]):
        valid = available[:, axis]
        observed = actual[valid, axis]
        residual = errors[valid, axis]
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        per_axis.append(dict(
            axis=axis, entries=int(valid.sum()), mae=float(np.abs(residual).mean()),
            r2=1 - float(np.sum(residual ** 2)) / denominator if denominator > 0 else None,
        ))
    return dict(
        mae=float(np.abs(errors[available]).mean()),
        rmse=float(np.sqrt(np.square(errors[available]).mean())),
        r2_uniform=float(np.mean([row["r2"] for row in per_axis if row["r2"] is not None])),
        per_axis=per_axis, entries=int(available.sum()),
    )


def report(run, replay, out):
    verify_manifest(run, "freeze.json")
    verify_manifest(replay, "freeze.json")
    verification = read_json(replay / "verification.json")
    if verification["status"] != "VERIFIED" or verification["run_manifest_sha256"] != sha(run / "freeze.json"):
        raise ValueError("A matching exact replay is required")
    out.mkdir(parents=True, exist_ok=False)
    protocol = read_json(run / "protocol.json")
    split = protocol["split"]
    banks = load_banks(Path(protocol["encoded"]), split)
    predictions = read_json(run / "predictions.json")
    results = read_json(run / "results.json")
    summary = read_json(run / "summary.json")
    indexed = {row["method"]: row for row in summary}
    families = {row["case_id"]: row["truth"]["family"] for row in
                read_json(Path(protocol["data"]) / "training-generation-ledger.json")}
    main = "query_shape_grounded"
    controls = ["query_shape_" + name for name in ("plain", "global", "shuffled", "numeric")]
    paired, totals, traces = [], [], {}
    for control in controls:
        for cid in split["validation"]:
            actual = banks[cid]["activity"]
            before = np.array(predictions[control][cid]["activity"])
            after = np.array(predictions[main][cid]["activity"])
            changes = dict(
                removed_fp=(actual == 0) & (before == 1) & (after != 1),
                added_fp=(actual == 0) & (before != 1) & (after == 1),
                gained_tp=(actual == 1) & (before != 1) & (after == 1),
                lost_tp=(actual == 1) & (before == 1) & (after != 1),
            )
            row = dict(comparator=control, case_id=cid, family=families[cid],
                       **{name: int(mask.sum()) for name, mask in changes.items()})
            row["net_correct"] = row["removed_fp"] + row["gained_tp"] - row["added_fp"] - row["lost_tp"]
            paired.append(row)
            traces[control + ":" + cid] = {name: runs(mask) for name, mask in changes.items()}
        subset = [row for row in paired if row["comparator"] == control]
        total = dict(comparator=control, **{name: sum(row[name] for row in subset) for name in
                                           ("removed_fp", "added_fp", "gained_tp", "lost_tp", "net_correct")})
        expected = indexed[control]["fp"] + indexed[control]["fn"] - indexed[main]["fp"] - indexed[main]["fn"]
        if expected != total["net_correct"]:
            raise ValueError("Daily correction counts disagree with the unified score")
        totals.append(total)
    write_csv(out / "paired-errors.csv", paired)
    write_csv(out / "correction-summary.csv", totals)
    write_json(out / "exact-error-traces.json", dict(interval_convention="Original zero-based half-open", traces=traces))
    family_rows = []
    for name in predictions:
        for family in sorted({families[cid] for cid in split["validation"]}):
            ids = [cid for cid in split["validation"] if families[cid] == family]
            aggregate = aggregate_scores([results[name]["per_case"][cid] for cid in ids])
            family_rows.append(dict(method=name, family=family, records=len(ids), **aggregate["daily"]))
    write_csv(out / "family-scores.csv", family_rows)

    targets = {}
    for cid in split["validation"]:
        with np.load(run / "targets" / (cid + ".npz")) as stored:
            targets[cid] = stored["query"]
    actual_shape = np.concatenate([targets[cid] for cid in split["validation"]])
    support = np.concatenate([banks[cid]["supported"] for cid in split["validation"]])
    available = np.isfinite(actual_shape) & support[:, None]
    auxiliary = read_json(run / "auxiliary.json")
    means = np.array(read_json(run / "scaling.json")["query"]["mean"])
    shape_scores = {
        "zero": shape_metrics(actual_shape, np.zeros_like(actual_shape), available),
        "train_mean": shape_metrics(actual_shape, np.broadcast_to(means, actual_shape.shape), available),
    }
    for name, values in auxiliary.items():
        if values:
            predicted = np.concatenate([values[cid] for cid in split["validation"]])
            shape_scores[name] = shape_metrics(actual_shape, predicted, available)
    write_json(out / "query-fidelity-baselines.json", dict(
        unit="Signed normalized raster ordinate;not millimeters or noise units",
        scope="Posthoc diagnostic, finite teacher dimensions and original supported dates;no activity-label filtering",
        warning="Global head was trained against global shape, so its query fidelity is cross-target, not matched reconstruction",
        scores=shape_scores,
    ))
    probability = read_json(run / "probabilities.json")
    known = np.concatenate([(banks[cid]["activity"] >= 0) & banks[cid]["supported"] for cid in split["validation"]])
    labels = np.concatenate([banks[cid]["activity"] for cid in split["validation"]])[known]
    aps = []
    for name, values in probability.items():
        scores = np.concatenate([values[cid] for cid in split["validation"]])[known]
        aps.append(dict(method=name, average_precision=float(average_precision_score(labels, scores)),
                        scored_days=int(known.sum()), positive_days=int(labels.sum())))
    write_csv(out / "posthoc-average-precision.csv", aps)

    candidates = sorted([row for row in paired if row["comparator"] == "query_shape_plain"],
                        key=lambda row: (row["net_correct"], row["case_id"]))
    review_ids = list(dict.fromkeys([candidates[0]["case_id"], candidates[-1]["case_id"]]))
    write_json(out / "diagnostic-plan.json", dict(
        review_ids=review_ids, selection="Posthoc minimum/maximum net correction against matched plain",
        reference_changes=0, threshold_changes=0, reserved_content_loaded=False,
    ))
    for cid in review_ids:
        case, _ = load_case_features(Path(protocol["data"]), cid)
        raw = np.array([value if value is not None else [np.nan] * 3 for value in case.displacement_mm])
        fig, panels = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
        for axis, label in enumerate(("N", "E", "U")):
            panels[axis].plot(raw[:, axis], lw=.8)
            panels[axis].set_ylabel(label + " (mm)")
            panels[axis].grid(alpha=.2)
        names = (("Process reference", banks[cid]["activity"]),
                 ("Plain", predictions["query_shape_plain"][cid]["activity"]),
                 ("QSG", predictions[main][cid]["activity"]),
                 ("Numeric", predictions["query_shape_numeric"][cid]["activity"]))
        for level, (_, values) in enumerate(names):
            for start, stop in runs(np.array(values) == 1):
                panels[3].plot([start, stop - 1], [level, level], lw=6)
        panels[3].set_yticks(range(len(names)), [name for name, _ in names])
        for name in (main, "query_shape_plain", "query_shape_numeric"):
            panels[4].plot(probability[name][cid], label=name, lw=.9)
        panels[4].set_ylabel("Probability")
        panels[4].legend(loc="upper left", fontsize=8)
        panels[4].set_xlabel("Original day index; posthoc diagnostic, reference unchanged")
        panels[4].set_xlim(0, len(raw) - 1)
        fig.suptitle(cid + " | developer diagnostic")
        fig.tight_layout()
        fig.savefig(out / (cid + "-diagnostic.png"), dpi=130)
        plt.close(fig)

    core = list(protocol["arms"])
    text = (
        "# 查询局部图形监督：固定QSG-VN完整开发结果\n\n## Material Passport\n\n"
        "- Origin Skill: academic-research-suite / experiment-agent。\n"
        "- Scope: 48TRAIN／24已暴露VALIDATION；96×730总规模，24保留输入及参考内容未加载。\n"
        "- Verification Status: VERIFIED（保留权重、教师、校准、输出与评分；未重跑优化器）。\n"
        "- Reference: 受控过程参考；事后AI看图为开发者分析，非独立领域复核。\n"
        "- Version Label: query-shape-grounding-report-v1。\n\n"
        "## 方法、选择机会与结论\n\n"
        "QSG-VN（Query-Shape-Grounded Visual-Numeric learning，查询局部图形监督的视觉数值学习）"
        "在已有小型两卷积活动分类器中加入一项局部图形变化辅助损失，辅助头只读取图像分支。"
        "它不是语言VLM或Agent。6通道32×61栅格、原数值输入和全部10187TRAIN中心不变；"
        "辅助目标是图上三组5列的NEU有符号变化，不是被逐窗缩放消除的物理幅度。"
        "每组至少4列，缺失教师只mask辅助损失，不删除活动样本。\n\n"
        "7臂各宽16／32、19阈值，共14模型／266校准配置，统一F1+IoU.3事件F1择优，"
        "同种子20261014、12epoch、batch128、Adam .001、weight decay .0001、SmoothL1系数1。"
        "没有种子、损失权重、训练轮数搜索。原18同划分重训强对照全部保留，25方法600输出。\n\n"
        "固定预设继续条件不满足，STOP_QUERY_SHAPE_GROUNDING。主相对plain少22FP但少90TP，"
        "净−68日；不能改选打乱控制、更名或换阈值来挽救本版。\n\n"
        + score_table(summary, core) + "\n\n## 同任务全部强基线\n\n"
        + score_table(summary, [name for name in predictions if name not in core])
        + "\n\n## 精确逐日修订\n\n"
        "| 相对控制 | 去FP | 加FP | 补TP | 撤TP | 净纠错 |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    )
    for row in totals:
        text += "| " + " | ".join(str(row[name]) for name in
                                    ("comparator", "removed_fp", "added_fp", "gained_tp", "lost_tp", "net_correct")) + " |\n"
    text += (
        "\n八族各3条验证记录，全25方法分族计数及F1保存于family-scores.csv；"
        "逐日半开区间与逐例计数另存，不以极值例代替总体。\n\n"
        "## 辅助读出是否有事实增益（事后诊断）\n\n"
        "| 预测 | MAE（栅格纵坐标单位） | RMSE | 六输出平均R² |\n"
        "| --- | ---: | ---: | ---: |\n"
    )
    for name, row in shape_scores.items():
        text += f"| {name} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['r2_uniform']:.6f} |\n"
    text += (
        "\n主／plain／打乱对局部教师可匹配比较；global头训练的目标不同，"
        "其表中局部误差是跨目标诊断，不能据此声称global目标重建较差。"
        "TRAIN均值使用原仅TRAIN维度均值，零与均值没有拟合验证标签。"
        "比未监督的随机辅助头误差低不等于优于简单事实预测，更不等于异常检出改善。"
        "单位与上一轮噪声单位目标不同，不能直接比较MAE。\n\n"
        "## 未解码排序（事后诊断）\n\n| 方法 | Average Precision |\n| --- | ---: |\n"
    )
    for row in aps:
        text += f"| {row['method']} | {row['average_precision']:.6f} |\n"
    text += (
        "\nAP只评价原supported且参考已知日，不经过持续期解码，也未重新选择权重或阈值。"
        "不能用它替换主活动和事件评分。\n\n## 原图检查与成本\n\n"
        f"事后按相对plain净纠错极小／极大选{', '.join(review_ids)}，原NEU与活动／概率图保存。"
        "实际视觉阅读记录由session保存，生成图片本身不冒称已看图。参考未修改。\n\n"
    )
    costs = read_json(run / "costs.json")
    text += (
        f"栅格与教师准备{costs['preparation_seconds']:.3f}秒，14模型含全部消融训练"
        f"{costs['training_seconds']:.3f}秒；0新API、0新记录、0新视觉编码。"
        "训练总成本不当作单方法推理／部署成本。72观测重构、14保留权重及266选择机会、"
        "25方法600输出、原阶段头与全部统一评分精确重放，最大概率误差0；"
        "没有重训优化器，因此不声称从头训练复现。验证器完成后E731由lambda改为def，"
        "重放冻结source.py保留实际运行版本；训练冻结源码未改。\n\n"
        "复现：run_landslide_query_shape_grounding.py、verify_landslide_query_shape_grounding.py、"
        "report_landslide_query_shape_grounding.py。保留data／encoded／parent参数及原冻结协议。"
        "当前开发验证已反复参与方法选择，不能用于独立效果结论；不扩停止版本数据来救分。\n"
    )
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    print(dict(review_ids=review_ids, corrections=totals, main_shape=shape_scores[main],
               zero_shape=shape_scores["zero"], train_mean_shape=shape_scores["train_mean"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("run", "replay", "out"):
        parser.add_argument("--" + argument, type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.run.resolve(), arguments.replay.resolve(), arguments.out.resolve())
