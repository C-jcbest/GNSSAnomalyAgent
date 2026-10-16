"""Describe the frozen pilot, including failures and post-score error inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from probe_landslide_nuisance_evidence import ARMS, DEVELOPMENT, OLD, write_csv
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import precision_recall_curve

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores


def f1_from_counts(counts):
    denominator = 2 * counts[0] + counts[1] + counts[2]
    return 2 * counts[0] / denominator if denominator else 0.0


def render_error(case, labels, predictions, path):
    values = np.array([
        row if row is not None else [np.nan] * 3 for row in case.displacement_mm
    ])
    figure, panels = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for axis in range(3):
        panels[axis].plot(values[:, axis], color="#466d7c", linewidth=.6)
        panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
        panels[axis].grid(alpha=.2)
    names = ("Process reference", "Raw84", "NCDA")
    activity = [np.array([int(row["activity_label"]) for row in labels])]
    activity.extend(np.array(predictions[arm][case.case_id]["activity"]) for arm in ARMS[:2])
    for index, (name, state) in enumerate(zip(names, activity, strict=True)):
        curve = state.astype(float)
        curve[state == -1] = np.nan
        panels[3].plot(curve + index * 1.4, label=name, linewidth=1)
    panels[3].set_yticks([0, 1.4, 2.8], names)
    panels[3].set_xlabel("Original day index; each row rises by 1 during activity; gaps remain unknown")
    panels[3].grid(alpha=.2)
    figure.suptitle(case.case_id + " | Post-score diagnostic; process labels, not independent expert reference")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def report(run, out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "analysis-plan.json", {
        "scope": "Post-score descriptive family/PR/bootstrap/error analysis; no threshold or model changes",
        "bootstrap": "2000 paired whole-record resamples, seed20261012; percentile intervals, no significance claim",
        "pr": "Common raw84 support; closest empirical recall to the raw84 probability threshold before minimum-run activity postprocessing; illustrative only, never deployment selection",
        "figures": "Maximum NCDA FP in quiet and instrument-step family, then maximum net correction among deformation families; selected after scoring, not representative sampling or new annotation",
    })
    results = read_json(run / "results.json")
    summaries = {row["method"]: row for row in read_json(run / "summary.json")}
    predictions = read_json(run / "new-predictions.json")
    labels = read_json(OLD / "reference-labels.json")
    decision = read_json(run / "decision.json")
    ledger = read_json(DEVELOPMENT / "mechanism-sealed/generation-ledger.json")
    families = {row["case_id"]: row["truth"]["family"] for row in ledger}
    family_rows = []
    for arm in ARMS:
        for family in sorted(set(families.values())):
            ids = [cid for cid in labels if families[cid] == family]
            score = aggregate_scores([results[arm]["per_case"][cid] for cid in ids])
            family_rows.append({"method": arm, "family": family, **score["daily"], "event_f1": score["events"]["f1"]})
    write_csv(out / "family-summary.csv", family_rows)
    ids = list(labels)
    generator = np.random.default_rng(20261012)
    draws = generator.integers(0, len(ids), size=(2000, len(ids)))
    bootstrap = {}
    for arm in ARMS[1:]:
        differences = []
        for key in ("daily", "events"):
            counts = {
                name: np.array([
                    [results[name]["per_case"][cid][key][field] for field in ("tp", "fp", "fn")]
                    for cid in ids
                ]) for name in (arm, ARMS[0])
            }
            scores = {
                name: np.array([f1_from_counts(rows[draw].sum(axis=0)) for draw in draws])
                for name, rows in counts.items()
            }
            difference = scores[arm] - scores[ARMS[0]]
            differences.append({"metric": key + "_f1", "low": float(np.quantile(difference, .025)), "high": float(np.quantile(difference, .975))})
        bootstrap[arm] = differences
    write_json(out / "paired-bootstrap.json", bootstrap)
    probabilities = read_json(run / "probabilities.json")
    actuals = []
    supported_probabilities = {arm: [] for arm in ARMS}
    for cid in ids:
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        with np.load(run / (cid + "-features.npz")) as bank:
            mask = bank["support"] & (actual >= 0)
        actuals.extend(actual[mask].tolist())
        for arm in ARMS:
            supported_probabilities[arm].extend(np.array(probabilities[arm][cid])[mask].tolist())
    actuals = np.array(actuals)
    raw_threshold = read_json(run / "selection.json")[ARMS[0]]["threshold"]
    base_positive = np.array(supported_probabilities[ARMS[0]]) >= raw_threshold
    target_recall = float(np.sum(base_positive & (actuals == 1)) / np.sum(actuals == 1))
    pr_rows = []
    matched = []
    for arm in ARMS:
        precision, recall, thresholds = precision_recall_curve(actuals, supported_probabilities[arm])
        closest = int(np.argmin(np.abs(recall - target_recall)))
        matched.append({"method": arm, "reference_recall": target_recall, "closest_recall": float(recall[closest]), "precision": float(precision[closest]), "scope": "post-score unprocessed probability PR illustration; not the postprocessed activity operating point"})
        for index in range(len(precision)):
            pr_rows.append({"method": arm, "precision": float(precision[index]), "recall": float(recall[index]), "threshold": float(thresholds[index]) if index < len(thresholds) else None})
    write_csv(out / "pr-curves.csv", pr_rows)
    write_json(out / "matched-recall-description.json", matched)
    negative_examples = [
        max((cid for cid in ids if families[cid] == family), key=lambda cid: results["numeric_ncda"]["per_case"][cid]["daily"]["fp"])
        for family in ("quiet", "instrument_step")
    ]
    positive_ids = [cid for cid in ids if families[cid] not in {"quiet", "instrument_step"}]
    best = max(positive_ids, key=lambda cid: sum(
        results["numeric_raw84"]["per_case"][cid]["daily"][key] - results["numeric_ncda"]["per_case"][cid]["daily"][key]
        for key in ("fp", "fn")
    ))
    examples = negative_examples + [best]
    for cid in examples:
        case, _ = load_case_features(DEVELOPMENT, cid)
        render_error(case, labels[cid], predictions, out / (cid + "-error.png"))
    write_json(out / "figure-selection.json", {"case_ids": examples, "selection_after_scoring": True, "labels_modified": False})
    lines = [
        "# 干扰条件化证据：首轮机制实验", "", "## Material Passport", "",
        "- Origin Skill: academic-research-suite；用户授权的方法开发与实验", "- Origin Date: 2026-10-12",
        "- Verification Status: 原始证据、概率、活动/阶段和全方法评分已重放核验；不是独立效果验证",
        "- Version Label: NCDA scalar pilot v1；历史结果不覆盖", "",
        "## 结果与决定", "",
        "新增一项联合干扰解释证据，在已暴露开发记录上改善整体日F1、事件F1和净纠错，但静稳/偏移误报增加，未达到预设减少干扰误报的条件。决定为 **STOP_NCDA_SCALAR_PILOT**（4/5条件通过）；停止推广这一单特征版本，没有进入新VLM复核或新记录确认。不改门槛、阈值或种子挽救该版本。", "",
        "| 方法 | Precision | Recall | 日F1 | 事件F1(.3) | AUPRC（共同支持） | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = summaries[arm]
        lines.append(f"| {arm} | {row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['event_f1_iou03']:.6f} | {row['auprc_common_support']:.6f} | {row['fp']} | {row['fn']} |")
    lines.extend([
        "", "NCDA相对同训练raw84增加188个真实活动日、增加55个误报日，净纠错+133；静稳/偏移8条的误报由41增至95。共享干扰相对互斥竞争F1更高，但AUPRC略低，不能宣称所有阈值都更好。三个新臂覆盖相同（0.914680），没有通过删除难日期提高F1。", "",
        "主方法起点MAE由33.14降至31.65日，终点MAE由16.90升至22.43日。匹配事件数量不同，不能将这组边界均值直接当作同事件配对改善。事件F1(.5)由0.545455到0.553846，较IoU .3的收益小，提示严格定位仍弱。", "",
        "## 机制与公平比较", "",
        "候选名NCDA（Nuisance-Conditioned Deformation Anomaly Detection），本轮只验证其数值证据，不是已实现的新Agent/VLM。", "",
        "$$", r"H_0:x_t=c+j_t+e_t,\qquad H_1:x_t=c+j_t+g_t+e_t.", "$$", "",
        "常量与最多一个偏移构成无形变解释；含查询日的连续饱和线性斜坡构成形变解释，可同时包含偏移。对相同14日原位移块中位数进行最多20轮Huber IRLS；参数及搜索节点惩罚固定。证据是", "",
        "$$", r"D=\frac{J_0-J_1}{3n},\qquad J_k=\sum_{i,a}\rho_{1.5}\!\left(\frac{x_{i,a}-\widehat x_{i,a}^{(k)}}{\widehat\sigma_a}\right)+(3p_k+q_k)\log(3n).", "$$", "",
        "其中n为支持块数，p为每轴拟合列数，q为搜索节点数。有限IRLS近似稳健拟合，D不是概率或显著性检验。缺测保持未知，长缺口或支持不足使证据为NaN，不扩大活动支持、不把拟合节点当标签。", "",
        "沿用原24训练/12验证，三个臂各6个深度/阈值机会；完整24已暴露开发记录。活动支持、原42/84日特征及数值阶段头相同；新模型和预测冻结后才在此执行中读取开发参考，但其历史暴露已知，不能据此声称盲测。重训raw84与历史强XGB84完整预测精确一致。数值无新增证据、共享干扰证据、互斥竞争证据三组只差一个标量，避免把新增数据或调参次数当机制收益。", "",
        "同时统一重评分37项历史对照，包括规则、PELT、XGBoost、直接VLM和固定/自适应复核，共40项，完整表在原实验summary.csv。历史视觉计算成本未在本轮重新装载回执，summary里的零请求表示本轮无新增请求，不能解释为历史VLM免费或零成本。", "",
        "## 描述性敏感性与错误检查", "",
        "2000次整记录配对bootstrap只描述这24条的波动，不修正开发选择、多重比较或生成分布偏差。", "",
        "| 比较 | 指标 | 95%描述性区间 |", "| --- | --- | --- |",
    ])
    for arm, values in bootstrap.items():
        for value in values:
            lines.append(f"| {arm} − raw84 | {value['metric']} | [{value['low']:.6f}, {value['high']:.6f}] |")
    lines.extend(["", "PR曲线与匹配召回比较只作为评分后的原始概率形状说明，未改变部署阈值。其参考召回在raw84固定概率阈值下、最短活动段后处理之前计算，不能等同于主表活动工作点；这也解释原始概率Precision与主表Precision不同。AUPRC使用三组共同支持的确定日期，主F1仍用统一完整参考及未知处理。v1表述没有注明这一区别；v2只澄清说明，概率、曲线点、模型和所有主指标不变，v1保留。族别表、完整PR点和三幅错误图随报告保存。图按评分后固定规则选出，不能作为随机代表样本，也不用于追改参考。", "", "实际看图及逐值复核：case_0172的平台偏移邻域新增23个误报日，新增证据均为负；case_0175的大阶跃附近新增46个误报日，其168日上下文均因超过7日连续缺口使证据不可用；case_0190保留了更多活动，同时仍漏检后续缓慢运动。视觉检查未追改任何过程参考，属于评分后的开发者错误分析。", "", "## 结论适用范围及下一项研究", "",
        "该证据提供了值得保留的整体识别开发信号，但未解决精确率/干扰误报，因此不能据本轮结果宣称新的可靠Agent/VLM贡献成立。单步+线性斜坡仍可能把相关噪声、边缘偏移或阶段转换拟合为运动；一个得分不能自动定位干扰或确保停止活动。瞬时物理位移与仪器偏移也可能观测不可辨。", "",
        "下一项优先检验VETime式时间对齐视觉/数值基础上的形变与干扰双输出监督，使用可同时阳性的标签；同增广单头及去视觉对照决定增益是否来自质量监督或视觉。新增数据应覆盖普通不同幅度/持续时间、快速连续起变、停滞再活动、形变+偏移/尖峰和缺测，而不是主要增加弱运动或固定A/S/D。此为新的机制设计，不能按本批误报调阈值后将失败v1改称成功。", "",
        "## 复现及产物", "",
        "本轮60条记录的新增证据计算合计34.007秒，原诊断/42/84特征14.846秒，树训练与新预测约1.197秒；只是当前CPU实测，不是部署端到端延迟或与VLM公平计时。6项业务测试通过，无新增API、生成记录或人工专家标注。", "",
        "```powershell", "$env:PYTHONPATH='src;scripts'",
        "& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/probe_landslide_nuisance_evidence.py --out artifacts/gnss-ncda-mechanism-2026-10-12/user-replay",
        "& artifacts/reproduction-2026-10-07/environment/Scripts/python.exe scripts/verify_landslide_nuisance_evidence.py --run artifacts/gnss-ncda-mechanism-2026-10-12/pilot-v1 --out artifacts/gnss-ncda-mechanism-2026-10-12/user-verification.json",
        "```", "",
        "原协议、模型、每记录证据、概率、预测、全方法评分及哈希manifest在 `../pilot-v1/`；重放核验在 `../pilot-v1-replay.json`。本报告为单独的机制结果，不修改封存v54论文初稿。文献依据见[候选调研](../../../docs/gnss-deformation-anomaly-method-directions-2026-10-12.md)。",
    ])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "manifest.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    print("Report saved", out, "examples", examples, "decision", decision["decision"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.out)


if __name__ == "__main__":
    main()
