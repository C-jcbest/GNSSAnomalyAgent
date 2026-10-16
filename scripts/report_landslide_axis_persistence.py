"""Score stopped axis-witness development and preserve its numerical controls."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_review_utility_v2 import verify_receipts
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from report_landslide_core_lockbox import full_table
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_core_lockbox import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_axis_persistence import axis_persistence
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_local_experiment import learned_stages

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
DATA = BASE / "core-lockbox-data-v1"
PRIOR = BASE / "feedback-placement-analysis-v1"
LABELS = BASE / "core-lockbox-analysis-v1"


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def render(cid, pair, witness, old_choices, new_choices, actual, threshold, path):
    case, _ = pair
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    fig, axes = plt.subplots(6, 1, figsize=(13, 11), sharex=True)
    for index, axis in enumerate(("N", "E", "U")):
        axes[index].plot(values[:, index], linewidth=0.7)
        axes[index].set_ylabel(axis + " (mm)")
        axes[index].grid(alpha=0.2)
    for index, axis in enumerate(("N", "E", "U")):
        axes[3].plot(witness["axis_score"][:, index], label=axis, linewidth=0.8)
    axes[3].axhline(threshold, color="black", linestyle="--", label="validation threshold")
    axes[3].set_ylabel("Axis score")
    axes[3].legend(ncol=4)
    for rep in (0, 1):
        panel = axes[4 + rep]
        for start, stop in actual:
            panel.broken_barh([(start, stop - start)], (0.65, 0.25), color="#69af97")
        for target in old_choices[f"{cid}-r{rep}"]["selected"]:
            panel.broken_barh(
                [(target["start"], target["stop"] - target["start"])], (0.35, 0.25), color="#5375a3"
            )
        for target in new_choices[f"{cid}-r{rep}"]["selected"]:
            panel.broken_barh(
                [(target["start"], target["stop"] - target["start"])], (0.05, 0.25), color="#c78f51"
            )
        panel.set_yticks([0.17, 0.47, 0.77], ["Axis views", "Original views", "Process activity"])
        panel.set_ylim(0, 1)
        panel.set_title(f"Repeat {rep}: observation coverage, not recognized activity", fontsize=10)
    axes[-1].set_xlabel("Original day; raw gaps retained; selection has no positive-label meaning")
    fig.suptitle(cid + " | Raw displacement, axis witness and bounded observation opportunities")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def report(probe, out):
    verify_manifest(probe)
    for filename, digest in read_json(probe / "source-bindings.json").items():
        if sha(Path(filename)) != digest:
            raise ValueError("Frozen witness input/source changed: " + filename)
    verify_manifest(PRIOR)
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    ids = read_json(DATA / "protocol.json")["case_ids"]
    pairs = {cid: load_case_features(DATA, cid) for cid in ids}
    witnesses = {
        cid: axis_persistence(case, feature.noise) for cid, (case, feature) in pairs.items()
    }
    predictions_by_target = {}
    for target in ("primary_process", "secondary_developer"):
        config = read_json(probe / target / "selection.json")
        stage = XGBClassifier()
        stage.load_model(probe / target / "stage-xgboost.json")
        activity_model = XGBClassifier()
        activity_model.load_model(probe / target / "activity-xgboost-axis.json")
        ids_for_target = ids if target == "primary_process" else ids[:12]
        predictions = {"axis_persistence_rule": {}, "xgboost_axis_persistence": {}}
        for cid in ids_for_target:
            _, feature = pairs[cid]
            witness = witnesses[cid]
            supported = feature.observed & np.isfinite(witness["score"])
            matrix = np.column_stack(
                (feature.matrix, witness["axis_score"], witness["rank_halves"])
            )
            codes = stage.predict(feature.matrix).astype(int) + 1
            masks = {
                "axis_persistence_rule": witness["score"]
                >= config["axis_persistence_rule"]["threshold"],
                "xgboost_axis_persistence": activity_model.predict_proba(matrix)[:, 1]
                >= config["xgboost_axis_persistence"]["threshold"],
            }
            for name, mask in masks.items():
                activity = activity_from_mask(mask, supported)
                predictions[name][cid] = learned_stages(
                    codes, activity, feature.supported
                ).to_dict()
        predictions_by_target[target] = predictions
    write_json(out / "numerical-predictions.json", predictions_by_target)
    write_json(
        out / "prediction-freeze.json",
        {
            "numerical-predictions.json": sha(out / "numerical-predictions.json"),
            "source.py": sha(out / "source.py"),
        },
    )
    receipts = verify_receipts(BASE / "core-lockbox-confirmation-v1")
    receipts.update(verify_receipts(BASE / "feedback-placement-development-v1"))
    comparisons = []
    for target, additions in predictions_by_target.items():
        previous = read_json(PRIOR / target / "predictions.json")
        labels = read_json(LABELS / target / "reference-labels.json")
        predictions = {**previous, **additions}
        observed = {cid: pairs[cid][1].observed for cid in labels}
        if target == "primary_process":
            weak = {
                cid: np.array(
                    [
                        int(row["activity_label"]) == 1 and int(row["weak_signal"]) == 1
                        for row in rows
                    ]
                )
                for cid, rows in labels.items()
            }
        else:
            spans = read_json(DATA / "codex-review/activity-decisions.json")[
                "weak_interiors_authored_before_predictions"
            ]
            weak = {}
            for cid, rows in labels.items():
                mask = np.zeros(len(rows), dtype=bool)
                for start, stop in spans.get(cid, []):
                    mask[start:stop] = True
                weak[cid] = mask & (np.array([int(row["activity_label"]) for row in rows]) == 1)
        results, summaries, records = score_target(
            labels, observed, weak, predictions, {}, receipts
        )
        # Numerical additions have no API requests. Preserve all authoritative old
        # cost rows instead of recomputing them without their request schedules.
        old_results = read_json(PRIOR / target / "results.json")
        old_summaries = {row["method"]: row for row in csv_rows(PRIOR / target / "summary.csv")}
        for name, result in old_results.items():
            if results[name] != result:
                raise ValueError("Original metric dictionary changed")
        summaries = [old_summaries.get(row["method"], row) for row in summaries]
        folder = out / target
        folder.mkdir()
        write_json(folder / "predictions.json", predictions)
        write_json(folder / "results.json", results)
        write_csv(folder / "summary.csv", summaries)
        write_csv(folder / "per-record.csv", records)
        table = {row["method"]: row for row in summaries}
        for name in additions:
            row = table[name]
            comparisons.append(
                {
                    "target": target,
                    "method": name,
                    "p": row["precision"],
                    "r": row["recall"],
                    "f1": row["f1"],
                    "weak_recall": row["weak_recall"],
                    "requests": row["requests"],
                }
            )
    write_csv(out / "new-numerical-summary.csv", comparisons)
    opportunities = csv_rows(probe / "observation-opportunities.csv")
    totals = []
    for rep in (0, 1):
        rows = [row for row in opportunities if int(row["repeat"]) == rep]
        negative = [row for row in rows if row["quiet_or_step"].lower() == "true"]
        totals.append(
            {
                "repeat": rep,
                **{
                    name: sum(int(row[name]) for row in rows)
                    for name in (
                        "old_reviews",
                        "new_reviews",
                        "old_weak_covered",
                        "new_weak_covered",
                    )
                },
                "old_negative_views": sum(int(row["old_reviews"]) for row in negative),
                "new_negative_views": sum(int(row["new_reviews"]) for row in negative),
            }
        )
    write_json(out / "opportunity-totals.json", totals)
    negative_ids = sorted(
        {
            row["case_id"]
            for row in opportunities
            if row["quiet_or_step"].lower() == "true"
            and int(row["new_reviews"]) > int(row["old_reviews"])
        }
    )
    inspection_ids = list(dict.fromkeys(["case_0187", "case_0171", *negative_ids]))[:4]
    from gnss_sim.landslide_evaluation import runs

    labels = read_json(LABELS / "primary_process/reference-labels.json")
    raw_blocks = {}
    old_choices = read_json(BASE / "core-lockbox-confirmation-v1/selections.json")
    new_choices = read_json(probe / "selections.json")
    threshold = read_json(probe / "primary_process/selection.json")["axis_persistence_rule"][
        "threshold"
    ]
    for cid in inspection_ids:
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        render(
            cid,
            pairs[cid],
            witnesses[cid],
            old_choices,
            new_choices,
            runs(actual == 1),
            threshold,
            out / (cid + "-opportunities.png"),
        )
        case, _ = pairs[cid]
        raw = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        raw_blocks[cid] = []
        for start in range(0, len(raw), 30):
            sample = raw[start : start + 30]
            complete = sample[np.isfinite(sample).all(axis=1)]
            raw_blocks[cid].append(
                {
                    "start": start,
                    "stop": min(start + 30, len(raw)),
                    "observed_days": len(complete),
                    "median_NEU_mm": np.median(complete, axis=0).tolist()
                    if len(complete)
                    else None,
                }
            )
    write_json(out / "raw-blocks.json", raw_blocks)
    write_json(
        out / "inspection-plan.json",
        {
            "images": [cid + "-opportunities.png" for cid in inspection_ids],
            "selection": "Weak-gap example and prior plateau, followed by first opaque negative IDs with increased observation count; post-score description only, no evaluation subset",
        },
    )
    text = """# 逐分量连续变化触发：观察机会增加，但阴性观察预算退化

本轮只替换原始运动触发证据，不修改图像、提示、预算或阶段头。每中心日比较七个互不重叠14日原始块，每块至少8个完整NEU观测。分别在每分量前/后四块计算六个时间顺序对的符号平均，以及各半段净变化/原日差MAD；只有两半秩符号及净变化同向才取非零分量分数，再取三分量最大值。没有固定mm/day下限，门槛只在原过程24训练/12验证记录上选择；它不是统计显著性或校准置信度。原14冲突日、最多两个不重叠180日窗及49日权限保持。孤立阶跃偶数中位数可产生一代理日，不能用它直接建立活动。

训练/验证选出过程门槛0.5，作者数值对照门槛0.75。同信息XGBoost各六候选，规则各九候选，选择均为验证日F1+事件F1；作者基线仍用其6训练/6验证目标匹配标签。视觉观察计划只使用过程校准门槛，作者参考不影响选窗。没有用当前24/12开发记录重调门槛或权重。

## 评分前固定的机制准入

两遍弱日被观察窗覆盖数都必须增加，静稳/单阶跃记录的观察次数不增加，总部署调用少于固定72。窗口覆盖只代表观察机会，不是VLM活动召回。该条件在本轮开始前已固定；没有将条件失败后的较好数字用来升级方法。

| 遍 | 原/新复核次数 | 原/新弱日覆盖 | 原/新阴性记录复核次数 |
|---|---|---|---|
"""
    for row in totals:
        text += f"| {row['repeat']} | {row['old_reviews']}/{row['new_reviews']} | {row['old_weak_covered']}/{row['new_weak_covered']} | {row['old_negative_views']}/{row['new_negative_views']} |\n"
    decision = read_json(probe / "decision.json")
    text += (
        "\n决定：**"
        + decision["decision"]
        + "**。弱日观察机会两遍增加，但两遍阴性记录观察次数均增加，条件失败。因此不执行原本缺少的17个问题，不构造新的Agent识别结果，不挑单轴/门槛/重复救分。原核心NOT_CONFIRMED及反馈STOP_FEEDBACK_PROMOTION不变。阴性记录被选中不是活动误报，只是准入与预算效果变差。\n"
    )
    text += "\n## 数值对照与全部统一结果\n\n实际运行新规则及同信息XGBoost，在完整24过程和预指定12作者目标评分；原45/42方法结果与成本保留，新增2纯数值方法后为47/44。更强纯数值分数也不能改称Agent贡献。所有事件、边界、阶段与成本在下表；新支撑需七原始块，覆盖与旧方法不同，分数差不是隔离的信息增量实验。0API不等于0CPU。\n"
    for target in ("primary_process", "secondary_developer"):
        text += (
            "\n### " + target + "\n\n" + full_table(csv_rows(out / target / "summary.csv")) + "\n"
        )
    text += "\n## 原始位移与观察机会图\n\n以下图只展示原始观测、触发强度和选窗，不是假造未执行的视觉预测；实际阅读另登记。\n"
    for cid in inspection_ids:
        text += f"\n![{cid}原始观测与选窗]({cid}-opportunities.png)\n"
    text += "\n## 来源与复现\n\n全部输入与生成/作者参考保持；当前开发者已知历史及当前结果，不能称独立专家或现场准确率。v1标签加载器读到旧测试编号而在拟合前停止；v2完成两套校准后NaN JSON导出停止；v3逐字节复用已冻校准，仅将不可用代理导出null，不重拟合或调分。失败目录保留。新增模型请求0，新记录0；累计3725启动/3723回执、至少16167212报告token，受控192/半合成60不变。\n\n复现只运行report_landslide_axis_persistence.py --probe <本轮probe-v3目录> --out <尚不存在的新目录>，不重新拟合或调用服务。本候选在请求前停止；未完成Agent活动增益确认。\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    write_json(
        out / "manifest.json",
        {
            "files": {
                str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()
            }
        },
    )
    print("Opportunity totals", totals, flush=True)
    print("Numerical results", comparisons, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.probe.resolve(), args.out.resolve())
