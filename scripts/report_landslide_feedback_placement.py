"""Export complete same-feedback controls and their actual correction transitions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from report_landslide_core_lockbox import full_table
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_review_utility import correction_utility

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
DATA = BASE / "core-lockbox-data-v1"
PRIOR = BASE / "core-lockbox-analysis-v1"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def render_comparison(case, labels, predictions, path):
    raw = np.asarray(
        [value if value is not None else [np.nan] * 3 for value in case.displacement_mm]
    )
    figure, axes = plt.subplots(5, 1, figsize=(13, 10), sharex=True)
    for axis, name in enumerate(("N", "E", "U")):
        axes[axis].plot(np.arange(len(raw)), raw[:, axis], linewidth=0.7)
        axes[axis].set_ylabel(name + " (mm)")
        axes[axis].grid(alpha=0.2)
    reference = [int(row["activity_label"]) for row in labels]
    names = ["Reference", "Direct", "Adaptive numeric", "Fixed numeric", "Same K early numeric"]
    for rep, axis in enumerate(axes[3:]):
        methods = [
            f"direct_visual_r{rep}",
            f"adaptive_numeric_feedback_r{rep}__conflict_scope",
            f"fixed_numeric_feedback_r{rep}__whole_target",
            f"same_count_fixed_early_numeric_r{rep}__conflict_scope",
        ]
        values = [reference] + [predictions[name][case.case_id]["activity"] for name in methods]
        axis.imshow(
            np.asarray(values) + 1,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=2,
            cmap=ListedColormap(["#cccccc", "#ffffff", "#327d89"]),
            extent=(-0.5, len(raw) - 0.5, 4.5, -0.5),
        )
        axis.set_yticks(range(5), names)
        axis.set_ylabel(f"Repeat {rep}")
    axes[-1].set_xlabel("Original day; teal=activity, white=stationary, gray=unknown/missing")
    figure.suptitle(
        case.case_id + " | Process reference; same numerical feedback, different view positions"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=145)
    plt.close(figure)


def report(analysis, replay, out):
    verify_manifest(analysis)
    verify_manifest(replay)
    expected = read_json(analysis / "manifest.json")["files"]
    for name, digest in expected.items():
        if sha(replay / name) != digest:
            raise ValueError("Offline result replay differs: " + name)
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "replay-audit.json",
        {"status": "passed", "byte_identical_files": len(expected), "new_remote_requests": 0},
    )
    transitions = []
    for target in ("primary_process", "secondary_developer"):
        predictions = read_json(analysis / target / "predictions.json")
        labels = read_json(PRIOR / target / "reference-labels.json")
        for rep in (0, 1):
            treatment = f"adaptive_numeric_feedback_r{rep}__conflict_scope"
            for control in (
                f"fixed_numeric_feedback_r{rep}__whole_target",
                f"same_count_fixed_early_numeric_r{rep}__conflict_scope",
            ):
                for cid, days in labels.items():
                    actual = np.array([int(day["activity_label"]) for day in days])
                    before = np.array(predictions[control][cid]["activity"])
                    after = np.array(predictions[treatment][cid]["activity"])
                    transitions.append(
                        {
                            "target": target,
                            "repeat": rep,
                            "case_id": cid,
                            "control": control,
                            "treatment": treatment,
                            **correction_utility(actual, before, after, actual >= 0),
                            "gained_tp": int(((actual == 1) & (before != 1) & (after == 1)).sum()),
                            "lost_tp": int(((actual == 1) & (before == 1) & (after != 1)).sum()),
                            "gained_fp": int(((actual == 0) & (before != 1) & (after == 1)).sum()),
                            "removed_fp": int(((actual == 0) & (before == 1) & (after != 1)).sum()),
                            "active_to_unknown": int(
                                ((actual == 1) & (before == 1) & (after == -1)).sum()
                            ),
                            "stable_to_unknown": int(
                                ((actual == 0) & (before == 0) & (after == -1)).sum()
                            ),
                        }
                    )
        additional = {
            name: cases
            for name, cases in predictions.items()
            if name.startswith(
                (
                    "fixed_numeric_feedback",
                    "same_count_fixed_early_numeric",
                    "same_count_fixed_late_numeric",
                )
            )
        }
        with (out / (target + "-additional-daily.csv")).open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["method", "case_id", "day", "activity", "stage"]
            )
            writer.writeheader()
            for method, cases in additional.items():
                for cid, prediction in cases.items():
                    writer.writerows(
                        {
                            "method": method,
                            "case_id": cid,
                            "day": day,
                            "activity": activity,
                            "stage": prediction["stage"][day],
                        }
                        for day, activity in enumerate(prediction["activity"])
                    )
    write_csv(out / "per-record-transitions.csv", transitions)
    predictions = read_json(analysis / "primary_process/predictions.json")
    labels = read_json(PRIOR / "primary_process/reference-labels.json")
    block_evidence = {}
    for cid in ("case_0171", "case_0174", "case_0177", "case_0187"):
        case, feature = load_case_features(DATA, cid)
        render_comparison(case, labels[cid], predictions, out / (cid + "-aligned.png"))
        raw = np.asarray(
            [value if value is not None else [np.nan] * 3 for value in case.displacement_mm]
        )
        block_evidence[cid] = [
            {
                "start": start,
                "stop": min(start + 30, len(raw)),
                "observed_days": int(feature.observed[start : start + 30].sum()),
                "median_NEU_mm": np.nanmedian(raw[start : start + 30], axis=0).tolist(),
            }
            for start in range(0, len(raw), 30)
        ]
    write_json(out / "raw-block-evidence.json", block_evidence)
    comparisons = rows(analysis / "placement-comparisons.csv")
    decision = read_json(analysis / "decision.json")
    cost = read_json(analysis / "cost.json")
    text = """# 同数值反馈的观察位置开发比较

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-10-11
- Verification Status: VERIFIED（冻结预测离线评分逐字节重放；不代表方法确认通过）
- Version Label: feedback-placement-development-v1

本实验补足公平控制：固定位置和自适应位置使用完全相同的原始持续速率数值摘要、图像画法、模型参数、共享全局初判及阶段头。原24条过程目标和预指定12条作者目标全部保留；每条每遍同K早/晚数值复核取相同次数。只新增缺少的固定位置回答，完全相同的已有回答复用，未改提示、阈值、参考、权重或种子。原核心NOT_CONFIRMED判定不变。

该人口已评分并参与本次问题选择，故这是开发研究。缓存自适应回答与新固定回答取得时间不同，服务模型漂移尚不能排除；不能当成新独立确认。完整人口为主，依据初判任务status排除失败的敏感性另报；不按活动标签或结果筛选记录。

## 位置与反馈的效果

原同位置反馈在正常初判过程记录上，相对纯图F1差+.0335/+.0156、净正确+358/+168；相对直接F1差+.0256/+.0511、净+301/+381。弱R相对直接下降，因此它主要减少误报，不是无损弱检出改进。

| 参考 | 遍 | 人口 | 对照 | 主/对照F1 | F1差 | 净确定纠错日 |
|---|---|---|---|---|---:|---:|
"""
    for row in comparisons:
        if row["control"].startswith(("fixed_numeric_feedback", "same_count_fixed_early_numeric")):
            text += f"| {row['target']} | {row['repeat']} | {row['population']} ({row['records']}) | {row['control']} | {float(row['treatment_f1']):.4f}/{float(row['control_f1']):.4f} | {float(row['f1_difference']):+.4f} | {row['net_correct']} |\n"
    text += "\n相同数值反馈下，过程同K位置两遍完整F1差+.0239/+.0476、净+183/+171；正常初判敏感性差+.0065/+.0215、净+79/+48。第一遍相对完整固定数值复核却净−131日，正常初判F1差−.0119、净−88日。作者第二遍对同K早净−25日，对完整固定两遍净−38/−166日。提高F1不能替代确定正确判断，已有收益也不能确认无损可靠性。\n"
    text += f"\n登记开发行动判定：**{decision['decision']}**。完整固定比较的第一遍条件未通过，停止将此反馈候选升级为主方法；不另抽种子、换对照或挑重复救分。原核心方法和原实验所有判定保持。\n"
    text += f"\n新增{cost['new_requests']}请求/{cost['responses']}回执、{cost['reported_tokens']}报告token，执行墙钟{cost['wall_seconds']:.3f}秒、{cost['task_failures']}新任务失败、0重试，复用{cost['cached_fixed_queries']}个相同固定问题。单遍部署过程主47/51调用、完整固定72/72、同K47/51；作者主24/24、固定36/36。全研究累计3725启动/3723回执、至少16167212报告token；受控192与半合成60不变。本轮无新记录、训练或参考标签。\n"
    text += "\n## 强基线与全部消融\n\n保留所有原37/34配置并分别增加8个控制，完整45/42配置如下。九/六数值方法按原目标匹配训练，所有旧结果字典保持；新控制的逐日结果单独导出，与旧完整逐日导出共同构成复现数据。0API不等于0CPU。\n"
    for target in ("primary_process", "secondary_developer"):
        text += (
            "\n### " + target + "\n\n" + full_table(rows(analysis / target / "summary.csv")) + "\n"
        )
    text += "\n## 实际错误图与复现\n\n以下四图来自完整评分后错误解释，未改变参考或用于挑评价子集；实际阅读情况另记developer-inspection.json。\n"
    for cid in block_evidence:
        text += f"\n![{cid}原始位移及同反馈位置对照]({cid}-aligned.png)\n"
    text += "\n源码probe_landslide_feedback_placement.py分prepare/run/score。已完成推理目录不得重启；离线重放只执行score并指定尚不存在的新analysis目录。report_landslide_feedback_placement.py核对离线字节一致性并生成这些图、逐记录转换和逐日附加控制。未运行最近邻作者模型，也未建立独立专家或现场金标准。\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    write_json(
        out / "manifest.json",
        {
            "files": {
                path.relative_to(out).as_posix(): sha(path)
                for path in out.iterdir()
                if path.is_file()
            }
        },
    )
    print("Wrote complete45/42 controls, four raw figures and additional daily exports")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("analysis", "replay", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(args.analysis, args.replay, args.out)
