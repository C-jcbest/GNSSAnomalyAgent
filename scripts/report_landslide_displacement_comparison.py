"""Export paired display diagnostics without changing references or model answers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from report_landslide_core_lockbox import csv_rows, full_table
from run_landslide_displacement_comparison import BASE, DATA, PRIOR, verify
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_review_utility import correction_utility


def plot_alignment(case, labels, predictions, path):
    values = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    figure, panels = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    for index, component in enumerate(("N", "E", "U")):
        panels[index].plot(values[:, index], linewidth=0.7)
        panels[index].set_ylabel(component + " (mm)")
        panels[index].grid(alpha=0.2)
    for repeat, panel in enumerate(panels[3:]):
        names = [
            f"direct_visual_r{repeat}",
            f"adaptive_no_feedback_r{repeat}__conflict_scope",
            f"displacement_median_trace_r{repeat}__conflict_scope",
            f"displacement_comparison_map_r{repeat}__conflict_scope",
            "raw_multiscale_rule",
            "xgboost",
        ]
        matrix = [[int(row["activity_label"]) for row in labels]]
        matrix.extend(predictions[name][case.case_id]["activity"] for name in names)
        panel.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            vmin=-1,
            vmax=1,
            cmap=ListedColormap(["#c9c9c9", "#fafafa", "#327b88"]),
            extent=(-0.5, len(values) - 0.5, len(matrix) - 0.5, -0.5),
        )
        panel.set_yticks(
            range(len(matrix)),
            [
                "Reference",
                "Direct",
                "Original local",
                "Median trace",
                "Difference map",
                "Raw rule",
                "XGB",
            ],
        )
        panel.set_ylabel(f"Repeat {repeat}")
    panels[-1].set_xlabel("Original day; teal=activity, white=stationary, gray=unknown/missing")
    panels[-1].set_xlim(-0.5, len(values) - 0.5)
    figure.suptitle(case.case_id + " | Exposed developer reference; post-score diagnostic only")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def report(run, analysis, out):
    verify(run)
    verify_manifest(analysis)
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    corrections, transitions = [], []
    checked_days = 0
    selections = read_json(BASE / "core-lockbox-confirmation-v1/selections.json")
    inherited_cases = 0
    text = [
        "# 同事实位移差图与中位数曲线：完整开发配对",
        "",
        "两组只改变第三张补图的表示，共享原始中位数/噪声尺度、全局初判、选窗、提示、"
        "两遍次数及49日修订权限。使用已有完整24过程记录与预指定12作者参考，均已开发暴露。"
        "不是独立确认，也不复活此前停止的候选。未知不获得正确静稳信用。",
        "",
        "## 预登记判定",
        "",
        "决定：" + read_json(analysis / "decision.json")["decision"],
        "",
    ]
    for target in ("primary_process", "secondary_developer"):
        folder = analysis / target
        predictions = read_json(folder / "predictions.json")
        labels = read_json(BASE / "core-lockbox-analysis-v1" / target / "reference-labels.json")
        rows = csv_rows(folder / "summary.csv")
        old_rows = {row["method"]: row for row in csv_rows(PRIOR / target / "summary.csv")}
        for row in rows:
            if row["method"] in old_rows and row != old_rows[row["method"]]:
                raise ValueError("Historical metrics or deployment costs changed")
        for name, cases in predictions.items():
            if not name.startswith("displacement_"):
                continue
            for cid, prediction in cases.items():
                case, feature = load_case_features(DATA, cid)
                activity = np.array(prediction["activity"])
                stages = np.array(prediction["stage"])
                if np.any(activity[~feature.observed] != -1) or np.any(
                    stages[~feature.observed] != -1
                ):
                    raise ValueError("Display predictions lost the raw missing mask")
                if np.any((stages > 0) & (activity != 1)):
                    raise ValueError("Display stages extend beyond confirmed activity")
                repeat = 0 if "_r0__" in name else 1
                if not selections[f"{cid}-r{repeat}"]["selected"]:
                    initial = predictions[f"direct_visual_r{repeat}"][cid]["activity"]
                    if not np.array_equal(activity, initial):
                        raise ValueError("Zero-review cases must inherit the global activity")
                    inherited_cases += 1
                checked_days += len(activity)
        for repeat in (0, 1):
            control = f"displacement_median_trace_r{repeat}__conflict_scope"
            treatment = f"displacement_comparison_map_r{repeat}__conflict_scope"
            for cid, reference in labels.items():
                actual = np.array([int(row["activity_label"]) for row in reference])
                before = np.array(predictions[control][cid]["activity"])
                after = np.array(predictions[treatment][cid]["activity"])
                corrections.append(
                    {
                        "target": target,
                        "repeat": repeat,
                        "case_id": cid,
                        **correction_utility(actual, before, after, actual >= 0),
                    }
                )
                for reference_state in (-1, 0, 1):
                    for before_state in (-1, 0, 1):
                        for after_state in (-1, 0, 1):
                            transitions.append(
                                {
                                    "target": target,
                                    "repeat": repeat,
                                    "case_id": cid,
                                    "reference": reference_state,
                                    "before": before_state,
                                    "after": after_state,
                                    "days": int(
                                        (
                                            (actual == reference_state)
                                            & (before == before_state)
                                            & (after == after_state)
                                        ).sum()
                                    ),
                                }
                            )
        text.extend(["## " + target, "", full_table(rows), ""])
        # Fixed diagnostic cases span weak constant, platform and weak tails.
        # Their selection does not alter the complete-population scoring gate.
        if target == "secondary_developer":
            image_folder = out / "figures"
            image_folder.mkdir()
            for cid in ("case_0169", "case_0171", "case_0174", "case_0178"):
                if cid in labels:
                    plot_alignment(
                        load_case_features(DATA, cid)[0],
                        labels[cid],
                        predictions,
                        image_folder / (cid + "-alignment.png"),
                    )
    write_csv(out / "per-record-corrections.csv", corrections)
    write_csv(out / "state-transitions.csv", transitions)
    (out / "all-results.md").write_text("\n".join(text), encoding="utf-8")
    write_json(
        out / "validation.json",
        {
            "new_prediction_days": checked_days,
            "missing_and_activity_stage_contract": "passed",
            "zero_review_inheritance_cases_checked": inherited_cases,
            "historical_summary_and_cost_rows": "exactly preserved",
            "new_model_requests": 0,
            "reference_changes": 0,
            "figures": "Generated, not yet claimed visually read",
        },
    )
    write_json(
        out / "manifest.json",
        {
            "files": {
                str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()
            }
        },
    )
    print("Paired diagnostic report:", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.analysis.resolve(), args.out.resolve())
