"""Replay temporal fusion and separate evidence changes from support coverage."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_temporal_fusion import PRIOR_ANALYSIS, REFERENCES, activity_for, score
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def case_figure(cid, case, predictions, labels, out):
    values = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True)
    for index, component in enumerate(("N", "E", "U")):
        axes[index].plot(np.arange(len(values)), values[:, index], linewidth=0.7)
        axes[index].set_ylabel(f"{component} (mm)")
        axes[index].grid(alpha=0.2)
    for rep, axis in enumerate(axes[3:]):
        methods = [
            f"direct_visual_r{rep}",
            f"temporal_joint_r{rep}",
            f"temporal_no_temporal_r{rep}",
            "temporal_numeric_temporal",
            "temporal_numeric_point",
            f"temporal_visual_temporal_r{rep}",
        ]
        matrix = [[row["activity_label"] for row in labels]]
        matrix += [predictions[name][cid]["activity"] for name in methods]
        axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            vmin=-1,
            vmax=1,
            cmap=ListedColormap(["#c9c9c9", "#fafafa", "#327b88"]),
            extent=[-0.5, len(values) - 0.5, len(matrix) - 0.5, -0.5],
        )
        axis.set_yticks(
            range(len(matrix)),
            [
                "Author-AI",
                "Global",
                "Joint",
                "No temporal",
                "Numeric temporal",
                "Numeric point",
                "Visual temporal",
            ],
        )
        axis.set_ylabel(f"Repeat {rep}")
    axes[-1].set_xlabel("Original day; teal=activity, white=stationary, gray=unknown/missing")
    axes[-1].set_xlim(-0.5, len(values) - 0.5)
    fig.suptitle(f"{cid} | Raw observations and frozen development predictions")
    fig.tight_layout()
    fig.savefig(out / f"{cid}.png", dpi=150)
    plt.close(fig)


def change_counts(actual, before, after, eligible):
    return {
        "eligible_days": int(eligible.sum()),
        "gained_correct": int((eligible & (before != actual) & (after == actual)).sum()),
        "lost_correct": int((eligible & (before == actual) & (after != actual)).sum()),
        "tp_gain": int(
            (eligible & (actual == 1) & (after == 1)).sum()
            - (eligible & (actual == 1) & (before == 1)).sum()
        ),
        "fp_gain": int(
            (eligible & (actual == 0) & (after == 1)).sum()
            - (eligible & (actual == 0) & (before == 1)).sum()
        ),
        "unknown_to_correct_stationary": int(
            (eligible & (actual == 0) & (before == -1) & (after == 0)).sum()
        ),
        "unknown_to_correct_activity": int(
            (eligible & (actual == 1) & (before == -1) & (after == 1)).sum()
        ),
    }


def report(run, analysis, replay, out):
    verify_manifest(analysis)
    score(run, replay)
    verify_manifest(replay)
    matches = {
        path.relative_to(analysis).as_posix(): sha(path) == sha(replay / path.relative_to(analysis))
        for path in analysis.rglob("*")
        if path.is_file()
    }
    if not all(matches.values()):
        raise ValueError("Every frozen score file must replay byte for byte")
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "exact-replay.json", matches)
    tables, unchanged, changes, transitions, matched = {}, [], [], [], []
    choices = read_json(run / "choices.json")
    probabilities = read_json(run / "numeric-probabilities.json")
    for target in ("primary_ai", "secondary_process"):
        tables[target] = {row["method"]: row for row in rows(analysis / target / "summary.csv")}
        for old in rows(PRIOR_ANALYSIS / target / "summary.csv"):
            if tables[target][old["method"]] != old:
                raise ValueError("A historical baseline metric or cost changed")
            unchanged.append({"target": target, "method": old["method"], "all_fields_equal": True})
        labels = read_json(REFERENCES / target / "reference-labels.json")
        predictions = read_json(analysis / target / "predictions.json")
        selected = choices[target]["joint"]
        for cid, reference in labels.items():
            case, feature = load_case_features(run, cid)
            if target == "primary_ai":
                case_figure(cid, case, predictions, reference, out)
            actual = np.array([row["activity_label"] for row in reference])
            eligible = feature.observed & (actual >= 0)
            # This post-score diagnostic never selects a new threshold or method.
            # It removes only the visual vote at the joint arm's frozen threshold.
            no_visual = activity_for(
                feature,
                np.array(probabilities[target][cid]),
                np.full(len(actual), -1),
                selected["threshold"],
                0,
                selected["cost"],
            )
            for rep in (0, 1):
                main = np.array(predictions[f"temporal_joint_r{rep}"][cid]["activity"])
                for scope, mask in (
                    ("all_known", eligible),
                    ("numeric_supported", eligible & feature.supported),
                    ("numeric_unsupported", eligible & ~feature.supported),
                ):
                    matched.append(
                        {
                            "target": target,
                            "case_id": cid,
                            "repeat": rep,
                            "scope": scope,
                            "threshold": selected["threshold"],
                            "diagnostic_only_not_selected": True,
                            **change_counts(actual, no_visual, main, mask),
                        }
                    )
                for control in (
                    f"direct_visual_r{rep}",
                    f"temporal_no_temporal_r{rep}",
                    "temporal_numeric_temporal",
                    f"temporal_visual_temporal_r{rep}",
                ):
                    before = np.array(predictions[control][cid]["activity"])
                    transitions.append(
                        {
                            "target": target,
                            "case_id": cid,
                            "repeat": rep,
                            "control": control,
                            **change_counts(actual, before, main, eligible),
                        }
                    )
                    for day in np.flatnonzero(eligible & (main != before)):
                        changes.append(
                            {
                                "target": target,
                                "case_id": cid,
                                "repeat": rep,
                                "control": control,
                                "day_index": int(day),
                                "date": case.dates[day],
                                "actual": int(actual[day]),
                                "before": int(before[day]),
                                "after": int(main[day]),
                                "numeric_supported": bool(feature.supported[day]),
                            }
                        )
    write_csv(out / "unchanged-baselines.csv", unchanged)
    write_csv(out / "changed-known-days.csv", changes)
    write_csv(out / "per-record-transitions.csv", transitions)
    write_csv(out / "matched-threshold-visual-diagnostic.csv", matched)
    compact = [
        "| 方法 | AI P r0/r1 | AI R r0/r1 | AI F1 r0/r1 | 事件F1(.3) r0/r1 | 弱R r0/r1 | 过程F1 r0/r1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, stem in (
        ("直接视觉", "direct_visual_r{}"),
        ("连续运动融合主方法", "temporal_joint_r{}"),
        ("去时间连续性", "temporal_no_temporal_r{}"),
        ("去视觉，独立验证阈值", "temporal_numeric_temporal"),
        ("数值逐日，独立验证阈值", "temporal_numeric_point"),
        ("去数值反馈", "temporal_visual_temporal_r{}"),
    ):
        cells = []
        for target, metric in (
            ("primary_ai", "precision"),
            ("primary_ai", "recall"),
            ("primary_ai", "f1"),
            ("primary_ai", "event_f1_iou03"),
            ("primary_ai", "weak_recall"),
            ("secondary_process", "f1"),
        ):
            cells.append(
                "/".join(f"{float(tables[target][stem.format(rep)][metric]):.4f}" for rep in (0, 1))
            )
        compact.append("| " + " | ".join([label] + cells) + " |")
    (out / "compact-table.md").write_text("\n".join(compact), encoding="utf-8")
    verification = {
        "exact_replay_files": len(matches),
        "unchanged_baseline_target_rows": len(unchanged),
        "all_record_figures": len(labels),
        "new_remote_calls_during_replay": 0,
        "matched_threshold_diagnostic_selected": False,
        "default_changed": False,
    }
    write_json(out / "verification.json", verification)
    write_json(
        out / "manifest.json",
        {
            "files": {
                path.relative_to(out).as_posix(): sha(path)
                for path in out.rglob("*")
                if path.is_file()
            }
        },
    )
    print(verification)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "analysis", "replay", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.analysis.resolve(), args.replay.resolve(), args.out.resolve())
