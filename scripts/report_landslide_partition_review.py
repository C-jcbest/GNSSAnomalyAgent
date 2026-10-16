"""Replay the frozen partition experiment and export observation/error alignments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_review_utility_v2 import verify_receipts
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_partition_review import PRIOR, REFERENCES, review_from_receipt, score
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_baselines import pelt_segment_rates
from gnss_sim.landslide_partition_review import rate_partition, uniform_partition


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def case_figure(cid, case, predictions, labels, out):
    values = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    fig, axes = plt.subplots(
        5, 1, figsize=(14, 11), sharex=True, gridspec_kw={"height_ratios": [2, 2, 2, 3, 3]}
    )
    for index, name in enumerate(("N", "E", "U")):
        axes[index].plot(np.arange(len(values)), values[:, index], linewidth=0.7)
        axes[index].set_ylabel(f"{name} (mm)")
        axes[index].grid(alpha=0.2)
    for rep, axis in enumerate(axes[3:]):
        methods = [f"direct_visual_r{rep}", f"fresh_norm_r{rep}__conflict_scope"]
        methods += [
            f"partition_{stem}_r{rep}__conflict_scope"
            for stem in ("free_norm", "pelt", "uniform_matched", "uniform30", "no_numeric", "first")
        ]
        matrix = [[row["activity_label"] for row in labels]]
        matrix += [predictions[name][cid]["activity"] for name in methods]
        axis.imshow(
            matrix,
            origin="upper",
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
                "Prior norm",
                "Fresh free norm",
                "PELT pieces",
                "Matched uniform",
                "Uniform 30",
                "No numeric",
                "First only",
            ],
        )
        axis.set_ylabel(f"Repeat {rep}")
    axes[-1].set_xlabel("Original day index; teal=activity, white=stationary, gray=unknown/missing")
    axes[-1].set_xlim(-0.5, len(values) - 0.5)
    fig.suptitle(f"{cid} | Developer reference and frozen predictions; no relabeling")
    fig.tight_layout()
    fig.savefig(out / f"{cid}.png", dpi=150)
    plt.close(fig)


def report(run, analysis, replay, out):
    verify_manifest(analysis)
    receipts = verify_receipts(run)
    partitions = read_json(run / "partitions.json")
    reviews = read_json(run / "reviews.json")
    penalty = read_json(run / "protocol.json")["penalty"]
    rates_by_case = {}
    checks = []
    for row in read_json(run / "query-plan.json"):
        receipt = receipts[row["request_id"]]
        case, feature = load_case_features(run, row["case_id"])
        if receipt["prompt"] != row["prompt"] or list(receipt["image_sha256"]) != row["images"]:
            raise ValueError("Actual question differs from frozen plan")
        try:
            if receipt["status"] != "returned_json":
                raise ValueError("Failed transport")
            activity = review_from_receipt(row, json.loads(receipt["output"]), feature, partitions)
            status = "success"
        except ValueError:
            activity = np.full(len(feature.observed), -1, dtype=int)
            status = "failed"
        if (
            reviews[row["request_id"]]["activity"] != activity.tolist()
            or reviews[row["request_id"]]["status"] != status
        ):
            raise ValueError("Stored decision does not reproduce from actual answer")
        start, stop = row["start"], row["stop"]
        if row["case_id"] not in rates_by_case:
            rates_by_case[row["case_id"]] = pelt_segment_rates(feature, penalty)
        pelt = rate_partition(rates_by_case[row["case_id"]], start, stop)
        geometries = {
            "pelt": pelt,
            "uniform_matched": uniform_partition(len(pelt), start, stop),
            "uniform30": uniform_partition((stop - start + 29) // 30, start, stop),
        }
        if geometries != partitions[f"{row['case_id']}-{row['index']}"]:
            raise ValueError("Partition does not reconstruct from observation-derived fits")
        checks.append({"request_id": row["request_id"], "actual_plan_answer_partition_equal": True})
    score(run, replay)
    verify_manifest(replay)
    matches = {
        path.relative_to(analysis).as_posix(): sha(path) == sha(replay / path.relative_to(analysis))
        for path in analysis.rglob("*")
        if path.is_file()
    }
    if not all(matches.values()):
        raise ValueError("Offline replay must match every file byte for byte")
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_csv(out / "actual-request-reconstruction.csv", checks)
    write_json(out / "exact-replay.json", matches)
    unchanged, tables = [], {}
    for target in ("primary_ai", "secondary_process"):
        tables[target] = {row["method"]: row for row in rows(analysis / target / "summary.csv")}
        for old in rows(PRIOR / target / "summary.csv"):
            if tables[target][old["method"]] != old:
                raise ValueError("Prior baseline score or cost changed")
            unchanged.append({"target": target, "method": old["method"], "all_fields_equal": True})
    write_csv(out / "unchanged-baselines.csv", unchanged)
    labels = read_json(REFERENCES / "primary_ai/reference-labels.json")
    predictions = read_json(analysis / "primary_ai/predictions.json")
    transitions, changes = [], []
    for cid, reference in labels.items():
        case, feature = load_case_features(run, cid)
        case_figure(cid, case, predictions, reference, out)
        actual = np.array([row["activity_label"] for row in reference])
        eligible = feature.observed & (actual >= 0)
        for rep in (0, 1):
            main = np.array(predictions[f"partition_pelt_r{rep}__conflict_scope"][cid]["activity"])
            for stem in ("free_norm", "uniform_matched", "uniform30", "no_numeric", "first"):
                before = np.array(
                    predictions[f"partition_{stem}_r{rep}__conflict_scope"][cid]["activity"]
                )
                changed = eligible & (main != before)
                transitions.append(
                    {
                        "case_id": cid,
                        "repeat": rep,
                        "control": stem,
                        "changed_known_days": int(changed.sum()),
                        "tp_gain": int(
                            (eligible & (actual == 1) & (main == 1)).sum()
                            - (eligible & (actual == 1) & (before == 1)).sum()
                        ),
                        "fp_gain": int(
                            (eligible & (actual == 0) & (main == 1)).sum()
                            - (eligible & (actual == 0) & (before == 1)).sum()
                        ),
                        "unknown_to_wrong_stationary": int(
                            (eligible & (actual == 1) & (before == -1) & (main == 0)).sum()
                        ),
                        "unknown_to_wrong_activity": int(
                            (eligible & (actual == 0) & (before == -1) & (main == 1)).sum()
                        ),
                    }
                )
                for day in np.flatnonzero(changed):
                    changes.append(
                        {
                            "case_id": cid,
                            "repeat": rep,
                            "control": stem,
                            "day_index": int(day),
                            "date": case.dates[day],
                            "actual": int(actual[day]),
                            "before": int(before[day]),
                            "after": int(main[day]),
                        }
                    )
    write_csv(out / "per-record-transitions.csv", transitions)
    write_csv(out / "changed-known-days.csv", changes)
    compact = [
        "| 方法 | AI F1 r0/r1 | 弱R r0/r1 | 过程F1 r0/r1 | 请求 r0/r1 | token r0/r1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, stem, variant in (
        ("同期自由区间模摘要", "free_norm", "conflict_scope"),
        ("PELT分段主方法", "pelt", "conflict_scope"),
        ("同段数均匀", "uniform_matched", "conflict_scope"),
        ("30日均匀", "uniform30", "conflict_scope"),
        ("去数值反馈", "no_numeric", "conflict_scope"),
        ("固定分段整窗", "fixed", "whole_target"),
        ("仅第一窗", "first", "conflict_scope"),
    ):
        cells = []
        for target, metric in (
            ("primary_ai", "f1"),
            ("primary_ai", "weak_recall"),
            ("secondary_process", "f1"),
            ("primary_ai", "requests"),
            ("primary_ai", "reported_tokens"),
        ):
            values = [
                tables[target][f"partition_{stem}_r{rep}__{variant}"][metric] for rep in (0, 1)
            ]
            cells.append(
                "/".join(
                    f"{float(value):.4f}" if metric in ("f1", "weak_recall") else value
                    for value in values
                )
            )
        compact.append("| " + " | ".join([label] + cells) + " |")
    (out / "compact-table.md").write_text("\n".join(compact), encoding="utf-8")
    write_json(
        out / "verification.json",
        {
            "exact_replay_files": len(matches),
            "unchanged_baseline_target_rows": len(unchanged),
            "all_record_figures": len(labels),
            "actual_requests_reconstructed": len(checks),
            "new_remote_calls_during_replay": 0,
            "default_changed": False,
        },
    )
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
    print(read_json(out / "verification.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "analysis", "replay", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.analysis.resolve(), args.replay.resolve(), args.out.resolve())
