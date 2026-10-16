"""Export all-record error alignment and verify unchanged baselines and replay."""

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
from probe_landslide_signed_evidence import ANALYSIS, PARENT, score
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_signed_evidence import signed_block_facts
from gnss_sim.landslide_window_experiment import parse_window_activity


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def case_figure(cid, case, predictions, labels, out):
    values = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    fig, axes = plt.subplots(
        5, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 2, 2, 2.5, 2.5]}
    )
    for axis, name in enumerate(("N", "E", "U")):
        axes[axis].plot(
            np.arange(len(values)),
            values[:, axis],
            linewidth=0.7,
            color=("#147d72", "#cc6d4b", "#a38a32")[axis],
        )
        axes[axis].set_ylabel(f"{name} (mm)")
        axes[axis].grid(alpha=0.2)
    for rep, axis in enumerate(axes[3:]):
        methods = [
            f"direct_visual_r{rep}",
            f"adaptive_no_feedback_r{rep}__conflict_scope",
            f"fresh_raw_r{rep}__conflict_scope",
            f"fresh_norm_r{rep}__conflict_scope",
            f"signed_adaptive_r{rep}__conflict_scope",
            f"signed_fixed_r{rep}__whole_target",
            f"signed_first_only_r{rep}__conflict_scope",
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
                "Historical raw",
                "Fresh raw",
                "Fresh norm",
                "Signed selected",
                "Signed fixed",
                "Signed first",
            ],
        )
        axis.set_ylabel(f"Repeat {rep}")
    axes[-1].set_xlabel("Original day index; teal=activity, white=stationary, gray=unknown/missing")
    axes[-1].set_xlim(-0.5, len(values) - 0.5)
    fig.suptitle(f"{cid} | Exposed author-AI reference and all core comparisons; no relabeling")
    fig.tight_layout()
    fig.savefig(out / f"{cid}.png", dpi=150)
    plt.close(fig)


def report(run, analysis, replay_out, out):
    verify_manifest(analysis)
    receipts = verify_receipts(run)
    saved_reviews = read_json(run / "reviews.json")
    facts = read_json(run / "raw-facts.json")
    request_checks = []
    for row in read_json(run / "query-plan.json"):
        case, feature = load_case_features(run, row["case_id"])
        receipt = receipts[row["request_id"]]
        if receipt["prompt"] != row["prompt"] or list(receipt["image_sha256"]) != row["images"]:
            raise ValueError("Actual request differs from its frozen plan")
        try:
            if receipt["status"] != "returned_json":
                raise ValueError("Failed transport")
            activity = parse_window_activity(
                json.loads(receipt["output"]), feature.observed, row["start"], row["stop"]
            )
            status = "success"
        except ValueError:
            activity = np.full(len(feature.observed), -1, dtype=int)
            status = "failed"
        saved = saved_reviews[row["request_id"]]
        if status != saved["status"] or activity.tolist() != saved["activity"]:
            raise ValueError("Saved review differs from the actual model answer")
        rebuilt = signed_block_facts(case, feature.noise, row["start"], row["stop"])
        if rebuilt != facts[f"{row['case_id']}-{row['index']}"]:
            raise ValueError("Raw fact card no longer reconstructs from observations")
        request_checks.append(
            {
                "request_id": row["request_id"],
                "actual_plan_and_answer_equal": True,
                "raw_facts_equal": True,
            }
        )
    score(run, replay_out)
    verify_manifest(replay_out)
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_csv(out / "actual-request-reconstruction.csv", request_checks)
    matches = {}
    for path in analysis.rglob("*"):
        if path.is_file():
            name = path.relative_to(analysis)
            matches[name.as_posix()] = sha(path) == sha(replay_out / name)
    if not all(matches.values()):
        raise ValueError("Offline score replay must match byte for byte")
    write_json(out / "exact-replay.json", matches)
    unchanged = []
    tables = {}
    for target in ("primary_ai", "secondary_process"):
        table = {row["method"]: row for row in rows(analysis / target / "summary.csv")}
        tables[target] = table
        for old in rows(ANALYSIS / target / "summary.csv"):
            if table[old["method"]] != old:
                raise ValueError(f"Original baseline scores changed: {target}/{old['method']}")
            unchanged.append(
                {"target": target, "method": old["method"], "all_score_and_cost_fields_equal": True}
            )
    write_csv(out / "unchanged-baselines.csv", unchanged)
    predictions = read_json(analysis / "primary_ai/predictions.json")
    labels = read_json(ANALYSIS / "primary_ai/reference-labels.json")
    transitions = []
    changes = []
    for cid, reference in labels.items():
        case, feature = load_case_features(run, cid)
        case_figure(cid, case, predictions, reference, out)
        actual = np.array([row["activity_label"] for row in reference])
        for rep in (0, 1):
            main = np.array(predictions[f"signed_adaptive_r{rep}__conflict_scope"][cid]["activity"])
            for stem in ("adaptive_no_feedback", "fresh_raw", "fresh_norm", "signed_first_only"):
                baseline = np.array(predictions[f"{stem}_r{rep}__conflict_scope"][cid]["activity"])
                eligible = feature.observed & (actual >= 0)
                changed = eligible & (main != baseline)
                row = {
                    "case_id": cid,
                    "repeat": rep,
                    "control": stem,
                    "changed_known_days": int(changed.sum()),
                    "tp_gain": int(
                        (eligible & (actual == 1) & (main == 1)).sum()
                        - (eligible & (actual == 1) & (baseline == 1)).sum()
                    ),
                    "fp_gain": int(
                        (eligible & (actual == 0) & (main == 1)).sum()
                        - (eligible & (actual == 0) & (baseline == 1)).sum()
                    ),
                    "unknown_to_wrong_stationary": int(
                        (eligible & (actual == 1) & (baseline == -1) & (main == 0)).sum()
                    ),
                    "unknown_to_wrong_activity": int(
                        (eligible & (actual == 0) & (baseline == -1) & (main == 1)).sum()
                    ),
                }
                transitions.append(row)
                for day in np.flatnonzero(changed):
                    changes.append(
                        {
                            "case_id": cid,
                            "repeat": rep,
                            "control": stem,
                            "day_index": int(day),
                            "date": case.dates[day],
                            "actual": int(actual[day]),
                            "before": int(baseline[day]),
                            "after": int(main[day]),
                        }
                    )
    write_csv(out / "per-record-transitions.csv", transitions)
    write_csv(out / "changed-known-days.csv", changes)
    compact = [
        "| 方法 | AI F1 r0/r1 | AI 弱R r0/r1 | 过程F1 r0/r1 | AI调用 r0/r1 | AI token r0/r1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    methods = [
        ("原冲突复核", "adaptive_no_feedback", "conflict_scope"),
        ("同期原图", "fresh_raw", "conflict_scope"),
        ("同期模摘要", "fresh_norm", "conflict_scope"),
        ("带符号块主方案", "signed_adaptive", "conflict_scope"),
        ("带符号固定", "signed_fixed", "whole_target"),
        ("带符号仅第一窗", "signed_first_only", "conflict_scope"),
    ]
    for label, stem, variant in methods:
        cells = []
        for target, metric in (
            ("primary_ai", "f1"),
            ("primary_ai", "weak_recall"),
            ("secondary_process", "f1"),
            ("primary_ai", "requests"),
            ("primary_ai", "reported_tokens"),
        ):
            cells.append(
                "/".join(
                    f"{float(tables[target][f'{stem}_r{rep}__{variant}'][metric]):.4f}"
                    if metric in {"f1", "weak_recall"}
                    else tables[target][f"{stem}_r{rep}__{variant}"][metric]
                    for rep in (0, 1)
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
            "new_remote_calls_during_replay": 0,
            "request_source": str(PARENT),
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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.analysis.resolve(), args.replay.resolve(), args.out.resolve())
