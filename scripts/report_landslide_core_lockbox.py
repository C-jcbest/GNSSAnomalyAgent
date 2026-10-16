"""Export all registered trial results, family breakdowns and aligned raw figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores, runs


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def full_table(rows):
    lines = [
        "| 方法 | P | R | F1 | 事件F1 .3/.5 | 弱R | FP/FN | 起/止MAE日 | 内部/端到端阶段F1 | 调用 | token |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:

        def number(key):
            value = row[key]
            return f"{float(value):.4f}" if value not in (None, "") else "—"

        lines.append(
            f"| {row['method']} | {number('precision')} | {number('recall')} | {number('f1')} | "
            f"{number('event_f1_iou03')}/{number('event_f1_iou05')} | {number('weak_recall')} | "
            f"{row['fp']}/{row['fn']} | {number('onset_core_mae_days')}/{number('offset_core_mae_days')} | "
            f"{number('stage_internal_macro_f1')}/{number('stage_end_to_end_macro_f1')} | "
            f"{row['requests']} | {row['reported_tokens']} |"
        )
    return "\n".join(lines)


def aligned_figure(case, labels, predictions, out):
    values = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    for axis, component in enumerate(("N", "E", "U")):
        axes[axis].plot(np.arange(len(values)), values[:, axis], linewidth=0.7)
        axes[axis].set_ylabel(component + " (mm)")
        axes[axis].grid(alpha=0.2)
    for rep, panel in enumerate(axes[3:]):
        names = [
            f"direct_visual_r{rep}",
            f"adaptive_no_feedback_r{rep}__conflict_scope",
            f"fixed_image_review_r{rep}__whole_target",
            f"same_count_fixed_early_r{rep}__conflict_scope",
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
            ["Reference", "Direct", "Main", "Fixed", "Same K early", "Raw rule", "XGB"],
        )
        panel.set_ylabel(f"Repeat {rep}")
    axes[-1].set_xlabel("Original day; teal=activity, white=stationary, gray=unknown/missing")
    axes[-1].set_xlim(-0.5, len(values) - 0.5)
    fig.suptitle(case.case_id + " | Frozen developer reference; raw observations and both repeats")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def report(data, run, analysis, replay, out):
    verify_manifest(analysis)
    verify_manifest(replay)
    matched = {}
    for path in analysis.rglob("*"):
        if path.is_file():
            name = path.relative_to(analysis).as_posix()
            matched[name] = sha(path) == sha(replay / name)
    if not all(matched.values()):
        raise ValueError("Deterministic replay differs")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    ledger = read_json(data / "mechanism-sealed/generation-ledger.json")
    families = {row["case_id"]: row["truth"]["family"] for row in ledger}
    tables, per_family, changes, all_daily = {}, [], [], []
    for target in ("primary_process", "secondary_developer"):
        folder = analysis / target
        summaries = csv_rows(folder / "summary.csv")
        tables[target] = {row["method"]: row for row in summaries}
        results = read_json(folder / "results.json")
        labels = read_json(folder / "reference-labels.json")
        predictions = read_json(folder / "predictions.json")
        per_record = csv_rows(folder / "per-record.csv")
        (out / f"{target}-all-methods.md").write_text(
            full_table(summaries) + "\n", encoding="utf-8"
        )
        for method, result in results.items():
            for family in sorted(set(families.values())):
                ids = [cid for cid in labels if families[cid] == family]
                if not ids:
                    continue
                aggregate = aggregate_scores([result["per_case"][cid] for cid in ids])
                subset = [
                    row for row in per_record if row["method"] == method and row["case_id"] in ids
                ]
                weak_count = sum(int(row["weak_days"]) for row in subset)
                found = sum(int(row["weak_detected_days"]) for row in subset)
                per_family.append(
                    {
                        "target": target,
                        "method": method,
                        "family": family,
                        "records": len(ids),
                        **aggregate["daily"],
                        "event_f1": aggregate["events"]["f1"],
                        "weak_days": weak_count,
                        "weak_detected_days": found,
                        "weak_recall": found / weak_count if weak_count else None,
                    }
                )
        for cid, rows in labels.items():
            actual = np.array([int(row["activity_label"]) for row in rows])
            for method, records in predictions.items():
                all_daily.extend(
                    {
                        "target": target,
                        "method": method,
                        "case_id": cid,
                        "day_index": day,
                        "date": row["date"],
                        "actual_activity": int(actual[day]),
                        "actual_feature": row["feature"],
                        "predicted_activity": records[cid]["activity"][day],
                        "predicted_stage": records[cid]["stage"][day],
                    }
                    for day, row in enumerate(rows)
                )
            for rep in (0, 1):
                main = f"adaptive_no_feedback_r{rep}__conflict_scope"
                after = np.array(predictions[main][cid]["activity"])
                for control in (
                    f"direct_visual_r{rep}",
                    f"fixed_image_review_r{rep}__whole_target",
                    f"same_count_fixed_early_r{rep}__conflict_scope",
                    f"adaptive_numeric_feedback_r{rep}__conflict_scope",
                ):
                    before = np.array(predictions[control][cid]["activity"])
                    eligible = actual >= 0
                    gained = eligible & (before != actual) & (after == actual)
                    lost = eligible & (before == actual) & (after != actual)
                    changes.append(
                        {
                            "target": target,
                            "case_id": cid,
                            "family": families[cid],
                            "control": control,
                            "treatment": main,
                            "gained_correct": int(gained.sum()),
                            "lost_correct": int(lost.sum()),
                            "net_correct": int(gained.sum() - lost.sum()),
                            "gained_tp": int(((before != 1) & (after == 1) & (actual == 1)).sum()),
                            "lost_tp": int(((before == 1) & (after != 1) & (actual == 1)).sum()),
                            "removed_fp": int(((before == 1) & (after != 1) & (actual == 0)).sum()),
                            "gained_fp": int(((before != 1) & (after == 1) & (actual == 0)).sum()),
                            "changed_known_runs": runs((before != after) & eligible),
                        }
                    )
            if target == "secondary_developer":
                case, _ = load_case_features(data, cid)
                aligned_figure(case, rows, predictions, out / f"{cid}-aligned.png")
    write_csv(out / "per-family.csv", per_family)
    write_csv(out / "paired-record-corrections.csv", changes)
    write_csv(out / "all-daily-predictions.csv", all_daily)
    selected = [
        ("Main r0", "adaptive_no_feedback_r0__conflict_scope"),
        ("Main r1", "adaptive_no_feedback_r1__conflict_scope"),
        ("Direct r0", "direct_visual_r0"),
        ("Direct r1", "direct_visual_r1"),
        ("Fixed r0", "fixed_image_review_r0__whole_target"),
        ("Fixed r1", "fixed_image_review_r1__whole_target"),
        ("Same K early r0", "same_count_fixed_early_r0__conflict_scope"),
        ("Same K early r1", "same_count_fixed_early_r1__conflict_scope"),
        ("Numeric feedback r0", "adaptive_numeric_feedback_r0__conflict_scope"),
        ("Numeric feedback r1", "adaptive_numeric_feedback_r1__conflict_scope"),
    ]
    selected += [
        (name, name)
        for name in tables["primary_process"]
        if not any(f"_r{rep}" in name for rep in (0, 1))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
    for panel, target in zip(axes, tables, strict=True):
        values = [
            float(tables[target][name]["f1"]) if name in tables[target] else np.nan
            for _, name in selected
        ]
        panel.barh(
            range(len(selected)),
            values,
            color=["#327b88" if label.startswith("Main") else "#87949b" for label, _ in selected],
        )
        panel.set_yticks(range(len(selected)), [label for label, _ in selected])
        panel.set_xlim(0, 1)
        panel.set_xlabel("Activity F1; unavailable controls left blank")
        panel.set_title(
            "Full24 designed-motion recovery"
            if target == "primary_process"
            else "Fixed12 developer observable target"
        )
        panel.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    receipts = list((run / "requests").glob("*.response.json"))
    cost = {
        "requests_started": len(list((run / "requests").glob("*.started.json"))),
        "responses": len(receipts),
        "reported_tokens": sum(
            read_json(path).get("usage", {}).get("total_tokens", 0) for path in receipts
        ),
        "inference_wall_seconds": read_json(run / "state.json")["seconds"],
        "task_failures": len(read_json(run / "failures.json")),
        "new_records": 24,
        "developer_calendar_days": 8760,
        "retries": 0,
        "new_remote_requests_for_report": 0,
    }
    write_json(out / "cost.json", cost)
    write_json(
        out / "replay-audit.json",
        {
            "files_byte_equal": matched,
            "all_equal": True,
            "new_remote_requests": 0,
            "zero_weak_denominator_marked_unavailable": True,
        },
    )
    write_json(
        out / "manifest.json",
        {
            "files": {
                path.relative_to(out).as_posix(): sha(path)
                for path in sorted(out.rglob("*"))
                if path.is_file()
            }
        },
    )
    print(cost)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "analysis", "replay", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report(args.data, args.run, args.analysis, args.replay, args.out)
