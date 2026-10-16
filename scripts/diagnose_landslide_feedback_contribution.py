"""Measure existing numerical feedback apart from global-output failure recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores
from gnss_sim.landslide_review_utility import correction_utility


def diagnose(analysis: Path, out: Path):
    verify_manifest(analysis)
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "protocol.json",
        {
            "question": "Does the existing numerical-text arm improve normal-output recognition, beyond pure local images?",
            "status": "Post-score exploratory development; not a new independent confirmation",
            "new_remote_requests": 0,
            "populations": ["complete", "valid_initial_repeat", "valid_initial_both_repeats"],
            "exclusion_rule": "Exclude initial status failed, by repeat or union of both repeats; never use labels to exclude",
            "next_action_rule": "Complete same-feedback fixed-placement control only if process valid-initial comparisons to direct and pure local each have positive F1 and net corrections in both repeats",
            "old_registered_decision_unchanged": True,
            "source_manifest_sha256": sha(analysis / "manifest.json"),
        },
    )
    rows, records = [], []
    for target in ("primary_process", "secondary_developer"):
        folder = analysis / target
        predictions = read_json(folder / "predictions.json")
        results = read_json(folder / "results.json")
        labels = read_json(folder / "reference-labels.json")
        per_record = read_record_rows(folder / "per-record.csv")
        failed_by_repeat = {
            rep: {
                cid
                for cid in labels
                if predictions[f"direct_visual_r{rep}"][cid]["status"] == "failed"
            }
            for rep in (0, 1)
        }
        failed_union = failed_by_repeat[0] | failed_by_repeat[1]
        for rep in (0, 1):
            treatment = f"adaptive_numeric_feedback_r{rep}__conflict_scope"
            controls = [
                f"direct_visual_r{rep}",
                f"adaptive_no_feedback_r{rep}__conflict_scope",
                f"fixed_image_review_r{rep}__whole_target",
                f"same_count_fixed_early_r{rep}__conflict_scope",
            ]
            controls.extend(
                method
                for method in predictions
                if not any(f"_r{repeat}" in method for repeat in (0, 1))
            )
            for population, excluded in (
                ("complete", set()),
                ("valid_initial_repeat", failed_by_repeat[rep]),
                ("valid_initial_both_repeats", failed_union),
            ):
                ids = [cid for cid in labels if cid not in excluded]
                for control in controls:
                    comparison = compare_cases(
                        labels, predictions, results, per_record, ids, control, treatment
                    )
                    rows.append(
                        {
                            "target": target,
                            "repeat": rep,
                            "population": population,
                            "excluded": ";".join(sorted(excluded)),
                            "records": len(ids),
                            "control": control,
                            "treatment": treatment,
                            **comparison,
                        }
                    )
                    if population == "complete":
                        for cid in ids:
                            actual = np.array([int(day["activity_label"]) for day in labels[cid]])
                            utility = correction_utility(
                                actual,
                                np.array(predictions[control][cid]["activity"]),
                                np.array(predictions[treatment][cid]["activity"]),
                                actual >= 0,
                            )
                            records.append(
                                {
                                    "target": target,
                                    "repeat": rep,
                                    "case_id": cid,
                                    "initial_failed": cid in failed_by_repeat[rep],
                                    "control": control,
                                    "treatment": treatment,
                                    **utility,
                                }
                            )
    write_csv(out / "comparisons.csv", rows)
    write_csv(out / "per-record-corrections.csv", records)
    required = [
        row
        for row in rows
        if row["target"] == "primary_process"
        and row["population"] == "valid_initial_repeat"
        and row["control"].startswith(("direct_visual", "adaptive_no_feedback"))
    ]
    proceed = len(required) == 4 and all(
        row["f1_difference"] > 0 and row["net_correct"] > 0 for row in required
    )
    write_json(
        out / "decision.json",
        {
            "next_action": "COMPLETE_SAME_FEEDBACK_CONTROL"
            if proceed
            else "DO_NOT_PROMOTE_FEEDBACK",
            "process_normal_initial_checks": required,
            "old_core_gate": "NOT_CONFIRMED; unchanged",
            "independent_confirmation": False,
        },
    )
    write_json(
        out / "manifest.json",
        {"files": {path.relative_to(out).as_posix(): sha(path) for path in sorted(out.iterdir())}},
    )
    print(
        "Feedback development decision:",
        "complete placement control" if proceed else "do not promote",
    )


def read_record_rows(path):
    import csv

    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {(row["method"], row["case_id"]): row for row in csv.DictReader(stream)}


def compare_cases(labels, predictions, results, per_record, ids, control, treatment):
    before = aggregate_scores([results[control]["per_case"][cid] for cid in ids])
    after = aggregate_scores([results[treatment]["per_case"][cid] for cid in ids])
    utility_rows = []
    for cid in ids:
        actual = np.array([int(day["activity_label"]) for day in labels[cid]])
        utility_rows.append(
            correction_utility(
                actual,
                np.array(predictions[control][cid]["activity"]),
                np.array(predictions[treatment][cid]["activity"]),
                actual >= 0,
            )
        )
    weak_days = sum(int(per_record[(treatment, cid)]["weak_days"]) for cid in ids)

    def weak_recall(method):
        found = sum(int(per_record[(method, cid)]["weak_detected_days"]) for cid in ids)
        return found / weak_days if weak_days else None

    return {
        "control_f1": before["daily"]["f1"],
        "treatment_f1": after["daily"]["f1"],
        "f1_difference": after["daily"]["f1"] - before["daily"]["f1"],
        "control_event_f1": before["events"]["f1"],
        "treatment_event_f1": after["events"]["f1"],
        "control_fp": before["daily"]["fp"],
        "treatment_fp": after["daily"]["fp"],
        "control_fn": before["daily"]["fn"],
        "treatment_fn": after["daily"]["fn"],
        "weak_days": weak_days,
        "control_weak_recall": weak_recall(control),
        "treatment_weak_recall": weak_recall(treatment),
        "gained_correct": sum(row["gained_correct"] for row in utility_rows),
        "lost_correct": sum(row["lost_correct"] for row in utility_rows),
        "net_correct": sum(row["net_correct"] for row in utility_rows),
        "benefited_records": sum(row["net_correct"] > 0 for row in utility_rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.analysis, args.out)
