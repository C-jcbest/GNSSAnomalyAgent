"""Export scenario tradeoffs and verify weak-motion denominators after scoring."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_observation_probe import csv_rows
from run_landslide_comparison import save_json, sha
from run_landslide_evidence_display_confirmation import verify
from run_landslide_window_optimization import read_json
from verify_landslide_evidence_display_confirmation import verify_live_sources

from gnss_sim.landslide_evaluation import aggregate_scores


def summarize(run, analysis, out):
    verify(run, generated=True)
    live_sources = verify_live_sources(run)
    for name, expected in read_json(analysis / "manifest.json")["files"].items():
        if sha(analysis / name) != expected:
            raise ValueError("Confirmation scoring artifact changed")
    labels = {}
    with (run / "reference-subset.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels.setdefault(row["case_id"], []).append(row)
    predictions = read_json(run / "predictions.json")
    results = read_json(analysis / "results.json")
    families = {cid: read_json(run / "mechanism-reference" / cid / "truth.json")["family"] for cid in labels}
    weak_checks, scenario_rows = {}, []
    for name, cases in predictions.items():
        actual_total, detected_total = 0, 0
        for cid, value in cases.items():
            weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            predicted = np.array(value["activity"])
            if np.any(weak & (actual != 1)):
                raise ValueError("Weak denominator contains missing or nonactive dates")
            counts = (int(weak.sum()), int(np.count_nonzero(weak & (predicted == 1))))
            measured = results[name]["per_case"][cid]
            if counts != (measured["weak_days"], measured["weak_detected_days"]):
                raise ValueError("Independent weak-motion per-record counts differ")
            actual_total += counts[0]
            detected_total += counts[1]
        aggregate = results[name]["aggregate"]
        if (actual_total, detected_total) != (aggregate["weak_days"], aggregate["weak_detected_days"]):
            raise ValueError("Independent weak-motion population differs")
        weak_checks[name] = {"weak_reference_days": actual_total, "detected_days": detected_total,
                             "recall": detected_total / actual_total}
        for family in sorted(set(families.values())):
            ids = [cid for cid in labels if families[cid] == family]
            score = aggregate_scores([results[name]["per_case"][cid] for cid in ids])
            scenario_rows.append({"method": name, "family": family, "records": len(ids),
                **score["daily"], "weak_recall": score["weak_recall"], "weak_days": score["weak_days"],
                "event_f1_iou03": score["events"]["f1"], "coverage": score["coverage"]})
    with (analysis / "changes.csv").open(encoding="utf-8", newline="") as handle:
        changes = list(csv.DictReader(handle))
    churn = []
    for rep in range(2):
        candidate = f"adaptive84_evidence_display_r{rep}"
        for control in sorted({row["control"] for row in changes if row["candidate"] == candidate}):
            rows = [row for row in changes if row["candidate"] == candidate and row["control"] == control]
            counts = {key: sum(int(row[key]) for row in rows) for key in ("lost_tp", "gained_tp", "removed_fp", "gained_fp")}
            a, b = results[candidate]["aggregate"]["daily"], results[control]["aggregate"]["daily"]
            if counts["gained_tp"] - counts["lost_tp"] != a["tp"] - b["tp"]:
                raise ValueError("TP change decomposition does not conserve totals")
            if counts["gained_fp"] - counts["removed_fp"] != a["fp"] - b["fp"]:
                raise ValueError("FP change decomposition does not conserve totals")
            churn.append({"repeat": rep + 1, "control": control, **counts})
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "independent-weak-verification.json", weak_checks)
    csv_rows(out / "family-summary.csv", scenario_rows)
    csv_rows(out / "whole-record-changes.csv", churn)
    selected = read_json(run / "selections.json")
    choices = []
    for key, row in selected.items():
        for value in row["display_choices"]:
            choices.append({"case_repeat": key, **value})
    if choices:
        csv_rows(out / "display-choices.csv", choices)
    fig, panels = plt.subplots(1, 2, figsize=(12, 4.5))
    family_names = sorted(set(families.values()))
    index = {(row["method"], row["family"]): row for row in scenario_rows}
    for rep, panel in enumerate(panels):
        candidate = f"adaptive84_evidence_display_r{rep}"
        control = f"adaptive84_narrow_raw_r{rep}"
        differences = {metric: [index[candidate, family][metric] - index[control, family][metric]
                                for family in family_names] for metric in ("tp", "fp")}
        for ordinal, (metric, values) in enumerate(differences.items()):
            panel.bar(np.arange(len(family_names)) + (ordinal - .5) * .35, values, width=.33,
                      label=f"delta {metric.upper()}")
        panel.axhline(0, color="#333333", linewidth=.7)
        panel.set_xticks(range(len(family_names)), family_names, rotation=35, ha="right", fontsize=9)
        panel.set_title(f"repeat {rep + 1}")
        panel.set_ylabel("Candidate minus same-window raw84 (days)")
        panel.legend()
        panel.grid(axis="y", alpha=.2)
    fig.suptitle("New controlled confirmation: display choice tradeoffs across all six families")
    fig.tight_layout()
    fig.savefig(out / "family-tradeoffs.png", dpi=180)
    plt.close(fig)
    save_json(out / "verification.json", {"live_source_count": len(live_sources),
        "independent_weak_groups": len(weak_checks), "scenario_groups": len(scenario_rows),
        "conserved_churn_comparisons": len(churn), "source_sha256": sha(Path(__file__))})
    save_json(out / "manifest.json", {"analysis_manifest_sha256": sha(analysis / "manifest.json"),
        "files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Weak/scenario/churn verification complete", len(weak_checks), len(scenario_rows), len(churn))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run, args.analysis, args.out)


if __name__ == "__main__":
    main()
