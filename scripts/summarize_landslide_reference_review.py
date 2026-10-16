"""Export full-record family results, permissions and reference-control diagnostics."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_observation_probe import csv_rows, evaluate_population
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_reference_review_development import BASE, verify
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import STAGES


def summarize(run, analysis, out):
    verify(run)
    for name, expected in read_json(analysis / "manifest.json")["files"].items():
        if sha(analysis / name) != expected:
            raise ValueError("Full-record analysis changed")
    config = read_json(run / "protocol.json")
    records = {cid: load_case_features(run, cid) for cid in config["case_ids"]}
    labels = load_labels(analysis / "reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    predictions = read_json(analysis / "predictions.json")
    initial = read_json(run / "initial-predictions.json")
    selections = read_json(run / "selections.json")
    permissions, changes, weak_checks = [], [], []
    for rep in (0, 1):
        for cid, (case, feature) in records.items():
            original = np.array(initial[f"direct_visual_r{rep}"][cid]["activity"])
            _, neighbourhood = conflict_neighbourhood(original, feature, raw_motion_evidence(case, feature), ReviewPolicy())
            windows = regular_windows(len(case.dates), 180, 90)
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            chosen = selections[f"{cid}-r{rep}"]
            for flow in ("adaptive", "fixed"):
                targets = [{"start": windows[index][0], "stop": windows[index][1]} for index in chosen[f"{flow}_indices"]]
                conflict, center, _ = target_scopes(neighbourhood, feature.observed, targets)
                if int(conflict.sum()) != int(center.sum()):
                    raise ValueError("Matched-center permission budget differs")
                for arm in ("raw", "reference_residual"):
                    for variant, scope in (("conflict_scope", conflict), ("matched_center_scope", center)):
                        method = f"{flow}_{arm}_r{rep}__{variant}"
                        result = np.array(predictions[method][cid]["activity"])
                        whole = np.array(predictions[f"{flow}_{arm}_r{rep}__whole_target"][cid]["activity"])
                        if not np.array_equal(result[~scope], original[~scope]) or not np.array_equal(result[scope], whole[scope]):
                            raise ValueError("Reference observation exceeded original raw revision permission")
                        permissions.append({"repeat": rep + 1, "case_id": cid, "method": method,
                                            "editable_observed_days": int(scope.sum())})
            comparisons = [
                ("reference_vs_raw", f"adaptive_reference_residual_r{rep}__conflict_scope", f"adaptive_raw_r{rep}__conflict_scope"),
                ("adaptive_vs_fixed_reference", f"adaptive_reference_residual_r{rep}__conflict_scope", f"fixed_reference_residual_r{rep}__conflict_scope"),
                ("scope_vs_whole_reference", f"adaptive_reference_residual_r{rep}__conflict_scope", f"adaptive_reference_residual_r{rep}__whole_target"),
                ("scope_vs_center_reference", f"adaptive_reference_residual_r{rep}__conflict_scope", f"adaptive_reference_residual_r{rep}__matched_center_scope"),
            ]
            for contrast, new_name, old_name in comparisons:
                new = np.array(predictions[new_name][cid]["activity"])
                old = np.array(predictions[old_name][cid]["activity"])
                removed, added = (old == 1) & (new != 1), (old != 1) & (new == 1)
                row = {"repeat": rep + 1, "case_id": cid, "contrast": contrast,
                       "lost_tp": int(np.count_nonzero(removed & (actual == 1))),
                       "gained_tp": int(np.count_nonzero(added & (actual == 1))),
                       "removed_fp": int(np.count_nonzero(removed & (actual == 0))),
                       "gained_fp": int(np.count_nonzero(added & (actual == 0)))}
                for reference_class, lost, gained in ((1, "lost_tp", "gained_tp"), (0, "removed_fp", "gained_fp")):
                    difference = np.count_nonzero((new == 1) & (actual == reference_class)) - np.count_nonzero((old == 1) & (actual == reference_class))
                    if difference != row[gained] - row[lost]:
                        raise ValueError("Full-record modification conservation failed")
                changes.append(row)
    for method, cases in predictions.items():
        denominator, detected = 0, 0
        for cid, value in cases.items():
            weak = np.array([row["weak_signal"] == "1" and int(row["activity_label"]) == 1 for row in labels[cid]])
            denominator += int(weak.sum())
            detected += int(np.count_nonzero(weak & (np.array(value["activity"]) == 1)))
        aggregate = read_json(analysis / "results.json")[method]["aggregate"]
        if denominator != aggregate["weak_days"] or not np.isclose(detected / denominator, aggregate["weak_recall"]):
            raise ValueError("Full-record weak population differs")
        weak_checks.append({"method": method, "weak_days": denominator, "weak_detected_days": detected})
    truth = {cid: read_json(BASE / "background-benchmark-data-v1/mechanism-reference" / cid / "truth.json") for cid in records}
    groups = {family: [cid for cid, row in truth.items() if row["family"] == family] for family in sorted({row["family"] for row in truth.values()})}
    groups.update({f"background{group}": [cid for cid, row in truth.items() if row["background_group"] == group]
                   for group in sorted({row["background_group"] for row in truth.values()})})
    group_rows = []
    for group, ids in groups.items():
        _, summary, _ = evaluate_population({method: {cid: cases[cid] for cid in ids} for method, cases in predictions.items()},
                                             {cid: labels[cid] for cid in ids})
        group_rows.extend({"group": group, **row} for row in summary)
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    selected = [method for rep in (0, 1) for method in
        (f"direct_visual_r{rep}", f"adaptive_raw_r{rep}__conflict_scope",
         f"fixed_reference_residual_r{rep}__conflict_scope", f"adaptive_reference_residual_r{rep}__conflict_scope")]
    selected += ["pelt_rule", "xgboost_84", "annual_residual_rule", "annual_augmented_xgboost"]
    for cid, (case, _) in records.items():
        from matplotlib.colors import ListedColormap

        values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        stage = np.array([STAGES[row["feature"]] if int(row["activity_label"]) == 1 else int(row["activity_label"]) for row in labels[cid]])
        matrix = np.array([actual] + [predictions[method][cid]["activity"] for method in selected])
        fig, panels = plt.subplots(5, 1, figsize=(13, 11), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1, 1, .3, 3]})
        for component, name in enumerate(("N", "E", "U")):
            panels[component].plot(values[:, component], color="#24887b", linewidth=.8)
            panels[component].set_ylabel(name + " (mm)")
            panels[component].grid(alpha=.2)
        panels[3].imshow(stage[None, :], aspect="auto", interpolation="nearest", vmin=-1, vmax=3,
                         cmap=ListedColormap(["#c5c9cc", "#fff4dc", "#c9584f", "#377dab", "#56a477"]),
                         extent=[-.5, len(stage) - .5, .5, -.5])
        panels[3].set_yticks([0], ["Injected ASD"], fontsize=8)
        panels[4].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
                         cmap=ListedColormap(["#c5c9cc", "#fff4dc", "#377dab"]),
                         extent=[-.5, len(stage) - .5, len(matrix) - .5, -.5])
        panels[4].set_yticks(range(len(matrix)), ["Injected activity"] + selected, fontsize=7)
        panels[4].set_xlabel("Original calendar index | blue active; cream quiet; gray unknown/missing")
        panels[4].set_xlim(0, len(stage) - 1)
        fig.suptitle(f"{cid} | {truth[cid]['family']} | background{truth[cid]['background_group']}\n"
                     "Full-record viewed development; injection recovery is not field activity truth")
        fig.tight_layout()
        fig.savefig(out / "images" / f"{cid}.png", dpi=130)
        plt.close(fig)
    destination = out / "source" / Path(__file__).relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), destination)
    csv_rows(out / "group-summary.csv", group_rows)
    csv_rows(out / "changes.csv", changes)
    csv_rows(out / "permissions.csv", permissions)
    csv_rows(out / "weak-denominators.csv", weak_checks)
    save_json(out / "verification.json", {"group_rows": len(group_rows), "all_record_figures": 12,
        "modification_conservation_rows": len(changes), "permission_checks": len(permissions),
        "weak_denominator_checks": len(weak_checks), "observed_group_count": 2})
    save_json(out / "manifest.json", {"analysis_manifest_sha256": sha(analysis / "manifest.json"),
        "files": {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Full-record plots, groups, permissions and conservation verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run, args.analysis, args.out)


if __name__ == "__main__":
    main()
