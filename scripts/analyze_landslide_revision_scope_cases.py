"""Export complete record contrasts and two disclosed post-score case illustrations."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_evaluation import runs


def analyze(run, scored, out):
    for file, expected in read_json(scored / "manifest.json")["files"].items():
        if sha(scored / file) != expected:
            raise ValueError("Scored comparison changed")
    predictions = read_json(scored / "predictions.json")
    ids = read_json(run / "protocol.json")["case_ids"]
    cases = {cid: load_case_features(run, cid)[0] for cid in ids}
    labels = load_labels(run / "reference-subset.csv", cases)
    details = read_json(run / "scope-v1" / run.name / "revision-scopes.json")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    changes, weak_rows = {}, []
    with (scored / "per-record.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            weak_rows.append(row)
    for cid in ids:
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        changes[cid] = {}
        for rep in range(2):
            method = f"adaptive_no_feedback_r{rep}"
            whole = np.array(predictions[method + "__whole_target"][cid]["activity"])
            scoped = np.array(predictions[method + "__conflict_scope"][cid]["activity"])
            changed = whole != scoped
            changes[cid][f"r{rep}"] = {
                "all_changed_observed_runs_half_open": runs(changed),
                "removed_false_positive_runs_half_open": runs(changed & (whole == 1) & (actual == 0)),
                "removed_true_positive_runs_half_open": runs(changed & (whole == 1) & (actual == 1)),
                "unknown_restored_to_negative_runs_half_open": runs((whole == -1) & (scoped == 0)),
                "weak_reference_days": sum(int(row["weak_signal"]) for row in labels[cid]),
            }
    save_json(out / "all-record-changes.json", changes)
    with (out / "complete-record-comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(weak_rows[0]))
        writer.writeheader()
        writer.writerows(weak_rows)
    cmap = ListedColormap(["#c1c1c1", "#f2f2f2", "#e37b38"])
    norm = BoundaryNorm([-1.5, -.5, .5, 1.5], cmap.N)
    for cid, limits in (("case_0078", (400, 660)), ("case_0084", (0, 730))):
        case = cases[cid]
        values = np.array([point if point is not None else [np.nan] * 3
                           for point in case.displacement_mm])
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1, 1, 1.7]})
        for component, name in enumerate(("N", "E", "U")):
            start, stop = limits
            axes[component].plot(np.arange(start, stop), values[start:stop, component],
                                 color=("#00897b", "#dd6031", "#a18c18")[component], linewidth=.9)
            for target in details["adaptive_no_feedback_r0"][cid]["targets"]:
                axes[component].axvspan(target["start"], target["stop"], color="#b7d8ef", alpha=.35)
            for start, stop in details["adaptive_no_feedback_r0"][cid]["conflict_scope_runs"]:
                axes[component].axvspan(start, stop, color="#3f8fc0", alpha=.12)
            axes[component].set_ylabel(f"{name} (mm)")
            axes[component].grid(alpha=.15)
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        strips = [actual]
        names = ["Controlled latent reference"]
        for rep in range(2):
            for variant, label in (("whole_target", "whole"), ("conflict_scope", "conflict"),
                                   ("matched_center_scope", "centre")):
                arm = f"adaptive_no_feedback_r{rep}__{variant}"
                strips.append(np.array(predictions[arm][cid]["activity"]))
                names.append(f"repeat {rep + 1}: {label}")
        axes[3].imshow(np.stack(strips), cmap=cmap, norm=norm, aspect="auto",
                       interpolation="nearest", extent=(-.5, len(actual) - .5, len(strips) - .5, -.5))
        axes[3].set_yticks(range(len(strips)), names, fontsize=8)
        axes[3].set_xlabel("Original day index; orange=activity, light=negative, grey=unknown/missing")
        axes[3].set_xlim(*limits)
        fig.suptitle(f"{cid}: post-score illustration; targets (light blue), editable scope (darker blue)\n"
                     "Selected gain and unresolved weak-tail examples; all records exported separately", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"{cid}-scope.png", dpi=180)
        plt.close(fig)
    save_json(out / "manifest.json", {
        "scope": "Descriptive post-score case selection, not independent evidence or a new method",
        "source_bindings": {str(scored / "manifest.json"): sha(scored / "manifest.json")},
        "files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.scored, args.out)


if __name__ == "__main__":
    main()
