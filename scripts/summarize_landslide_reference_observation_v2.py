"""Add matched target baselines and diagnostics without changing frozen inference."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_background_benchmark import verify_data
from analyze_landslide_observation_probe import csv_rows, evaluate_population
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_reference_observation_probe_v2 import BASE, DATA, TARGETS, verify
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_background_benchmark import recompute_numerical
from summarize_landslide_motion_competition import target_predictions, verify_metrics

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_seasonal_reference import fit_annual_reference


def summarize(run, analysis, out):
    verify(run)
    for name, expected in read_json(analysis / "manifest.json")["files"].items():
        if sha(analysis / name) != expected:
            raise ValueError("Frozen scoring changed")
    source = BASE / "background-benchmark-data-v1"
    verify_data(source)
    model = read_json(run / "annual-reference.json")
    training = [LandslideInput.model_validate_json((source / "inputs" / cid / "input.json").read_text(encoding="utf-8"))
                for cid in model["training_case_ids"]]
    if fit_annual_reference(training) != model:
        raise ValueError("Train-only reference reconstruction differs")
    records = {target["case_id"]: load_case_features(run, target["case_id"]) for target in TARGETS}
    labels = load_labels(run / "reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    started = time.monotonic()
    numeric = recompute_numerical(source, records)
    numeric_seconds = time.monotonic() - started
    previous = read_json(DATA / "numeric-predictions.json")
    for method, cases in numeric.items():
        for cid, value in cases.items():
            if value != previous[method][cid]:
                raise ValueError(f"Same-split numerical reconstruction differs: {method} {cid}")
    predictions = target_predictions(read_json(run / "predictions.json") | numeric, TARGETS)
    results, summary, per_record = evaluate_population(predictions, labels)
    stages = verify_metrics(predictions, results, labels, records)
    plan = read_json(run / "query-plan.json")
    for target in TARGETS:
        rows = [row for row in plan if row["case_id"] == target["case_id"]]
        if len(rows) != 6 or len({row["prompt"] for row in rows}) != 1:
            raise ValueError("Matched target prompt population differs")
        if any(len(row["images"]) != 3 for row in rows) or len({tuple(row["images"][:2]) for row in rows}) != 1:
            raise ValueError("Raw observations or image count differ across arms")
    changes = []
    for rep in (0, 1):
        for arm in ("reference_curve", "reference_residual"):
            for target in TARGETS:
                cid = target["case_id"]
                actual = np.array([int(row["activity_label"]) for row in labels[cid]])
                old = np.array(predictions[f"raw_r{rep}"][cid]["activity"])
                new = np.array(predictions[f"{arm}_r{rep}"][cid]["activity"])
                known = actual >= 0
                removed, added = (old == 1) & (new != 1), (old != 1) & (new == 1)
                row = {"repeat": rep + 1, "arm": arm, "case_id": cid, "family": target["family"],
                       "lost_tp": int(np.count_nonzero(removed & (actual == 1))),
                       "gained_tp": int(np.count_nonzero(added & (actual == 1))),
                       "removed_fp": int(np.count_nonzero(removed & (actual == 0))),
                       "gained_fp": int(np.count_nonzero(added & (actual == 0)))}
                old_tp = np.count_nonzero((old == 1) & (actual == 1))
                new_tp = np.count_nonzero((new == 1) & (actual == 1))
                old_fp = np.count_nonzero((old == 1) & (actual == 0))
                new_fp = np.count_nonzero((new == 1) & (actual == 0))
                if new_tp - old_tp != row["gained_tp"] - row["lost_tp"]:
                    raise ValueError("TP conservation failed")
                if new_fp - old_fp != row["gained_fp"] - row["removed_fp"]:
                    raise ValueError("FP conservation failed")
                row["new_unknown_known_days"] = int(np.count_nonzero(known & (new == -1)))
                changes.append(row)
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    for target in TARGETS:
        cid = target["case_id"]
        values = np.array([row if row is not None else [np.nan] * 3 for row in records[cid][0].displacement_mm])
        methods = [f"{arm}_r{rep}" for rep in (0, 1) for arm in ("raw", "reference_curve", "reference_residual")]
        methods += ["raw_multiscale_rule", "xgboost_84"]
        left, right = target["start"], target["stop"]
        fig, panels = plt.subplots(4, 1, figsize=(13, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1, 1, 2]})
        for component, name in enumerate(("N", "E", "U")):
            panels[component].plot(np.arange(left, right), values[left:right, component], linewidth=.8)
            panels[component].set_ylabel(name + " (mm)")
            panels[component].grid(alpha=.2)
        actual = [int(row["activity_label"]) for row in labels[cid]][left:right]
        matrix = np.array([actual] + [predictions[name][cid]["activity"][left:right] for name in methods])
        from matplotlib.colors import ListedColormap

        panels[3].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
                         cmap=ListedColormap(["#c5c9cc", "#fff4dc", "#377dab"]),
                         extent=[left - .5, right - .5, len(matrix) - .5, -.5])
        panels[3].set_yticks(range(len(matrix)), ["Known injection"] + methods, fontsize=8)
        panels[3].set_xlabel("Original calendar index | blue active; cream quiet; gray missing/unknown")
        panels[3].set_xlim(left, right - 1)
        fig.suptitle(f"{cid} | {target['family']} | Viewed target only; not field ground truth")
        fig.tight_layout()
        fig.savefig(out / "images" / f"{cid}.png", dpi=130)
        plt.close(fig)
    for path in (Path(__file__), ROOT / "scripts/summarize_landslide_background_benchmark.py"):
        destination = out / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    save_json(out / "predictions.json", predictions)
    save_json(out / "results.json", results)
    save_json(out / "stage-metrics.json", stages)
    csv_rows(out / "summary.csv", summary)
    csv_rows(out / "per-target.csv", per_record)
    csv_rows(out / "changes.csv", changes)
    save_json(out / "verification.json", {"train_reference_exactly_reproduced": True,
        "numeric_models_same_train_validation": True, "numeric_predictions_recomputed": sum(map(len, numeric.values())),
        "independent_activity_stage_groups": len(predictions), "paired_conservation_rows": len(changes),
        "matched_three_images_and_prompt_targets": 8, "target_diagnostic_plots": 8,
        "numeric_model_load_and_predict_seconds": numeric_seconds,
        "numeric_scope": "Original-input controls, do not receive new annual reference; context only, not fair full-method superiority"})
    save_json(out / "manifest.json", {"analysis_manifest_sha256": sha(analysis / "manifest.json"),
        "files": {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Verified 15 target configurations, 72 numerical predictions, 32 change rows and 8 plots", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run, args.analysis, args.out)


if __name__ == "__main__":
    main()
