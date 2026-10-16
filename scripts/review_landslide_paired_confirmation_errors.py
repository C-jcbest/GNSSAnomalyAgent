"""Prepare a small, explicitly post-score visual and raw numerical error review."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from probe_landslide_endpoint_fit import export
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha

METHODS = (
    "visual_numeric_pair", "visual_numeric_augmentation", "visual_numeric_plain",
    "numeric_pair", "xgb_expert24", "direct_visual_r0",
    "fixed_image_review_r0__whole_target",
)


def prepare(data, analysis, out):
    labels = read_json(analysis / "primary_process/reference-labels.json")
    predictions = read_json(analysis / "primary_process/predictions.json")
    corrections = read_json(analysis / "primary_process/corrections.json")
    generation = read_json(data / "mechanism-sealed/generation-ledger.json")
    families = {row["case_id"]: row["truth"]["family"] for row in generation}
    selection = {}
    augmentation = [row for row in corrections if row["control"] == "visual_numeric_augmentation"]
    for row, reason in (
        (min(augmentation, key=lambda row: (row["net"], row["case_id"])), "Lowest net correction versus identical augmentation"),
        (max(augmentation, key=lambda row: (row["net"], row["case_id"])), "Highest net correction versus identical augmentation"),
        (min((row for row in corrections if row["control"] == "numeric_pair"),
             key=lambda row: (row["net"], row["case_id"])), "Lowest net correction versus numerical pair control"),
    ):
        selection.setdefault(row["case_id"], []).append(reason)
    metrics = {}
    for cid, rows in labels.items():
        actual = np.array([int(row["activity_label"]) for row in rows])
        weak = np.array([int(row["weak_signal"]) == 1 for row in rows]) & (actual == 1)
        metrics[cid] = {}
        for name in METHODS:
            activity = np.array(predictions[name][cid]["activity"])
            metrics[cid][name] = {
                "FP": int(((actual == 0) & (activity == 1)).sum()),
                "FN": int(((actual == 1) & (activity != 1)).sum()),
                "weak_total": int(weak.sum()),
                "weak_found": int((weak & (activity == 1)).sum()),
                "weak_missed": int((weak & (activity != 1)).sum()),
            }
    for eligible, key, reason in (
        ([cid for cid in labels if families[cid] in ("quiet", "instrument_step")], "FP", "Highest stationary/offset false-positive count"),
        ([cid for cid in labels if metrics[cid]["visual_numeric_pair"]["weak_total"] > 0], "weak_missed", "Highest weak-motion missed count"),
    ):
        cid = max(eligible, key=lambda cid: (metrics[cid]["visual_numeric_pair"][key], cid))
        selection.setdefault(cid, []).append(reason)
    out.mkdir(exist_ok=False)
    evidence = []
    for cid, reasons in selection.items():
        case = load_case_features(data, cid)[0]
        values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        blocks = []
        for start in range(0, len(values), 30):
            stop = min(start + 30, len(values))
            block = values[start:stop]
            complete = np.isfinite(block).all(axis=1)
            blocks.append({"days": [start, stop], "observed_days": int(complete.sum()),
                           "median_NEU_mm": np.median(block[complete], axis=0).tolist() if complete.any() else None})
        figure, panels = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1, 1, 1.6]})
        for component, name in enumerate(("N", "E", "U")):
            panels[component].plot(values[:, component], linewidth=.65)
            panels[component].set_ylabel(name + " (mm)")
            panels[component].grid(alpha=.2)
        matrix = [[int(row["activity_label"]) for row in labels[cid]]]
        matrix.extend(predictions[name][cid]["activity"] for name in METHODS)
        panels[3].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
                         cmap=ListedColormap(["#bbbbbb", "#fafafa", "#227e8b"]),
                         extent=(-.5, len(values) - .5, len(matrix) - .5, -.5))
        panels[3].set_yticks(range(len(matrix)), ["Controlled reference"] + list(METHODS), fontsize=7)
        panels[3].set_xlabel("Original day | gray unknown, white stationary, teal continuing activity")
        figure.suptitle(cid + " | Post-score new-batch error explanation; references unchanged")
        figure.tight_layout()
        filename = cid + "-diagnostic.png"
        figure.savefig(out / filename, dpi=120)
        plt.close(figure)
        evidence.append({"case_id": cid, "family": families[cid], "selection_reasons": reasons,
                         "image": filename, "image_sha256": sha(out / filename), "blocks": blocks,
                         "metrics": metrics[cid], "post_score": True, "reference_changed": False})
    export(out / "prepared-evidence.json", evidence)
    export(out / "scope.json", {"actual_image_review": "pending", "independent_expert": False,
                                "case_ids": list(selection), "references_changed": False,
                                "selection": "Post-score extrema; complete scored population unchanged",
                                "source_sha256": sha(Path(__file__))})
    print("Prepared post-score views:", list(selection), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "analysis", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.data.resolve(), arguments.analysis.resolve(), arguments.out.resolve())
