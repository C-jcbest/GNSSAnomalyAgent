"""Render all activity/stage outputs for the same-data nearest-design supplement."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image
from run_landslide_comparison import STAGES, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json


def render(run, out):
    analysis = run / "analysis-v3"
    for name, expected in read_json(analysis / "prediction-freeze.json").items():
        if sha(analysis / name) != expected:
            raise ValueError("Scored predictions changed")
    predictions = read_json(analysis / "predictions.json")
    ids = read_json(run / "protocol.json")["case_ids"]
    cases = {cid: load_case_features(run, cid)[0] for cid in ids}
    labels = load_labels(run / "reference-subset.csv", cases)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    activity_colours = ListedColormap(["#c5c5c5", "#ffffff", "#238b72"])
    stage_colours = ListedColormap(["#c5c5c5", "#ffffff", "#e68635", "#238b72", "#8a70ba"])
    figure_paths = []
    for cid, case in cases.items():
        raw = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        actual_activity = [int(row["activity_label"]) for row in labels[cid]]
        actual_stage = [STAGES[row["feature"]] if int(row["activity_label"]) == 1
                        else int(row["activity_label"]) for row in labels[cid]]
        for kind, reference, cmap in (("activity", actual_activity, activity_colours),
                                      ("stage", actual_stage, stage_colours)):
            values = np.array([reference] + [predictions[method][cid][kind] for method in predictions])
            fig, panels = plt.subplots(4, 1, figsize=(14, 13), sharex=True,
                gridspec_kw={"height_ratios": [1, 1, 1, 5]})
            for axis, name in enumerate(("N", "E", "U")):
                panels[axis].plot(np.arange(len(raw)), raw[:, axis], linewidth=.7)
                panels[axis].set_ylabel(name + " (mm)")
                panels[axis].grid(alpha=.2)
            boundaries = np.arange(-1.5, (1.6 if kind == "activity" else 3.6), 1)
            panels[3].imshow(values, cmap=cmap, norm=BoundaryNorm(boundaries, cmap.N),
                aspect="auto", interpolation="nearest", extent=[-.5, len(raw)-.5, len(values)-.5, -.5])
            panels[3].set_yticks(range(len(values)), ["Known controlled latent reference"] + list(predictions), fontsize=6.5)
            explanation = "green=active; white=inactive; grey=unknown/missing" if kind == "activity" else (
                "orange=A; green=S; purple=D; white=inactive; grey=unknown/missing")
            panels[3].set_xlabel("Original day index | " + explanation)
            fig.suptitle(cid + " | Same-data nearest-design supplement | retrospective\n"
                         "Controlled mechanism recovery, not published TAMA or expert field labels", fontsize=11)
            fig.tight_layout()
            path = out / f"{cid}-{kind}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            with Image.open(path) as picture:
                picture.verify()
            figure_paths.append(path)
    save_json(out / "manifest.json", {"source_predictions_sha256": sha(analysis / "predictions.json"),
        "source_reference_sha256": sha(run / "reference-subset.csv"),
        "files": {path.name: sha(path) for path in [out / "source.py", *figure_paths]},
        "records": len(ids), "methods_per_figure": len(predictions), "new_api": 0})
    save_json(out / "presentation-correction.json", {
        "old_figures_retained": {str(path): sha(path) for pattern in ("*-activity.png", "*-stages.png")
                                  for path in run.glob(pattern)},
        "reason": "Old generic renderer says AI development label/DEVELOPMENT; actual reference "
                  "is controlled latent mechanism on pre-frozen new-seed inputs",
        "new_figures": len(figure_paths), "predictions_scores_protocol_and_gates_unchanged": True,
        "png_files_verified": len(figure_paths), "source_sha256": sha(Path(__file__)),
    })
    print({"corrected_figures": len(figure_paths), "new_api": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.run.resolve(), args.out.resolve())
