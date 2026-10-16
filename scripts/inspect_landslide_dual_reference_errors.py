"""Localize fixed-response errors without changing labels or inference choices."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_dual_reference_confirmation import DATA
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import runs
from gnss_sim.landslide_review_utility import correction_utility

BASE = DATA.parent
ANALYSIS = BASE / "dual-reference-confirmation-analysis-v1"
OUTPUT = BASE / "dual-reference-confirmation-errors-v1"


def render(cid, reference, predictions, output):
    case = read_json(DATA / "inputs" / cid / "input.json")
    values = np.array([[np.nan] * 3 if row is None else row for row in case["displacement_mm"]])
    arms = {
        "AI reference": [row["activity_label"] for row in reference],
        "Direct r1": predictions["direct_visual_r1"][cid]["activity"],
        "Fixed late same K r1": predictions["same_count_fixed_late_r1__conflict_scope"][cid]["activity"],
        "Main r1": predictions["adaptive_no_feedback_r1__conflict_scope"][cid]["activity"],
        "Numeric feedback r1": predictions["adaptive_numeric_feedback_r1__conflict_scope"][cid]["activity"],
    }
    figure, axes = plt.subplots(8, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 3, 1, 1, 1, 1, 1]})
    for axis, component in enumerate(("N", "E", "U")):
        axes[axis].plot(values[:, axis], linewidth=.7)
        axes[axis].set_ylabel(component + " mm")
        axes[axis].grid(alpha=.2)
    colors = ListedColormap(["#edc46a", "#e0e7e0", "#407ea0", "#777777"])
    for axis, (label, activity) in zip(axes[3:], arms.items(), strict=True):
        codes = np.array(activity) + 1
        codes[[row is None for row in case["displacement_mm"]]] = 3
        axis.imshow([codes], aspect="auto", interpolation="nearest", cmap=colors, vmin=0, vmax=3,
            extent=(-.5, len(values) - .5, .5, -.5))
        axis.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        axis.set_yticks([])
    axes[-1].set_xlabel("Original day; blue=activity, pale=stationary, amber=unknown, gray=missing")
    axes[-1].set_xlim(0, 729)
    figure.suptitle(cid + " | post hoc author-AI error inspection; frozen labels unchanged")
    figure.tight_layout()
    figure.savefig(output / (cid + ".png"), dpi=140)
    plt.close(figure)


def inspect():
    verify_manifest(ANALYSIS)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "source.py").write_bytes(Path(__file__).read_bytes())
    rows, intervals = [], []
    for target in ("primary_ai", "secondary_process"):
        reference = read_json(ANALYSIS / target / "reference-labels.json")
        predictions = read_json(ANALYSIS / target / "predictions.json")
        for rep in (0, 1):
            main = f"adaptive_no_feedback_r{rep}__conflict_scope"
            for control in (f"direct_visual_r{rep}", f"same_count_fixed_late_r{rep}__conflict_scope",
                    f"adaptive_numeric_feedback_r{rep}__conflict_scope"):
                for cid, record in reference.items():
                    actual = np.array([row["activity_label"] for row in record])
                    before = np.array(predictions[control][cid]["activity"])
                    after = np.array(predictions[main][cid]["activity"])
                    observed = np.array([row is not None for row in read_json(
                        DATA / "inputs" / cid / "input.json")["displacement_mm"]])
                    utility = correction_utility(actual, before, after, observed)
                    rows.append({"target": target, "repeat": rep, "case_id": cid,
                        "control": control, "treatment": main, **utility,
                        "tp_difference": int(((actual == 1) & (after == 1)).sum() - ((actual == 1) & (before == 1)).sum()),
                        "fp_difference": int(((actual == 0) & (after == 1)).sum() - ((actual == 0) & (before == 1)).sum())})
                    for expected in (0, 1):
                        for old in (-1, 0, 1):
                            for new in (-1, 0, 1):
                                if old == new:
                                    continue
                                for start, stop in runs(observed & (actual == expected) & (before == old) & (after == new)):
                                    intervals.append({"target": target, "repeat": rep, "case_id": cid,
                                        "control": control, "treatment": main, "reference_activity": expected,
                                        "control_activity": old, "main_activity": new, "start": start, "stop": stop})
        if target == "primary_ai":
            for cid in ("case_0148", "case_0152", "case_0153", "case_0154"):
                render(cid, reference[cid], predictions, OUTPUT)
    write_csv(OUTPUT / "per-record-corrections.csv", rows)
    write_csv(OUTPUT / "changed-intervals.csv", intervals)
    write_json(OUTPUT / "manifest.json", {"files": {
        path.name: sha(path) for path in sorted(OUTPUT.iterdir()) if path.is_file()}})


if __name__ == "__main__":
    inspect()
