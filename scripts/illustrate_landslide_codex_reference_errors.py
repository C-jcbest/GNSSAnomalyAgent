"""Save disclosed post-score examples without modifying the frozen scoring run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_codex_observation_reference import BASE, read_json, verify_hashes
from matplotlib.colors import ListedColormap

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_evaluation import runs


def illustrate(out):
    if out.exists():
        raise FileExistsError("Preserve the existing error analysis")
    comparison = BASE / "codex-observation-comparison-v1"
    reference = BASE / "codex-observation-review-v1"
    scope = BASE / "revision-scope-probe-v1/confirmation-v1"
    run = BASE / "confirmation-v1"
    verify_hashes(comparison, "manifest.json", "files")
    verify_hashes(reference, "delivery-manifest.json", "files")
    predictions = read_json(scope / "predictions.json")
    daily = read_json(reference / "compiled/daily-reference.json")
    selections = read_json(run / "selections.json")
    methods = ("direct_visual_r0", "adaptive_no_feedback_r0__conflict_scope",
               "adaptive_no_feedback_r1__conflict_scope",
               "adaptive_numeric_feedback_r0__conflict_scope",
               "adaptive_numeric_feedback_r1__conflict_scope",
               "raw_multiscale_rule", "xgboost_raw_persistence")
    # Examples chosen after scoring: expose the largest weak-record miss and a
    # repeat-sensitive weak-record boundary error; the full 12-record table stays available.
    out.mkdir(parents=True)
    examples = {}
    for case_id in ("case_0057", "case_0051"):
        case = LandslideInput.model_validate_json((reference / case_id / "input.json").read_bytes())
        values = np.asarray([value if value is not None else [np.nan] * 3 for value in case.displacement_mm])
        rows = [row for row in daily if row["case_id"] == case_id]
        actual = np.array([row["activity_label"] for row in rows])
        observed = np.array([value is not None for value in case.displacement_mm])
        labels = ["Codex development reference", *methods]
        strips = [actual, *[np.array(predictions[method][case_id]["activity"]) for method in methods]]
        codes = np.stack(strips) + 1
        codes[:, ~observed] = 3
        figure, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 2, 2, 3]})
        for component, name in enumerate(("N", "E", "U")):
            axes[component].plot(np.arange(len(values)), values[:, component], linewidth=.7)
            axes[component].set_ylabel(name + " (mm)")
            axes[component].grid(alpha=.2)
        axes[-1].imshow(codes, aspect="auto", interpolation="nearest", vmin=0, vmax=3,
                        cmap=ListedColormap(["#ecc168", "#e2e8e2", "#3c7ca0", "#777777"]),
                        extent=(-.5, len(values) - .5, len(labels) - .5, -.5))
        axes[-1].set_yticks(range(len(labels)), labels, fontsize=8)
        axes[-1].set_xlim(0, len(values) - 1)
        axes[-1].set_xlabel("Original day index | blue=activity, pale=stationary, amber=unknown, gray=missing")
        figure.suptitle(case_id + " | Post-score failure example; AI reference, NOT independent truth")
        figure.tight_layout()
        figure.savefig(out / f"{case_id}-errors.png", dpi=130)
        plt.close(figure)
        examples[case_id] = {"reference_activity_runs_half_open": runs(actual == 1),
                            "prediction_activity_runs_half_open": {
                                method: runs(np.array(predictions[method][case_id]["activity"]) == 1)
                                for method in methods},
                            "selected_targets": {f"r{repeat}": selections[f"{case_id}-r{repeat}"]["selected"]
                                                 for repeat in (0, 1)}}
    answers = {}
    for repeat in (0, 1):
        for representation in ("image", "numeric"):
            name = f"case_0057-window-03-{representation}-r{repeat}.response.json"
            receipt = read_json(run / "requests" / name)
            answers[name] = {"response_sha256": sha(run / "requests" / name), "output": receipt["output"]}
    write_json(out / "error-evidence.json", {
        "scope": "Disclosed post-score case selection; no method changes or independent claim",
        "examples": examples, "original_weak_window_answers": answers,
        "finding": "0057 selected window [270,450) was acquired in both repeats. Image-only answer describes gradual N/E motion but returns empty activity; numeric text answers retain a partial weak interval. Failure includes judgment, not just missing review acquisition.",
        "comparison_manifest_sha256": sha(comparison / "manifest.json"),
        "reference_manifest_sha256": sha(reference / "delivery-manifest.json"),
        "implementation_sha256": sha(Path(__file__)), "new_remote_model_requests": 0,
    })
    write_json(out / "manifest.json", {"reference_use": "development_only", "files": {
        path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()}})
    print("Saved two post-score examples with actual weak-window answer hashes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    illustrate(parser.parse_args().out.resolve())
