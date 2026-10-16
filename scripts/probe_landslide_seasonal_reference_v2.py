"""Fit a training-only seasonal reference and screen persistence evidence."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_background_benchmark import verify_data
from analyze_landslide_observation_probe import csv_rows
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_seasonal_reference import (
    apply_annual_reference,
    fit_annual_reference,
    observed_values,
)

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
SOURCE = BASE / "background-benchmark-data-v1"
TRAIN_REFERENCES = ("case_2001", "case_2007", "case_2037", "case_2043")


def positive_days(short, long, observed, policy):
    short_positive = (short["rate"] >= policy.minimum_rate) & (short["signal_to_noise"] >= policy.minimum_signal_to_noise)
    long_positive = (long["rate"] >= policy.minimum_rate) & (long["signal_to_noise"] >= policy.minimum_signal_to_noise)
    return observed & (short_positive | long_positive)


def render_reference(case, corrected, predicted, path):
    raw, _ = observed_values(case)
    residual, _ = observed_values(corrected)
    fig, panels = plt.subplots(3, 2, figsize=(13, 7), sharex=True)
    for axis, name in enumerate(("N", "E", "U")):
        panels[axis, 0].plot(raw[:, axis], color="#24887b", linewidth=.8, label="Observed raw")
        panels[axis, 0].plot(predicted[:, axis], color="#c78635", linewidth=1.2, label="Frozen training annual reference")
        panels[axis, 1].plot(residual[:, axis], color="#24887b", linewidth=.8)
        panels[axis, 0].set_ylabel(name + " (mm)")
        for panel in panels[axis]:
            panel.grid(alpha=.2)
    panels[0, 0].legend(fontsize=8)
    panels[0, 1].set_title("Observed minus historical reference; not a physical activity label")
    for panel in panels[-1]:
        panel.set_xlabel("Original daily calendar index")
    fig.suptitle(case.case_id + " | One annual harmonic fitted on four train backgrounds only")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def probe(out):
    verify_data(SOURCE)
    split = read_json(SOURCE / "protocol.json")["split"]
    if any(cid not in split["train"] for cid in TRAIN_REFERENCES):
        raise ValueError("Historical references must be entirely in train")
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"scope": "Evidence mechanism development; old24 test records are now viewed and are not independent confirmation",
        "train_reference_ids": list(TRAIN_REFERENCES), "validation_ids": split["validation"],
        "viewed_development_ids": split["mechanism_test"], "period_days": 365.25,
        "harmonics": 1, "fitting": "Projected annual sine/cosine jointly fitted from four train quiet inputs; no current displacement fit",
        "assumption": "Train records are no-target-injection normal references; not a claim of stationary physical GNSS field motion",
        "feedback": "A frozen historical annual curve may aid observation; no activity labels from residual magnitude",
        "evidence": "Original0.03mm/day/2MAD policy,21/42 scales, same RAW per-record daily noise proxy; no threshold search",
        "primary_diagnostic": "On validation, both quiet backgrounds lose positive trigger days, and neither weak-after-strong record loses weak positive support. Passing only justifies a bounded visual probe, not method efficacy.",
        "review_screen": "No exact current background or injected latent trajectory available to fit/predict. Paired records are used only in diagnostics.",
        "maximum_api_requests": 0, "no_model_default_switch": True}
    save_json(out / "protocol.json", protocol)
    source_paths = [Path(__file__), ROOT / "src/gnss_sim/landslide_seasonal_reference.py",
                    ROOT / "tests/test_landslide_seasonal_reference.py"]
    for path in source_paths:
        target = out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    save_json(out / "pre-fit-freeze.json", {str(path): sha(path) for path in source_paths}
              | {str(SOURCE / "inputs" / cid / "input.json"): sha(SOURCE / "inputs" / cid / "input.json")
                 for cid in TRAIN_REFERENCES})
    references = [LandslideInput.model_validate_json((SOURCE / "inputs" / cid / "input.json").read_text(encoding="utf-8"))
                  for cid in TRAIN_REFERENCES]
    model = fit_annual_reference(references)
    save_json(out / "annual-reference.json", model)
    save_json(out / "fitting-freeze.json", {"annual-reference.json": sha(out / "annual-reference.json")})
    ids = split["validation"] + split["mechanism_test"]
    records = {cid: load_case_features(SOURCE, cid) for cid in ids}
    with (SOURCE / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        subset = [row for row in csv.DictReader(handle) if row["case_id"] in records]
    csv_rows(out / "reference-subset.csv", subset)
    labels = load_labels(out / "reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    policy = ReviewPolicy()
    rows, reference_rmse = [], []
    (out / "images").mkdir()
    for cid, (case, feature) in records.items():
        corrected, prediction = apply_annual_reference(case, model)
        values, observed = observed_values(case)
        residual, _ = observed_values(corrected)
        raw = {"rate": feature.progressive_rate, "signal_to_noise": feature.signal_to_noise}
        raw_long = raw_motion_evidence(case, feature)
        adjusted_short = raw_motion_evidence(corrected, feature, 21)
        adjusted_long = raw_motion_evidence(corrected, feature)
        original = positive_days(raw, raw_long, observed, policy)
        adjusted = positive_days(adjusted_short, adjusted_long, observed, policy)
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
        truth = read_json(SOURCE / "mechanism-reference" / cid / "truth.json")
        row = {"case_id": cid, "split": "validation" if cid in split["validation"] else "viewed_development",
               "background": truth["background_group"], "family": truth["family"],
               "raw_positive_days": int(original.sum()), "adjusted_positive_days": int(adjusted.sum()),
               "raw_active_support": int(np.count_nonzero(original & (actual == 1))),
               "adjusted_active_support": int(np.count_nonzero(adjusted & (actual == 1))),
               "raw_negative_support": int(np.count_nonzero(original & (actual == 0))),
               "adjusted_negative_support": int(np.count_nonzero(adjusted & (actual == 0))),
               "weak_reference_days": int(weak.sum()), "raw_weak_support": int(np.count_nonzero(original & weak)),
               "adjusted_weak_support": int(np.count_nonzero(adjusted & weak))}
        rows.append(row)
        if truth["family"] == "quiet":
            reference_rmse.append({"case_id": cid, "split": row["split"],
                "raw_rmse_NEU": np.sqrt(np.mean(values[observed] ** 2, axis=0)).tolist(),
                "adjusted_rmse_NEU": np.sqrt(np.mean(residual[observed] ** 2, axis=0)).tolist()})
        folder = out / "adjusted-inputs" / cid
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(corrected.model_dump_json(indent=2), encoding="utf-8")
        if cid in ("case_2013", "case_2019", "case_2018", "case_2024"):
            render_reference(case, corrected, prediction, out / "images" / f"{cid}.png")
    validation = [row for row in rows if row["split"] == "validation"]
    quiet = [row for row in validation if row["family"] == "quiet"]
    tails = [row for row in validation if row["family"] == "weak_after_strong"]
    decision = {"quiet_positive_support_reduced_each_background": all(
        row["adjusted_positive_days"] < row["raw_positive_days"] for row in quiet),
        "weak_tail_positive_support_not_lost_each_background": all(
        row["adjusted_weak_support"] >= row["raw_weak_support"] for row in tails),
        "scope": "Validation diagnostic of trigger evidence only; not VLM activity accuracy"}
    decision["allows_visual_probe"] = (decision["quiet_positive_support_reduced_each_background"]
                                       and decision["weak_tail_positive_support_not_lost_each_background"])
    csv_rows(out / "evidence-support.csv", rows)
    save_json(out / "reference-rmse.json", reference_rmse)
    save_json(out / "decision.json", decision)
    save_json(out / "manifest.json", {"source_manifest_sha256": sha(SOURCE / "manifest.json"),
        "files": {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()}})
    print("Train-only annual coefficients:", model["coefficients_sin_cos_NEU_mm"], flush=True)
    print("Validation evidence:", validation, flush=True)
    print("Decision:", decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    probe(args.out)


if __name__ == "__main__":
    main()
