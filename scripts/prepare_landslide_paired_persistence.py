"""Prepare equal-displacement TRAINING pairs for frozen visual adaptation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from probe_landslide_contrastive_exemplars import SOURCE, raw_values
from probe_landslide_endpoint_fit import BASE, ROOT, export
from probe_landslide_visual_invariance import calibration_labels
from run_landslide_comparison import save_json
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha
from gnss_sim.landslide_baselines import observation_features
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_paired_persistence import offset_counterpart, render_encoder_chart
from gnss_sim.landslide_visual_invariance import invariance_inputs


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    split = read_json(SOURCE / "protocol.json")["split"]
    pairs = {cid: load_case_features(SOURCE, cid) for cid in split["train"] + split["validation"]}
    labels = calibration_labels(SOURCE, pairs)
    runtime = BASE / "paired-persistence-runtime-v1"
    if read_json(runtime / "encoder-availability.json")["status"] != "available":
        raise ValueError("The selected pretrained encoder has not loaded successfully")
    model_path = runtime / "mobileclip2-s0-vision.pt"
    export(out / "protocol.json", {
        "scope": "Primary controlled calibration; exposed development, not field confirmation",
        "change": "Frozen MobileCLIP2-S0 image-text vision encoder plus equal-displacement matched-offset supervision",
        "backbone": {"model": "MobileCLIP2-S0", "pretrained": "dfndr2b", "frozen": True,
                     "weights_sha256": sha(model_path), "embedding_width": 512},
        "split": {key: split[key] for key in ("train", "validation")},
        "rendering": "Original61 disjoint3-day slots and availability, original32-row translation-centered raster; "
                     "three NEU panels, fixed query marker; no time compression or gap filling",
        "pair_selection": "Each28-day grid center with14 homogeneous observed active query,>=10 observed query dates, "
                          "original raw84 support and observed endpoints t+-91; original24 TRAIN records only",
        "counterpart": "Original observations minus TRAIN latent plus constant origin and one instrument offset; "
                       "exact same observed endpoints t+-91 and same observational errors/missing dates; "
                       "step at center-42/center/center+42 cyclic, negative center; not a new independent record",
        "truth_access": "Only original training controlled latent components for intervention; never validation truth or inference",
        "arms": ["visual_numeric_pair", "visual_numeric_augmentation", "visual_numeric_plain",
                 "visual_numeric_shuffled_pairs", "visual_pair_no_numeric", "numeric_pair", "numeric_augmentation"],
        "training": {"seed": 20261020, "epochs": 12, "batch_size": 128, "optimizer": "Adam",
                     "learning_rate": .001, "weight_decay": .0001, "head_widths": [16, 32],
                     "train_visual_stride": 2, "pair_weight": 1., "pair_margin_logit": 1.,
                     "loss": "Balanced BCE original+negative augmentation plus mean softplus(1-positive_logit+matched_negative_logit)",
                     "threads": 2, "device": "cpu"},
        "calibration": "Each arm19 thresholds .05..95 x2 head widths=38 opportunities; original dailyF1+eventF1 selection",
        "numeric_controls": "Refit18raw84,expert24 and384chart-information XGB with SAME new negative centers "
                            "and ALL original known supported train days;2 depths x19 thresholds=38 opportunities; "
                            "original rules/PELT retained. Neural numeric arms use same sampled original centers as visual arms",
        "stage": "Original shared numerical stage head and original activity/phase support gate, no new native vision stages",
        "calibration_gate": "MainF1>=augmentation and shuffled; main-plain>=.005; net definite corrections>plain and augmentation; "
                            "weakRecall>=plain/augmentation; main>=numeric_pair F1 or eventF1 improvement with dailyF1 loss<=.01; "
                            "noninferiority alone does not prove pairing, main>augmentation F1 or eventF1",
        "next_if_pass": "Freeze encoder/head/thresholds then complete development against VLM/fixedAgent/strongnumeric and unseen controlled confirmation",
        "stop": "Stop fixed candidate if gate fails, no seed/width/grid/margin/render/arm rescue",
        "new_api_calls": 0, "validation_truth_used_for_pairing": False, "independent_confirmation": False,
    })
    save_json(out / "state.json", {"status": "preparing", "elapsed_seconds": 0})
    selected = []
    bindings = {model_path.relative_to(ROOT).as_posix(): sha(model_path)}
    original_train_samples = 0
    for cid, (case, features) in pairs.items():
        images, numbers, supported = invariance_inputs(case, features)
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        group = "train" if cid in split["train"] else "validation"
        directory = out / "original" / group
        directory.mkdir(parents=True, exist_ok=True)
        sample = supported & (actual >= 0)
        if group == "train":
            sample &= np.arange(len(actual)) % 2 == 0
            original_train_samples += int(sample.sum())
        np.savez_compressed(directory / (cid + ".npz"), images=images, numbers=numbers,
                            supported=supported, observed=features.observed, target=actual,
                            sample=sample, weak=np.array([int(row.get("weak_signal", 0)) == 1 for row in labels[cid]]))
        for path in (SOURCE / "inputs" / cid).iterdir():
            if path.is_file():
                bindings[path.relative_to(ROOT).as_posix()] = sha(path)
        if group == "validation":
            continue
        truth_path = SOURCE / "mechanism-reference" / cid / "truth.json"
        truth = read_json(truth_path)
        bindings[truth_path.relative_to(ROOT).as_posix()] = sha(truth_path)
        latent = np.asarray(truth["latent_displacement_mm"], dtype=float)
        values = raw_values(case)
        for day in range(98, len(actual) - 98, 28):
            query = slice(day - 7, day + 7)
            observed = features.observed[query]
            if (observed.sum() < 10 or not np.all(actual[query][observed] == 1)
                    or not supported[day] or not features.observed[[day - 91, day + 91]].all()):
                continue
            step_day = day + (-42, 0, 42)[len(selected) % 3]
            negative, replacement = offset_counterpart(values, latent, day - 91, day + 91, step_day)
            counterpart = case.model_copy(update={"case_id": f"pair_{len(selected):04d}",
                "displacement_mm": [row.tolist() if np.isfinite(row).all() else None for row in negative]})
            diagnostics = {window: derive_motion(counterpart, window) for window in (31, 61, 91)}
            negative_features = observation_features(counterpart, diagnostics)
            negative_images, negative_numbers, negative_support = invariance_inputs(counterpart, negative_features)
            # Numerical and visual classifiers see the same center, even when
            # a derivative is unavailable; original raw84 support is explicit.
            if not negative_support[day]:
                raise ValueError("Counterpart lost eligible raw84 center support")
            key = f"pair_{len(selected):04d}"
            (out / "pairs").mkdir(exist_ok=True)
            np.savez_compressed(out / "pairs" / (key + ".npz"),
                positive_image=images[day], negative_image=negative_images[day],
                positive_numbers=numbers[day], negative_numbers=negative_numbers[day],
                original_observations=values, original_latent=latent,
                negative_observations=negative, negative_latent=replacement,
                positive_noise=features.noise, negative_noise=negative_features.noise)
            row = {"pair_id": key, "case_id": cid, "day": day, "step_day": step_day,
                   "endpoints": [day - 91, day + 91], "query": [day - 7, day + 7],
                   "weak_center": bool(int(labels[cid][day].get("weak_signal", 0))),
                   "positive_target": 1, "negative_target": 0}
            selected.append(row)
            if len(selected) <= 4:
                render_encoder_chart(images[day]).save(out / (key + "-positive.png"))
                render_encoder_chart(negative_images[day]).save(out / (key + "-negative.png"))
            print(cid, key, day, flush=True)
            save_json(out / "state.json", {"status": "preparing", "prepared_pairs": len(selected),
                                          "elapsed_seconds": time.perf_counter() - started})
    if len(selected) < 12 or not any(row["weak_center"] for row in selected):
        raise ValueError("Insufficient training-only matched supervision")
    export(out / "selected-training-pairs.json", selected)
    export(out / "summary.json", {"matched_pairs": len(selected), "parent_train_records": len(set(row["case_id"] for row in selected)),
        "new_negative_training_centers": len(selected), "original_sampled_train_centers": original_train_samples,
        "validation_truth_used": False, "new_independent_records": 0, "new_api_calls": 0,
        "preparation_seconds": time.perf_counter() - started})
    sources = [Path(__file__), ROOT / "src/gnss_sim/landslide_paired_persistence.py",
        ROOT / "tests/test_landslide_paired_persistence.py", ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "src/gnss_sim/landslide_diagnostics.py", SOURCE / "protocol.json", SOURCE / "reference-labels.csv",
        runtime / "observed-versions.json", runtime / "encoder-availability.json"]
    for index, path in enumerate(sources):
        copy = out / "source" / f"{index:02d}-{path.name}"
        copy.parent.mkdir(exist_ok=True)
        copy.write_bytes(path.read_bytes())
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    export(out / "root-bindings.json", bindings)
    save_json(out / "state.json", {"status": "prepared", "pairs": len(selected),
                                  "elapsed_seconds": time.perf_counter() - started})
    export(out / "pre-encoding-freeze.json", {path.relative_to(out).as_posix(): sha(path)
                                               for path in out.rglob("*") if path.is_file()})
    print("Prepared", len(selected), "matched pairs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out.resolve())
