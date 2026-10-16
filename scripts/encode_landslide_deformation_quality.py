"""Cache frozen visual/numeric observations on the new training/validation split."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open_clip
import torch
from probe_landslide_nuisance_evidence import BASE, ROOT, write_csv
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_deformation_quality import query_quality_targets
from gnss_sim.landslide_paired_persistence import render_encoder_chart
from gnss_sim.landslide_visual_invariance import (
    chart_numeric_descriptors,
    fit_numeric_scaling,
    invariance_inputs,
)


def encode(data, out):
    # Check all bytes, including sealed files, without reading evaluation labels
    # into model preparation or exposing any specific evaluation truth.
    for relative, expected in read_json(data / "data-freeze.json").items():
        if sha(data / relative) != expected:
            raise ValueError("Frozen data changed: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    source_split = read_json(data / "protocol.json")["split"]
    split = {group: source_split[group] for group in ("train", "validation")}
    runtime = BASE / "paired-persistence-runtime-v1"
    weights = runtime / "mobileclip2-s0-vision.pt"
    protocol = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_name": "DQ-VN / Deformation-Quality Separated Visual-Numeric Learning",
        "scope": "New48train/24validation; retained24test labels unopened",
        "change": "One auxiliary observed-artifact query head on a shared trunk; activity and quality nonexclusive",
        "split": split,
        "reserved_test": source_split["mechanism_test"],
        "arms": [
            "visual_numeric_dual",
            "visual_numeric_single",
            "numeric_dual",
            "numeric_single",
            "visual_single",
        ],
        "baselines": "Refit rules/PELT/originalXGB and raw84/chart-information XGB on same48/24; old models cannot be the new primary controls",
        "encoder": "Frozen MobileCLIP2-S0/dfndr2b visual branch, exact original-day NEU plots/preprocessing; no language generation, no full VETime reproduction",
        "weights_sha256": sha(weights),
        "data_manifest_sha256": sha(data / "data-freeze.json"),
        "sampling": "Known supported train centers every3 original days; all supported validation centers; identical centers and labels in neural ablations",
        "quality_target": "Any observed original-day offset/spike in central[day-7,day+7); >=8 observed query dates, otherwise auxiliary unknown",
        "training": "seed2026101249;12 epochs, Adam lr.001 decay.0001,batch128; balanced BCE activity plus balanced BCE quality with fixed coefficient1 in dual arms, coefficient0 in single arms; no added images or negative samples",
        "calibration": "Hidden widths16/32 and19 thresholds.05..95, 38 equal chances; tree depths2/4 and same19 thresholds; stable first max dailyF1+eventF1; numeric stages shared",
        "development_gate": "Validation visual_numeric_dual F1>visual_numeric_single; eventF1>=single; positive net correction vs single; nuisance-only FP<=single; mixed-motion recall>=single; dual F1>numeric_dual or dual eventF1>numeric_dual with dailyF1 loss<=.01; report strong numerical gaps and all conditions. Otherwise stop this version without seed/threshold/loss rescue or test scoring",
        "confirmation": "Only after gate: freeze selected models/calibration; raw Codex review before test predictions/process truth exposure; infer all24 reserved test records once and score full population; same generator is not field independence",
        "new_api_calls": 0,
    }
    write_json(out / "protocol.json", protocol)
    sources = [
        Path(__file__),
        ROOT / "src/gnss_sim/landslide_deformation_quality.py",
        ROOT / "src/gnss_sim/landslide_paired_persistence.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "tests/test_landslide_deformation_quality.py",
    ]
    bindings = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    bindings[str(weights.relative_to(ROOT))] = sha(weights)
    for source in sources:
        destination = out / "source" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    write_json(out / "source-bindings.json", bindings)
    write_json(
        out / "pre-encoding-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    labels = {cid: [] for ids in split.values() for cid in ids}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            labels[row["case_id"]].append(row)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = open_clip.get_pretrained_cfg("MobileCLIP2-S0", "dfndr2b")
    encoder, _, transform = open_clip.create_model_and_transforms(
        "MobileCLIP2-S0",
        pretrained=None,
        image_mean=config["mean"],
        image_std=config["std"],
        image_interpolation=config["interpolation"],
        image_resize_mode=config["resize_mode"],
    )
    encoder.visual.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    encoder.eval()
    encoder.requires_grad_(False)
    if str(transform) != read_json(runtime / "encoder-availability.json")["image_transform"]:
        raise ValueError("Pretrained image transform changed")
    timings = []
    training_numbers = []
    started = time.perf_counter()
    for group, ids in split.items():
        folder = out / group
        folder.mkdir()
        for cid in ids:
            before = time.perf_counter()
            case, feature = load_case_features(data, cid)
            images, numbers, supported = invariance_inputs(case, feature)
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            quality = query_quality_targets(labels[cid])
            selected = supported & (actual >= 0)
            if group == "train":
                selected &= np.arange(len(selected)) % 3 == 0
            indices = np.flatnonzero(selected)
            embeddings = np.zeros((len(selected), 512), dtype=np.float32)
            numeric_seconds = time.perf_counter() - before
            before = time.perf_counter()
            with torch.inference_mode():
                for first in range(0, len(indices), 32):
                    batch = indices[first : first + 32]
                    tensor = torch.stack(
                        [transform(render_encoder_chart(images[index])) for index in batch]
                    )
                    embeddings[batch] = encoder.encode_image(tensor, normalize=True).numpy()
            encoding_seconds = time.perf_counter() - before
            if group == "train":
                training_numbers.append(numbers[selected])
            np.savez_compressed(
                folder / (cid + ".npz"),
                embeddings=embeddings,
                numbers=numbers,
                selected=selected,
                supported=supported,
                observed=feature.observed,
                stage_supported=feature.supported,
                stage_matrix=feature.matrix,
                activity=actual,
                quality=quality,
                chart_descriptors=chart_numeric_descriptors(images),
            )
            if cid == split[group][0]:
                render_encoder_chart(images[indices[len(indices) // 2]]).save(
                    out / (group + "-example.png")
                )
            timings.append(
                {
                    "group": group,
                    "case_id": cid,
                    "encoded_centers": len(indices),
                    "numeric_seconds": numeric_seconds,
                    "encoding_seconds": encoding_seconds,
                }
            )
            print(group, cid, len(indices), f"encoding{encoding_seconds:.2f}s", flush=True)
    write_json(out / "numeric-scaling.json", fit_numeric_scaling(np.concatenate(training_numbers)))
    write_csv(out / "encoding-timings.csv", timings)
    write_json(
        out / "encoding-cost.json",
        {
            "seconds": time.perf_counter() - started,
            "transform": str(transform),
            "encoded_centers": sum(row["encoded_centers"] for row in timings),
            "test_centers_encoded": 0,
            "test_truth_opened": False,
            "new_api_calls": 0,
        },
    )
    for relative, expected in bindings.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Encoding source changed: " + relative)
    write_json(
        out / "encoding-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(
        "Frozen train/validation encodings; no heads trained and no test labels opened", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    encode(args.data.resolve(), args.out.resolve())
