"""Extend the established PP-VN intervention to mixed-nuisance training records."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open_clip
import torch
from probe_landslide_nuisance_evidence import BASE, ROOT
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_baselines import observation_features
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_paired_persistence import offset_counterpart, render_encoder_chart
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors, invariance_inputs


def prepare(data, encoding, out):
    for folder, manifest in ((data, "data-freeze.json"), (encoding, "encoding-freeze.json")):
        for relative, expected in read_json(folder / manifest).items():
            if sha(folder / relative) != expected:
                raise ValueError("Frozen input changed: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    split = read_json(encoding / "protocol.json")["split"]
    runtime = BASE / "paired-persistence-runtime-v1"
    weights = runtime / "mobileclip2-s0-vision.pt"
    protocol = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "name": "PP-VN diverse-population development extension",
        "scope": "48train/24exposedvalidation; same96-record population,24reserved references remain sealed",
        "intervention": "Existing same-net-displacement matched motion/offset pairs extended to new diverse training. Remove only TRAIN latent deformation; preserve all original measurement errors, actual instrument offsets/spikes and missing dates. No new claimed pair-loss architecture",
        "split": split,
        "selection": "28-day grid centers snapped down to nearest multiple3, endpoints center+-91 observed, >=10 observed central14 dates and every known query date active, original supported training sample; all eligible kept, no weak quota or morphology selection",
        "negative": "Step at center-42/center/center+42 cyclic; exact observed endpoints. Other injected nuisances retained, so negative does not mean pristine stable. Counterfactuals stay with TRAIN parent",
        "training": "Reuse established PP-VN seed20261020,12epochs Adam.001 decay.0001 batch128; coefficient1 softplus margin1; widths16/32,19 thresholds. No seed or loss search",
        "arms": "Seven established PP-VN neural ablations and same-augmentation raw84/chart384 trees; all38 chances. Original 12DQ controls retained with new population training/calibration",
        "gate": "Main dailyF1>augmentation and >=shuffled, net corrections>plain/augmentation; eventF1>=augmentation; nuisanceFP<=plain; mixedRecall>=augmentation; visual dailyF1>numericpair or visual eventF1>numericpair with dailyF1 loss<=.01. All8 required. Strong numerical differences fully reported; do not rename their improvement as visual",
        "support": "Same original support/minimum14 rule, unsupported positives remain FN; all short events kept; old shared new48-trained stage head",
        "independence": "Validation already selected DQ and axis pilots, not independent. No new test generation, label exposure or changes to stopped versions",
        "backbone": "Unchanged frozen MobileCLIP2-S0/dfndr2b; reuse original cached embeddings, encode ONLY new counterpart centers",
        "weights_sha256": sha(weights),
        "data_manifest_sha256": sha(data / "data-freeze.json"),
        "encoding_manifest_sha256": sha(encoding / "encoding-freeze.json"),
        "new_api_calls": 0,
        "test_truth_opened": False,
    }
    write_json(out / "protocol.json", protocol)
    sources = [
        Path(__file__),
        ROOT / "scripts/train_landslide_paired_persistence.py",
        ROOT / "src/gnss_sim/landslide_paired_persistence.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "src/gnss_sim/landslide_diagnostics.py",
    ]
    for source in sources:
        target = out / "source" / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
    write_json(
        out / "preparation-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    ledger = {
        row["case_id"]: row["truth"]
        for row in read_json(data / "training-generation-ledger.json")
        if row["group"] == "train"
    }
    started = time.perf_counter()
    selected, negative_images = [], []
    positive_embeddings, positive_numbers, negative_numbers, negative_charts = [], [], [], []
    for cid in split["train"]:
        case, features = load_case_features(data, cid)
        values = np.array(
            [row if row is not None else [np.nan] * 3 for row in case.displacement_mm]
        )
        latent = np.array(ledger[cid]["latent_displacement_mm"])
        with np.load(encoding / "train" / (cid + ".npz")) as archive:
            bank = {key: archive[key] for key in archive.files}
        for grid_day in range(98, len(values) - 98, 28):
            day = grid_day - grid_day % 3
            query = bank["activity"][day - 7 : day + 7]
            if not bank["selected"][day] or np.sum(query >= 0) < 10:
                continue
            if np.any(query[query >= 0] != 1):
                continue
            if not np.isfinite(values[[day - 91, day + 91]]).all():
                continue
            if np.linalg.norm(latent[day + 91] - latent[day - 91]) <= 1e-10:
                continue
            step_day = day + (-42, 0, 42)[len(selected) % 3]
            negative, replacement = offset_counterpart(values, latent, day - 91, day + 91, step_day)
            # Both original error and nuisance components are retained exactly
            # in the counterfactual, apart from endpoint roundoff protection.
            if not np.array_equal(np.isnan(negative), np.isnan(values)):
                raise ValueError("Counterfactual changed missing observations")
            counterpart = case.model_copy(
                update={
                    "case_id": f"pair_{len(selected):04d}",
                    "displacement_mm": [
                        tuple(row.tolist()) if np.isfinite(row).all() else None for row in negative
                    ],
                }
            )
            diagnostics = {window: derive_motion(counterpart, window) for window in (31, 61, 91)}
            negative_features = observation_features(counterpart, diagnostics)
            images, numbers, support = invariance_inputs(counterpart, negative_features)
            if not support[day]:
                raise ValueError("Counterfactual lost original center support")
            pair_id = f"pair_{len(selected):04d}"
            folder = out / "pairs"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(
                folder / (pair_id + ".npz"),
                negative_observations=negative,
                negative_latent=replacement,
                negative_image=images[day],
                negative_numbers=numbers[day],
                positive_numbers=bank["numbers"][day],
            )
            selected.append(
                {
                    "pair_id": pair_id,
                    "case_id": cid,
                    "day": day,
                    "step_day": step_day,
                    "endpoints": [day - 91, day + 91],
                    "family": ledger[cid]["family"],
                }
            )
            positive_embeddings.append(bank["embeddings"][day])
            positive_numbers.append(bank["numbers"][day])
            negative_images.append(images[day])
            negative_numbers.append(numbers[day])
            negative_charts.append(chart_numeric_descriptors(images[day][None])[0])
            if len(selected) <= 2:
                render_encoder_chart(images[day]).save(out / (pair_id + "-negative.png"))
        print("Prepared", cid, "cumulative pairs", len(selected), flush=True)
    if not selected:
        raise ValueError("No eligible training-only pairs")
    write_json(out / "selected-training-pairs.json", selected)
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
        raise ValueError("Counterpart transform differs from original embedding transform")
    encoded = []
    before = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(negative_images), 32):
            tensor = torch.stack(
                [
                    transform(render_encoder_chart(patch))
                    for patch in negative_images[first : first + 32]
                ]
            )
            encoded.append(encoder.encode_image(tensor, normalize=True).numpy())
    negative_embeddings = np.concatenate(encoded).astype(np.float32)
    if not np.isfinite(negative_embeddings).all():
        raise ValueError("Nonfinite counterpart embeddings")
    np.savez_compressed(
        out / "pair-bank.npz",
        positive_embeddings=np.stack(positive_embeddings),
        negative_embeddings=negative_embeddings,
        positive_numbers=np.stack(positive_numbers),
        negative_numbers=np.stack(negative_numbers),
        negative_charts=np.stack(negative_charts),
    )
    write_json(
        out / "preparation-cost.json",
        {
            "pairs": len(selected),
            "parents": len(set(row["case_id"] for row in selected)),
            "families": sorted(set(row["family"] for row in selected)),
            "encoder_seconds": time.perf_counter() - before,
            "total_seconds": time.perf_counter() - started,
            "reused_original_centers": 25586,
            "new_encoded_centers": len(selected),
            "new_independent_records": 0,
            "new_api_calls": 0,
        },
    )
    for source in sources:
        if sha(source) != sha(out / "source" / source.name):
            raise ValueError("Preparation source changed: " + str(source))
    write_json(
        out / "encoding-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print("Frozen diverse PP-VN training pairs", len(selected), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoding", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data.resolve(), args.encoding.resolve(), args.out.resolve())
