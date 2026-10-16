"""Prepare a fixed local-versus-global persistence intervention on TRAIN only."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import BASE, ROOT
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_baselines import observation_features
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_local_persistence import local_offset_counterpart
from gnss_sim.landslide_paired_persistence import offset_counterpart, render_encoder_chart
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors, invariance_inputs


def prepare(data, encoded, out):
    verify_manifest(data, "data-freeze.json")
    verify_manifest(encoded, "encoding-freeze.json")
    if (
        sha(data / "data-freeze.json")
        != read_json(encoded / "protocol.json")["data_manifest_sha256"]
    ):
        raise ValueError("Original cache belongs to different data")
    out.mkdir(parents=True, exist_ok=False)
    split = read_json(encoded / "protocol.json")["split"]
    sources = [
        Path(__file__),
        ROOT / "src/gnss_sim/landslide_local_persistence.py",
        ROOT / "tests/test_landslide_local_persistence.py",
        ROOT / "scripts/train_landslide_local_persistence.py",
        ROOT / "scripts/train_landslide_paired_persistence.py",
        ROOT / "src/gnss_sim/landslide_paired_persistence.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "src/gnss_sim/landslide_observed_duration.py",
    ]
    bindings = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    write_json(
        out / "protocol.json",
        dict(
            candidate="CP-PP / Context-Preserving Local Persistence Pairing",
            scope="48train/24exposedvalidation; reserved24 not loaded",
            split=split,
            data_manifest_sha256=sha(data / "data-freeze.json"),
            encoded_manifest_sha256=sha(encoded / "encoding-freeze.json"),
            source_bindings=bindings,
            selection="28-day original grid98..T-98 snapped down to multiple3; known active and originally sampled/supported center; original endpoints c+-21 and c+-91 observed; both latent displacements nonzero; all eligible retained",
            local_interval="[c-21,c+21], keep ALL latent motion outside; step c-14/+14 alternating, central[c-7,c+7) has zero latent increments",
            global_control="Same parents/centers/step days but original full-calendar PP replacement with endpoints c+-91; not an isolated step-magnitude ablation",
            training="same seed20261020,12epochs,batch128,Adam.001,weight_decay.0001,balancedBCE,softplus margin1/coefficient1; original train-only scaling; widths16/32 and19 thresholds",
            decoder="All arms including parent controls use14 positive observations and<=7 raw-missing bridges, no imputation",
            gate=[
                "dailyF1 greater than local augmentation and global pair, and at least plain",
                "eventF1 nondecrease vs local augmentation and global pair",
                "positive net correct days vs augmentation and global pair",
                "nuisance-only FP no higher than augmentation and plain",
                "mixed motion recall at least augmentation",
                "visual value vs numeric pair: daily improvement or event improvement with daily loss<=.01",
                "report strong raw84 and same-augmentation tree gaps, no generic tree gain renamed as VLM",
                "all invariant/replay checks pass",
            ],
            novelty_scope="Pair learning and context anchoring already exist (CAPS); only local GNSS intervention and actual date-attribution gains are hypotheses",
            test_truth_opened=False,
            new_api_calls=0,
        ),
    )
    write_json(out / "preparation-freeze.json", {p.name: sha(p) for p in out.iterdir()})
    truths = {
        row["case_id"]: row["truth"]
        for row in read_json(data / "training-generation-ledger.json")
        if row["group"] == "train"
    }
    pair_folder = out / "pairs"
    pair_folder.mkdir()
    selected, images = [], {"local": [], "global": []}
    arrays = {
        name: []
        for name in (
            "positive_embeddings",
            "positive_numbers",
            "local_numbers",
            "global_numbers",
            "local_charts",
            "global_charts",
        )
    }
    started = time.perf_counter()
    for cid in split["train"]:
        case, _ = load_case_features(data, cid)
        values = np.array(
            [row if row is not None else [np.nan] * 3 for row in case.displacement_mm]
        )
        latent = np.array(truths[cid]["latent_displacement_mm"])
        with np.load(encoded / "train" / (cid + ".npz")) as archive:
            bank = {key: archive[key] for key in archive.files}
        for grid in range(98, len(values) - 98, 28):
            day = grid - grid % 3
            if not bank["selected"][day] or bank["activity"][day] != 1:
                continue
            if not np.isfinite(values[[day - 91, day - 21, day + 21, day + 91]]).all():
                continue
            if any(
                np.linalg.norm(latent[day + half] - latent[day - half]) <= 1e-10
                for half in (21, 91)
            ):
                continue
            step = day + (-14, 14)[len(selected) % 2]
            local, local_latent = local_offset_counterpart(values, latent, day - 21, day + 21, step)
            global_values, global_latent = offset_counterpart(
                values, latent, day - 91, day + 91, step
            )
            pair_id = f"pair_{len(selected):04d}"
            saved = dict(
                local_observations=local,
                local_latent=local_latent,
                global_observations=global_values,
                global_latent=global_latent,
            )
            for kind, negative in (("local", local), ("global", global_values)):
                counterpart = case.model_copy(
                    update={
                        "case_id": pair_id + "_" + kind,
                        "displacement_mm": [
                            tuple(row) if np.isfinite(row).all() else None for row in negative
                        ],
                    }
                )
                diagnostics = {
                    window: derive_motion(counterpart, window) for window in (31, 61, 91)
                }
                features = observation_features(counterpart, diagnostics)
                charts, numbers, support = invariance_inputs(counterpart, features)
                if not support[day]:
                    raise ValueError("Counterpart has no original-center support")
                images[kind].append(charts[day])
                arrays[kind + "_numbers"].append(numbers[day])
                arrays[kind + "_charts"].append(chart_numeric_descriptors(charts[day][None])[0])
                saved[kind + "_image"] = charts[day]
                saved[kind + "_numbers"] = numbers[day]
                if len(selected) < 2:
                    render_encoder_chart(charts[day]).save(out / (pair_id + "-" + kind + ".png"))
            np.savez_compressed(pair_folder / (pair_id + ".npz"), **saved)
            arrays["positive_embeddings"].append(bank["embeddings"][day])
            arrays["positive_numbers"].append(bank["numbers"][day])
            selected.append(
                dict(
                    pair_id=pair_id,
                    case_id=cid,
                    day=day,
                    step_day=step,
                    local_endpoints=[day - 21, day + 21],
                    global_endpoints=[day - 91, day + 91],
                    family=truths[cid]["family"],
                )
            )
        print("Prepared", cid, "pairs", len(selected), flush=True)
    if not selected:
        raise ValueError("No eligible training pairs")
    write_json(out / "selected-training-pairs.json", selected)
    runtime = BASE / "paired-persistence-runtime-v1"
    weights = runtime / "mobileclip2-s0-vision.pt"
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
        raise ValueError("Visual preprocessing differs from original cache")
    before = time.perf_counter()
    with torch.inference_mode():
        for kind in images:
            parts = []
            for first in range(0, len(selected), 32):
                tensor = torch.stack(
                    [
                        transform(render_encoder_chart(patch))
                        for patch in images[kind][first : first + 32]
                    ]
                )
                parts.append(encoder.encode_image(tensor, normalize=True).numpy())
            arrays[kind + "_embeddings"] = np.concatenate(parts).astype(np.float32)
    np.savez_compressed(
        out / "pair-bank.npz", **{name: np.array(values) for name, values in arrays.items()}
    )
    write_json(
        out / "preparation-cost.json",
        dict(
            pairs=len(selected),
            parents=len({row["case_id"] for row in selected}),
            families=sorted({row["family"] for row in selected}),
            new_encoded_centers=2 * len(selected),
            new_independent_records=0,
            encoder_seconds=time.perf_counter() - before,
            total_seconds=time.perf_counter() - started,
            weights_sha256=sha(weights),
            test_truth_opened=False,
        ),
    )
    if {str(path.relative_to(ROOT)): sha(path) for path in sources} != bindings:
        raise ValueError("Prepared source changed during execution")
    write_json(
        out / "encoding-freeze.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print("Frozen local/global pairs", len(selected), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data.resolve(), args.encoded.resolve(), args.out.resolve())
