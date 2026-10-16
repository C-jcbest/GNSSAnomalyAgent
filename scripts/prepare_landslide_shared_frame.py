"""Freeze and encode one shared-coordinate intervention on existing TRAIN pairs."""

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
from gnss_sim.landslide_paired_persistence import render_encoder_chart
from gnss_sim.landslide_shared_frame import shared_frame_pair
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors, raw_chart_patches


def prepare(data, encoded, parent_pairs, out):
    for folder, manifest in (
        (data, "data-freeze.json"),
        (encoded, "encoding-freeze.json"),
        (parent_pairs, "encoding-freeze.json"),
    ):
        verify_manifest(folder, manifest)
    old_protocol = read_json(parent_pairs / "protocol.json")
    if old_protocol["encoded_manifest_sha256"] != sha(encoded / "encoding-freeze.json"):
        raise ValueError("Local pairs do not belong to this original encoding")
    if old_protocol["data_manifest_sha256"] != sha(data / "data-freeze.json"):
        raise ValueError("Local pairs do not belong to this observation set")
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "scripts/train_landslide_shared_frame.py",
        ROOT / "src/gnss_sim/landslide_shared_frame.py",
        ROOT / "tests/test_landslide_shared_frame.py",
        ROOT / "scripts/train_landslide_paired_persistence.py",
        ROOT / "src/gnss_sim/landslide_paired_persistence.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "src/gnss_sim/landslide_observed_duration.py",
    ]
    bindings = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    write_json(
        out / "protocol.json",
        dict(
            candidate="SC-PP / Shared-Coordinate Persistence Pairing (development candidate)",
            split=old_protocol["split"],
            data_manifest_sha256=sha(data / "data-freeze.json"),
            encoded_manifest_sha256=sha(encoded / "encoding-freeze.json"),
            parent_pairs_manifest_sha256=sha(parent_pairs / "encoding-freeze.json"),
            source_bindings=bindings,
            physical_pairs="Exactly all127 frozen local counterparts; no new centers, spans or latent intervention",
            rendering="Positive slot median reference; shared span=max(2*max absolute deviation of BOTH members,6*ORIGINAL noise); no clipping",
            supervision="Every new neural arm: original10187 +127shared positives +127shared negatives +127independent negatives, identical order and labels; independent positives are original training centers",
            auxiliary="Only auxiliary positive/negative chart frame or modality differs; same softplus margin1, weight1, full paired mean per minibatch",
            training="seed20261020,12epochs,batch128,Adam.001,weight_decay.0001,balancedBCE; widths16/32,19 thresholds; original train-only scaling",
            trees="Same381 augmented supervised rows on all30659 original eligible training days; depths2/4,120estimators,19thresholds",
            gate=[
                "dailyF1 above matched augmentation AND independent-coordinate pair and >=plain",
                "eventF1 nonloss vs augmentation AND independent pair",
                "net correct days positive vs both",
                "nuisance-only FP <=augmentation and plain",
                "mixed motion recall >=augmentation",
                "visual value vs numeric pair: daily improvement OR event improvement with daily loss<=.01",
                "invariant/replay pass; report strong numerical gaps",
            ],
            decoder=old_protocol["decoder"],
            novelty_scope="Shared frames and context anchoring are general existing ideas; only GNSS intervention-specific pixel context isolation and verified recognition benefit are hypotheses",
            inherited_stop="CP-PP stays stopped; this is a rendering-specific new hypothesis, no rescue parameter tuning",
            test_truth_opened=False,
            new_api_calls=0,
        ),
    )
    write_json(out / "preparation-freeze.json", {p.name: sha(p) for p in out.iterdir()})
    selected = read_json(parent_pairs / "selected-training-pairs.json")
    if any(row["case_id"] not in old_protocol["split"]["train"] for row in selected):
        raise ValueError("Shared-coordinate pairing must use TRAIN parents only")
    write_json(out / "selected-training-pairs.json", selected)
    with np.load(parent_pairs / "pair-bank.npz") as archive:
        bank = {
            key: archive[key]
            for key in (
                "positive_embeddings",
                "positive_numbers",
                "local_embeddings",
                "local_numbers",
                "local_charts",
            )
        }
    arrays = dict(
        independent_positive_embeddings=bank["positive_embeddings"],
        independent_negative_embeddings=bank["local_embeddings"],
        positive_numbers=bank["positive_numbers"],
        negative_numbers=bank["local_numbers"],
        independent_negative_charts=bank["local_charts"],
    )
    images = {"shared_positive": [], "shared_negative": []}
    originals = {}
    original_charts = {}
    started = time.perf_counter()
    (out / "pairs").mkdir()
    for row in selected:
        cid, day = row["case_id"], row["day"]
        if cid not in originals:
            case, features = load_case_features(data, cid)
            values = np.array([v if v is not None else [np.nan] * 3 for v in case.displacement_mm])
            originals[cid] = (values, features.noise)
            original_charts[cid] = raw_chart_patches(case, features.noise)
        values, noise = originals[cid]
        with np.load(parent_pairs / "pairs" / (row["pair_id"] + ".npz")) as archive:
            negative = archive["local_observations"]
        positive_patch, negative_patch, frame = shared_frame_pair(values, negative, day, noise)
        outside = np.abs(np.arange(-90, 91, 3)) > 21
        np.testing.assert_array_equal(positive_patch[:, :, outside], negative_patch[:, :, outside])
        np.testing.assert_array_equal(positive_patch[3:], negative_patch[3:])
        for patch in (positive_patch, negative_patch):
            available = patch[3:, 0] > 0
            np.testing.assert_allclose(patch[:3].sum(axis=1)[available], 1, atol=1e-6)
        images["shared_positive"].append(positive_patch)
        images["shared_negative"].append(negative_patch)
        np.savez_compressed(
            out / "pairs" / (row["pair_id"] + ".npz"),
            positive_image=positive_patch,
            negative_image=negative_patch,
            reference=frame["reference"],
            span=frame["span"],
        )
    for key, patches in images.items():
        arrays[key + "_charts"] = chart_numeric_descriptors(np.stack(patches))
    arrays["independent_positive_charts"] = np.stack(
        [
            chart_numeric_descriptors(original_charts[row["case_id"]][row["day"]][None])[0]
            for row in selected
        ]
    )
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
        raise ValueError("Shared charts changed encoder preprocessing")
    before = time.perf_counter()
    with torch.inference_mode():
        for key, patches in images.items():
            parts = []
            for first in range(0, len(patches), 32):
                tensor = torch.stack(
                    [transform(render_encoder_chart(p)) for p in patches[first : first + 32]]
                )
                parts.append(encoder.encode_image(tensor, normalize=True).numpy())
            arrays[key + "_embeddings"] = np.concatenate(parts).astype(np.float32)
    np.savez_compressed(out / "pair-bank.npz", **arrays)
    write_json(
        out / "preparation-cost.json",
        dict(
            pairs=len(selected),
            parents=len(originals),
            new_encoded_centers=2 * len(selected),
            new_independent_records=0,
            encoder_seconds=time.perf_counter() - before,
            total_seconds=time.perf_counter() - started,
            weights_sha256=sha(weights),
            all_outside_pixels_exact=True,
            no_clipping=True,
            missing_mask_preserved=True,
            test_truth_opened=False,
        ),
    )
    if {str(p.relative_to(ROOT)): sha(p) for p in sources} != bindings:
        raise ValueError("Preparation source changed during execution")
    write_json(
        out / "encoding-freeze.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(read_json(out / "preparation-cost.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "parent-pairs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(
        args.data.resolve(), args.encoded.resolve(), args.parent_pairs.resolve(), args.out.resolve()
    )
