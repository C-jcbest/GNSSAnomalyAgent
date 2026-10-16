"""Freeze and encode a same-observation continuity-guided pooling pilot."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open_clip
import torch
from encode_landslide_spatial_readout import pilot_split, verify_manifest
from probe_landslide_nuisance_evidence import BASE, ROOT, write_csv
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evidence_pooling import (
    continuity_guides,
    evidence_weighted_embedding,
    feature_weights,
)
from gnss_sim.landslide_paired_persistence import render_encoder_chart
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling, invariance_inputs

MAIN = "continuity_pool_numeric"
ARMS = {
    MAIN: "continuity",
    "uniform_pool_numeric": "uniform",
    "permuted_pool_numeric": "permuted",
    "jump_pool_numeric": "jump",
    "continuity_pool_no_branch": "no_branch",
    "uniform_pool_visual": "visual",
    "guide_numeric_mlp": "numeric",
}
TREES = ("guide_information_xgb", "all_guide_information_xgb")


def encode(data, existing, evidence_source, parent, out):
    for folder, manifest in (
        (data, "data-freeze.json"), (existing, "encoding-freeze.json"),
        (evidence_source, "freeze.json"), (parent, "freeze.json"),
    ):
        verify_manifest(folder, manifest)
    split = pilot_split(data)
    if split != read_json(parent / "protocol.json")["split"]:
        raise ValueError("Previous controls do not use the same fixed16/8 population")
    if read_json(existing / "protocol.json")["data_manifest_sha256"] != sha(data / "data-freeze.json"):
        raise ValueError("Original visual cache belongs to different data")
    stage_source = data.parent / "spatial-pilot-calibration-v1"
    verify_manifest(stage_source, "calibration-freeze.json")
    out.mkdir(parents=True, exist_ok=False)
    runtime = BASE / "paired-persistence-runtime-v1"
    weights = runtime / "mobileclip2-s0-vision.pt"
    sources = [
        Path(__file__), ROOT / "scripts/train_landslide_evidence_pooling.py",
        ROOT / "src/gnss_sim/landslide_evidence_pooling.py",
        ROOT / "tests/test_landslide_evidence_pooling.py",
        ROOT / "src/gnss_sim/landslide_motion_mass.py",
        ROOT / "src/gnss_sim/landslide_paired_persistence.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "scripts/train_landslide_spatial_readout.py",
        ROOT / "scripts/train_landslide_paired_persistence.py",
        ROOT / "scripts/score_landslide_claim_review.py",
        ROOT / "src/gnss_sim/landslide_observed_duration.py",
    ]
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    protocol = dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),
        candidate="SCVP / Sustained-Change Visual Pooling;exploratory GNSS candidate",
        hypothesis="The original frozen map may contain useful shape detail not expressed by uniform global pooling. Select coarse date/axis footprints with same-sign adjacent raw change continuity rather than isolated jump magnitude",
        mechanism="12 original14-day block changes;11 same-axis adjacent harmonic changes;map to exact256px source-chart footprints;uniform weight floor1noise unit;pretrained1024->512projection andL2unchanged",
        geometry="8x8map;32px footprints;x29..245 corresponds -90..90;N/E/U plot top12/85/158,height60;token context already mixed,not isolated raw-day attention",
        controls="Uniform native global;per-query shuffle of64weights same multisets;absolute-change jump guide;remove explicit numeric branch;fully uniform pure visual;numericalMLP and same-guide/all-observation trees",
        fairness="All4main fusion arms1198dims:global512+pool512+87numbers/flags174;same numerical guide values supplied to every numeric contestant.1024no-branch still gets raw numeric guidance through pooling;uniform1024visual has no numeric feedback",
        split=split, scope="Fixed16TRAIN/8exposedVALIDATION,eight families2/1;reserved24content not loaded",
        arms=ARMS, trees=TREES,
        training="Same TRAIN stride3selected centers,12epochs,seed20261020,Adam.001/wd.0001,widths16/32;trees120/seed20261007/depths2/4/lr.05/all supported known TRAIN days;19thresholds per2params=>38opportunities per9newarms,342new",
        selection="First maximal dailyF1+eventF1 on same exposed validation;21previous controls retained;16TRAIN stage unchanged",
        decoder="14positive observed support;bridge only<=7raw missing;no interpolation or filling unknown",
        gate=[
            "main dailyF1 above uniform,permuted and jump",
            "positive net correction vs uniform,permuted and jump",
            "main eventF1>=uniform,permuted and jump",
            "pure nuisance FP<=uniform",
            "mixed deformation recall>=uniform",
            "visual value: dailyF1 above best same-information numeric OR higher eventF1 with dailyF1 within.01",
        ],
        failure="If any gate fails stop this exact version;no new floor,guide widths,epoch,seed or threshold rescue;no reserved evaluation;do not rename a best control as novel",
        next="Only all supports48/24development expansion;new independent confirmation only after expansion",
        weights_sha256=sha(weights),
        input_bindings={str(folder.resolve()): sha(folder / manifest) for folder, manifest in (
            (data, "data-freeze.json"), (existing, "encoding-freeze.json"),
            (evidence_source, "freeze.json"), (parent, "freeze.json"),
            (stage_source, "calibration-freeze.json"),
        )},
        source_bindings=bindings, reserved_content_loaded=False, new_api_calls=0,
    )
    write_json(out / "protocol.json", protocol)
    write_json(out / "source-bindings.json", bindings)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "pre-encoding-freeze.json", {p.name: sha(p) for p in out.iterdir()})
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = open_clip.get_pretrained_cfg("MobileCLIP2-S0", "dfndr2b")
    encoder, _, transform = open_clip.create_model_and_transforms(
        "MobileCLIP2-S0", pretrained=None, image_mean=config["mean"], image_std=config["std"],
        image_interpolation=config["interpolation"], image_resize_mode=config["resize_mode"],
    )
    encoder.visual.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    encoder.eval()
    encoder.requires_grad_(False)
    if str(transform) != read_json(runtime / "encoder-availability.json")["image_transform"]:
        raise ValueError("Pretrained image transform changed")
    timings, training_numbers, maximum_error = [], [], 0.
    started = time.perf_counter()
    (out / "progress").mkdir()
    for group, ids in split.items():
        (out / group).mkdir()
        for cid in ids:
            before = time.perf_counter()
            case, features = load_case_features(data, cid)
            images, numbers, support = invariance_inputs(case, features)
            with np.load(existing / group / (cid + ".npz")) as archive:
                bank = {key: archive[key] for key in archive.files}
            with np.load(evidence_source / "evidence" / (cid + ".npz")) as archive:
                evidence = {key: archive[key] for key in archive.files}
            if not np.array_equal(support, bank["supported"]) or not np.array_equal(numbers, bank["numbers"], equal_nan=True):
                raise ValueError("Original support or numbers changed")
            continuous, jumps = continuity_guides(evidence)
            weight_continuous = feature_weights(continuous)
            weight_permuted = feature_weights(continuous, permuted=True)
            weight_jump = feature_weights(jumps)
            shared = np.column_stack((
                bank["numbers"], evidence["global_distribution"],
                continuous.reshape(len(images), 24), jumps.reshape(len(images), 24),
            ))
            bank["shared_numbers"] = shared
            bank["raw_profile"] = evidence["raw_profile"]
            bank["continuity_guide"] = continuous
            bank["jump_guide"] = jumps
            bank["continuity_weights"] = weight_continuous
            bank["permuted_weights"] = weight_permuted
            bank["jump_weights"] = weight_jump
            bank["continuity_embedding"] = np.zeros_like(bank["embeddings"])
            bank["permuted_embedding"] = np.zeros_like(bank["embeddings"])
            bank["jump_embedding"] = np.zeros_like(bank["embeddings"])
            indices = np.flatnonzero(bank["selected"])
            with torch.inference_mode():
                for first in range(0, len(indices), 32):
                    batch = indices[first:first + 32]
                    tensors = torch.stack([transform(render_encoder_chart(images[index])) for index in batch])
                    maps = encoder.visual.trunk.forward_features(tensors)
                    native = torch.nn.functional.normalize(encoder.visual.trunk.forward_head(maps), dim=-1)
                    original = bank["embeddings"][batch]
                    error = float(np.max(np.abs(native.numpy() - original)))
                    maximum_error = max(maximum_error, error)
                    if not np.allclose(native.numpy(), original, atol=1e-6, rtol=1e-5):
                        raise ValueError("Original pretrained global vectors changed")
                    for destination, weight in (
                        ("continuity_embedding", weight_continuous),
                        ("permuted_embedding", weight_permuted),
                        ("jump_embedding", weight_jump),
                    ):
                        bank[destination][batch] = evidence_weighted_embedding(
                            maps, encoder.visual.trunk.head.fc, weight[batch]
                        ).numpy()
            np.savez_compressed(out / group / (cid + ".npz"), **bank)
            if group == "train":
                training_numbers.append(shared[bank["selected"]])
            if cid == ids[0]:
                center = indices[len(indices) // 2]
                render_encoder_chart(images[center]).save(out / (group + "-example.png"))
                write_json(out / (group + "-example-weights.json"), dict(
                    case_id=cid, day=int(center), continuity=weight_continuous[center].tolist(),
                    permuted=weight_permuted[center].tolist(), jump=weight_jump[center].tolist(),
                ))
            timings.append(dict(group=group, case_id=cid, centers=len(indices),
                                seconds=time.perf_counter() - before))
            write_csv(out / "progress" / (cid + ".csv"), [timings[-1]])
            print(group, cid, len(indices), f"{timings[-1]['seconds']:.2f}s", flush=True)
    write_csv(out / "encoding-timings.csv", timings)
    write_json(out / "numeric-scaling.json", fit_numeric_scaling(np.concatenate(training_numbers)))
    write_json(out / "encoding-cost.json", dict(
        seconds=time.perf_counter() - started, original_image_forwards=sum(row["centers"] for row in timings),
        auxiliary_pool_vectors=3 * sum(row["centers"] for row in timings),
        original_global_maximum_error=maximum_error, new_images=0, new_api_calls=0,
        reserved_content_loaded=False,
    ))
    for relative, expected in bindings.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Prepared source changed: " + relative)
    write_json(out / "encoding-freeze.json", {
        str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()
    })
    print("Evidence pooling encoding frozen", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "existing", "evidence-source", "parent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    encode(args.data.resolve(), args.existing.resolve(), args.evidence_source.resolve(),
           args.parent.resolve(), args.out.resolve())
