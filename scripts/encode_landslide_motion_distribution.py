"""Freeze an ordered physical-evidence visual pilot with same-information controls."""

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
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_motion_distribution_view import (
    distribution_view_inputs,
    render_distribution_view,
)
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling

MAIN = "ordered_distribution_numeric"
ARMS = {
    MAIN: "ordered",
    "duplicate_motion_global": "duplicate",
    "permuted_distribution_numeric": "permuted",
    "ordered_distribution_visual": "visual",
    "auxiliary_only_numeric": "auxiliary",
    "same_information_numeric_mlp": "same_numeric",
}
TREES = (
    "distribution_information_xgb",
    "all_distribution_information_xgb",
    "pilot_global_mass_xgb",
    "pilot_query_mass_xgb",
)


def encode(data, existing, controls, out):
    for folder, manifest in (
        (data, "data-freeze.json"),
        (existing, "encoding-freeze.json"),
        (controls, "calibration-freeze.json"),
    ):
        verify_manifest(folder, manifest)
    split = pilot_split(data)
    previous = read_json(controls / "training-plan.json")
    previous_encoding = data.parent / "spatial-pilot-encoding-v2"
    if split != read_json(previous_encoding / "protocol.json")["split"]:
        raise ValueError("The frozen small pilot uses a different split")
    if previous["encoding_manifest_sha256"] != sha(previous_encoding / "encoding-freeze.json"):
        raise ValueError("Control models are not bound to the known small pilot")
    out.mkdir(parents=True, exist_ok=False)
    weights = BASE / "paired-persistence-runtime-v1/mobileclip2-s0-vision.pt"
    sources = [
        Path(__file__),
        ROOT / "scripts/train_landslide_motion_distribution.py",
        ROOT / "src/gnss_sim/landslide_motion_distribution_view.py",
        ROOT / "tests/test_landslide_motion_distribution_view.py",
        ROOT / "src/gnss_sim/landslide_feature_discovery.py",
        ROOT / "src/gnss_sim/landslide_motion_mass.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "scripts/train_landslide_paired_persistence.py",
        ROOT / "scripts/train_landslide_spatial_readout.py",
        ROOT / "scripts/score_landslide_claim_review.py",
        ROOT / "src/gnss_sim/landslide_observed_duration.py",
    ]
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    write_json(out / "source-bindings.json", bindings)
    protocol = dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),
        candidate="MD-VN / Motion-Distribution Visual-Numeric Recognition; development candidate",
        hypothesis="Expose distribution of displacement changes and adjacent direction agreements to a pretrained visual branch; ordered physical evidence must outperform duplicate and permuted observations and equally informed numerical learning",
        split=split,
        scope="Fixed eight families, first sorted2TRAIN/1VALIDATION per family,16/8; development selection only; no reserved inputs/reference access",
        data_manifest_sha256=sha(data / "data-freeze.json"),
        original_encoding_manifest_sha256=sha(existing / "encoding-freeze.json"),
        controls_manifest_sha256=sha(controls / "calibration-freeze.json"),
        weights_sha256=sha(weights),
        arms=ARMS,
        trees=TREES,
        observation="Same13 disjoint14day blocks,8complete NEU required;12 adjacent vector magnitudes in original noise units,11 neighboring vector cosines; no gap interpolation, no activity/stage annotations; magnitude axis max(6,max) prevents clipping",
        permutation="Fixed2026101325, independent order permutations of12magnitudes/11cosines; exact value/missing multisets, explicit rank axis; both chronology AND relational alignment removed, not a pure temporal-order isolation",
        fairness="Main/duplicate/permuted1060 inputs; no-numeric1024; auxiliary+numeric548; all neural same stride3TRAIN centers/order/labels; sameinfo MLP18+46+52=>232inputs, tree116raw dimensions; numeric arms additionally have original block vectors, at least auxiliary information; no full-calendar scaling",
        training="Original seed20261020,12epochs,Adam.001/wd.0001,widths16/32;tree120/lr.05/depth2/4/seed20261007;19thresholds,38opportunities per10new arms,380total; first max dailyF1+eventF1",
        decoder="Common14positive supported observations; bridge<=7original missing only; original numerical stage head fitted16TRAIN unchanged",
        controls="Frozen original16/8 spatial pilot11methods,38opportunities per learned arm,9per rule; no48TRAIN models masquerading as16TRAIN competitors",
        gate=[
            "main dailyF1 above duplicate and original global+numeric",
            "main dailyF1 above permuted auxiliary",
            "main eventF1 >= duplicate, original global+numeric and permuted",
            "positive net corrected known days versus duplicate and permuted",
            "quiet/step-only/spike-only FP <= duplicate",
            "mixed deformation recall >= duplicate",
            "visual necessity: beat best sameinfo numerical F1 OR greater eventF1 with dailyF1 within.01",
        ],
        failure="Only all seven supports48/24expansion; otherwise stop fixed view version, no scale/seed/threshold rescue and no reserved evaluation",
        limitations="Visualization plus multimodal concatenation are not general novelty; this is not a new Agent runtime; developer exploratory reference and selected validation do not establish independent field accuracy",
        test_truth_opened=False,
        new_api_calls=0,
    )
    write_json(out / "protocol.json", protocol)
    write_json(
        out / "pre-encoding-freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    )
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
    if (
        str(transform)
        != read_json(BASE / "paired-persistence-runtime-v1/encoder-availability.json")[
            "image_transform"
        ]
    ):
        raise ValueError("Pretrained image transform changed")
    started = time.perf_counter()
    timings, training_numbers, training_information = [], [], []
    (out / "progress").mkdir()
    for group, ids in split.items():
        (out / group).mkdir()
        for cid in ids:
            before = time.perf_counter()
            case, features = load_case_features(data, cid)
            raw = np.array([v if v is not None else [np.nan] * 3 for v in case.displacement_mm])
            view = distribution_view_inputs(raw, features.noise)
            blocks, available, _, _ = discovery_block_matrix(raw, features.noise)
            profile = np.column_stack((blocks.reshape(len(raw), -1), available))
            with np.load(existing / group / (cid + ".npz")) as archive:
                bank = {key: archive[key] for key in archive.files}
            bank["distribution_descriptors"] = view["descriptors"]
            bank["raw_profile"] = profile
            indices = np.flatnonzero(bank["selected"])
            for permuted, key in ((False, "distribution"), (True, "permuted")):
                embeddings = np.zeros_like(bank["embeddings"])
                with torch.inference_mode():
                    for first in range(0, len(indices), 32):
                        batch = indices[first : first + 32]
                        tensor = torch.stack(
                            [
                                transform(
                                    render_distribution_view(
                                        view["magnitude"][index], view["agreement"][index], permuted
                                    )
                                )
                                for index in batch
                            ]
                        )
                        embeddings[batch] = torch.nn.functional.normalize(
                            encoder.visual(tensor), dim=-1
                        ).numpy()
                bank[key] = embeddings
            np.savez_compressed(out / group / (cid + ".npz"), **bank)
            if group == "train":
                training_numbers.append(bank["numbers"][indices])
                training_information.append(
                    np.column_stack((bank["numbers"], view["descriptors"], profile))[indices]
                )
            timings.append(
                dict(
                    group=group,
                    case_id=cid,
                    centers=len(indices),
                    seconds=time.perf_counter() - before,
                )
            )
            write_csv(out / "progress" / (cid + ".csv"), [timings[-1]])
            print(group, cid, len(indices), f"{timings[-1]['seconds']:.2f}s", flush=True)
    write_csv(out / "encoding-timings.csv", timings)
    scaling = fit_numeric_scaling(np.concatenate(training_numbers))
    if scaling != read_json(previous_encoding / "numeric-scaling.json"):
        raise ValueError("Original small-pilot numeric scaling changed")
    write_json(out / "numeric-scaling.json", scaling)
    write_json(
        out / "information-scaling.json", fit_numeric_scaling(np.concatenate(training_information))
    )
    write_json(
        out / "encoding-cost.json",
        dict(
            seconds=time.perf_counter() - started,
            original_centers=sum(row["centers"] for row in timings),
            new_visual_encodings=2 * sum(row["centers"] for row in timings),
            new_independent_records=0,
            new_api_calls=0,
            test_truth_opened=False,
        ),
    )
    for relative, expected in bindings.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Bound source changed: " + relative)
    write_json(
        out / "encoding-freeze.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print("Motion-distribution pilot encoding frozen", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "existing", "controls", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    encode(
        args.data.resolve(), args.existing.resolve(), args.controls.resolve(), args.out.resolve()
    )
