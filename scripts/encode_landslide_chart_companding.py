"""Freeze a target-anchored plotting pilot and encode its two changed views."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open_clip
import torch
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import BASE, ROOT, write_csv
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_companded_chart import companded_chart_patches, raw_window_medians
from gnss_sim.landslide_paired_persistence import render_encoder_chart
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors

ARMS = {
    "target_companded_numeric": {"view": "target", "numeric": True},
    "global_companded_numeric": {"view": "global", "numeric": True},
    "target_companded_visual": {"view": "target", "numeric": False},
    "original_visual": {"view": "original", "numeric": False},
}
EXISTING_CONTROLS = [
    "global_numeric",
    "numeric",
    "xgb_raw12",
    "xgb_raw84",
    "xgb_chart384",
    "robust_rule",
    "pelt_rule",
]


def encode(data, existing, previous, out):
    for folder, manifest in (
        (data, "data-freeze.json"),
        (existing, "encoding-freeze.json"),
        (previous, "calibration-freeze.json"),
    ):
        verify_manifest(folder, manifest)
    original_protocol = read_json(existing / "protocol.json")
    if original_protocol["data_manifest_sha256"] != sha(data / "data-freeze.json"):
        raise ValueError("Original pilot cache belongs to a different dataset")
    if read_json(previous / "training-plan.json")["encoding_manifest_sha256"] != sha(
        existing / "encoding-freeze.json"
    ):
        raise ValueError("Existing controls did not train on exactly this pilot cache")
    out.mkdir(parents=True, exist_ok=False)
    split = original_protocol["split"]
    runtime = BASE / "paired-persistence-runtime-v1"
    weights = runtime / "mobileclip2-s0-vision.pt"
    sources = [
        Path(__file__),
        ROOT / "scripts/train_landslide_chart_companding.py",
        ROOT / "src/gnss_sim/landslide_companded_chart.py",
        ROOT / "src/gnss_sim/landslide_paired_persistence.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "scripts/train_landslide_paired_persistence.py",
        ROOT / "scripts/train_landslide_spatial_readout.py",
        ROOT / "src/gnss_sim/landslide_observed_duration.py",
        ROOT / "scripts/score_landslide_claim_review.py",
        ROOT / "src/gnss_sim/landslide_evaluation.py",
    ]
    protocol = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "TAC-VN / Target-Anchored Companding Visual-Numeric Detection",
        "scope": "Same stratified16train/8exposedvalidation; original centers, support and references unchanged; not reserved evaluation",
        "split": split,
        "arms": ARMS,
        "existing_controls": EXISTING_CONTROLS,
        "data_manifest_sha256": sha(data / "data-freeze.json"),
        "original_encoding_manifest_sha256": sha(existing / "encoding-freeze.json"),
        "original_encoding_path": str(existing),
        "original_control_manifest_sha256": sha(previous / "calibration-freeze.json"),
        "weights_sha256": sha(weights),
        "change": "Use asinh((original3-daymedians - central14observedrawmedian)/(3originaldiffMADnoise)); same61slots/32rows; span=max(abs(transformed),asinh(1)); no clipping/jump subtraction/gap filling",
        "anchor_ablation": "Identical asinh/noise/masks/encoder/head, replacing target anchor with median of all61 available slots",
        "limits": "Asinh/MAD normalization already used in GNSS-FM2606.07725 III-C; image preprocessing already in VLM4TS/ITF-TAD. Candidate only tests target-local/noise-difference adaptation as a chart input. Warped curve derivatives cannot be physical ASD evidence",
        "training": "Reuse original train_head: fixed seed20261020,12epochs, balanced BCE, no pairing/augmentation; widths16,32 and thresholds.05..95;38opportunities each; new two chart-information trees depths2,4/19thresholds, full supported TRAIN dates",
        "fairness": "Main/global-asinh/original-global each548inputs, identical model capacity and same3382TRAINcenters. Numerical scaling and stage head trained solely on same16records. Existing controls are exact applicable same-split fits under common missing decoder, not old-population models",
        "decoder": "Same14positive observations, bridge only<=7rawmissing, missing remainsunknown; all methods use identical original activity and stage support",
        "gate": [
            "dailyF1 strictly improves original-global and global-asinh",
            "eventF1 not lower than either",
            "positive net correction vs both",
            "pure nuisance FP no greater than original-global",
            "mixed recall no lower than original-global",
            "visual value vs same numeric head",
            "numerical feedback value vs target-only visual",
        ],
        "next": "All7 permits48/24expansion only; any failure stops this registered plotting version without tuning3sigma/anchor length/seed. Reserved24 untouched",
        "test_truth_opened": False,
        "new_api_calls": 0,
    }
    write_json(out / "protocol.json", protocol)
    for source in sources:
        target = out / "source" / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
    bindings = {str(source.relative_to(ROOT)): sha(source) for source in sources}
    write_json(out / "source-bindings.json", bindings)
    shutil.copyfile(existing / "numeric-scaling.json", out / "numeric-scaling.json")
    write_json(
        out / "pre-encoding-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
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
    if str(transform) != read_json(runtime / "encoder-availability.json")["image_transform"]:
        raise ValueError("Pretrained transform changed")
    started = time.perf_counter()
    timings = []
    (out / "progress").mkdir()
    for group, ids in split.items():
        (out / group).mkdir()
        for cid in ids:
            before = time.perf_counter()
            case, feature = load_case_features(data, cid)
            medians, anchor = raw_window_medians(case)
            with np.load(existing / group / (cid + ".npz")) as archive:
                # Do not duplicate obsolete center/context pilot representations.
                bank = {
                    key: archive[key] for key in archive.files if key not in {"center", "contrast"}
                }
            indices = np.flatnonzero(bank["selected"])
            for name, local in (("target", True), ("global", False)):
                patches = companded_chart_patches(
                    medians, anchor, feature.noise, target_anchored=local
                )
                embeddings = np.zeros_like(bank["embeddings"])
                with torch.inference_mode():
                    for first in range(0, len(indices), 32):
                        batch = indices[first : first + 32]
                        tensor = torch.stack(
                            [transform(render_encoder_chart(patches[index])) for index in batch]
                        )
                        embeddings[batch] = encoder.encode_image(tensor, normalize=True).numpy()
                if not np.isfinite(embeddings).all() or not np.allclose(
                    np.linalg.norm(embeddings[indices], axis=1), 1, atol=1e-5
                ):
                    raise ValueError("Companded embeddings must be finite and normalized")
                bank[name + "_embeddings"] = embeddings
                bank[name + "_descriptors"] = chart_numeric_descriptors(patches)
                if cid == ids[0]:
                    render_encoder_chart(patches[indices[len(indices) // 2]]).save(
                        out / (group + "-" + name + "-example.png")
                    )
            np.savez_compressed(out / group / (cid + ".npz"), **bank)
            timings.append(
                {
                    "group": group,
                    "case_id": cid,
                    "original_centers": len(indices),
                    "new_view_centers": 2 * len(indices),
                    "seconds": time.perf_counter() - before,
                }
            )
            write_csv(out / "progress" / (cid + ".csv"), [timings[-1]])
            print(group, cid, len(indices), f"two-view{timings[-1]['seconds']:.2f}s", flush=True)
    write_csv(out / "encoding-timings.csv", timings)
    write_json(
        out / "encoding-cost.json",
        {
            "seconds": time.perf_counter() - started,
            "original_centers_reused": sum(row["original_centers"] for row in timings),
            "new_view_centers": sum(row["new_view_centers"] for row in timings),
            "test_truth_opened": False,
            "new_api_calls": 0,
        },
    )
    for relative, expected in bindings.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Bound source changed during encoding: " + relative)
    write_json(
        out / "encoding-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print("Companded pilot encoding frozen", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "existing", "previous", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    encode(
        args.data.resolve(), args.existing.resolve(), args.previous.resolve(), args.out.resolve()
    )
