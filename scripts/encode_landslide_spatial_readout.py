"""Freeze and encode a stratified center/context visual mechanism pilot."""

from __future__ import annotations

import argparse
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
from gnss_sim.landslide_paired_persistence import render_encoder_chart
from gnss_sim.landslide_spatial_readout import spatial_chart_readouts
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling, invariance_inputs

ARMS = {
    "global_numeric": "global",
    "duplicate_global_numeric": "duplicate",
    "global_center_numeric": "center",
    "global_contrast_numeric": "contrast",
    "global_contrast_visual": "visual",
    "numeric": "numeric",
}


def verify_manifest(folder, filename):
    for relative, expected in read_json(folder / filename).items():
        if sha(folder / relative) != expected:
            raise ValueError("Frozen artifact changed: " + str(folder / relative))


def pilot_split(data):
    protocol = read_json(data / "protocol.json")
    ledger = read_json(data / "training-generation-ledger.json")
    families = {row["case_id"]: row["truth"]["family"] for row in ledger}
    split = {}
    for group, count in (("train", 2), ("validation", 1)):
        ids = []
        for family in protocol["families"]:
            candidates = [cid for cid in protocol["split"][group] if families[cid] == family]
            ids.extend(sorted(candidates)[:count])
        split[group] = sorted(ids)
    return split


def encode(data, existing, out):
    verify_manifest(data, "data-freeze.json")
    verify_manifest(existing, "encoding-freeze.json")
    if read_json(existing / "protocol.json")["data_manifest_sha256"] != sha(
        data / "data-freeze.json"
    ):
        raise ValueError("Original embedding cache belongs to a different dataset")
    out.mkdir(parents=True, exist_ok=False)
    split = pilot_split(data)
    runtime = BASE / "paired-persistence-runtime-v1"
    weights = runtime / "mobileclip2-s0-vision.pt"
    sources = [
        Path(__file__),
        ROOT / "scripts/train_landslide_spatial_readout.py",
        ROOT / "src/gnss_sim/landslide_spatial_readout.py",
        ROOT / "src/gnss_sim/landslide_observed_duration.py",
        ROOT / "src/gnss_sim/landslide_paired_persistence.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "scripts/train_landslide_paired_persistence.py",
        ROOT / "scripts/run_landslide_comparison.py",
        ROOT / "scripts/score_landslide_claim_review.py",
        ROOT / "src/gnss_sim/landslide_evaluation.py",
        ROOT / "src/gnss_sim/landslide_local_experiment.py",
    ]
    protocol = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "CC-VN / Center-Context Visual-Numeric Readout (pilot only)",
        "scope": "16 training /8 exposed validation records; first sorted2/1 per eight families; no reserved input or reference access",
        "split": split,
        "data_manifest_sha256": sha(data / "data-freeze.json"),
        "original_encoding_manifest_sha256": sha(existing / "encoding-freeze.json"),
        "weights_sha256": sha(weights),
        "arms": ARMS,
        "geometry": "8x8 feature map: center rows0:7 cols3:5; flanks rows0:7 cols1,2,5,6; global all8x8; same pretrained1024->512 fc, L2 normalize each pool",
        "contrast": "(normalized center - normalized flanks)/2; concatenate global; no contrast renormalization or learned pooling",
        "limits": "Attention mixes global context; stride32 and plot geometry do not support exact daily alignment. Patch readout itself is not novel relative to VLM4TS/VETime",
        "fairness": "Main/center/duplicate have identical1060 inputs and head sizes; global-only548, visual1024, numeric36 disclosed. All neural arms use identical stride3 known TRAIN centers; trees all supported known TRAIN dates",
        "training": "Fixed existing seed20261020/12epochs/Adam lr.001/wd.0001; widths16,32 or tree depths2,4; thresholds.05..95;38 opportunities per learned arm; first max dailyF1+eventF1 selects",
        "decoder": "Common14 positive supported observations, bridge only<=7 raw missing dates, no observed negatives/unsupported observed days; unknown days never filled",
        "gate": [
            "main dailyF1 higher than same-dimension duplicate",
            "main dailyF1 higher than center without subtraction",
            "main eventF1 not lower than duplicate or center",
            "main positive net corrected days versus duplicate and center",
            "quiet/step-only/spike-only FP no greater than duplicate",
            "mixed deformation+artifact recall no lower than duplicate",
            "visual necessity: beat numeric dailyF1, or better eventF1 with dailyF1 within.01",
        ],
        "next": "Only all checks permits expansion to48/24; not reserved confirmation. If not, stop this readout, no seed/window/threshold rescue",
        "original_strong_methods": "Refit raw12/raw84/chart384 XGBoost plus robust rules and PELT on identical16/8; reuse existing implementations but recalibrate all with common decoder",
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
        raise ValueError("Pretrained image transform changed")
    timings, training_numbers = [], []
    (out / "progress").mkdir()
    maximum_error = 0.0
    started = time.perf_counter()
    for group, ids in split.items():
        (out / group).mkdir()
        for cid in ids:
            before = time.perf_counter()
            case, features = load_case_features(data, cid)
            images, numbers, supported = invariance_inputs(case, features)
            with np.load(existing / group / (cid + ".npz")) as archive:
                bank = {key: archive[key] for key in archive.files}
            if not np.array_equal(supported, bank["supported"]):
                raise ValueError("Original feature support changed")
            indices = np.flatnonzero(bank["selected"])
            center = np.zeros_like(bank["embeddings"])
            contrast = np.zeros_like(center)
            global_embedding = np.zeros_like(center)
            with torch.inference_mode():
                for first in range(0, len(indices), 32):
                    batch = indices[first : first + 32]
                    tensor = torch.stack(
                        [transform(render_encoder_chart(images[index])) for index in batch]
                    )
                    maps = encoder.visual.trunk.forward_features(tensor)
                    global_part, center_part, difference = spatial_chart_readouts(
                        maps, encoder.visual.trunk.head.fc
                    )
                    # Native forward_head is the official global-pool path.
                    native = torch.nn.functional.normalize(
                        encoder.visual.trunk.forward_head(maps), dim=-1
                    )
                    if not torch.allclose(global_part, native, atol=1e-6, rtol=1e-5):
                        raise ValueError(
                            "Spatial global pool does not match pretrained native head"
                        )
                    global_embedding[batch] = global_part.numpy()
                    center[batch] = center_part.numpy()
                    contrast[batch] = difference.numpy()
            error = float(np.max(np.abs(global_embedding[indices] - bank["embeddings"][indices])))
            maximum_error = max(maximum_error, error)
            if not np.allclose(
                global_embedding[indices], bank["embeddings"][indices], atol=1e-6, rtol=1e-5
            ):
                raise ValueError("Original frozen global embeddings changed")
            bank["center"] = center
            bank["contrast"] = contrast
            np.savez_compressed(out / group / (cid + ".npz"), **bank)
            if group == "train":
                training_numbers.append(numbers[bank["selected"]])
            if cid == ids[0]:
                render_encoder_chart(images[indices[len(indices) // 2]]).save(
                    out / (group + "-example.png")
                )
            timings.append(
                {
                    "group": group,
                    "case_id": cid,
                    "centers": len(indices),
                    "seconds": time.perf_counter() - before,
                }
            )
            write_csv(out / "progress" / (cid + ".csv"), [timings[-1]])
            print(group, cid, len(indices), f"{timings[-1]['seconds']:.2f}s", flush=True)
    write_csv(out / "encoding-timings.csv", timings)
    write_json(out / "numeric-scaling.json", fit_numeric_scaling(np.concatenate(training_numbers)))
    write_json(
        out / "encoding-cost.json",
        {
            "seconds": time.perf_counter() - started,
            "centers": sum(row["centers"] for row in timings),
            "global_native_and_original_verified": True,
            "maximum_global_embedding_absolute_error": maximum_error,
            "new_api_calls": 0,
            "test_truth_opened": False,
        },
    )
    for relative, expected in bindings.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Bound source changed during encoding: " + relative)
    write_json(
        out / "encoding-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print("Spatial pilot encoding frozen", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "existing", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    encode(args.data.resolve(), args.existing.resolve(), args.out.resolve())
