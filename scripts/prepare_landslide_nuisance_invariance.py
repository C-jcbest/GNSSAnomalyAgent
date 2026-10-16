"""Prepare frozen VLM-vision counterfactual views with additive instrument noise."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open_clip
import torch
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import BASE, ROOT, write_csv
from run_landslide_comparison import save_json
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_baselines import observation_features
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_motion_mass import motion_mass_descriptors
from gnss_sim.landslide_nuisance_invariance import nuisance_counterpart
from gnss_sim.landslide_paired_persistence import render_encoder_chart
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors, invariance_inputs


def prepare(data, encoded, parent, out):
    for folder, manifest in ((data, "data-freeze.json"), (encoded, "encoding-freeze.json"), (parent, "freeze.json")):
        verify_manifest(folder, manifest)
    split = read_json(encoded / "protocol.json")["split"]
    if split != read_json(parent / "protocol.json")["split"]:
        raise ValueError("Existing 48TRAIN/24VALIDATION split is required")
    out.mkdir(parents=True, exist_ok=False)
    weights = BASE / "paired-persistence-runtime-v1/mobileclip2-s0-vision.pt"
    sources = [Path(__file__), ROOT / "scripts/train_landslide_nuisance_invariance.py",
               ROOT / "src/gnss_sim/landslide_nuisance_invariance.py", ROOT / "tests/test_landslide_nuisance_invariance.py",
               ROOT / "src/gnss_sim/landslide_visual_invariance.py", ROOT / "src/gnss_sim/landslide_motion_mass.py",
               ROOT / "src/gnss_sim/landslide_diagnostics.py", ROOT / "src/gnss_sim/landslide_paired_persistence.py"]
    protocol = dict(
        candidate="NPI-VN / Nuisance-Paired Invariance Visual-Numeric Learning",
        registered_utc=datetime.now(timezone.utc).isoformat(), data=str(data), encoded=str(encoded), parent=str(parent), split=split,
        change="Preserve complete physical motion/calendar and add only two instrument offsets/four isolated spikes; activity consistency across same-parent counterfactuals",
        injection="TRAINseed20261015/VALIDATIONstressseed20261016; record-number independent streams; offsets2 unique observed dates70..T-70, directionsnormal/magnitudeUniform8..40mm; spikes4 separate observed dates20..T-20, normal vector times original noise timesUniform6..20",
        label_status="Process activity/stages invariant by additive intervention;observability can deteriorate;not independent field or new original records;no labels or latent components used to choose injections",
        visual="Frozen MobileCLIP2-S0/dfndr2b vision encoder512;original NEU chart/pretrained transform;not language generation or Agent",
        numeric="Original18 raw features plus27 existing query-mass descriptors,45physical/availability variables;original TRAIN-only scaling;same information to all numerical competitors",
        samples="All original10187 supported known stride3TRAIN centers plus corresponding augmented centers;no filtering by visual change or teacher availability",
        training="Seed20261015/12epochs/Adam.001/wd.0001/batch128/class-balanced BCE average original and counterpart;probability MSE consistency coefficient1;no loss/seed/epoch search",
        controls="Same samples without consistency;no augmentation;within-activity-class shuffled parent correspondence;visual and numeric consistency/augmentation pairs;numeric plain;query-mass XGB same augmentation",
        calibration="Nine neural arms widths16/32 and19thresholds,38opportunities each;strong augmented tree depths2/4 and19thresholds;stable first max clean dayF1+IoU.3eventF1;stress never selects thresholds/weights",
        stress="24same-parent synthetic nuisance counterparts,generated before fitting without activity labels;clean selection/predictions frozen before stress scoring;not independent data or field evaluation",
        continuation=[
            "Main clean dailyF1>=matched augmentation-.01",
            "Main stress dailyF1>matched augmentation,positive stress net correction,and stressFP<=augmentation",
            "Main stress dailyF1>same-class shuffled correspondence",
            "Main clean IoU.3eventF1>=augmentation-.02",
            "Main stressF1>numeric consistency OR cleanF1>numeric consistency with stressF1loss<=.01",
            "Mixed deformation+nuisance stress recall>=augmentation-.02",
        ],
        failure="Stop fixed version if continuation fails;do not rename stronger controls or tune stress/seed/loss;reserved24 input/reference content stays unloaded",
        quality="Instrument events are independent labels and may coexist with activity;no invariant quality head or forced exclusivity;existing quality method remains separate",
        source_bindings={str(path.resolve()): sha(path) for path in sources},
        input_bindings={str(folder / manifest): sha(folder / manifest) for folder, manifest in
                        ((data, "data-freeze.json"), (encoded, "encoding-freeze.json"), (parent, "freeze.json"))},
        weights_sha256=sha(weights), reserved_content_loaded=False, new_api_calls=0,
    )
    write_json(out / "protocol.json", protocol)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "pre-encoding-freeze.json", {path.name: sha(path) for path in out.iterdir()})
    save_json(out / "state.json", dict(status="encoding", completed_records=0, reserved_content_loaded=False))
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
    if str(transform) != read_json(BASE / "paired-persistence-runtime-v1/encoder-availability.json")["image_transform"]:
        raise ValueError("Existing pretrained image transform changed")
    banks = load_banks(encoded, split)
    rows, events = [], []
    for group, ids in split.items():
        (out / group).mkdir()
        (out / "inputs" / group).mkdir(parents=True)
        for cid in ids:
            started = time.perf_counter()
            case, features = load_case_features(data, cid)
            values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
            altered, artifact, added_events = nuisance_counterpart(
                values, features.noise, 20261015 if group == "train" else 20261016, int(cid[-4:]),
            )
            counterpart_input = case.model_dump()
            counterpart_input["displacement_mm"] = [
                row.tolist() if np.isfinite(row).all() else None for row in altered
            ]
            counterpart = LandslideInput.model_validate(counterpart_input)
            (out / "inputs" / group / (cid + ".json")).write_text(counterpart.model_dump_json(), encoding="utf-8")
            motions = {window: derive_motion(counterpart, window) for window in (31, 61, 91)}
            changed_features = observation_features(counterpart, motions)
            images, numbers, supported = invariance_inputs(counterpart, changed_features)
            np.testing.assert_array_equal(supported, banks[cid]["supported"])
            np.testing.assert_array_equal(changed_features.observed, banks[cid]["observed"])
            np.testing.assert_array_equal(changed_features.supported, banks[cid]["stage_supported"])
            mass = motion_mass_descriptors(altered, changed_features.noise)["query_distribution"]
            selected = banks[cid]["selected"] if group == "train" else supported
            indices = np.flatnonzero(selected)
            embeddings = np.zeros((len(values), 512), dtype=np.float32)
            with torch.inference_mode():
                for start in range(0, len(indices), 32):
                    batch = indices[start:start + 32]
                    tensor = torch.stack([transform(render_encoder_chart(images[index])) for index in batch])
                    embeddings[batch] = encoder.encode_image(tensor, normalize=True).numpy()
            np.savez_compressed(out / group / (cid + ".npz"), embeddings=embeddings, numbers=np.column_stack((numbers, mass)),
                                selected=selected, supported=supported, observed=changed_features.observed,
                                stage_supported=changed_features.supported, stage_matrix=changed_features.matrix,
                                chart_descriptors=chart_numeric_descriptors(images), artifact=artifact,
                                noise=changed_features.noise, activity=banks[cid]["activity"])
            events.append(dict(group=group, case_id=cid, parent_input_sha256=sha(data / "inputs" / cid / "input.json"),
                               counterpart_input_sha256=sha(out / "inputs" / group / (cid + ".json")),
                               seed=20261015 if group == "train" else 20261016, events=added_events,
                               activity_source="Exact unchanged process activity labels from parent bank"))
            rows.append(dict(group=group, case_id=cid, images=len(indices), seconds=time.perf_counter() - started))
            save_json(out / "state.json", dict(status="encoding", completed_records=len(rows), last_record=cid,
                                              encoded_images=sum(row["images"] for row in rows), reserved_content_loaded=False))
            print(rows[-1], flush=True)
    write_csv(out / "encoding-costs.csv", rows)
    write_json(out / "injection-ledger.json", events)
    write_json(out / "costs.json", dict(seconds=sum(row["seconds"] for row in rows), images=sum(row["images"] for row in rows),
                                      derived_train_records=48, derived_stress_records=24, new_independent_records=0, new_api_calls=0))
    for path, expected in protocol["source_bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Prepared source changed during encoding: " + path)
    save_json(out / "state.json", dict(status="encoded", records=len(rows), reserved_content_loaded=False))
    write_json(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    print(dict(status="encoded", derived_records=len(rows), independent_new_records=0), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "parent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.data.resolve(), arguments.encoded.resolve(), arguments.parent.resolve(), arguments.out.resolve())
