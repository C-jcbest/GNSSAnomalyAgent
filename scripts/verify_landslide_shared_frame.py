"""Replay shared rendering, physical TRAIN interventions and selected predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import open_clip
import torch
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import BASE, ROOT
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_paired_persistence import modality_inputs, probabilities
from train_landslide_shared_frame import ARMS
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_local_persistence import local_offset_counterpart
from gnss_sim.landslide_paired_persistence import pretrained_activity_head, render_encoder_chart
from gnss_sim.landslide_shared_frame import shared_frame_pair
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors, numeric_input


def verify(data, encoded, pairs, parent_pairs, physical_replay, parent, run, out):
    for folder, manifest in (
        (data, "data-freeze.json"),
        (encoded, "encoding-freeze.json"),
        (pairs, "encoding-freeze.json"),
        (parent_pairs, "encoding-freeze.json"),
        (physical_replay, "freeze.json"),
        (parent, "freeze.json"),
        (run, "freeze.json"),
    ):
        verify_manifest(folder, manifest)
    protocol = read_json(run / "protocol.json")
    if protocol["parent_manifest_sha256"] != sha(parent / "freeze.json") or protocol[
        "pairs_manifest_sha256"
    ] != sha(pairs / "encoding-freeze.json"):
        raise ValueError("Replay references changed")
    if protocol["parent_pairs_manifest_sha256"] != sha(parent_pairs / "encoding-freeze.json"):
        raise ValueError("Physical intervention source changed")
    physical = read_json(physical_replay / "verification.json")
    physical_run = data.parent / "local-persistence-calibration-v1"
    physical_protocol = read_json(physical_run / "protocol.json")
    if (
        physical["status"] != "VERIFIED"
        or physical["calibration_manifest_sha256"] != sha(physical_run / "freeze.json")
        or physical_protocol["pairs_manifest_sha256"] != sha(parent_pairs / "encoding-freeze.json")
    ):
        raise ValueError("Physical numeric reconstruction receipt does not belong to these pairs")
    for relative, expected in protocol["source_bindings"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen method source changed: " + relative)
    split = protocol["split"]
    banks = load_banks(encoded, split)
    with np.load(pairs / "pair-bank.npz") as archive:
        pair_bank = {key: archive[key] for key in archive.files}
    with np.load(parent_pairs / "pair-bank.npz") as archive:
        old_bank = {key: archive[key] for key in archive.files}
    selected = read_json(pairs / "selected-training-pairs.json")
    if selected != read_json(parent_pairs / "selected-training-pairs.json"):
        raise ValueError("Shared-frame training centers changed")
    truths = {
        row["case_id"]: row["truth"]
        for row in read_json(data / "training-generation-ledger.json")
        if row["group"] == "train"
    }
    out.mkdir(parents=True, exist_ok=False)
    pair_images = {"shared_positive": [], "shared_negative": []}
    originals = {}
    for index, row in enumerate(selected):
        cid, day = row["case_id"], row["day"]
        if cid not in split["train"]:
            raise ValueError("Intervention outside TRAIN")
        if cid not in originals:
            case, features = load_case_features(data, cid)
            values = np.array([v if v is not None else [np.nan] * 3 for v in case.displacement_mm])
            originals[cid] = (values, features.noise)
        values, noise = originals[cid]
        latent = np.array(truths[cid]["latent_displacement_mm"])
        negative, replacement = local_offset_counterpart(
            values, latent, day - 21, day + 21, row["step_day"]
        )
        with np.load(parent_pairs / "pairs" / (row["pair_id"] + ".npz")) as archive:
            np.testing.assert_array_equal(negative, archive["local_observations"])
            np.testing.assert_array_equal(replacement, archive["local_latent"])
        left, right, frame = shared_frame_pair(values, negative, day, noise)
        with np.load(pairs / "pairs" / (row["pair_id"] + ".npz")) as archive:
            np.testing.assert_array_equal(left, archive["positive_image"])
            np.testing.assert_array_equal(right, archive["negative_image"])
            for key in ("span", "reference"):
                np.testing.assert_array_equal(frame[key], archive[key])
        outside = np.abs(np.arange(-90, 91, 3)) > 21
        np.testing.assert_array_equal(left[:, :, outside], right[:, :, outside])
        np.testing.assert_array_equal(left[3:], right[3:])
        for kind, image in (("shared_positive", left), ("shared_negative", right)):
            available = image[3:, 0] > 0
            np.testing.assert_allclose(image[:3].sum(axis=1)[available], 1, atol=1e-6)
            np.testing.assert_array_equal(
                chart_numeric_descriptors(image[None])[0], pair_bank[kind + "_charts"][index]
            )
            pair_images[kind].append(image)
        for new, old in (
            ("positive_numbers", "positive_numbers"),
            ("negative_numbers", "local_numbers"),
            ("independent_positive_embeddings", "positive_embeddings"),
            ("independent_negative_embeddings", "local_embeddings"),
            ("independent_negative_charts", "local_charts"),
        ):
            np.testing.assert_array_equal(pair_bank[new][index], old_bank[old][index])
    runtime = BASE / "paired-persistence-runtime-v1"
    weights = runtime / "mobileclip2-s0-vision.pt"
    config = open_clip.get_pretrained_cfg("MobileCLIP2-S0", "dfndr2b")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
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
    with torch.inference_mode():
        for kind, images in pair_images.items():
            parts = []
            for first in range(0, len(images), 32):
                tensor = torch.stack(
                    [transform(render_encoder_chart(image)) for image in images[first : first + 32]]
                )
                parts.append(encoder.encode_image(tensor, normalize=True).numpy())
            np.testing.assert_array_equal(np.concatenate(parts), pair_bank[kind + "_embeddings"])
    labels = {cid: [] for cid in split["validation"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    choices = read_json(run / "selection.json")
    saved_probability = read_json(run / "probabilities.json")
    saved_predictions = read_json(run / "predictions.json")
    scaling = read_json(run / "numeric-scaling.json")
    stage = XGBClassifier()
    stage.load_model(run / "stage-xgboost.json")
    codes = {cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1 for cid in labels}
    predictions = {}
    for name, choice in choices.items():
        probability = {}
        if name in ARMS:
            arm = ARMS[name]
            dimension = 512 * arm["image"] + 36 * arm["numeric"]
            model = pretrained_activity_head(dimension, 0, choice["parameter"])
            model.load_state_dict(
                torch.load(
                    run / f"{name}-width{choice['parameter']}.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            for cid in labels:
                inputs = modality_inputs(
                    banks[cid]["embeddings"], numeric_input(banks[cid]["numbers"], scaling), arm
                )
                probability[cid] = probabilities(model, inputs)
        else:
            model = XGBClassifier()
            model.load_model(run / f"{name}-depth{choice['parameter']}.json")
            for cid in labels:
                bank = banks[cid]
                inputs = (
                    np.column_stack((bank["numbers"], bank["chart_descriptors"]))
                    if "chart384" in name
                    else bank["numbers"]
                )
                probability[cid] = model.predict_proba(inputs)[:, 1]
        for cid in labels:
            np.testing.assert_array_equal(probability[cid], np.array(saved_probability[name][cid]))
        predictions[name] = {
            cid: decode(probability[cid], choice["threshold"], banks[cid], codes[cid]).to_dict()
            for cid in labels
        }
        if predictions[name] != saved_predictions[name]:
            raise ValueError("Activity/stage output replay differs: " + name)
    parent_predictions = read_json(parent / "predictions.json")
    if any(saved_predictions[name] != cases for name, cases in parent_predictions.items()):
        raise ValueError("Reused original methods changed")
    predictions.update(parent_predictions)
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in labels}
    results, summary, _ = score_target(labels, observed, weak, predictions, {}, {})
    if results != read_json(run / "results.json"):
        raise ValueError("Unified result replay differs")
    if {row["method"]: row for row in summary} != {
        row["method"]: row for row in read_json(run / "summary.json")
    }:
        raise ValueError("Unified summary replay differs")
    write_json(
        out / "verification.json",
        dict(
            status="VERIFIED",
            calibration_manifest_sha256=sha(run / "freeze.json"),
            all_eligible_pairs=len(selected),
            train_parents=len({row["case_id"] for row in selected}),
            local_outer_motion_error_missing_preserved=True,
            shared_images_embeddings_and_descriptors_exact=True,
            outer_pixels_exact=True,
            no_clipping=True,
            physical_numeric_replay_manifest_sha256=sha(physical_replay / "freeze.json"),
            selected_new_models=len(choices),
            methods=len(predictions),
            record_outputs=len(predictions) * len(labels),
            probability_values=len(choices) * len(labels) * 730,
            all_activity_stage_scores_exact=True,
            test_truth_opened=False,
            no_training=True,
            new_api_calls=0,
        ),
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(read_json(out / "verification.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data",
        "encoded",
        "pairs",
        "parent-pairs",
        "physical-replay",
        "parent",
        "run",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(
        args.data.resolve(),
        args.encoded.resolve(),
        args.pairs.resolve(),
        args.parent_pairs.resolve(),
        args.physical_replay.resolve(),
        args.parent.resolve(),
        args.run.resolve(),
        args.out.resolve(),
    )
