"""Encode frozen original-date charts without any evaluation-label fitting."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from probe_landslide_endpoint_fit import BASE, ROOT, export
from run_landslide_comparison import save_json
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha
from gnss_sim.landslide_paired_persistence import render_encoder_chart


def verify_inputs(run):
    for name, expected in read_json(run / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Bound preparation source/input changed: " + name)
    for name, expected in read_json(run / "pre-encoding-freeze.json").items():
        if name != "state.json" and sha(run / name) != expected:
            raise ValueError("Frozen prepared input changed: " + name)


def encode(run):
    verify_inputs(run)
    if read_json(run / "state.json")["status"] != "prepared":
        raise ValueError("Encode only never-started prepared inputs")
    directory = run / "encoded"
    directory.mkdir(exist_ok=False)
    export(run / "encoding-plan.json", {
        "batch_size": 32, "threads": 2, "device": "cpu", "normalize_embedding": True,
        "preprocess": "Exact dfndr2b pretrained configuration, eval transform",
        "eligible": "Original sampled known train centers, all original supported validation dates, all pair negatives",
        "labels_used_to_fit_encoder": False, "encoder_frozen": True,
        "sources": {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))},
    })
    save_json(run / "state.json", {"status": "encoding"})
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = open_clip.get_pretrained_cfg("MobileCLIP2-S0", "dfndr2b")
    model, _, transform = open_clip.create_model_and_transforms(
        "MobileCLIP2-S0", pretrained=None, image_mean=config["mean"],
        image_std=config["std"], image_interpolation=config["interpolation"],
        image_resize_mode=config["resize_mode"],
    )
    weights = BASE / "paired-persistence-runtime-v1/mobileclip2-s0-vision.pt"
    model.visual.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    model.requires_grad_(False)
    expected_transform = read_json(BASE / "paired-persistence-runtime-v1/encoder-availability.json")["image_transform"]
    if str(transform) != expected_transform:
        raise ValueError("Actual transform differs from loaded pretrained transform")
    rows = []

    def encode_images(images):
        outputs = []
        with torch.inference_mode():
            for first in range(0, len(images), 32):
                tensors = torch.stack([transform(render_encoder_chart(chart)) for chart in images[first:first + 32]])
                embeddings = model.encode_image(tensors, normalize=True).cpu().numpy()
                if embeddings.shape[1] != 512 or not np.isfinite(embeddings).all():
                    raise ValueError("Invalid frozen image embeddings")
                outputs.append(embeddings)
        return np.concatenate(outputs).astype(np.float32)

    for group in ("train", "validation"):
        (directory / group).mkdir()
        for path in sorted((run / "original" / group).glob("*.npz")):
            before = time.perf_counter()
            with np.load(path) as data:
                selected = data["sample"] if group == "train" else data["supported"]
                indices = np.flatnonzero(selected)
                encoded = np.zeros((len(selected), 512), dtype=np.float32)
                encoded[indices] = encode_images(data["images"][indices])
            np.savez_compressed(directory / group / path.name, embeddings=encoded, encoded_mask=selected)
            rows.append({"group": group, "case_id": path.stem, "images": len(indices),
                         "seconds_render_preprocess_encoder": time.perf_counter() - before})
            save_json(run / "state.json", {"status": "encoding", "records": len(rows),
                                          "encoded_images": sum(row["images"] for row in rows)})
            print(group, path.stem, len(indices), flush=True)
    selected_pairs = read_json(run / "selected-training-pairs.json")
    before = time.perf_counter()
    negatives = []
    positives = []
    for row in selected_pairs:
        with np.load(run / "pairs" / (row["pair_id"] + ".npz")) as data:
            negatives.append(data["negative_image"])
        with np.load(directory / "train" / (row["case_id"] + ".npz")) as data:
            if not data["encoded_mask"][row["day"]]:
                raise ValueError("Pair positive must be an original sampled training center")
            positives.append(data["embeddings"][row["day"]])
    np.savez_compressed(directory / "pairs.npz", positive=np.stack(positives),
                        negative=encode_images(np.stack(negatives)))
    rows.append({"group": "pair_negative", "images": len(negatives),
                 "seconds_render_preprocess_encoder": time.perf_counter() - before})
    export(run / "encoding-cost.json", {"rows": rows, "wall_seconds": time.perf_counter() - started,
                                         "actual_transform": str(transform), "new_api_calls": 0})
    verify_inputs(run)
    export(run / "encoding-freeze.json", {path.relative_to(run).as_posix(): sha(path)
                                           for path in directory.rglob("*") if path.is_file()})
    save_json(run / "state.json", {"status": "encoded", "encoded_images": sum(row["images"] for row in rows)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    encode(parser.parse_args().run.resolve())
