"""Infer the registered new batch after Codex observation reference is frozen."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from encode_landslide_paired_persistence import verify_inputs
from probe_landslide_contrastive_exemplars import SOURCE, raw_values
from probe_landslide_endpoint_fit import BASE, ROOT, export
from probe_landslide_feature_discovery import expert_features
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import save_json
from run_landslide_window_optimization import load_case_features, read_json
from train_landslide_paired_persistence import (
    ARMS,
    calibrated_prediction,
    modality_inputs,
    probabilities,
)
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_paired_persistence import pretrained_activity_head, render_encoder_chart
from gnss_sim.landslide_visual_invariance import (
    chart_numeric_descriptors,
    invariance_inputs,
    numeric_input,
)


def run_confirmation(calibration, data, out):
    verify_inputs(calibration)
    verify_bindings(calibration, "calibration-freeze.json")
    if read_json(calibration / "calibration-decision.json")["decision"] != "PROCEED_PAIRED_PERSISTENCE_DEVELOPMENT":
        raise ValueError("Complete development only after the registered calibration gate")
    verify_bindings(data, "data-freeze.json")
    review = data / "codex-review/compiled/reference-freeze.json"
    for filename, expected in read_json(review)["files"].items():
        if sha(review.parent / filename) != expected:
            raise ValueError("Registered observation reference changed")
    out.mkdir(exist_ok=False)
    ids = read_json(data / "protocol.json")["case_ids"]
    selection = read_json(calibration / "selection.json")
    if selection != read_json(data / "protocol.json")["selection"]:
        raise ValueError("Pre-generation model selection changed")
    export(out / "protocol.json", {
        "scope": "Registered unseen24 controlled batch; no fitting or retuning",
        "case_ids": ids, "calibration": calibration.relative_to(ROOT).as_posix(),
        "data": data.relative_to(ROOT).as_posix(), "selection": selection,
        "review_binding": sha(review), "refitting": False, "new_api_calls": 0,
    })
    sources = [Path(__file__), ROOT / "scripts/train_landslide_paired_persistence.py",
               ROOT / "src/gnss_sim/landslide_paired_persistence.py", ROOT / "scripts/score_landslide_claim_review.py"]
    bindings = {path.relative_to(ROOT).as_posix(): sha(path) for path in sources}
    bindings.update(read_json(data / "training-bindings.json"))
    for name, expected in bindings.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Pre-generation training dependency changed: " + name)
    for index, path in enumerate(sources):
        destination = out / "source" / f"{index:02d}-{path.name}"
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(path.read_bytes())
    for cid in ids:
        for path in (data / "inputs" / cid).iterdir():
            if path.is_file():
                bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (review, calibration / "selection.json",
                 calibration / "numeric-scaling.json", SOURCE / "stage-xgboost.json",
                 BASE / "paired-persistence-runtime-v1/mobileclip2-s0-vision.pt"):
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    export(out / "root-bindings.json", bindings)
    export(out / "pre-inference-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "inference", "completed_records": 0})
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    config = open_clip.get_pretrained_cfg("MobileCLIP2-S0", "dfndr2b")
    encoder, _, transform = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained=None,
        image_mean=config["mean"], image_std=config["std"], image_interpolation=config["interpolation"],
        image_resize_mode=config["resize_mode"])
    encoder.visual.load_state_dict(torch.load(BASE / "paired-persistence-runtime-v1/mobileclip2-s0-vision.pt", weights_only=True))
    encoder.eval()
    encoder.requires_grad_(False)
    if str(transform) != read_json(calibration / "encoding-cost.json")["actual_transform"]:
        raise ValueError("Pretrained preprocessing changed")
    models = {}
    for name, chosen in selection.items():
        if name in ARMS:
            arm = ARMS[name]
            model = pretrained_activity_head(512 if arm["image"] else 0, 36 if arm["numeric"] else 0, chosen["width"])
            model.load_state_dict(torch.load(calibration / "models" / f"{name}-width{chosen['width']}.pt", weights_only=True))
        else:
            model = XGBClassifier()
            model.load_model(calibration / "models" / f"{name}-depth{chosen['depth']}.json")
        models[name] = model
    scaling = read_json(calibration / "numeric-scaling.json")
    stage_head = XGBClassifier()
    stage_head.load_model(SOURCE / "stage-xgboost.json")
    predictions = {name: {} for name in list(ARMS) + ["xgb_raw18", "xgb_expert24", "xgb_chart384"]}
    costs, observed = [], {}
    for cid in ids:
        case, features = load_case_features(data, cid)
        before = time.perf_counter()
        charts, numbers, support = invariance_inputs(case, features)
        feature_seconds = time.perf_counter() - before
        embeddings = np.zeros((len(support), 512), dtype=np.float32)
        indices = np.flatnonzero(support)
        before = time.perf_counter()
        with torch.inference_mode():
            for first in range(0, len(indices), 32):
                selected = indices[first:first + 32]
                inputs = torch.stack([transform(render_encoder_chart(charts[index])) for index in selected])
                embeddings[selected] = encoder.encode_image(inputs, normalize=True).numpy()
        encoding_seconds = time.perf_counter() - before
        descriptors = chart_numeric_descriptors(charts)
        expert = expert_features(discovery_block_matrix(raw_values(case), features.noise)[0])
        normalized = numeric_input(numbers, scaling)
        codes = stage_head.predict(features.matrix).astype(int) + 1
        probabilities_by_method = {}
        head_costs = {}
        for name, model in models.items():
            before = time.perf_counter()
            if name in ARMS:
                probability = probabilities(model, modality_inputs(embeddings, normalized, ARMS[name]))
            else:
                matrix = numbers
                if name == "xgb_expert24":
                    matrix = np.column_stack((numbers, expert))
                elif name == "xgb_chart384":
                    matrix = np.column_stack((numbers, descriptors))
                probability = model.predict_proba(matrix)[:, 1]
            predictions[name][cid] = calibrated_prediction(probability, support, codes, features.supported,
                                                         selection[name]["threshold"]).to_dict()
            head_costs[name] = time.perf_counter() - before
            probabilities_by_method[name] = probability.tolist()
        np.savez_compressed(out / (cid + "-inputs.npz"), embeddings=embeddings, numbers=numbers,
            supported=support, observed=features.observed, stage_support=features.supported, codes=codes,
            chart_descriptors=descriptors, expert_features=expert)
        export(out / (cid + "-probabilities.json"), probabilities_by_method)
        observed[cid] = features.observed
        costs.append({"case_id": cid, "original_dates": len(case.dates), "encoded_images": len(indices),
                      "feature_seconds": feature_seconds, "render_encoder_seconds": encoding_seconds, "heads": head_costs})
        save_json(out / "state.json", {"status": "inference", "completed_records": len(costs),
                                      "encoded_images": sum(row["encoded_images"] for row in costs)})
        print(cid, len(indices), round(encoding_seconds, 3), flush=True)
    export(out / "new-predictions.json", predictions)
    export(out / "inference-cost.json", {"rows": costs, "wall_seconds": time.perf_counter() - started, "new_api_calls": 0})
    export(out / "prediction-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file() and path.name != "state.json"})
    for name, expected in bindings.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Confirmation inference source/evidence changed: " + name)
    save_json(out / "state.json", {"status": "predictions_complete", "process_reference_opened": False})
    export(out / "result-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()})
    print("All ten frozen heads complete; process labels remain unopened", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    run_confirmation(arguments.calibration.resolve(), arguments.data.resolve(), arguments.out.resolve())
