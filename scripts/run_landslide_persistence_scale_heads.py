"""Run the ten frozen learning models on every registered scale-extension record."""

from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
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

WORKER = {}


def initialize_models(calibration: str) -> None:
    """Each worker owns its models; original two-thread numerical settings stay fixed."""
    directory = Path(calibration)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    config = open_clip.get_pretrained_cfg("MobileCLIP2-S0", "dfndr2b")
    encoder, _, transform = open_clip.create_model_and_transforms(
        "MobileCLIP2-S0",
        pretrained=None,
        image_mean=config["mean"],
        image_std=config["std"],
        image_interpolation=config["interpolation"],
        image_resize_mode=config["resize_mode"],
    )
    weights = BASE / "paired-persistence-runtime-v1/mobileclip2-s0-vision.pt"
    encoder.visual.load_state_dict(torch.load(weights, weights_only=True), strict=True)
    encoder.eval()
    encoder.requires_grad_(False)
    if str(transform) != read_json(directory / "encoding-cost.json")["actual_transform"]:
        raise ValueError("Frozen visual preprocessing changed")
    selection = read_json(directory / "selection.json")
    models = {}
    for name, chosen in selection.items():
        if name in ARMS:
            arm = ARMS[name]
            model = pretrained_activity_head(
                512 if arm["image"] else 0, 36 if arm["numeric"] else 0, chosen["width"]
            )
            model.load_state_dict(torch.load(
                directory / "models" / f"{name}-width{chosen['width']}.pt", weights_only=True
            ))
        else:
            model = XGBClassifier()
            model.load_model(directory / "models" / f"{name}-depth{chosen['depth']}.json")
        models[name] = model
    stage_head = XGBClassifier()
    stage_head.load_model(SOURCE / "stage-xgboost.json")
    WORKER.update(
        encoder=encoder,
        transform=transform,
        models=models,
        stage_head=stage_head,
        selection=selection,
        scaling=read_json(directory / "numeric-scaling.json"),
        initialization_seconds=time.perf_counter() - started,
    )


def predict_record(data: str, cid: str):
    """Use the same calendar, batches, embeddings and postprocessing as old confirmation."""
    started = time.perf_counter()
    case, features = load_case_features(Path(data), cid)
    load_seconds = time.perf_counter() - started
    before = time.perf_counter()
    charts, numbers, support = invariance_inputs(case, features)
    feature_seconds = time.perf_counter() - before
    embeddings = np.zeros((len(support), 512), dtype=np.float32)
    indices = np.flatnonzero(support)
    before = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(indices), 32):
            selected = indices[first:first + 32]
            tensors = torch.stack([
                WORKER["transform"](render_encoder_chart(charts[index])) for index in selected
            ])
            embeddings[selected] = WORKER["encoder"].encode_image(tensors, normalize=True).numpy()
    encoding_seconds = time.perf_counter() - before
    before = time.perf_counter()
    descriptors = chart_numeric_descriptors(charts)
    expert = expert_features(discovery_block_matrix(raw_values(case), features.noise)[0])
    normalized = numeric_input(numbers, WORKER["scaling"])
    codes = WORKER["stage_head"].predict(features.matrix).astype(int) + 1
    auxiliary_seconds = time.perf_counter() - before
    outputs, probability_values, costs = {}, {}, {}
    for name, model in WORKER["models"].items():
        before = time.perf_counter()
        if name in ARMS:
            probability = probabilities(
                model, modality_inputs(embeddings, normalized, ARMS[name])
            )
        else:
            matrix = numbers
            if name == "xgb_expert24":
                matrix = np.column_stack((numbers, expert))
            elif name == "xgb_chart384":
                matrix = np.column_stack((numbers, descriptors))
            probability = model.predict_proba(matrix)[:, 1]
        prediction = calibrated_prediction(
            probability, support, codes, features.supported, WORKER["selection"][name]["threshold"]
        )
        if np.any(prediction.activity[~features.observed] != -1):
            raise ValueError("Missing day acquired an activity decision")
        if np.any(prediction.stage[~features.observed] != -1):
            raise ValueError("Missing day acquired a stage")
        outputs[name] = prediction.to_dict()
        probability_values[name] = probability.tolist()
        costs[name] = time.perf_counter() - before
    arrays = dict(
        embeddings=embeddings, numbers=numbers, supported=support, observed=features.observed,
        stage_support=features.supported, codes=codes, chart_descriptors=descriptors,
        expert_features=expert,
    )
    cost = dict(
        case_id=cid, original_dates=len(case.dates), encoded_images=len(indices),
        load_seconds=load_seconds, feature_seconds=feature_seconds,
        render_encoder_seconds=encoding_seconds, auxiliary_seconds=auxiliary_seconds,
        heads=costs, record_wall_seconds=time.perf_counter() - started,
        worker_initialization_seconds=WORKER["initialization_seconds"],
    )
    return arrays, outputs, probability_values, cost


def verify_old_record(calibration: Path) -> dict:
    """Check the new execution path against a fixed old record without refitting."""
    initialize_models(str(calibration))
    previous = BASE / "paired-persistence-confirmation-inference-v1"
    cid = "case_0193"
    started = time.perf_counter()
    arrays, outputs, values, _ = predict_record(
        str(BASE / "paired-persistence-confirmation-data-v1"), cid
    )
    with np.load(previous / f"{cid}-inputs.npz") as saved:
        for name, value in arrays.items():
            np.testing.assert_array_equal(value, saved[name], err_msg=name)
    if values != read_json(previous / f"{cid}-probabilities.json"):
        raise ValueError("Old-record probabilities differ")
    saved_predictions = read_json(previous / "new-predictions.json")
    if any(outputs[name] != saved_predictions[name][cid] for name in outputs):
        raise ValueError("Old-record activity or stage differs")
    WORKER.clear()
    return dict(
        status="passed", case_id=cid, input_arrays="exact", probabilities="exact",
        activity_and_stage="exact", methods=len(outputs), seconds=time.perf_counter() - started,
    )


def execute_record(data: str, out: str, cid: str) -> tuple:
    arrays, outputs, values, cost = predict_record(data, cid)
    folder = Path(out)
    np.savez_compressed(folder / f"{cid}-inputs.npz", **arrays)
    export(folder / f"{cid}-probabilities.json", values)
    export(folder / f"{cid}-predictions.json", outputs)
    export(folder / f"{cid}-cost.json", cost)
    return cid, outputs, cost


def run(calibration: Path, data: Path, out: Path, workers: int) -> None:
    verify_inputs(calibration)
    verify_bindings(calibration, "calibration-freeze.json")
    verify_bindings(data, "data-freeze.json")
    protocol = read_json(data / "protocol.json")
    if protocol["experiment"] != "persistence-scale-v1" or len(protocol["case_ids"]) != 120:
        raise ValueError("This runner requires the registered complete 120-record extension")
    if protocol["selection"] != read_json(calibration / "selection.json"):
        raise ValueError("Pre-generation model selection changed")
    review = data / "codex-review/compiled/reference-freeze.json"
    for filename, expected in read_json(review)["files"].items():
        if sha(review.parent / filename) != expected:
            raise ValueError("Frozen observation reference changed")
    bindings = read_json(data / "training-bindings.json")
    for name, expected in bindings.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen training dependency changed: " + name)
    out.mkdir(parents=True, exist_ok=False)
    bindings[Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    bindings[data.relative_to(ROOT).as_posix() + "/data-freeze.json"] = sha(data / "data-freeze.json")
    bindings[review.relative_to(ROOT).as_posix()] = sha(review)
    export(out / "root-bindings.json", bindings)
    shutil.copyfile(Path(__file__), out / "source.py")
    export(out / "protocol.json", dict(
        experiment="persistence-scale-heads-v1",
        started_at=datetime.now(timezone.utc).isoformat(),
        scope="All 120 registered controlled records, unchanged frozen ten learning models",
        case_ids=protocol["case_ids"], calibration=calibration.relative_to(ROOT).as_posix(),
        data=data.relative_to(ROOT).as_posix(), selection=protocol["selection"],
        review_binding=sha(review), workers=workers, threads_per_worker=2,
        parallelism="Independent complete records; original batch32 and two-thread inference unchanged",
        refitting=False, new_api_calls=0, labels_used_for_inference=False,
    ))
    started = time.perf_counter()
    save_json(out / "state.json", dict(status="preflight", completed_records=0))
    try:
        preflight = verify_old_record(calibration)
        export(out / "old-record-preflight.json", preflight)
        predictions = {name: {} for name in protocol["selection"]}
        costs = []
        save_json(out / "state.json", dict(status="inference", completed_records=0))
        with ProcessPoolExecutor(
            max_workers=workers, initializer=initialize_models, initargs=(str(calibration),)
        ) as executor:
            futures = {
                executor.submit(execute_record, str(data), str(out), cid): cid
                for cid in protocol["case_ids"]
            }
            for future in as_completed(futures):
                cid, outputs, cost = future.result()
                for name, value in outputs.items():
                    predictions[name][cid] = value
                costs.append(cost)
                save_json(out / "state.json", dict(
                    status="inference", completed_records=len(costs),
                    encoded_images=sum(row["encoded_images"] for row in costs),
                    elapsed_seconds=time.perf_counter() - started,
                ))
                print(f"{cid}: {len(costs)}/120, {cost['encoded_images']} images", flush=True)
        for name, expected in bindings.items():
            if sha(ROOT / name) != expected:
                raise ValueError("Bound source or frozen evidence changed: " + name)
        export(out / "new-predictions.json", predictions)
        export(out / "inference-cost.json", dict(
            rows=sorted(costs, key=lambda row: row["case_id"]),
            wall_seconds=time.perf_counter() - started,
            preflight_seconds=preflight["seconds"], workers=workers,
            timing="Parallel wall clock differs from summed record times; includes old-record preflight",
            new_api_calls=0,
        ))
        save_json(out / "state.json", dict(
            status="predictions_complete", completed_records=len(costs),
            process_reference_used_for_prediction=False, failures=0,
        ))
        export(out / "result-freeze.json", {
            path.relative_to(out).as_posix(): sha(path)
            for path in out.rglob("*") if path.is_file()
        })
    except Exception as error:
        export(out / "failure.json", dict(error_type=type(error).__name__, detail=str(error)))
        save_json(out / "state.json", dict(
            status="failed", completed_records=len(list(out.glob("case_*-predictions.json"))),
            elapsed_seconds=time.perf_counter() - started,
        ))
        raise
    print("All 120 records and ten frozen learning models completed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    run(args.calibration.resolve(), args.data.resolve(), args.out.resolve(), args.workers)
