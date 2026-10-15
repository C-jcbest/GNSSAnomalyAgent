"""Large independent station cohorts; fit once and reuse unchanged numerical models."""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np

from gnss_research.baselines import fit_baselines, predict_numerical
from gnss_research.data import SCENARIOS, simulate
from gnss_research.experiment import register, verify_dataset
from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.segmentation import segment_proposals

DATASET = Path("data/exp28-expanded-1528-1530")
TRAINING = Path("runs/exp28-shared-training-v1")
METHODS = ["rf_split_temporal", "segment_numeric", "random_forest", "rf_balanced", "extra_trees", "isolation_forest"]
TRAIN_SEED = 2026101531
COHORT_SEEDS = (2026101528, 2026101529, 2026101530)
_MODELS = None


def generate():
    DATASET.mkdir(parents=True, exist_ok=False)
    assignments = []
    for index, stream in enumerate(np.random.SeedSequence(TRAIN_SEED).spawn(96)):
        assignments.append((f"t{index:04d}", "train", TRAIN_SEED, index, stream))
    for group, cohort_seed in enumerate(COHORT_SEEDS):
        streams = np.random.SeedSequence(cohort_seed).spawn(240)
        for split_index, split in enumerate(("validation", "test")):
            prefix = "v" if split == "validation" else "e"
            for index in range(120):
                assignments.append((f"{prefix}{group * 120 + index:04d}", split, cohort_seed,
                                    index, streams[split_index * 120 + index]))
    records = []
    for station, split, cohort_seed, index, stream in assignments:
        case_seed = int(stream.generate_state(1)[0])
        observation, truth = simulate(case_seed, station, SCENARIOS[index % 6], 730, 365)
        observed_path, truth_path = Path("observations") / f"{station}.json", Path("truth") / f"{station}.json"
        write_json(DATASET / observed_path, observation)
        write_json(DATASET / truth_path, truth)
        records.append({"station_id": station, "split": split, "cohort_seed": cohort_seed,
                        "observation": str(observed_path), "truth": str(truth_path),
                        "observation_sha256": sha256(DATASET / observed_path),
                        "truth_sha256": sha256(DATASET / truth_path)})
    write_json(DATASET / "manifest.json", {
        "schema": "gnss-research-dataset-v1", "seed": COHORT_SEEDS[0], "days": 730,
        "origin": "mechanism-inspired simulation, not field measurements",
        "per_split": 360, "split_counts": {"train": 96, "validation": 360, "test": 360},
        "cohort_seeds": list(COHORT_SEEDS), "training_seed": TRAIN_SEED,
        "sources": source_hashes(), "background_days": 365, "records": records,
        "design": "unchanged simulate; independent streams; training size decoupled from evaluation size",
        "generator_script_sha256": sha256(Path(__file__))})
    verify_dataset(DATASET)
    print("generated", len(records), "independent stations", flush=True)


def train():
    TRAINING.mkdir(parents=True, exist_ok=False)
    manifest = verify_dataset(DATASET)
    write_json(TRAINING / "started.json", {"training_seed": TRAIN_SEED, "source_hashes": source_hashes()})
    models, receipt = fit_baselines(DATASET, manifest["records"], TRAIN_SEED)
    joblib.dump(models, TRAINING / "models.joblib", compress=3)
    receipt.update({"model_sha256": sha256(TRAINING / "models.joblib"), "source_hashes": source_hashes(),
                    "script_sha256": sha256(Path(__file__)), "manifest_sha256": sha256(DATASET / "manifest.json")})
    write_json(TRAINING / "training.json", receipt)
    print("fitted once", len(receipt["training_stations"]), "stations", flush=True)


def initialize_worker(model_path):
    global _MODELS
    _MODELS = joblib.load(model_path)


def predict_station(job):
    record, output = job
    station = record["station_id"]
    observation = read_json(DATASET / record["observation"])
    evidence = numerical_evidence(observation)
    first = None
    for method in METHODS:
        start = time.perf_counter()
        if method == "segment_numeric":
            result = segment_proposals(observation, first, evidence)
        else:
            result = predict_numerical(_MODELS[method], method, observation, evidence)
            if method == "rf_split_temporal":
                first = result
        result.update(status="success", request_count=0, method=method,
                      elapsed_seconds=time.perf_counter() - start)
        write_json(output / "predictions" / method / station / "result.json", result)
    return station


def predict(split):
    receipt = read_json(TRAINING / "training.json")
    if receipt["source_hashes"] != source_hashes() or receipt["model_sha256"] != sha256(TRAINING / "models.joblib"):
        raise ValueError("frozen numerical model/source changed")
    manifest = verify_dataset(DATASET)
    output = Path(f"runs/exp28-{split}-baseline-v1")
    if (output / "execution-started.json").exists():
        raise ValueError("prediction already started; no implicit replay")
    if not (output / "registration.json").exists():
        register(DATASET, output, METHODS, split, TRAIN_SEED, None, 0, 2)
    registration = read_json(output / "registration.json")
    if registration["source_hashes"] != source_hashes() or registration["manifest_sha256"] != sha256(DATASET / "manifest.json"):
        raise ValueError("registered baseline inputs changed")
    write_json(output / "shared-training-link.json", {"shared_training": str(TRAINING),
        "shared_model_sha256": receipt["model_sha256"], "protocol_path": "docs/experiment-exp28.md",
        "protocol_sha256": sha256(Path("docs/experiment-exp28.md")), "script_sha256": sha256(Path(__file__))})
    write_json(output / "training.json", receipt)
    write_json(output / "execution-started.json", {"source": "shared frozen fit", "retries": 0})
    records = [record for record in manifest["records"] if record["split"] == split]
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(max_workers=2, initializer=initialize_worker,
                             initargs=(TRAINING / "models.joblib",)) as executor:
        for number, station in enumerate(executor.map(predict_station, [(record, output) for record in records]), 1):
            if number % 20 == 0:
                print(split, number, "/", len(records), station, flush=True)
    write_json(output / "execution-complete.json", {"stations": len(records), "methods": METHODS, "requests": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("generate", "train", "predict"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    if args.stage == "generate":
        generate()
    elif args.stage == "train":
        train()
    else:
        predict(args.split)
