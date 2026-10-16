"""Check frozen head settings across five seeds without selecting another model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import train_landslide_paired_persistence as training
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import export
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import read_json
from score_landslide_claim_review import score_target

from gnss_sim.artifacts import sha
from gnss_sim.landslide_visual_invariance import numeric_input

SEEDS = (20261020, 20261021, 20261022, 20261023, 20261024)


def run(calibration, heads, analysis, out):
    verify_bindings(calibration, "calibration-freeze.json")
    verify_bindings(heads, "result-freeze.json")
    for name, expected in read_json(analysis / "manifest.json")["files"].items():
        if sha(analysis / name) != expected:
            raise ValueError("Original confirmation score changed")
    out.mkdir(exist_ok=False)
    selection = read_json(calibration / "selection.json")
    export(out / "protocol.json", {
        "registered_before_new_training_at": datetime.now(timezone.utc).isoformat(),
        "seeds": SEEDS, "arms": training.ARMS,
        "selection": selection, "retuning": False, "new_api_requests": 0,
        "scope": "Post-confirmation exploratory training-seed sensitivity on already-scored records; not a new independent test",
        "rule": "Retain all seeds; original width, threshold, training centers, encoder, losses and optimizer remain fixed. Do not replace the registered primary model with any seed.",
        "original_seed": "Retrain solely to check exact equality to original selected frozen weights and probabilities, not to recover old timing",
        "source_sha256": sha(Path(__file__)),
        "frozen_training_source_sha256": sha(Path(training.__file__)),
        "calibration_freeze_sha256": sha(calibration / "calibration-freeze.json"),
        "confirmation_freeze_sha256": sha(heads / "result-freeze.json"),
    })
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    (out / "models").mkdir()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    scaling = read_json(calibration / "numeric-scaling.json")
    originals, embeddings, targets = [], [], []
    for cid in read_json(calibration / "protocol.json")["split"]["train"]:
        with np.load(calibration / "original/train" / (cid + ".npz")) as data:
            sample = data["sample"]
            originals.append(data["numbers"][sample])
            targets.append(data["target"][sample])
        with np.load(calibration / "encoded/train" / (cid + ".npz")) as data:
            if not np.all(data["encoded_mask"][sample]):
                raise ValueError("Training center has no saved image encoding")
            embeddings.append(data["embeddings"][sample])
    original_numbers = numeric_input(np.concatenate(originals), scaling)
    original_embeddings = np.concatenate(embeddings)
    target = np.concatenate(targets)
    positives, negatives = [], []
    for row in read_json(calibration / "selected-training-pairs.json"):
        with np.load(calibration / "pairs" / (row["pair_id"] + ".npz")) as data:
            positives.append(data["positive_numbers"])
            negatives.append(data["negative_numbers"])
    positive_numbers = numeric_input(np.stack(positives), scaling)
    negative_numbers = numeric_input(np.stack(negatives), scaling)
    with np.load(calibration / "encoded/pairs.npz") as data:
        positive_embeddings = data["positive"]
        negative_embeddings = data["negative"]
    banks = {}
    for cid in read_json(heads / "protocol.json")["case_ids"]:
        with np.load(heads / (cid + "-inputs.npz")) as data:
            banks[cid] = {name: data[name] for name in (
                "embeddings", "numbers", "supported", "codes", "stage_support", "observed")}
            banks[cid]["normalized"] = numeric_input(data["numbers"], scaling)
    references = {name: read_json(analysis / name / "reference-labels.json")
                  for name in ("primary_process", "secondary_observation")}
    original_predictions = read_json(heads / "new-predictions.json")
    original_probabilities = {cid: read_json(heads / (cid + "-probabilities.json")) for cid in banks}
    summaries, costs = [], []
    for seed in SEEDS:
        # The frozen research runner exposes its one random seed as a module
        # constant. Change that process-local setting; no source file is edited.
        training.SEED = seed
        predictions, all_probabilities = {}, {}
        for name, arm in training.ARMS.items():
            chosen = selection[name]
            original = training.modality_inputs(original_embeddings, original_numbers, arm)
            positive = training.modality_inputs(positive_embeddings, positive_numbers, arm)
            negative = training.modality_inputs(negative_embeddings, negative_numbers, arm)
            model, history, seconds, count = training.train_head(
                original, target, positive, negative, arm, chosen["width"])
            if seed == SEEDS[0]:
                frozen = torch.load(calibration / "models" / f"{name}-width{chosen['width']}.pt", weights_only=True)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
            torch.save(model.state_dict(), out / "models" / f"{seed}-{name}.pt")
            export(out / "models" / f"{seed}-{name}-history.json", history)
            costs.append({"seed": seed, "method": name, "new_train_seconds": seconds, "training_centers": count})
            predictions[name], all_probabilities[name] = {}, {}
            for cid, bank in banks.items():
                probability = training.probabilities(model, training.modality_inputs(
                    bank["embeddings"], bank["normalized"], arm))
                if seed == SEEDS[0]:
                    np.testing.assert_array_equal(probability, original_probabilities[cid][name])
                predictions[name][cid] = training.calibrated_prediction(
                    probability, bank["supported"], bank["codes"], bank["stage_support"], chosen["threshold"]).to_dict()
                all_probabilities[name][cid] = probability.tolist()
            if seed == SEEDS[0] and predictions[name] != original_predictions[name]:
                raise ValueError("Original-seed fitted forecast differs")
        export(out / f"{seed}-predictions.json", predictions)
        export(out / f"{seed}-probabilities.json", all_probabilities)
        for target_name, labels in references.items():
            values = {name: {cid: cases[cid] for cid in labels} for name, cases in predictions.items()}
            observed = {cid: banks[cid]["observed"] for cid in labels}
            if target_name == "primary_process":
                weak = {cid: np.array([int(row["weak_signal"]) == 1 and int(row["activity_label"]) == 1
                                      for row in rows]) for cid, rows in labels.items()}
            else:
                # This supplement does not redefine the manually frozen weak
                # subset; secondary weak recall remains in the original report.
                weak = {cid: np.zeros(len(rows), dtype=bool) for cid, rows in labels.items()}
            results, rows, records = score_target(labels, observed, weak, values, {}, {})
            export(out / f"{seed}-{target_name}-results.json", results)
            write_csv(out / f"{seed}-{target_name}-per-record.csv", records)
            summaries.extend(dict(row, seed=seed, target=target_name) for row in rows)
        current = {row["method"]: row for row in summaries if row["seed"] == seed and row["target"] == "primary_process"}
        print(seed, "main", round(current["visual_numeric_pair"]["f1"], 6),
              "augmentation", round(current["visual_numeric_augmentation"]["f1"], 6), flush=True)
    export(out / "summary.json", summaries)
    write_csv(out / "summary.csv", summaries)
    export(out / "new-training-cost.json", {"rows": costs, "new_api_calls": 0,
        "scope": "Purposeful seed-robustness training costs; not reconstruction of original training costs"})
    export(out / "verification.json", {"original_seed_weights_probabilities_predictions": "exact",
        "models_replaced": False, "retuned": False, "new_encoder_passes": 0, "new_api_calls": 0})
    export(out / "result-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("calibration", "heads", "analysis", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.calibration.resolve(), arguments.heads.resolve(), arguments.analysis.resolve(), arguments.out.resolve())
