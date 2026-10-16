"""Test additive-nuisance invariance with frozen visual features and strong controls."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_comparison import calibration_objective, save_json
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import read_json
from score_landslide_claim_review import score_target
from train_landslide_deformation_quality import THRESHOLDS
from train_landslide_paired_persistence import probabilities
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_nuisance_invariance import (
    nuisance_consistency_loss,
    shuffled_within_activity,
)
from gnss_sim.landslide_paired_persistence import pretrained_activity_head
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling, numeric_input

SEED = 20261015
ARMS = {
    "npi_visual_numeric": dict(visual=True, numeric=True, augmentation=True, consistency=True, shuffle=False),
    "npi_augmentation": dict(visual=True, numeric=True, augmentation=True, consistency=False, shuffle=False),
    "npi_plain": dict(visual=True, numeric=True, augmentation=False, consistency=False, shuffle=False),
    "npi_shuffled": dict(visual=True, numeric=True, augmentation=True, consistency=True, shuffle=True),
    "npi_visual": dict(visual=True, numeric=False, augmentation=True, consistency=True, shuffle=False),
    "npi_visual_augmentation": dict(visual=True, numeric=False, augmentation=True, consistency=False, shuffle=False),
    "npi_numeric": dict(visual=False, numeric=True, augmentation=True, consistency=True, shuffle=False),
    "npi_numeric_augmentation": dict(visual=False, numeric=True, augmentation=True, consistency=False, shuffle=False),
    "npi_numeric_plain": dict(visual=False, numeric=True, augmentation=False, consistency=False, shuffle=False),
}


def inputs(bank, arm, scaling):
    parts = []
    if arm["visual"]:
        parts.append(bank["embeddings"])
    if arm["numeric"]:
        parts.append(numeric_input(bank["numbers"], scaling))
    return np.column_stack(parts).astype(np.float32)


def fit(original, counterpart, actual, width, arm):
    torch.manual_seed(SEED)
    model = pretrained_activity_head(original.shape[1], 0, width)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(actual) - actual.sum()) / actual.sum()))
    values = torch.from_numpy(original)
    changed = torch.from_numpy(counterpart)
    target = torch.from_numpy(actual.astype(np.float32))
    paired = shuffled_within_activity(actual, SEED + 1) if arm["shuffle"] else np.arange(len(actual))
    generator = np.random.default_rng(SEED)
    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(12):
        supervised_total = consistency_total = 0.
        order = generator.permutation(len(actual))
        for first in range(0, len(order), 128):
            index = order[first:first + 128]
            logits = model(values[index]).squeeze(1)
            primary = criterion(logits, target[index])
            secondary = torch.tensor(0.)
            if arm["augmentation"]:
                altered_logits = model(changed[index]).squeeze(1)
                primary = (primary + criterion(altered_logits, target[index])) / 2
                if arm["consistency"]:
                    comparison = model(changed[paired[index]]).squeeze(1) if arm["shuffle"] else altered_logits
                    secondary = nuisance_consistency_loss(logits, comparison)
            optimizer.zero_grad()
            (primary + secondary).backward()
            optimizer.step()
            supervised_total += float(primary.detach()) * len(index)
            consistency_total += float(secondary.detach()) * len(index)
        history.append(dict(epoch=epoch + 1, classification=supervised_total / len(actual),
                            consistency=consistency_total / len(actual)))
    return model, history, time.perf_counter() - started


def train(prepared, out):
    verify_manifest(prepared, "freeze.json")
    protocol = read_json(prepared / "protocol.json")
    for path, expected in protocol["source_bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Frozen preparation source changed: " + path)
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    write_json(out / "protocol.json", dict(prepared=str(prepared), prepared_manifest_sha256=sha(prepared / "freeze.json"),
                                          arms=ARMS, **protocol))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "pre-fit-freeze.json", {path.name: sha(path) for path in out.iterdir() if path.is_file()})
    save_json(out / "state.json", dict(status="training", completed_models=0, reserved_content_loaded=False))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    split = protocol["split"]
    banks = load_banks(Path(protocol["encoded"]), split)
    for cid, bank in banks.items():
        with np.load(Path(protocol["parent"]) / "evidence" / (cid + ".npz")) as mass:
            bank["numbers"] = np.column_stack((bank["numbers"], mass["query_distribution"]))
    changed = load_banks(prepared, split)
    ids = split["train"]
    matrix = np.concatenate([banks[cid]["numbers"][banks[cid]["selected"]] for cid in ids])
    scaling = fit_numeric_scaling(matrix)
    write_json(out / "scaling.json", scaling)
    actual = np.concatenate([banks[cid]["activity"][banks[cid]["selected"]] for cid in ids])
    labels = {cid: [] for cid in split["validation"]}
    with (Path(protocol["data"]) / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(Path(protocol["data"]).parent / "quality-calibration-v1/stage-xgboost.json")
    codes = {cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1 for cid in labels}
    stress_codes = {cid: stage.predict(changed[cid]["stage_matrix"]).astype(int) + 1 for cid in labels}
    write_json(out / "stage-codes.json", dict(clean={cid: value.tolist() for cid, value in codes.items()},
                                             stress={cid: value.tolist() for cid, value in stress_codes.items()}))
    trials, selection, clean_predictions, stress_predictions, clean_probabilities, stress_probabilities, costs = [], {}, {}, {}, {}, {}, []
    for name, arm in ARMS.items():
        originals = np.concatenate([inputs(banks[cid], arm, scaling)[banks[cid]["selected"]] for cid in ids])
        counterparts = np.concatenate([inputs(changed[cid], arm, scaling)[banks[cid]["selected"]] for cid in ids])
        candidates = {}
        for width in (16, 32):
            model, history, seconds = fit(originals, counterparts, actual, width, arm)
            torch.save(model.state_dict(), out / "models" / f"{name}-width{width}.pt")
            write_json(out / "models" / f"{name}-width{width}.loss.json", history)
            clean = {cid: probabilities(model, inputs(banks[cid], arm, scaling)) for cid in labels}
            stress = {cid: probabilities(model, inputs(changed[cid], arm, scaling)) for cid in labels}
            candidates[width] = clean, stress
            for threshold in THRESHOLDS:
                decoded = {cid: decode(clean[cid], threshold, banks[cid], codes[cid]) for cid in labels}
                score = aggregate_scores([evaluate_case(labels[cid], decoded[cid]) for cid in labels])
                trials.append(dict(method=name, parameter=width, threshold=threshold,
                                   objective=calibration_objective(score), validation=score))
            costs.append(dict(method=name, width=width, seconds=seconds, original_samples=len(actual),
                              augmented_samples=len(actual) if arm["augmentation"] else 0, dimensions=originals.shape[1]))
            save_json(out / "state.json", dict(status="training", completed_models=len(costs), last_model=name,
                                              reserved_content_loaded=False))
            print(costs[-1], flush=True)
        choice = max((row for row in trials if row["method"] == name), key=lambda row: row["objective"])
        selection[name] = choice
        clean, stress = candidates[choice["parameter"]]
        clean_probabilities[name] = {cid: value.tolist() for cid, value in clean.items()}
        stress_probabilities[name] = {cid: value.tolist() for cid, value in stress.items()}
        clean_predictions[name] = {cid: decode(clean[cid], choice["threshold"], banks[cid], codes[cid]).to_dict() for cid in labels}
        stress_predictions[name] = {cid: decode(stress[cid], choice["threshold"], changed[cid], stress_codes[cid]).to_dict() for cid in labels}

    # The augmented strong tree retains all original supported TRAIN dates, as
    # did its original query-mass control. Neural arms retain the same stride3.
    original_matrix = np.concatenate([banks[cid]["numbers"] for cid in ids])
    counterpart_matrix = np.concatenate([changed[cid]["numbers"] for cid in ids])
    original_actual = np.concatenate([banks[cid]["activity"] for cid in ids])
    eligible = np.concatenate([banks[cid]["supported"] for cid in ids]) & (original_actual >= 0)
    name = "npi_mass_augmented"
    candidates = {}
    for depth in (2, 4):
        started = time.perf_counter()
        model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=.05, random_state=20261007, n_jobs=2)
        model.fit(np.concatenate((original_matrix[eligible], counterpart_matrix[eligible])),
                  np.tile(original_actual[eligible], 2))
        model.save_model(out / "models" / f"{name}-depth{depth}.json")
        clean = {cid: model.predict_proba(banks[cid]["numbers"])[:, 1] for cid in labels}
        stress = {cid: model.predict_proba(changed[cid]["numbers"])[:, 1] for cid in labels}
        candidates[depth] = clean, stress
        for threshold in THRESHOLDS:
            decoded = {cid: decode(clean[cid], threshold, banks[cid], codes[cid]) for cid in labels}
            score = aggregate_scores([evaluate_case(labels[cid], decoded[cid]) for cid in labels])
            trials.append(dict(method=name, parameter=depth, threshold=threshold,
                               objective=calibration_objective(score), validation=score))
        costs.append(dict(method=name, depth=depth, seconds=time.perf_counter() - started,
                          original_samples=int(eligible.sum()), augmented_samples=int(eligible.sum()), dimensions=45))
    choice = max((row for row in trials if row["method"] == name), key=lambda row: row["objective"])
    selection[name] = choice
    clean, stress = candidates[choice["parameter"]]
    clean_probabilities[name] = {cid: value.tolist() for cid, value in clean.items()}
    stress_probabilities[name] = {cid: value.tolist() for cid, value in stress.items()}
    clean_predictions[name] = {cid: decode(clean[cid], choice["threshold"], banks[cid], codes[cid]).to_dict() for cid in labels}
    stress_predictions[name] = {cid: decode(stress[cid], choice["threshold"], changed[cid], stress_codes[cid]).to_dict() for cid in labels}
    parent = Path(protocol["parent"])
    parent_choice = read_json(parent / "selection.json")["query_motion_mass"]
    model = XGBClassifier()
    model.load_model(parent / f"query_motion_mass-depth{parent_choice['parameter']}.json")
    stress = {cid: model.predict_proba(changed[cid]["numbers"])[:, 1] for cid in labels}
    stress_probabilities["query_motion_mass"] = {cid: value.tolist() for cid, value in stress.items()}
    stress_predictions["query_motion_mass"] = {cid: decode(stress[cid], parent_choice["threshold"], changed[cid], stress_codes[cid]).to_dict() for cid in labels}
    clean_predictions.update(read_json(parent / "predictions.json"))
    for filename, value in (("selection", selection), ("trials", trials), ("clean-probabilities", clean_probabilities),
                            ("stress-probabilities", stress_probabilities), ("clean-predictions", clean_predictions),
                            ("stress-predictions", stress_predictions)):
        write_json(out / (filename + ".json"), value)
    write_csv(out / "training-costs.csv", costs)
    write_json(out / "prediction-freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*")
                                              if path.is_file() and path.name != "state.json"})
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in values]) for cid, values in labels.items()}
    clean_results, clean_summary, clean_per_case = score_target(labels, observed, weak, clean_predictions, {}, {})
    stress_results, stress_summary, stress_per_case = score_target(labels, observed, weak, stress_predictions, {}, {})
    for scope, results, summary, per_case in (("clean", clean_results, clean_summary, clean_per_case),
                                           ("stress", stress_results, stress_summary, stress_per_case)):
        write_json(out / (scope + "-results.json"), results)
        write_json(out / (scope + "-summary.json"), summary)
        write_csv(out / (scope + "-summary.csv"), summary)
        write_csv(out / (scope + "-per-case.csv"), per_case)
    clean_rows, stress_rows = ({row["method"]: row for row in summary} for summary in (clean_summary, stress_summary))
    main, aug, shuffle, numeric = "npi_visual_numeric", "npi_augmentation", "npi_shuffled", "npi_numeric"
    families = {row["case_id"]: row["truth"]["family"] for row in
                read_json(Path(protocol["data"]) / "training-generation-ledger.json")}
    mixed = [cid for cid in labels if families[cid] in {"deformation_step", "deformation_spike"}]
    def mixed_recall(name):
        return aggregate_scores([stress_results[name]["per_case"][cid] for cid in mixed])["daily"]["recall"]
    checks = dict(
        clean_nonloss=clean_rows[main]["f1"] >= clean_rows[aug]["f1"] - .01,
        stress_gain=stress_rows[main]["f1"] > stress_rows[aug]["f1"] and
        stress_rows[main]["fp"] + stress_rows[main]["fn"] < stress_rows[aug]["fp"] + stress_rows[aug]["fn"] and
        stress_rows[main]["fp"] <= stress_rows[aug]["fp"],
        correspondence=stress_rows[main]["f1"] > stress_rows[shuffle]["f1"],
        clean_event_nonloss=clean_rows[main]["event_f1_iou03"] >= clean_rows[aug]["event_f1_iou03"] - .02,
        visual_increment=stress_rows[main]["f1"] > stress_rows[numeric]["f1"] or (
            clean_rows[main]["f1"] > clean_rows[numeric]["f1"] and stress_rows[main]["f1"] >= stress_rows[numeric]["f1"] - .01),
        mixed_nonloss=mixed_recall(main) >= mixed_recall(aug) - .02,
    )
    decision = dict(status="NPI_CONFIRMATION_PENDING_REPLAY" if all(checks.values()) else "STOP_NUISANCE_INVARIANCE",
                    checks=checks, clean_methods=len(clean_predictions), stress_methods=len(stress_predictions),
                    clean_outputs=len(clean_predictions) * len(labels), stress_outputs=len(stress_predictions) * len(labels),
                    stress_reused_parent_population=True, independent_new_records=0, new_api_calls=0,
                    reserved_content_loaded=False, stable_contribution_established=False)
    write_json(out / "decision.json", decision)
    save_json(out / "state.json", dict(status="complete", decision=decision["status"], reserved_content_loaded=False))
    write_json(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    print(dict(decision=decision, clean_main=clean_rows[main], stress_main=stress_rows[main],
               stress_aug=stress_rows[aug], stress_numeric=stress_rows[numeric]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    train(arguments.prepared.resolve(), arguments.out.resolve())
