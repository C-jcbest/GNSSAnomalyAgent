"""Calibrate the fixed persistence-pair candidate and same-data strong controls."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from encode_landslide_paired_persistence import verify_inputs
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_contrastive_exemplars import SOURCE, raw_values
from probe_landslide_endpoint_fit import ROOT, export
from probe_landslide_feature_discovery import expert_features
from probe_landslide_visual_invariance import calibration_labels
from probe_landslide_visual_residual import THRESHOLDS, numerical_model
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import calibration_objective, predict_baselines, save_json
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_paired_persistence import pair_order_loss, pretrained_activity_head
from gnss_sim.landslide_visual_invariance import (
    chart_numeric_descriptors,
    fit_numeric_scaling,
    numeric_input,
)

ARMS = {
    "visual_numeric_pair": {"image": True, "numeric": True, "augment": True, "pair": True, "shuffle": False},
    "visual_numeric_augmentation": {"image": True, "numeric": True, "augment": True, "pair": False, "shuffle": False},
    "visual_numeric_plain": {"image": True, "numeric": True, "augment": False, "pair": False, "shuffle": False},
    "visual_numeric_shuffled_pairs": {"image": True, "numeric": True, "augment": True, "pair": True, "shuffle": True},
    "visual_pair_no_numeric": {"image": True, "numeric": False, "augment": True, "pair": True, "shuffle": False},
    "numeric_pair": {"image": False, "numeric": True, "augment": True, "pair": True, "shuffle": False},
    "numeric_augmentation": {"image": False, "numeric": True, "augment": True, "pair": False, "shuffle": False},
}
SEED = 20261020


def modality_inputs(embeddings, numbers, arm):
    arrays = []
    if arm["image"]:
        arrays.append(embeddings)
    if arm["numeric"]:
        arrays.append(numbers)
    if not arrays or any(len(value) != len(embeddings) for value in arrays):
        raise ValueError("Each selected observation modality must share original centers")
    return np.column_stack(arrays).astype(np.float32)


def train_head(original, target, positive, negative, arm, width):
    """Keep original examples and augmentation identical across paired controls.

    The full small paired set contributes one mean auxiliary loss per ordinary
    supervised minibatch. Shuffling changes only the negative pair identities,
    never the supervised examples, labels, order, or class balancing.
    """
    torch.manual_seed(SEED)
    model = pretrained_activity_head(original.shape[1], 0, width)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    if arm["augment"]:
        inputs = np.concatenate((original, negative))
        actual = np.concatenate((target, np.zeros(len(negative))))
    else:
        inputs, actual = original, target
    if not 0 < actual.sum() < len(actual):
        raise ValueError("Both activity and stationary training centers are required")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(actual) - actual.sum()) / actual.sum()))
    paired_negative = negative
    if arm["shuffle"]:
        paired_negative = negative[np.random.default_rng(SEED + 1).permutation(len(negative))]
    left = torch.from_numpy(positive)
    right = torch.from_numpy(paired_negative)
    values = torch.from_numpy(inputs)
    labels = torch.from_numpy(actual.astype(np.float32))
    shuffle = np.random.default_rng(SEED)
    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(12):
        supervised_total = paired_total = 0.
        order = shuffle.permutation(len(actual))
        for first in range(0, len(order), 128):
            indices = order[first:first + 128]
            logits = model(values[indices]).squeeze(1)
            supervised = criterion(logits, labels[indices])
            if arm["pair"]:
                paired = pair_order_loss(model(left).squeeze(1), model(right).squeeze(1))
            else:
                paired = torch.tensor(0.)
            optimizer.zero_grad()
            (supervised + paired).backward()
            optimizer.step()
            supervised_total += float(supervised.detach()) * len(indices)
            paired_total += float(paired.detach()) * len(indices)
        history.append({"epoch": epoch + 1, "balanced_bce": supervised_total / len(actual),
                        "pair_order_loss": paired_total / len(actual)})
    return model, history, time.perf_counter() - started, len(actual)


def probabilities(model, inputs):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for first in range(0, len(inputs), 256):
            logits = model(torch.from_numpy(inputs[first:first + 256])).squeeze(1)
            outputs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(outputs)


def calibrated_prediction(probability, support, stage_codes, stage_support, threshold):
    activity = activity_from_mask(probability >= threshold, support)
    return learned_stages(stage_codes, activity, stage_support)


def calibrate(run):
    verify_inputs(run)
    verify_bindings(run, "encoding-freeze.json")
    if read_json(run / "state.json")["status"] != "encoded":
        raise ValueError("Calibrate only never-trained encoded data")
    protocol = read_json(run / "protocol.json")
    if protocol["arms"] != list(ARMS) or protocol["training"]["seed"] != SEED:
        raise ValueError("Training implementation differs from prepared plan")
    sources = [Path(__file__), ROOT / "scripts/encode_landslide_paired_persistence.py",
               ROOT / "scripts/score_landslide_claim_review.py", ROOT / "scripts/probe_landslide_feature_discovery.py",
               ROOT / "scripts/probe_landslide_visual_residual.py", ROOT / "scripts/run_landslide_comparison.py",
               ROOT / "src/gnss_sim/landslide_evaluation.py", ROOT / "src/gnss_sim/landslide_local_experiment.py",
               ROOT / "src/gnss_sim/landslide_feature_discovery.py",
               ROOT / "tests/test_landslide_paired_persistence_learning.py", SOURCE / "stage-xgboost.json",
               SOURCE / "activity-xgboost.json", SOURCE / "frozen-selection.json"]
    for index, path in enumerate(sources):
        (run / "source" / f"learning-{index:02d}-{path.name}").write_bytes(path.read_bytes())
    export(run / "learning-bindings.json", {path.relative_to(ROOT).as_posix(): sha(path) for path in sources})
    export(run / "learning-plan.json", {
        "arms": ARMS, "thresholds": THRESHOLDS,
        "numeric_scaling": "Same original sampled training-only means/std/missing indicators in every neural arm",
        "embedding_scaling": "Pretrained L2-normalized embeddings; no fitted validation transform",
        "pair_batch": "Full paired mean once per supervised minibatch; fixed shuffle SEED+1 only in shuffled arm",
        "selection": "Stable first maximum of original daily F1+event F1; widths16,32/depths2,4 then thresholds ascending",
        "numerical_controls": ["xgb_raw18", "xgb_expert24", "xgb_chart384"],
        "independence": "Training descendants grouped with parent; no new independent data",
    })
    save_json(run / "state.json", {"status": "training"})
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    split = protocol["split"]
    case_features = {cid: load_case_features(SOURCE, cid) for cid in split["train"] + split["validation"]}
    labels = calibration_labels(SOURCE, case_features)
    pair_rows = read_json(run / "selected-training-pairs.json")
    if any(row["case_id"] not in split["train"] for row in pair_rows):
        raise ValueError("Paired supervision must belong to training parents only")
    banks = {}
    for group in ("train", "validation"):
        for cid in split[group]:
            with np.load(run / "original" / group / (cid + ".npz")) as data:
                bank = {key: data[key] for key in ("numbers", "supported", "sample", "target", "observed", "weak")}
                bank["chart"] = chart_numeric_descriptors(data["images"])
            with np.load(run / "encoded" / group / (cid + ".npz")) as data:
                bank["embeddings"] = data["embeddings"]
                required = bank["sample"] if group == "train" else bank["supported"]
                if not np.all(data["encoded_mask"][required]):
                    raise ValueError("A classified center has no frozen image encoding")
            case, features = case_features[cid]
            blocks = discovery_block_matrix(raw_values(case), features.noise)[0]
            bank["expert"] = expert_features(blocks)
            banks[cid] = bank
    positives, negatives, negative_expert, negative_chart = [], [], [], []
    for row in pair_rows:
        with np.load(run / "pairs" / (row["pair_id"] + ".npz")) as data:
            positives.append(data["positive_numbers"])
            negatives.append(data["negative_numbers"])
            blocks = discovery_block_matrix(data["negative_observations"], data["negative_noise"])[0]
            negative_expert.append(expert_features(blocks)[row["day"]])
            negative_chart.append(chart_numeric_descriptors(data["negative_image"][None])[0])
    negative_matrix = np.stack(negatives)
    original_matrix = np.concatenate([banks[cid]["numbers"][banks[cid]["sample"]] for cid in split["train"]])
    original_embeddings = np.concatenate([banks[cid]["embeddings"][banks[cid]["sample"]] for cid in split["train"]])
    target = np.concatenate([banks[cid]["target"][banks[cid]["sample"]] for cid in split["train"]])
    scaling = fit_numeric_scaling(original_matrix)
    export(run / "numeric-scaling.json", scaling)
    original_numbers = numeric_input(original_matrix, scaling)
    positive_numbers = numeric_input(np.stack(positives), scaling)
    negative_numbers = numeric_input(negative_matrix, scaling)
    with np.load(run / "encoded/pairs.npz") as data:
        positive_embeddings, negative_embeddings = data["positive"], data["negative"]
    stage_head = XGBClassifier()
    stage_head.load_model(SOURCE / "stage-xgboost.json")
    codes = {cid: stage_head.predict(case_features[cid][1].matrix).astype(int) + 1 for cid in split["validation"]}
    trials, selections, all_predictions, costs = [], {}, {}, []
    (run / "models").mkdir(exist_ok=False)
    (run / "probabilities").mkdir(exist_ok=False)

    def evaluate_probabilities(name, parameter, value, probability):
        rows, outputs = [], {}
        for threshold in THRESHOLDS:
            predictions = {cid: calibrated_prediction(probability[cid], banks[cid]["supported"], codes[cid],
                            case_features[cid][1].supported, threshold) for cid in split["validation"]}
            scores = aggregate_scores([evaluate_case(labels[cid], predictions[cid]) for cid in predictions])
            rows.append({"arm": name, parameter: value, "threshold": threshold,
                         "objective": calibration_objective(scores), "scores": scores})
            outputs[threshold] = {cid: prediction.to_dict() for cid, prediction in predictions.items()}
        export(run / "probabilities" / f"{name}-{parameter}{value}.json", {cid: values.tolist() for cid, values in probability.items()})
        return rows, outputs

    for name, arm in ARMS.items():
        original = modality_inputs(original_embeddings, original_numbers, arm)
        positive = modality_inputs(positive_embeddings, positive_numbers, arm)
        negative = modality_inputs(negative_embeddings, negative_numbers, arm)
        arm_trials, candidates = [], {}
        for width in (16, 32):
            model, history, seconds, train_count = train_head(original, target, positive, negative, arm, width)
            torch.save(model.state_dict(), run / "models" / f"{name}-width{width}.pt")
            export(run / "models" / f"{name}-width{width}-history.json", history)
            before = time.perf_counter()
            probability = {cid: probabilities(model, modality_inputs(banks[cid]["embeddings"],
                            numeric_input(banks[cid]["numbers"], scaling), arm)) for cid in split["validation"]}
            inference_seconds = time.perf_counter() - before
            rows, outputs = evaluate_probabilities(name, "width", width, probability)
            arm_trials.extend(rows)
            candidates[width] = outputs
            costs.append({"arm": name, "width": width, "training_centers": train_count,
                          "parameters": sum(parameter.numel() for parameter in model.parameters()),
                          "train_seconds": seconds, "validation_head_seconds": inference_seconds})
            print(name, width, round(seconds, 3), flush=True)
        chosen = max(arm_trials, key=lambda row: row["objective"])
        selections[name] = chosen
        all_predictions[name] = candidates[chosen["width"]][chosen["threshold"]]
        trials.extend(arm_trials)
    for name in ("xgb_raw18", "xgb_expert24", "xgb_chart384"):
        fields = ["numbers"]
        if name == "xgb_expert24":
            fields.append("expert")
            negative_input = np.column_stack((negative_matrix, negative_expert))
        elif name == "xgb_chart384":
            fields.append("chart")
            negative_input = np.column_stack((negative_matrix, negative_chart))
        else:
            negative_input = negative_matrix
        train_inputs, train_targets = [], []
        for cid in split["train"]:
            bank = banks[cid]
            eligible = bank["supported"] & (bank["target"] >= 0)
            train_inputs.append(np.column_stack([bank[field][eligible] for field in fields]))
            train_targets.append(bank["target"][eligible])
        matrix = np.concatenate(train_inputs + [negative_input])
        actual = np.concatenate(train_targets + [np.zeros(len(negative_input))])
        validation = {cid: np.column_stack([banks[cid][field] for field in fields]) for cid in split["validation"]}
        arm_trials, candidates = [], {}
        for depth in (2, 4):
            before = time.perf_counter()
            model = numerical_model(depth)
            model.fit(matrix, actual)
            train_seconds = time.perf_counter() - before
            model.save_model(run / "models" / f"{name}-depth{depth}.json")
            before = time.perf_counter()
            probability = {cid: model.predict_proba(value)[:, 1] for cid, value in validation.items()}
            inference_seconds = time.perf_counter() - before
            rows, outputs = evaluate_probabilities(name, "depth", depth, probability)
            arm_trials.extend(rows)
            candidates[depth] = outputs
            costs.append({"arm": name, "depth": depth, "training_centers": len(actual),
                          "train_seconds": train_seconds, "validation_head_seconds": inference_seconds})
        chosen = max(arm_trials, key=lambda row: row["objective"])
        selections[name] = chosen
        all_predictions[name] = candidates[chosen["depth"]][chosen["threshold"]]
        trials.extend(arm_trials)
        print(name, chosen["depth"], chosen["threshold"], flush=True)
    original_activity_head = XGBClassifier()
    original_activity_head.load_model(SOURCE / "activity-xgboost.json")
    original_baselines = predict_baselines({cid: value[1] for cid, value in case_features.items()},
        split["validation"], read_json(SOURCE / "frozen-selection.json"), original_activity_head, stage_head)
    for name in ("robust_rule", "pelt_rule", "xgboost"):
        all_predictions["original_" + name] = {cid: value.to_dict() for cid, value in original_baselines[name].items()}
    observed = {cid: banks[cid]["observed"] for cid in split["validation"]}
    weak = {cid: banks[cid]["weak"] for cid in split["validation"]}
    validation_labels = {cid: labels[cid] for cid in split["validation"]}
    results, summary, records = score_target(validation_labels, observed, weak, all_predictions, {}, {})
    corrections = []
    for name, cases in all_predictions.items():
        if name == "visual_numeric_pair":
            continue
        for cid, value in cases.items():
            actual = banks[cid]["target"]
            known = actual >= 0
            before = np.asarray(value["activity"])
            after = np.asarray(all_predictions["visual_numeric_pair"][cid]["activity"])
            gained = int((known & (before != actual) & (after == actual)).sum())
            lost = int((known & (before == actual) & (after != actual)).sum())
            corrections.append({"control": name, "case_id": cid, "gained": gained, "lost": lost, "net": gained - lost})
    lookup = {row["method"]: row for row in summary}
    main = lookup["visual_numeric_pair"]
    plain, augment, shuffled, numeric = [lookup[name] for name in (
        "visual_numeric_plain", "visual_numeric_augmentation", "visual_numeric_shuffled_pairs", "numeric_pair")]
    checks = {
        "f1_nondecrease_augmentation": main["f1"] >= augment["f1"],
        "f1_nondecrease_shuffled": main["f1"] >= shuffled["f1"],
        "f1_gain_plain_at_least_005": main["f1"] - plain["f1"] >= .005,
        "strict_pairing_gain": main["f1"] > augment["f1"] or main["event_f1_iou03"] > augment["event_f1_iou03"],
        "weak_nondecrease": main["weak_recall"] >= max(plain["weak_recall"], augment["weak_recall"]),
        "visual_value_vs_numeric": main["f1"] >= numeric["f1"] or (
            main["event_f1_iou03"] > numeric["event_f1_iou03"] and main["f1"] >= numeric["f1"] - .01),
    }
    for control in ("visual_numeric_plain", "visual_numeric_augmentation"):
        checks["net_correction_" + control] = sum(row["net"] for row in corrections if row["control"] == control) > 0
    decision = "PROCEED_PAIRED_PERSISTENCE_DEVELOPMENT" if all(checks.values()) else "STOP_PAIRED_PERSISTENCE_CALIBRATION"
    export(run / "validation-trials.json", trials)
    export(run / "selection.json", selections)
    export(run / "predictions.json", all_predictions)
    export(run / "validation-labels.json", validation_labels)
    export(run / "results.json", results)
    export(run / "summary.json", summary)
    export(run / "corrections.json", corrections)
    write_csv(run / "summary.csv", summary)
    write_csv(run / "per-case.csv", records)
    export(run / "training-cost.json", {"rows": costs, "wall_seconds": time.perf_counter() - started,
                                         "shared_encoder_cost": "encoding-cost.json; shared once, charge to visual deployment",
                                         "new_api_calls": 0, "numeric_features_cost": "preparation and common source features"})
    export(run / "calibration-decision.json", {"decision": decision, "checks": checks,
        "independent_confirmation": False, "passed": sum(checks.values()), "total": len(checks)})
    verify_inputs(run)
    verify_bindings(run, "encoding-freeze.json")
    for name, expected in read_json(run / "learning-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Learning source changed during experiment")
    save_json(run / "state.json", {"status": "calibration_complete", "decision": decision})
    export(run / "calibration-freeze.json", {path.relative_to(run).as_posix(): sha(path)
        for path in run.rglob("*") if path.is_file() and path.name not in {"calibration-freeze.json", "state.json"}})
    print(decision, sum(checks.values()), "/", len(checks), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    calibrate(parser.parse_args().run.resolve())
