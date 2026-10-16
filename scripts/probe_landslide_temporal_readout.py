"""Freeze and calibrate date-aligned visual activity with masked dense supervision."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import BASE, ROOT, export
from probe_landslide_visual_invariance import TRAINING, calibration_labels
from probe_landslide_visual_residual import numerical_model
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import calibration_objective, save_json
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_temporal_readout import (
    column_numeric_input,
    column_supervision,
    temporal_activity_model,
)
from gnss_sim.landslide_visual_invariance import (
    chart_numeric_descriptors,
    fit_numeric_scaling,
    invariance_inputs,
)

SEED = 20261013
THRESHOLDS = [round(index / 20, 2) for index in range(1, 20)]
ARMS = {
    "aligned_dense_fusion": {"readout": "aligned", "numeric": True, "dense": True},
    "aligned_center_fusion": {"readout": "aligned", "numeric": True, "dense": False},
    "pooled_dense_fusion": {"readout": "pooled", "numeric": True, "dense": True},
    "aligned_dense_vision": {"readout": "aligned", "numeric": False, "dense": True},
    "numeric_dense": {"readout": "numeric", "numeric": True, "dense": True},
}


def temporal_probabilities(model, images, numbers):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(images), 128):
            value = model(torch.from_numpy(images[start : start + 128]),
                          torch.from_numpy(numbers[start : start + 128]))[:, 30]
            result.extend(torch.sigmoid(value).numpy().tolist())
    return np.array(result)


def train_temporal(images, numbers, target, mask, class_weight, arm, width):
    torch.manual_seed(SEED)
    model = temporal_activity_model(width, numbers.shape[2], arm["readout"], arm["numeric"])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weight), reduction="none")
    if not mask[:, 30].all():
        raise ValueError("Every selected training center must have its original supported label")
    auxiliary_mask = mask.copy()
    auxiliary_mask[:, 30] = False
    shuffle = np.random.default_rng(SEED)
    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(12):
        center_total = auxiliary_total = 0.
        order = shuffle.permutation(len(images))
        for start in range(0, len(order), 128):
            indices = order[start : start + 128]
            predicted = model(torch.from_numpy(images[indices]), torch.from_numpy(numbers[indices]))
            losses = criterion(predicted, torch.from_numpy(target[indices]))
            center = losses[:, 30].mean()
            valid = torch.from_numpy(auxiliary_mask[indices])
            if valid.any():
                auxiliary = losses[valid].mean()
            else:
                auxiliary = center
            loss = (center + auxiliary) / 2 if arm["dense"] else center
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            center_total += float(center.detach()) * len(indices)
            auxiliary_total += float(auxiliary.detach()) * len(indices)
        history.append({"epoch": epoch + 1, "center_balanced_bce": center_total / len(images),
                        "auxiliary_balanced_bce": auxiliary_total / len(images)})
    return model, history, time.perf_counter() - started


def prepare(out):
    prior = BASE / "visual-residual-development-v1"
    verify_bindings(prior, "freeze.json")
    verify_bindings(prior, "calibration-freeze.json")
    out.mkdir(parents=True, exist_ok=False)
    splits = {target: read_json(source / "protocol.json")["split"] for target, source in TRAINING.items()}
    export(out / "protocol.json", {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Supervised chart/numeric development; not a language model or field confirmation",
        "hypothesis": "Preserving original chart-column dates and supervised activity at those dates separates past-window motion from current motion more reliably than whole-window pooling",
        "arms": ARMS,
        "calibration_splits": {target: {key: split[key] for key in ("train", "validation")}
                               for target, split in splits.items()},
        "unchanged_input": "Existing raw chart6x32x61; three-day median columns centered -90..90 at stride3, original mask/noise normalization, no interpolation or chart edits; original18raw84 numerical features plus18 unavailable indicators",
        "alignment": "Columnj refers exactly to original day t+3(j-30). Two image CNN3x3 layers8/16 stride(2,1); retain61 time columns, pool height2; temporal1D CNN32->16 kernel9 dilation1,16->16 kernel9 dilation4; shared point head hidden16/32. Numbers belong to each column's own date; relative query position -1..1 exposed equally to all arms",
        "pooled_control": "Same image CNN, pool2x5, broadcast160 global visual features to all dated numeric/position inputs; same61 labels/loss. Distinguishes exact readout alignment from general dense supervision",
        "supervision": "Training patches use everysecond known/supported original center; center label unchanged. Auxiliary labels are actual original t+3k known/supported days in SAME training record; outside/unknown/unsupported masked, center excluded. Dense loss0.5centerBCE+0.5maskedauxBCE; center-only ablation onlycenterBCE. No synthetic label interpolation or cross-record neighbor gathering",
        "baseline_fairness": "Refit numeric XGB and384D chart-information XGB on ALL supported known training days, since dense arms see neighboring training labels. Same original train/validation, target-specific frozen stage head. Numerical dense NN has same label exposure as main. Do not reuse stride2 weights as strong controls",
        "training": {"seed": SEED, "epochs": 12, "batch": 128, "widths": [16, 32],
                     "optimizer": "Adam", "lr": .001, "weight_decay": .0001,
                     "class_weight": "One positive weight from all supported known original training dates, shared arms",
                     "threads": 2, "device": "cpu", "deterministic": True,
                     "hard_timeout_seconds": 1800},
        "selection": "DailyF1+eventF1; first maximal objective; same2widths or2XGBdepths x19thresholds=38 opportunities everyarm",
        "thresholds": THRESHOLDS,
        "xgboost": "Same XGB120trees,depth2/4,lr.05,seed20261007,threads2; no reflection, jump-view augmentation or teacher",
        "calendar_stage": "Original support84 and14-day persistence; missingunknown; existing target stage head only inside final activity",
        "calibration_gate": "All primary checks: main dailyF1 strictly above alignedcenter,pooleddense,numericdense; positive net correct vs numericdense;weakRecall>=numericdense;>=2 records net benefit vs numericdense. Strong-control usefulness: main F1>=max(rawXGB,chartXGB)-.02 AND either weakRecall>=max(XGBweak)+.03 OR eventF1>=max(XGBevent)+.03. This permits an explicitly reported small dailyF1 tradeoff for a preregistered scenario benefit, not universal dominance. Author scores not used to choose process model",
        "failure": "Stop fixed candidate if any gate fails. No seed/loss/kernel/readout/denseweight/threshold/subset rescue or best-ablation promotion",
        "next_if_pass": "Complete24process/12author frozen development and all old strong/VLM/fixed/Agent comparators, then fresh controlled confirmation with generator/reference frozen before observations. Calibration alone not a contribution confirmation",
        "old_decisions": "All prior stopped candidates remain stopped; this changes date-aligned recognition/supervision, not old residual bounds",
        "literature_basis": "VETime3.3 and relevant source read before this freeze; generic temporal visual alignment not claimed novel",
        "new_api_calls": 0, "new_records": 0,
    })
    sources = [Path(__file__), ROOT / "src/gnss_sim/landslide_temporal_readout.py",
               ROOT / "tests/test_landslide_temporal_readout.py",
               ROOT / "src/gnss_sim/landslide_visual_invariance.py",
               ROOT / "scripts/probe_landslide_visual_invariance.py",
               ROOT / "scripts/probe_landslide_visual_residual.py",
               ROOT / "src/gnss_sim/landslide_visual_residual.py",
               ROOT / "src/gnss_sim/landslide_adaptive_review.py",
               ROOT / "src/gnss_sim/landslide_baselines.py",
               ROOT / "src/gnss_sim/landslide_local_experiment.py",
               ROOT / "src/gnss_sim/landslide_evaluation.py",
               ROOT / "scripts/score_landslide_core_lockbox.py",
               ROOT / "scripts/run_landslide_comparison.py",
               ROOT / "scripts/run_landslide_window_optimization.py"]
    (out / "source").mkdir()
    for path in sources:
        shutil.copyfile(path, out / "source" / path.name)
    bindings = list(sources)
    for target, source in TRAINING.items():
        bindings.extend([source / "protocol.json", source / "reference-labels.csv", source / "stage-xgboost.json"])
        for cid in splits[target]["train"] + splits[target]["validation"]:
            bindings.extend(path for path in (source / "inputs" / cid).iterdir() if path.is_file())
    literature = BASE / "temporal-readout-literature-v1"
    bindings.extend(path for path in literature.rglob("*") if path.is_file())
    export(out / "source-bindings.json", {str(path): sha(path) for path in bindings})
    export(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    export(out / "state.json", {"status": "prepared"})
    print("Frozen original-date supervision, temporal readout and fair strong controls", flush=True)


def verify(out):
    verify_bindings(out, "source-bindings.json")
    verify_bindings(out, "freeze.json")


def calibrate(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Only a never-started calibration may train")
    protocol = read_json(out / "protocol.json")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    save_json(out / "state.json", {"status": "training"})
    trials, selections, costs, correction_rows = [], {}, [], []
    for target, source in TRAINING.items():
        started = time.perf_counter()
        split = protocol["calibration_splits"][target]
        ids = split["train"] + split["validation"]
        pairs = {cid: load_case_features(source, cid) for cid in ids}
        labels = calibration_labels(source, pairs)
        inputs = {cid: invariance_inputs(*pairs[cid]) for cid in ids}
        actual = {cid: np.array([int(row["activity_label"]) for row in labels[cid]]) for cid in ids}
        all_eligible = {cid: inputs[cid][2] & (actual[cid] >= 0) for cid in split["train"]}
        matrix = np.concatenate([inputs[cid][1][all_eligible[cid]] for cid in split["train"]])
        train_y = np.concatenate([actual[cid][all_eligible[cid]] for cid in split["train"]])
        if train_y.sum() == 0 or train_y.sum() == len(train_y):
            raise ValueError("Training needs both original activity classes")
        class_weight = (len(train_y) - train_y.sum()) / train_y.sum()
        scaling = fit_numeric_scaling(matrix)
        folder = out / target
        export(folder / "numeric-scaling.json", scaling)
        numeric_bank = {cid: column_numeric_input(inputs[cid][1], scaling) for cid in ids}
        image_bank, number_bank, target_bank, mask_bank = [], [], [], []
        exposure = []
        for cid in split["train"]:
            targets, mask = column_supervision(actual[cid], inputs[cid][2])
            centers = all_eligible[cid] & (np.arange(len(actual[cid])) % 2 == 0)
            image_bank.append(inputs[cid][0][centers])
            number_bank.append(numeric_bank[cid][centers])
            target_bank.append(targets[centers])
            mask_bank.append(mask[centers])
            auxiliary = mask[centers].copy()
            auxiliary[:, 30] = False
            exposure.append({"case_id": cid, "all_eligible_original_days": int(all_eligible[cid].sum()),
                             "training_centers": int(centers.sum()), "auxiliary_label_occurrences": int(auxiliary.sum()),
                             "unknown_or_unsupported_supervision": 0})
        images = np.concatenate(image_bank)
        numbers = np.concatenate(number_bank)
        targets = np.concatenate(target_bank)
        masks = np.concatenate(mask_bank)
        del image_bank, number_bank, target_bank, mask_bank
        export(folder / "label-exposure.json", exposure)
        export(folder / "preprocessing-cost.json", {"wall_seconds": time.perf_counter() - started,
                                                   "all_original_training_days": len(train_y),
                                                   "neural_training_centers": len(images)})
        stage = XGBClassifier()
        stage.load_model(source / "stage-xgboost.json")
        codes = {cid: stage.predict(pairs[cid][1].matrix).astype(int) + 1 for cid in split["validation"]}
        weak = {cid: np.array([int(row.get("weak_signal", 0)) == 1 and int(row["activity_label"]) == 1
                               for row in labels[cid]]) for cid in split["validation"]}
        validation_predictions, validation_probabilities = {}, {}

        def select(name, candidates):
            arm_trials, outputs, probabilities = [], {}, {}
            for configuration, probability in candidates:
                for threshold in THRESHOLDS:
                    predictions, scores = {}, []
                    weak_tp = weak_total = 0
                    for cid in split["validation"]:
                        activity = activity_from_mask(probability[cid] >= threshold, inputs[cid][2])
                        prediction = learned_stages(codes[cid], activity, pairs[cid][1].supported)
                        predictions[cid] = prediction.to_dict()
                        scores.append(evaluate_case(labels[cid], prediction))
                        weak_tp += int((weak[cid] & (activity == 1)).sum())
                        weak_total += int(weak[cid].sum())
                    score = aggregate_scores(scores)
                    row = {"target": target, "arm": name, **configuration, "threshold": threshold,
                           "scores": score, "objective": calibration_objective(score),
                           "weak_tp": weak_tp, "weak_total": weak_total,
                           "weak_recall": weak_tp / weak_total if weak_total else None}
                    arm_trials.append(row)
                    key = (tuple(configuration.items()), threshold)
                    outputs[key] = predictions
                    probabilities[key] = probability
            chosen = max(arm_trials, key=lambda row: row["objective"])
            config_key = tuple((key, chosen[key]) for key in candidates[0][0])
            selected_key = (config_key, chosen["threshold"])
            selections.setdefault(target, {})[name] = chosen
            validation_predictions[name] = outputs[selected_key]
            validation_probabilities[name] = {cid: value.tolist() for cid, value in probabilities[selected_key].items()}
            trials.extend(arm_trials)
            print(f"Selected {target} {name}: F1={chosen['scores']['daily']['f1']:.6f}", flush=True)

        for name, arm in ARMS.items():
            candidates = []
            for width in (16, 32):
                model, history, seconds = train_temporal(images, numbers, targets, masks, class_weight, arm, width)
                torch.save(model.state_dict(), folder / f"{name}-width{width}.pt")
                export(folder / f"{name}-width{width}-training.json", history)
                costs.append({"target": target, "arm": name, "configuration": width,
                              "training_center_or_original_days": len(images), "wall_seconds": seconds,
                              "parameters": sum(value.numel() for value in model.parameters())})
                candidates.append(({"width": width}, {cid: temporal_probabilities(model, inputs[cid][0], numeric_bank[cid])
                                                     for cid in split["validation"]}))
                print(f"Trained {target} {name} width{width}: {seconds:.2f}s", flush=True)
            select(name, candidates)
        descriptors = np.concatenate([chart_numeric_descriptors(inputs[cid][0][all_eligible[cid]])
                                      for cid in split["train"]])
        information = np.column_stack((matrix, descriptors))
        for name, observations in (("xgboost_raw84", matrix), ("xgboost_chart_information", information)):
            candidates = []
            for depth in (2, 4):
                started = time.perf_counter()
                model = numerical_model(depth)
                model.fit(observations, train_y)
                model.save_model(folder / f"{name}-depth{depth}.json")
                probability = {}
                for cid in split["validation"]:
                    validation = inputs[cid][1]
                    if name == "xgboost_chart_information":
                        validation = np.column_stack((validation, chart_numeric_descriptors(inputs[cid][0])))
                    probability[cid] = model.predict_proba(validation)[:, 1]
                candidates.append(({"depth": depth}, probability))
                costs.append({"target": target, "arm": name, "configuration": depth,
                              "training_center_or_original_days": len(train_y),
                              "wall_seconds": time.perf_counter() - started, "parameters": ""})
            select(name, candidates)
        export(folder / "validation-predictions.json", validation_predictions)
        export(folder / "validation-probabilities.json", validation_probabilities)
        for control in validation_predictions:
            if control == "aligned_dense_fusion":
                continue
            for cid in split["validation"]:
                known = actual[cid] >= 0
                before = np.array(validation_predictions[control][cid]["activity"])
                after = np.array(validation_predictions["aligned_dense_fusion"][cid]["activity"])
                gained = int((known & (before != actual[cid]) & (after == actual[cid])).sum())
                lost = int((known & (before == actual[cid]) & (after != actual[cid])).sum())
                correction_rows.append({"target": target, "control": control, "case_id": cid,
                                        "gained": gained, "lost": lost, "net": gained - lost})
        del images, numbers, targets, masks, matrix, descriptors, information
    export(out / "selection.json", selections)
    export(out / "validation-trials.json", trials)
    write_csv(out / "training-cost.csv", costs)
    write_csv(out / "per-record-corrections.csv", correction_rows)
    selected = selections["primary_process"]
    main = selected["aligned_dense_fusion"]
    checks = {"f1_strictly_above_" + name: main["scores"]["daily"]["f1"] > selected[name]["scores"]["daily"]["f1"]
              for name in ("aligned_center_fusion", "pooled_dense_fusion", "numeric_dense")}
    relevant = [row for row in correction_rows if row["target"] == "primary_process" and row["control"] == "numeric_dense"]
    checks["positive_net_vs_numeric_dense"] = sum(row["net"] for row in relevant) > 0
    checks["at_least_two_records_benefit"] = sum(row["net"] > 0 for row in relevant) >= 2
    checks["weak_recall_at_least_numeric_dense"] = main["weak_recall"] >= selected["numeric_dense"]["weak_recall"]
    strong = [selected[name] for name in ("xgboost_raw84", "xgboost_chart_information")]
    checks["daily_f1_within_two_points_strong_xgb"] = main["scores"]["daily"]["f1"] >= max(row["scores"]["daily"]["f1"] for row in strong) - .02
    scene_checks = {
        "weak_recall_gain_three_points": main["weak_recall"] >= max(row["weak_recall"] for row in strong) + .03,
        "event_f1_gain_three_points": main["scores"]["events"]["f1"] >= max(row["scores"]["events"]["f1"] for row in strong) + .03,
    }
    checks["predeclared_strong_control_scene_benefit"] = any(scene_checks.values())
    export(out / "calibration-decision.json", {
        "decision": "CONTINUE_TEMPORAL_READOUT_DEVELOPMENT" if all(checks.values()) else "STOP_TEMPORAL_READOUT_CALIBRATION",
        "checks": checks, "strong_control_scene_checks": scene_checks,
        "independent_validation": False, "new_api_calls": 0,
    })
    verify(out)
    export(out / "calibration-freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*")
                                             if path.is_file() and path.name != "state.json"})
    save_json(out / "state.json", {"status": "calibration_complete"})
    print(read_json(out / "calibration-decision.json")["decision"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "calibrate"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out.resolve())
    else:
        calibrate(args.out.resolve())
