"""Freeze and calibrate a cross-fitted, numerically anchored visual correction."""

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
from probe_landslide_visual_invariance import TRAINING, calibration_labels, probabilities
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import calibration_objective, save_json
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual_invariance import (
    activity_model,
    chart_numeric_descriptors,
    fit_numeric_scaling,
    invariance_inputs,
    numeric_input,
)
from gnss_sim.landslide_visual_residual import anchor_logits, grouped_folds, residual_model

PRIOR = BASE / "visual-invariance-development-v1"
SEED = 20261012
THRESHOLDS = [round(value / 20, 2) for value in range(1, 20)]
ARMS = {
    "bounded_visual_oof": {"image": True, "bounded": True, "oof": True},
    "bounded_numeric_oof": {"image": False, "bounded": True, "oof": True},
    "unbounded_visual_oof": {"image": True, "bounded": False, "oof": True},
    "bounded_visual_in_sample": {"image": True, "bounded": True, "oof": False},
}
REUSED = {
    "chart_plain": (True, True),
    "numeric_mlp": (False, True),
    "chart_vision_only": (True, False),
}


def numerical_model(depth):
    return XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=.05,
                         n_jobs=2, random_state=20261007)


def residual_probabilities(model, images, numbers, logits):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(images), 256):
            value = model(torch.from_numpy(images[start : start + 256]),
                          torch.from_numpy(numbers[start : start + 256]),
                          torch.from_numpy(logits[start : start + 256]))
            result.extend(torch.sigmoid(value).numpy().tolist())
    return np.array(result)


def train_residual(images, numbers, actual, logits, arm, width):
    torch.manual_seed(SEED)
    model = residual_model(width, numbers.shape[1], arm["image"], arm["bounded"])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    positives = float(actual.sum())
    if positives == 0 or positives == len(actual):
        raise ValueError("Both motion and stationary training examples are required")
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(actual) - positives) / positives))
    shuffle = np.random.default_rng(SEED)
    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(12):
        total = 0.
        order = shuffle.permutation(len(actual))
        for start in range(0, len(order), 128):
            indices = order[start : start + 128]
            predicted = model(torch.from_numpy(images[indices]), torch.from_numpy(numbers[indices]),
                              torch.from_numpy(logits[indices]))
            loss = criterion(predicted, torch.from_numpy(actual[indices].astype(np.float32)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append({"epoch": epoch + 1, "balanced_bce": total / len(actual)})
    return model, history, time.perf_counter() - started


def prepare(out):
    verify_bindings(PRIOR, "freeze.json")
    verify_bindings(PRIOR, "calibration-freeze.json")
    out.mkdir(parents=True, exist_ok=False)
    splits = {target: read_json(source / "protocol.json")["split"]
              for target, source in TRAINING.items()}
    folds = {target: grouped_folds(split["train"], 6 if target == "primary_process" else 2)
             for target, split in splits.items()}
    protocol = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Exposed development calibration, supervised chart CNN; not an LLM/Agent or field confirmation",
        "hypothesis": "A fixed strong numerical anchor plus a bounded learned visual correction preserves useful numerical decisions and adds independent visual corrections",
        "arms": ARMS, "reused_controls": REUSED,
        "calibration_splits": {target: {key: split[key] for key in ("train", "validation")}
                               for target, split in splits.items()},
        "folds": folds,
        "anchor": "Fixed depth4 XGB120trees lr.05 seed20261007; each OOF record predicted only by other whole records; all eligible training then full depth4 anchor for validation. No validation choice of anchor depth for residual arms; depth2 and depth4 both available to standalone control",
        "inputs": "Unmodified existing raw84 18 features and support; identical eligible dates (known, supported, every second day) to old neural calibration. Six-channel 32x61 original raw chart, no reflection/jump editing; train-only numeric normalization plus18 unavailable flags. Anchor logit added to both residual heads",
        "formula": "logit(p)=clipped_anchor_logit+log(4)*tanh(r); fixed1e-6 probability clipping; zero final correction head; no trainable anchor",
        "training": {"seed": SEED, "epochs": 12, "batch": 128, "widths": [16, 32],
                     "optimizer": "Adam", "lr": .001, "weight_decay": .0001,
                     "loss": "Same class-balanced BCE in all new residual arms", "torch_threads": 2,
                     "device": "cpu", "deterministic": True, "hard_timeout_seconds": 1800},
        "thresholds": THRESHOLDS,
        "selection": "DailyF1+eventF1, first maximal objective; 2 widths or2 XGB depths x19 thresholds=38 opportunities each. Same expanded threshold grid also reselects old frozen plain/image-only/numericNN widths; original six-opportunity results remain untouched",
        "information_control": "Refit XGB depths2/4, same dates/settings, all18raw84+366 original chart column positions/availability. No reflection augmentation; thus distinct from prior chartXGB. Chart encodes no added observation channel",
        "stage": "Existing target-specific frozen stage XGB gated by final activity;14-day activity persistence, original observation/support/unknown retained; no derivative expansion",
        "gate": "Primary: main dailyF1>=anchor,nonvisual_residual,plain,unbounded,in_sample; main net correct>0 against both anchor and nonvisual_residual; weakRecall>=anchor; at least2 records net benefit vs anchor. All controls reported, no author-score selection of primary",
        "failure": "Stop fixed candidate if any gate fails; no seed/epochs/loss/bound/threshold/subset adjustment, no substitute best arm as contribution",
        "next_if_pass": "Complete existing24 process/12 author development with old numerical/PELT/VLM/fixed/Agent scores; then unseen independent controlled validation. Exposed calibration never constitutes confirmation",
        "new_api_calls": 0, "new_records": 0,
    }
    export(out / "protocol.json", protocol)
    sources = [Path(__file__), ROOT / "src/gnss_sim/landslide_visual_residual.py",
               ROOT / "tests/test_landslide_visual_residual.py",
               ROOT / "scripts/probe_landslide_visual_invariance.py",
               ROOT / "src/gnss_sim/landslide_visual_invariance.py",
               ROOT / "src/gnss_sim/landslide_baselines.py",
               ROOT / "src/gnss_sim/landslide_local_experiment.py",
               ROOT / "src/gnss_sim/landslide_evaluation.py",
               ROOT / "src/gnss_sim/landslide_adaptive_review.py",
               ROOT / "scripts/score_landslide_core_lockbox.py",
               ROOT / "scripts/run_landslide_comparison.py",
               ROOT / "scripts/run_landslide_window_optimization.py"]
    (out / "source").mkdir()
    for path in sources:
        shutil.copyfile(path, out / "source" / path.name)
    bindings = list(sources)
    bindings.extend(path for path in PRIOR.rglob("*") if path.is_file())
    for target, source in TRAINING.items():
        bindings.extend([source / "protocol.json", source / "reference-labels.csv", source / "stage-xgboost.json"])
        for cid in splits[target]["train"] + splits[target]["validation"]:
            bindings.extend(path for path in (source / "inputs" / cid).iterdir() if path.is_file())
    export(out / "source-bindings.json", {str(path): sha(path) for path in bindings})
    export(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    export(out / "state.json", {"status": "prepared"})
    print("Frozen record cross-fitting, residual arms and common threshold opportunities", flush=True)


def verify(out):
    verify_bindings(out, "source-bindings.json")
    verify_bindings(out, "freeze.json")


def calibrate(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Only a never-started calibration may run")
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
        weak = {cid: np.array([int(row.get("weak_signal", 0)) == 1 and int(row["activity_label"]) == 1
                               for row in labels[cid]]) for cid in split["validation"]}
        eligible = {cid: inputs[cid][2] & (actual[cid] >= 0) & (np.arange(len(actual[cid])) % 2 == 0)
                    for cid in split["train"]}
        images = np.concatenate([inputs[cid][0][eligible[cid]] for cid in split["train"]])
        matrix = np.concatenate([inputs[cid][1][eligible[cid]] for cid in split["train"]])
        train_y = np.concatenate([actual[cid][eligible[cid]] for cid in split["train"]])
        record = np.concatenate([np.full(eligible[cid].sum(), cid) for cid in split["train"]])
        scaling = fit_numeric_scaling(matrix)
        if scaling != read_json(PRIOR / target / "numeric-scaling.json"):
            raise ValueError("Reused models must see the original training normalization")
        numbers = numeric_input(matrix, scaling)
        folder = out / target
        export(folder / "numeric-scaling.json", scaling)
        export(folder / "preprocessing-cost.json", {"wall_seconds": time.perf_counter() - started})
        stage = XGBClassifier()
        stage.load_model(source / "stage-xgboost.json")
        codes = {cid: stage.predict(pairs[cid][1].matrix).astype(int) + 1 for cid in split["validation"]}
        validation_numbers = {cid: numeric_input(inputs[cid][1], scaling) for cid in split["validation"]}
        validation_predictions, validation_probabilities = {}, {}

        def select(name, candidates):
            arm_trials, outputs, probability_bank = [], {}, {}
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
                           "objective": calibration_objective(score), "scores": score,
                           "weak_tp": weak_tp, "weak_total": weak_total,
                           "weak_recall": weak_tp / weak_total if weak_total else None}
                    arm_trials.append(row)
                    key = (tuple(configuration.items()), threshold)
                    outputs[key] = predictions
                    probability_bank[key] = probability
            chosen = max(arm_trials, key=lambda row: row["objective"])
            key = (tuple((key, chosen[key]) for key in candidates[0][0]), chosen["threshold"])
            selections.setdefault(target, {})[name] = chosen
            validation_predictions[name] = outputs[key]
            validation_probabilities[name] = {cid: value.tolist() for cid, value in probability_bank[key].items()}
            trials.extend(arm_trials)
            print(f"Selected {target} {name}: F1={chosen['scores']['daily']['f1']:.6f}", flush=True)

        numerical_candidates = []
        for depth in (2, 4):
            started = time.perf_counter()
            model = numerical_model(depth)
            model.fit(matrix, train_y)
            model.save_model(folder / f"anchor-depth{depth}.json")
            probability = {cid: model.predict_proba(inputs[cid][1])[:, 1] for cid in split["validation"]}
            numerical_candidates.append(({"depth": depth}, probability))
            costs.append({"target": target, "arm": "numeric_anchor", "configuration": depth,
                          "training_days": len(train_y), "wall_seconds": time.perf_counter() - started})
            if depth == 4:
                full_anchor = model
        select("numeric_anchor", numerical_candidates)
        full_logits = anchor_logits(full_anchor.predict_proba(matrix)[:, 1])
        oof_probability = np.full(len(train_y), np.nan)
        fold_audit = []
        for fold_index, held_out in enumerate(protocol["folds"][target]):
            test_mask = np.isin(record, held_out)
            train_mask = ~test_mask
            started = time.perf_counter()
            model = numerical_model(4)
            model.fit(matrix[train_mask], train_y[train_mask])
            model.save_model(folder / f"anchor-oof-fold{fold_index}.json")
            oof_probability[test_mask] = model.predict_proba(matrix[test_mask])[:, 1]
            fold_audit.append({"fold": fold_index, "held_out_records": held_out,
                               "fit_records": sorted(set(record[train_mask].tolist())),
                               "training_days": int(train_mask.sum()), "held_out_days": int(test_mask.sum())})
            costs.append({"target": target, "arm": "anchor_oof", "configuration": fold_index,
                          "training_days": int(train_mask.sum()), "wall_seconds": time.perf_counter() - started})
        oof_logits = anchor_logits(oof_probability)
        export(folder / "cross-fitting-audit.json", fold_audit)
        np.savez_compressed(folder / "training-anchor-predictions.npz", record=record, labels=train_y,
                            oof_probability=oof_probability, full_probability=full_anchor.predict_proba(matrix)[:, 1])
        validation_anchor = {cid: anchor_logits(full_anchor.predict_proba(inputs[cid][1])[:, 1])
                             for cid in split["validation"]}
        for name, arm in ARMS.items():
            candidates = []
            logits = oof_logits if arm["oof"] else full_logits
            for width in (16, 32):
                model, history, seconds = train_residual(images, numbers, train_y, logits, arm, width)
                torch.save(model.state_dict(), folder / f"{name}-width{width}.pt")
                export(folder / f"{name}-width{width}-training.json", history)
                costs.append({"target": target, "arm": name, "configuration": width,
                              "training_days": len(train_y), "wall_seconds": seconds})
                probability = {cid: residual_probabilities(model, inputs[cid][0], validation_numbers[cid],
                                                           validation_anchor[cid]) for cid in split["validation"]}
                candidates.append(({"width": width}, probability))
            select(name, candidates)
        for name, (image, numeric) in REUSED.items():
            candidates = []
            for width in (16, 32):
                model = activity_model(width, 36, image, numeric)
                model.load_state_dict(torch.load(PRIOR / target / f"{name}-width{width}.pt", weights_only=True))
                probability = {cid: probabilities(model, inputs[cid][0], validation_numbers[cid])
                               for cid in split["validation"]}
                candidates.append(({"width": width}, probability))
            select(name, candidates)
        information = np.column_stack((matrix, chart_numeric_descriptors(images)))
        candidates = []
        for depth in (2, 4):
            started = time.perf_counter()
            model = numerical_model(depth)
            model.fit(information, train_y)
            model.save_model(folder / f"chart-information-depth{depth}.json")
            probability = {cid: model.predict_proba(np.column_stack((inputs[cid][1], chart_numeric_descriptors(inputs[cid][0]))))[:, 1]
                           for cid in split["validation"]}
            candidates.append(({"depth": depth}, probability))
            costs.append({"target": target, "arm": "xgboost_chart_information", "configuration": depth,
                          "training_days": len(train_y), "wall_seconds": time.perf_counter() - started})
        select("xgboost_chart_information", candidates)
        export(folder / "validation-predictions.json", validation_predictions)
        export(folder / "validation-probabilities.json", validation_probabilities)
        for control in validation_predictions:
            if control == "bounded_visual_oof":
                continue
            for cid in split["validation"]:
                known = actual[cid] >= 0
                before = np.array(validation_predictions[control][cid]["activity"])
                after = np.array(validation_predictions["bounded_visual_oof"][cid]["activity"])
                gained = int((known & (before != actual[cid]) & (after == actual[cid])).sum())
                lost = int((known & (before == actual[cid]) & (after != actual[cid])).sum())
                correction_rows.append({"target": target, "control": control, "case_id": cid,
                                        "gained": gained, "lost": lost, "net": gained - lost})
        del images, matrix, numbers, information
    export(out / "selection.json", selections)
    export(out / "validation-trials.json", trials)
    write_csv(out / "training-cost.csv", costs)
    write_csv(out / "per-record-corrections.csv", correction_rows)
    selected = selections["primary_process"]
    main = selected["bounded_visual_oof"]
    checks = {"f1_at_least_" + name: main["scores"]["daily"]["f1"] >= selected[name]["scores"]["daily"]["f1"]
              for name in ("numeric_anchor", "bounded_numeric_oof", "chart_plain",
                           "unbounded_visual_oof", "bounded_visual_in_sample")}
    for name in ("numeric_anchor", "bounded_numeric_oof"):
        rows = [row for row in correction_rows if row["target"] == "primary_process" and row["control"] == name]
        checks["positive_net_vs_" + name] = sum(row["net"] for row in rows) > 0
        if name == "numeric_anchor":
            checks["at_least_two_records_benefit"] = sum(row["net"] > 0 for row in rows) >= 2
    checks["weak_recall_at_least_anchor"] = main["weak_recall"] >= selected["numeric_anchor"]["weak_recall"]
    export(out / "calibration-decision.json", {
        "decision": "CONTINUE_VISUAL_RESIDUAL_DEVELOPMENT" if all(checks.values()) else "STOP_VISUAL_RESIDUAL_CALIBRATION",
        "checks": checks, "independent_validation": False, "new_api_calls": 0,
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
