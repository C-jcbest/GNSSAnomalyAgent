"""Freeze and calibrate a bounded alternative-view activity classifier."""

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
from probe_landslide_visual_invariance import (
    SEED,
    TRAINING,
    calibration_labels,
    probabilities,
)
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import calibration_objective, save_json
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_jump_view import jump_view_inputs
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual_invariance import (
    activity_model,
    chart_numeric_descriptors,
    fit_numeric_scaling,
    invariance_inputs,
    numeric_input,
)

PRIOR = BASE / "visual-invariance-development-v1"
ARMS = {
    "adaptive_jump_pair": {"fixed": False, "numeric": True, "consistency": 1.0},
    "fixed_jump_pair": {"fixed": True, "numeric": True, "consistency": 1.0},
    "adaptive_no_consistency": {"fixed": False, "numeric": True, "consistency": 0.0},
    "adaptive_vision_only": {"fixed": False, "numeric": False, "consistency": 1.0},
}


def inputs(case, features):
    raw, adaptive, fixed, evidence = jump_view_inputs(case, features.noise)
    original, matrix, support = invariance_inputs(case, features)
    if not np.array_equal(raw, original):
        raise ValueError("Raw observation geometry changed")
    return raw, adaptive, fixed, evidence, matrix, support


def numerical_chart_information(raw, adaptive, evidence, matrix):
    """The numerical control gets both charts and all view-selection evidence."""
    metadata = np.column_stack(
        (evidence["dominant"], evidence["largest_noise_units"], evidence["second_noise_units"])
    )
    return np.column_stack(
        (matrix, chart_numeric_descriptors(raw), chart_numeric_descriptors(adaptive), metadata)
    )


def train_pair(raw, alternative, numbers, target, arm, width):
    torch.manual_seed(SEED)
    model = activity_model(width, numbers.shape[1], True, arm["numeric"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    positive = float(target.sum())
    if positive == 0 or positive == len(target):
        raise ValueError("Training requires positive and negative activity")
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(target) - positive) / positive)
    )
    rng = np.random.default_rng(SEED)
    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(12):
        supervised_sum = consistency_sum = 0.0
        order = rng.permutation(len(target))
        for start in range(0, len(order), 128):
            indices = order[start : start + 128]
            numeric = torch.from_numpy(numbers[indices])
            actual = torch.from_numpy(target[indices].astype(np.float32))
            left = model(torch.from_numpy(raw[indices]), numeric)
            right = model(torch.from_numpy(alternative[indices]), numeric)
            supervised = (criterion(left, actual) + criterion(right, actual)) / 2
            consistency = torch.mean((torch.sigmoid(left) - torch.sigmoid(right)) ** 2)
            loss = supervised + arm["consistency"] * consistency
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            supervised_sum += float(supervised.detach()) * len(indices)
            consistency_sum += float(consistency.detach()) * len(indices)
        history.append({"epoch": epoch + 1, "supervised_loss": supervised_sum / len(target),
                        "paired_probability_loss": consistency_sum / len(target)})
    return model, history, time.perf_counter() - started


def paired_probabilities(model, raw, alternative, numbers, requested):
    """Only requested days acquire a second observation; no recurrent loop."""
    left = probabilities(model, raw, numbers)
    right = left.copy()
    if np.any(requested):
        right[requested] = probabilities(model, alternative[requested], numbers[requested])
    return (left + right) / 2, left, right


def prepare(out):
    verify_bindings(PRIOR, "calibration-freeze.json")
    out.mkdir(parents=True, exist_ok=False)
    source_paths = [
        Path(__file__), ROOT / "src/gnss_sim/landslide_jump_view.py",
        ROOT / "tests/test_landslide_jump_view.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "scripts/probe_landslide_visual_invariance.py",
        ROOT / "scripts/probe_landslide_endpoint_fit.py",
        ROOT / "scripts/score_landslide_core_lockbox.py",
        ROOT / "scripts/run_landslide_window_optimization.py",
        ROOT / "scripts/run_landslide_comparison.py",
        ROOT / "src/gnss_sim/landslide_baselines.py",
        ROOT / "src/gnss_sim/landslide_evaluation.py",
        ROOT / "src/gnss_sim/landslide_local_experiment.py",
    ]
    (out / "source").mkdir()
    for path in source_paths:
        shutil.copyfile(path, out / "source" / path.name)
    split = read_json(PRIOR / "protocol.json")["calibration_splits"]
    export(out / "protocol.json", {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Supervised multimodal visual activity; no LLM, no independent field confirmation",
        "prior_candidate": "STOP_VISUAL_INVARIANCE_CALIBRATION remains stopped",
        "change": "A dominant adjacent raw-block jump requests one jump-subtracted chart; train matched raw/alternative activity",
        "arms": ARMS, "calibration_splits": split,
        "jump_rule": "Original61 disjoint3day medians>=2 completeNEU;largest norm(delta/noise)>=6 and >2*second;no difference across an unavailable column;subtract exactlyone largest adjacent observed increment from all later available columns;keep raw",
        "interpretation": "Alternative observation only; no instrument attribution, raw correction, added label or interpolation. Large real motion can also trigger",
        "inference": "Mean raw/alternative probability only on supported dominant-jump dates;at most2 network passes per date. Fixed bypasses dominance and uses one alternative on every supported date",
        "training": {"seed": SEED, "epochs": 12, "batch": 128, "threads": 2,
                     "optimizer": "Adam lr.001 wd.0001", "stride": 2, "positive_weight": "Train-only",
                     "loss": "Mean positive-balanced BCE(raw,alternative)+lambda probabilityMSE;lambda1 main/fixed/noNumeric,lambda0 noConsistency;no coordinate reflection"},
        "selection": "Same2 widths16/32 x3 thresholds.35/.5/.65 per arm;dailyF1+eventF1;no early stopping or new seed/lambda/jump-threshold/raster search",
        "controls": "Reuse frozen raw chart_plain/numeric_mlp from exactlysame train/validation,seed,epochs,normalization and6 selection chances;never substitute process fitted model for author",
        "strong_numeric": "New XGB120trees depth2/4 lr.05 seed20261007,6 chances,same eligibletraining dates;18 originalraw84+2*(183 rasterpositions+183availability)+3 exact selectionfacts=753 features",
        "activity_stage": "Original support84 and14day persistence;missing unknown;target-specific originalstage head;no native visualstage claim",
        "calibration_gate": "Processmain F1>=rawplain andnumericMLP;weakRecall>=numericMLP;instrumentstepFP<rawplain;F1>=noConsistency;mean paired probabilitydisagreement at same dominant supported dates strictly lower than noConsistency;F1>=fixed and fewer secondviews. Report author and same-informationXGB completely but no universal superiority requirement",
        "development_gate": "Only after calibrationpasses: all24/12 complete development predictionsfreeze before scoring;process mainF1>rawplain,netcorrect>0,weakRecall>=numericMLP,stepFP<rawplain;mainF1>=noConsistency andfixed,fewer secondviews thanfixed;authorfully report;prior108/96all results/costretain",
        "failure": "STOP_JUMP_VIEW_CALIBRATION or STOP_JUMP_VIEW_DEVELOPMENT at gate failure;no parameter,seed,subset,arm,thresholdgrid or stopping-rule rescue",
        "new_api_calls": 0,
    })
    bound = list(source_paths)
    bound.extend(path for path in PRIOR.rglob("*") if path.is_file())
    for target, source in TRAINING.items():
        bound.extend([source / "protocol.json", source / "reference-labels.csv", source / "stage-xgboost.json"])
        for cid in split[target]["train"] + split[target]["validation"]:
            bound.extend(path for path in (source / "inputs" / cid).iterdir() if path.is_file())
    export(out / "source-bindings.json", {str(path): sha(path) for path in bound})
    export(out / "freeze.json", {str(path.relative_to(out)): sha(path)
                               for path in out.rglob("*") if path.is_file()})
    export(out / "state.json", {"status": "prepared"})
    print("Prepared bounded jump-view calibration", flush=True)


def verify(out):
    verify_bindings(out, "source-bindings.json")
    verify_bindings(out, "freeze.json")


def calibrate(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Only a never-started candidate may train")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    save_json(out / "state.json", {"status": "training"})
    protocol = read_json(out / "protocol.json")
    trials, chosen_by_target, costs, selected_predictions = [], {}, [], {}
    preprocessing_costs = []
    all_views, selected_disagreements = {}, {}
    for target, source in TRAINING.items():
        preprocessing_started = time.perf_counter()
        split = protocol["calibration_splits"][target]
        pairs = {cid: load_case_features(source, cid) for cid in split["train"] + split["validation"]}
        labels = calibration_labels(source, pairs)
        train_raw, train_adaptive, train_fixed, train_matrix, train_y, train_information = [], [], [], [], [], []
        for cid in split["train"]:
            raw, adaptive, fixed, evidence, matrix, support = inputs(*pairs[cid])
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            eligible = support & (actual >= 0) & (np.arange(len(actual)) % 2 == 0)
            train_raw.append(raw[eligible])
            train_adaptive.append(adaptive[eligible])
            train_fixed.append(fixed[eligible])
            train_matrix.append(matrix[eligible])
            train_y.append(actual[eligible])
            information = numerical_chart_information(raw, adaptive, evidence, matrix)
            train_information.append(information[eligible])
        raw_training = np.concatenate(train_raw)
        adaptive_training = np.concatenate(train_adaptive)
        fixed_training = np.concatenate(train_fixed)
        matrix_training = np.concatenate(train_matrix)
        actual_training = np.concatenate(train_y)
        information_training = np.concatenate(train_information)
        del train_raw, train_adaptive, train_fixed, train_matrix, train_y, train_information
        scaling = fit_numeric_scaling(matrix_training)
        if scaling != read_json(PRIOR / target / "numeric-scaling.json"):
            raise ValueError("Training dates or normalization differ from the frozen raw controls")
        numbers_training = numeric_input(matrix_training, scaling)
        folder = out / target
        export(folder / "numeric-scaling.json", scaling)
        validation = {cid: inputs(*pairs[cid]) for cid in split["validation"]}
        preprocessing_costs.append({"target": target, "training_records": len(split["train"]),
                                    "validation_records": len(split["validation"]),
                                    "load_features_charts_and_scaling_wall_seconds": time.perf_counter() - preprocessing_started})
        stage = XGBClassifier()
        stage.load_model(source / "stage-xgboost.json")
        codes = {cid: stage.predict(pairs[cid][1].matrix).astype(int) + 1 for cid in validation}
        target_predictions, target_choice, target_differences = {}, {}, {}
        for control in ("chart_plain", "numeric_mlp"):
            target_predictions[control] = read_json(PRIOR / target / f"{control}-validation-predictions.json")
            target_choice[control] = read_json(PRIOR / "selection.json")[target][control]
        numerical_trials = []
        for depth in (2, 4):
            started = time.perf_counter()
            model = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=0.05,
                                  n_jobs=2, random_state=20261007)
            model.fit(information_training, actual_training)
            model.save_model(folder / f"xgboost-paired-depth{depth}.json")
            costs.append({"target": target, "arm": "xgboost_paired_information", "width": depth,
                          "training_days": len(actual_training), "training_wall_seconds": time.perf_counter() - started})
            validation_started = time.perf_counter()
            probability = {cid: model.predict_proba(numerical_chart_information(v[0], v[1], v[3], v[4]))[:, 1]
                           for cid, v in validation.items()}
            costs[-1]["validation_probability_wall_seconds"] = time.perf_counter() - validation_started
            for threshold in (0.35, 0.5, 0.65):
                predictions, scores = {}, []
                for cid, v in validation.items():
                    activity = activity_from_mask(probability[cid] >= threshold, v[5])
                    prediction = learned_stages(codes[cid], activity, pairs[cid][1].supported)
                    predictions[cid] = prediction.to_dict()
                    scores.append(evaluate_case(labels[cid], prediction))
                score = aggregate_scores(scores)
                row = {"target": target, "arm": "xgboost_paired_information", "depth": depth,
                       "threshold": threshold, "objective": calibration_objective(score), "scores": score}
                numerical_trials.append((row, predictions))
        best, predictions = max(numerical_trials, key=lambda pair: pair[0]["objective"])
        target_choice["xgboost_paired_information"] = best
        target_predictions["xgboost_paired_information"] = predictions
        trials.extend(row for row, _ in numerical_trials)
        for name, arm in ARMS.items():
            arm_trials, banks, disagreements = [], {}, {}
            alternative_training = fixed_training if arm["fixed"] else adaptive_training
            for width in (16, 32):
                model, history, seconds = train_pair(
                    raw_training, alternative_training, numbers_training, actual_training, arm, width,
                )
                torch.save(model.state_dict(), folder / f"{name}-width{width}.pt")
                export(folder / f"{name}-width{width}-training.json", history)
                costs.append({"target": target, "arm": name, "width": width,
                              "training_days": len(actual_training), "training_wall_seconds": seconds})
                probability, difference = {}, []
                validation_started = time.perf_counter()
                for cid, v in validation.items():
                    raw, adaptive, fixed, evidence, matrix, support = v
                    requested = support if arm["fixed"] else support & evidence["dominant"]
                    alternative = fixed if arm["fixed"] else adaptive
                    probability[cid], left, right = paired_probabilities(
                        model, raw, alternative, numeric_input(matrix, scaling), requested,
                    )
                    same_requested = support & evidence["dominant"]
                    difference.extend(np.abs(left[same_requested] - right[same_requested]).tolist())
                costs[-1]["validation_probability_wall_seconds"] = time.perf_counter() - validation_started
                disagreements[width] = float(np.mean(difference)) if difference else None
                for threshold in (0.35, 0.5, 0.65):
                    predictions, scores = {}, []
                    for cid, v in validation.items():
                        activity = activity_from_mask(probability[cid] >= threshold, v[5])
                        prediction = learned_stages(codes[cid], activity, pairs[cid][1].supported)
                        predictions[cid] = prediction.to_dict()
                        scores.append(evaluate_case(labels[cid], prediction))
                    score = aggregate_scores(scores)
                    row = {"target": target, "arm": name, "width": width, "threshold": threshold,
                           "objective": calibration_objective(score), "scores": score}
                    arm_trials.append(row)
                    banks[(width, threshold)] = predictions
                print(f"Trained {target} {name} width{width}: {seconds:.2f}s", flush=True)
            chosen = max(arm_trials, key=lambda row: row["objective"])
            target_choice[name] = chosen
            target_predictions[name] = banks[(chosen["width"], chosen["threshold"])]
            target_differences[name] = disagreements[chosen["width"]]
            trials.extend(arm_trials)
        views = {}
        for cid, v in validation.items():
            evidence, support = v[3], v[5]
            views[cid] = {"supported_dates": int(support.sum()),
                          "adaptive_second_views": int((support & evidence["dominant"]).sum()),
                          "fixed_second_views": int(support.sum()),
                          "dominant_days": np.flatnonzero(evidence["dominant"]).tolist(),
                          "largest_noise_units": evidence["largest_noise_units"].tolist(),
                          "second_noise_units": evidence["second_noise_units"].tolist(),
                          "jump_right_column": evidence["jump_right_column"].tolist()}
        chosen_by_target[target] = target_choice
        selected_predictions[target] = target_predictions
        selected_disagreements[target] = target_differences
        all_views[target] = views
        export(folder / "validation-predictions.json", target_predictions)
        export(folder / "selection.json", target_choice)
        del validation, raw_training, adaptive_training, fixed_training, matrix_training, numbers_training, information_training
    export(out / "validation-trials.json", trials)
    export(out / "selection.json", chosen_by_target)
    export(out / "view-evidence.json", all_views)
    export(out / "paired-disagreement.json", selected_disagreements)
    write_csv(out / "training-cost.csv", costs)
    write_csv(out / "preprocessing-cost.csv", preprocessing_costs)
    target = "primary_process"
    choice, predictions = chosen_by_target[target], selected_predictions[target]
    labels = calibration_labels(TRAINING[target], {
        cid: load_case_features(TRAINING[target], cid)
        for cid in protocol["calibration_splits"][target]["validation"]
    })

    def weak_recall(name):
        found = total = 0
        for cid, value in predictions[name].items():
            weak = np.array([int(row["weak_signal"]) == 1 and int(row["activity_label"]) == 1 for row in labels[cid]])
            total += int(weak.sum())
            found += int((weak & (np.array(value["activity"]) == 1)).sum())
        return found / total

    def step_false_positives(name):
        return sum(sum(day == 1 for day in value["activity"]) for cid, value in predictions[name].items()
                   if (int(cid.split("_")[1]) - 1) % 6 == 1)

    main = "adaptive_jump_pair"
    f1 = choice[main]["scores"]["daily"]["f1"]
    disagreement = selected_disagreements[target]
    checks = {
        "f1_at_least_raw_and_numeric": all(f1 >= choice[name]["scores"]["daily"]["f1"] for name in ("chart_plain", "numeric_mlp")),
        "weak_at_least_numeric": weak_recall(main) >= weak_recall("numeric_mlp"),
        "step_fp_below_raw": step_false_positives(main) < step_false_positives("chart_plain"),
        "f1_at_least_no_consistency": f1 >= choice["adaptive_no_consistency"]["scores"]["daily"]["f1"],
        "paired_disagreement_below_no_consistency": disagreement[main] is not None
            and disagreement[main] < disagreement["adaptive_no_consistency"],
        "f1_at_least_fixed": f1 >= choice["fixed_jump_pair"]["scores"]["daily"]["f1"],
        "fewer_second_views_than_fixed": sum(row["adaptive_second_views"] for row in all_views[target].values())
            < sum(row["fixed_second_views"] for row in all_views[target].values()),
    }
    export(out / "calibration-decision.json", {
        "decision": "CONTINUE_JUMP_VIEW_DEVELOPMENT" if all(checks.values()) else "STOP_JUMP_VIEW_CALIBRATION",
        "checks": checks, "process_weak_recall": {name: weak_recall(name) for name in predictions},
        "process_step_fp": {name: step_false_positives(name) for name in predictions},
        "new_api_calls": 0, "independent_validation": False,
    })
    verify(out)
    export(out / "calibration-freeze.json", {str(path.relative_to(out)): sha(path)
                                           for path in out.rglob("*") if path.is_file() and path.name != "state.json"})
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
