"""Train and freeze a chart/numerical activity model with physical consistency."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_endpoint_fit import (
    BASE,
    DATA,
    ROOT,
    csv_rows,
    export,
)
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import calibration_objective, save_json
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual_invariance import (
    activity_model,
    chart_numeric_descriptors,
    fit_numeric_scaling,
    invariance_inputs,
    numeric_input,
    reflect_chart_axes,
)

PRIOR = BASE / "endpoint-fit-analysis-v1"
TRAINING = {
    "primary_process": BASE / "motion-probe-v2",
    "secondary_developer": BASE / "ai-observable-calibration-v3",
}
ARMS = {
    "chart_consistency": {"use_image": True, "use_numeric": True, "augment": True, "consistency": 1.0},
    "chart_augmentation": {"use_image": True, "use_numeric": True, "augment": True, "consistency": 0.0},
    "chart_plain": {"use_image": True, "use_numeric": True, "augment": False, "consistency": 0.0},
    "chart_vision_only": {"use_image": True, "use_numeric": False, "augment": True, "consistency": 1.0},
    "numeric_mlp": {"use_image": False, "use_numeric": True, "augment": False, "consistency": 0.0},
}
SEED = 20261011


def calibration_labels(source, pairs):
    """Filter the original CSV to the exact existing training/validation records."""
    # The process CSV also contains old mechanism records. Select only the
    # registered calibration split before checking its calendar.
    rows = csv_rows(source / "reference-labels.csv")
    labels = {cid: [] for cid in pairs}
    for row in rows:
        cid = row["case_id"]
        if cid in labels:
            labels[cid].append(row)
    for cid, (case, _) in pairs.items():
        if len(labels[cid]) != len(case.dates):
            raise ValueError("Calibration labels must cover the original calendar")
        for day, row in enumerate(labels[cid]):
            if int(row["day_index"]) != day or row["date"] != str(case.dates[day]):
                raise ValueError("Calibration calendar mismatch")
    return labels


def probabilities(model, images, numbers):
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), 256):
            image = torch.from_numpy(images[start : start + 256])
            numeric = torch.from_numpy(numbers[start : start + 256])
            result.extend(torch.sigmoid(model(image, numeric)).numpy().tolist())
    return np.array(result)


def train_model(images, numbers, target, arm, width):
    """Use one fixed seed, optimizer, epoch budget and training order in every arm."""
    torch.manual_seed(SEED)
    model = activity_model(width, numbers.shape[1], arm["use_image"], arm["use_numeric"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    positives = float(target.sum())
    if positives == 0 or positives == len(target):
        raise ValueError("Training requires both motion and stationary examples")
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(target) - positives) / positives)
    )
    shuffle = np.random.default_rng(SEED)
    augmentation = np.random.default_rng(SEED + 1)
    epoch_rows = []
    started = time.perf_counter()
    model.train()
    for epoch in range(12):
        supervised_sum = consistency_sum = 0.0
        order = shuffle.permutation(len(target))
        for start in range(0, len(order), 128):
            indices = order[start : start + 128]
            original = images[indices]
            if arm["augment"]:
                signs = augmentation.choice([-1, 1], size=(len(indices), 3))
                transformed = reflect_chart_axes(original, signs)
            else:
                transformed = original
            numeric = torch.from_numpy(numbers[indices])
            labels = torch.from_numpy(target[indices].astype(np.float32))
            left = model(torch.from_numpy(original), numeric)
            right = model(torch.from_numpy(transformed), numeric)
            supervised = (criterion(left, labels) + criterion(right, labels)) / 2
            consistency = torch.mean((torch.sigmoid(left) - torch.sigmoid(right)) ** 2)
            loss = supervised + arm["consistency"] * consistency
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            supervised_sum += float(supervised.detach()) * len(indices)
            consistency_sum += float(consistency.detach()) * len(indices)
        epoch_rows.append(
            {"epoch": epoch + 1, "supervised_loss": supervised_sum / len(target),
             "probability_consistency_loss": consistency_sum / len(target)}
        )
    return model, epoch_rows, time.perf_counter() - started


def prepare(out):
    verify_manifest(PRIOR)
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "tests/test_landslide_visual_invariance.py",
        ROOT / "src/gnss_sim/landslide_baselines.py",
        ROOT / "src/gnss_sim/landslide_evaluation.py",
        ROOT / "src/gnss_sim/landslide_adaptive_review.py",
        ROOT / "scripts/run_landslide_comparison.py",
        ROOT / "scripts/run_landslide_window_optimization.py",
        ROOT / "scripts/score_landslide_core_lockbox.py",
        ROOT / "scripts/probe_landslide_endpoint_fit.py",
    ]
    plans = {target: read_json(source / "protocol.json")["split"] for target, source in TRAINING.items()}
    export(
        out / "protocol.json",
        {
            "registered_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Trainable image/numerical supervised development; not an LLM or independent field evaluation",
            "change": "Train under coordinate sign reflection consistency rather than abstaining by test-time hard consensus",
            "arms": ARMS,
            "calibration_splits": {target: {key: split[key] for key in ("train", "validation")} for target, split in plans.items()},
            "raster": "3-day disjoint medians >=2 completeNEU;61 columns centered-90..90,height32;translation-centered,minimum6noise vertical span;3 traces+3 availability masks;no gap filling or time reversal",
            "numerical": "Same18 raw84 features as existing process XGB84;train-only mean/std and18 unavailable indicators",
            "architecture": "Two 3x3 stride2 CNNs8/16,adaptive2x5 pooling;concatenate numeric observations;one hidden head16or32",
            "training": {"seed": SEED, "epochs": 12, "stride_days": 2, "batch_size": 128,
                         "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0001,
                         "loss": "Mean positive-balanced BCE of original/reflected views + lambda*probability MSE;lambda1 only in registered consistency arms",
                         "torch_threads": 2, "device": "cpu", "deterministic": True},
            "selection": "Exactly2 head widths x3 probability thresholds .35/.5/.65 per arm;dailyF1+eventF1 as original baseline;fixed12epochs,no early-stopping/seed/lambda/raster search",
            "information_matched_baseline": "Refit XGB120trees,depth2or4,learning_rate.05,seed20261007 on18raw84+183raster column ordinates+183availability;SAME eligible train dates with all8 coordinate sign versions;3 thresholds.35/.5/.65. Bitmap position recovered by first moment,so numerical control gets the new temporal chart information. Author uses own6/6 calibration;original baselines retained",
            "calendar_and_stage": "Exact existing support84,14-day persistence;missing unknown;original target stage head gated by final activity",
            "calibration_gate": "Primary validation main dailyF1>=plain,augmentation and numeric-only;strictF1 gain over at least one of plain/numeric-only;mean reflection probability disagreement strictly below plain and augmentation. Author validation reported,not used to select a process model",
            "development_gate": "Only if calibration gate passes: complete24 main F1>plain and numeric-only, net definite corrections>both, weakRecall>=plain,>=2 benefiting records vs plain;F1>=augmentation and vision-only;strictly lower average reflected probability disagreement than augmentation;author complete12 and all strong baselines fully report;no universal numeric dominance required",
            "failure": "Stop this fixed candidate at failing calibration/development gate;no seed,lambda,epochs,raster,threshold grid,arm/subset replacement",
            "next_if_pass": "Unseen controlled records and independent observable reference validation;this exposed development cannot confirm contribution",
            "old_decisions": ["NOT_CONFIRMED", "STOP_FEEDBACK_PROMOTION", "STOP_AXIS_WITNESS_BEFORE_CALLS", "STOP_COMPARISON_DISPLAY", "STOP_ENDPOINT_FIT"],
            "new_api_calls": 0,
        },
    )
    (out / "source").mkdir()
    for path in sources:
        shutil.copyfile(path, out / "source" / path.name)
    bound = list(sources)
    bound.extend(path for path in PRIOR.rglob("*") if path.is_file())
    for target, source in TRAINING.items():
        bound.extend([source / "protocol.json", source / "reference-labels.csv", source / "stage-xgboost.json"])
        for cid in plans[target]["train"] + plans[target]["validation"]:
            bound.extend(path for path in (source / "inputs" / cid).iterdir() if path.is_file())
    bound.append(DATA / "stage-xgboost.json")
    bound.extend(path for path in (DATA / "inputs").rglob("*") if path.is_file())
    export(out / "source-bindings.json", {str(path): sha(path) for path in bound})
    export(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    export(out / "state.json", {"status": "prepared"})
    print("Prepared fixed supervised visual-invariance probe", flush=True)


def verify(out):
    verify_bindings(out, "source-bindings.json")
    verify_bindings(out, "freeze.json")


def calibrate(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Only a never-started calibration plan may train")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    save_json(out / "state.json", {"status": "training"})
    trials, selections, symmetry, costs = [], {}, [], []
    for target, source in TRAINING.items():
        split = read_json(out / "protocol.json")["calibration_splits"][target]
        ids = split["train"] + split["validation"]
        pairs = {cid: load_case_features(source, cid) for cid in ids}
        labels = calibration_labels(source, pairs)
        inputs = {cid: invariance_inputs(*pairs[cid]) for cid in ids}
        train_images, train_matrix, train_y = [], [], []
        for cid in split["train"]:
            images, matrix, support = inputs[cid]
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            eligible = support & (actual >= 0) & (np.arange(len(actual)) % 2 == 0)
            train_images.append(images[eligible])
            train_matrix.append(matrix[eligible])
            train_y.append(actual[eligible])
        images = np.concatenate(train_images)
        matrix = np.concatenate(train_matrix)
        actual = np.concatenate(train_y)
        scaling = fit_numeric_scaling(matrix)
        numeric = numeric_input(matrix, scaling)
        folder = out / target
        export(folder / "numeric-scaling.json", scaling)
        head = XGBClassifier()
        head.load_model(source / "stage-xgboost.json")
        codes = {cid: head.predict(pairs[cid][1].matrix).astype(int) + 1 for cid in split["validation"]}
        numerical_trials = []
        numerical_images = []
        numerical_labels = []
        for sign_code in range(8):
            signs = np.tile([(-1 if sign_code & (1 << axis) else 1) for axis in range(3)], (len(images), 1))
            descriptor = chart_numeric_descriptors(reflect_chart_axes(images, signs))
            numerical_images.append(np.column_stack((matrix, descriptor)))
            numerical_labels.append(actual)
        numerical_matrix = np.concatenate(numerical_images)
        numerical_target = np.concatenate(numerical_labels)
        validation_matrix = {cid: np.column_stack((inputs[cid][1], chart_numeric_descriptors(inputs[cid][0]))) for cid in split["validation"]}
        for depth in (2, 4):
            numerical = XGBClassifier(n_estimators=120, max_depth=depth, learning_rate=0.05,
                                      n_jobs=2, random_state=20261007)
            numerical.fit(numerical_matrix, numerical_target)
            numerical.save_model(folder / f"xgboost-information-depth{depth}.json")
            probability = {cid: numerical.predict_proba(validation_matrix[cid])[:, 1] for cid in split["validation"]}
            for threshold in (0.35, 0.5, 0.65):
                scores = []
                for cid in split["validation"]:
                    activity = activity_from_mask(probability[cid] >= threshold, inputs[cid][2])
                    scores.append(evaluate_case(labels[cid], learned_stages(codes[cid], activity, pairs[cid][1].supported)))
                score = aggregate_scores(scores)
                numerical_trials.append({"target": target, "arm": "xgboost_chart_information", "depth": depth,
                                         "threshold": threshold, "objective": calibration_objective(score), "scores": score})
        selections.setdefault(target, {})["xgboost_chart_information"] = max(numerical_trials, key=lambda row: row["objective"])
        trials.extend(numerical_trials)
        del numerical_matrix, numerical_target, numerical_images, numerical_labels
        for name, arm in ARMS.items():
            arm_trials = []
            validation_predictions = {}
            for width in (16, 32):
                model, history, seconds = train_model(images, numeric, actual, arm, width)
                model_path = folder / f"{name}-width{width}.pt"
                torch.save(model.state_dict(), model_path)
                export(folder / f"{name}-width{width}-training.json", history)
                costs.append({"target": target, "arm": name, "width": width, "training_days": len(actual),
                              "parameters": sum(p.numel() for p in model.parameters()), "training_wall_seconds": seconds})
                probability = {}
                for cid in split["validation"]:
                    charts, raw, support = inputs[cid]
                    numbers = numeric_input(raw, scaling)
                    probability[cid] = probabilities(model, charts, numbers)
                    differences = []
                    for sign_code in range(1, 8):
                        signs = np.tile([(-1 if sign_code & (1 << axis) else 1) for axis in range(3)], (len(charts), 1))
                        reflected = probabilities(model, reflect_chart_axes(charts, signs), numbers)
                        differences.extend(np.abs(reflected[support] - probability[cid][support]).tolist())
                    symmetry.append({"target": target, "arm": name, "width": width, "case_id": cid,
                                     "supported_days": int(support.sum()), "mean_probability_disagreement": float(np.mean(differences))})
                for threshold in (0.35, 0.5, 0.65):
                    predictions = {}
                    for cid in split["validation"]:
                        activity = activity_from_mask(probability[cid] >= threshold, inputs[cid][2])
                        predictions[cid] = learned_stages(codes[cid], activity, pairs[cid][1].supported).to_dict()
                    case_scores = []
                    for cid, value in predictions.items():
                        prediction = ActivityPrediction(np.array(value["activity"]), np.array(value["stage"]))
                        case_scores.append(evaluate_case(labels[cid], prediction))
                    scores = aggregate_scores(case_scores)
                    row = {"target": target, "arm": name, "width": width, "threshold": threshold,
                           "objective": calibration_objective(scores), "scores": scores}
                    arm_trials.append(row)
                    validation_predictions[(width, threshold)] = predictions
                print(f"Trained {target} {name} width{width}: {seconds:.2f}s", flush=True)
            chosen = max(arm_trials, key=lambda row: row["objective"])
            selections.setdefault(target, {})[name] = chosen
            export(folder / f"{name}-validation-predictions.json",
                   validation_predictions[(chosen["width"], chosen["threshold"])])
            trials.extend(arm_trials)
        # Release the full chart bank between the two target-specific calibrations.
        del inputs, images, matrix, numeric, train_images, train_matrix, train_y
    export(out / "validation-trials.json", trials)
    export(out / "selection.json", selections)
    write_csv(out / "validation-reflection.csv", symmetry)
    write_csv(out / "training-cost.csv", costs)
    selection = selections["primary_process"]
    main_f1 = selection["chart_consistency"]["scores"]["daily"]["f1"]

    def disagreement(name):
        width = selection[name]["width"]
        rows = [row for row in symmetry if row["target"] == "primary_process" and row["arm"] == name and row["width"] == width]
        return float(np.average([row["mean_probability_disagreement"] for row in rows], weights=[row["supported_days"] for row in rows]))

    controls = [selection[name]["scores"]["daily"]["f1"] for name in ("chart_plain", "chart_augmentation", "numeric_mlp")]
    checks = {"main_f1_nondecrease": all(main_f1 >= value for value in controls),
              "main_f1_strict_gain": any(main_f1 > selection[name]["scores"]["daily"]["f1"] for name in ("chart_plain", "numeric_mlp")),
              "reflection_below_plain": disagreement("chart_consistency") < disagreement("chart_plain"),
              "reflection_below_augmentation": disagreement("chart_consistency") < disagreement("chart_augmentation")}
    export(out / "calibration-decision.json", {"decision": "CONTINUE_VISUAL_INVARIANCE_DEVELOPMENT" if all(checks.values()) else "STOP_VISUAL_INVARIANCE_CALIBRATION",
                                              "checks": checks, "mean_probability_disagreement": {name: disagreement(name) for name in ("chart_consistency", "chart_plain", "chart_augmentation")},
                                              "independent_validation": False, "new_api_calls": 0})
    verify(out)
    export(out / "calibration-freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file() and path.name != "state.json"})
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
