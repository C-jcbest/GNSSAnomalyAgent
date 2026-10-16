"""Calibrate one quality-supervised visual/numeric mechanism without test access."""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_comparison import calibration_objective, fit_baselines, predict_baselines
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_deformation_quality import deformation_quality_head
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual_invariance import numeric_input

ARMS = {
    "visual_numeric_dual": {"visual": True, "numeric": True, "quality": True},
    "visual_numeric_single": {"visual": True, "numeric": True, "quality": False},
    "numeric_dual": {"visual": False, "numeric": True, "quality": True},
    "numeric_single": {"visual": False, "numeric": True, "quality": False},
    "visual_single": {"visual": True, "numeric": False, "quality": False},
}
THRESHOLDS = [number / 20 for number in range(1, 20)]
SEED = 2026101249


def modality_inputs(bank, arm, scaling):
    parts = []
    if arm["visual"]:
        parts.append(bank["embeddings"])
    if arm["numeric"]:
        parts.append(numeric_input(bank["numbers"], scaling))
    return np.column_stack(parts).astype(np.float32)


def train_head(values, activity, quality, width, auxiliary):
    torch.manual_seed(SEED)
    model = deformation_quality_head(values.shape[1], width)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    activity_loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(activity) - activity.sum()) / activity.sum())
    )
    known_quality = quality >= 0
    quality_positive = quality[known_quality].sum()
    if auxiliary and not 0 < quality_positive < known_quality.sum():
        raise ValueError("Quality training needs observed positive and negative queries")
    quality_loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((known_quality.sum() - quality_positive) / max(quality_positive, 1))
    )
    inputs = torch.from_numpy(values)
    activity_labels = torch.from_numpy(activity.astype(np.float32))
    quality_labels = torch.from_numpy(quality.astype(np.float32))
    generator = np.random.default_rng(SEED)
    history = []
    started = time.perf_counter()
    for epoch in range(12):
        activity_total = quality_total = 0.0
        order = generator.permutation(len(values))
        for start in range(0, len(order), 128):
            indices = order[start : start + 128]
            deformation_logits, quality_logits = model(inputs[indices])
            primary = activity_loss(deformation_logits, activity_labels[indices])
            known = quality_labels[indices] >= 0
            secondary = torch.tensor(0.0)
            if auxiliary and known.any():
                secondary = quality_loss(quality_logits[known], quality_labels[indices][known])
            optimizer.zero_grad()
            (primary + secondary).backward()
            optimizer.step()
            activity_total += float(primary.detach()) * len(indices)
            quality_total += float(secondary.detach()) * len(indices)
        history.append(
            {
                "epoch": epoch + 1,
                "activity_bce": activity_total / len(values),
                "quality_bce": quality_total / len(values),
            }
        )
    return model, history, time.perf_counter() - started


def probabilities(model, values):
    model.eval()
    activity = []
    quality = []
    with torch.inference_mode():
        for start in range(0, len(values), 256):
            left, right = model(torch.from_numpy(values[start : start + 256]))
            activity.extend(torch.sigmoid(left).numpy().tolist())
            quality.extend(torch.sigmoid(right).numpy().tolist())
    return np.array(activity), np.array(quality)


def prediction(probability, bank, codes, threshold):
    gate = activity_from_mask(probability >= threshold, bank["supported"])
    return learned_stages(codes, gate, bank["stage_supported"])


def quality_query_scores(banks, probability):
    """Score the separate quality task without dropping unsupported positives.

    Missing/unknown reference queries are not negatives. A supported auxiliary
    probability uses the preregistered .5 threshold; unsupported predictions
    remain unknown and a known positive there remains a false negative.
    AUPRC is explicitly restricted to known, supported query probabilities.
    """
    actual_all = []
    predicted_all = []
    ranked_actual = []
    ranked_probability = []
    for cid, values in probability.items():
        bank = banks[cid]
        values = np.asarray(values)
        actual = bank["quality"]
        if values.shape != actual.shape:
            raise ValueError("Quality probabilities do not match the original calendar")
        supported = bank["supported"]
        if not np.isfinite(values[supported]).all():
            raise ValueError("Supported quality probabilities must be finite")
        predicted = np.full(len(actual), -1, dtype=int)
        predicted[supported] = values[supported] >= 0.5
        known = bank["observed"] & (actual >= 0)
        actual_all.extend(actual[known].tolist())
        predicted_all.extend(predicted[known].tolist())
        ranked = known & supported
        ranked_actual.extend(actual[ranked].tolist())
        ranked_probability.extend(values[ranked].tolist())
    actual = np.array(actual_all)
    predicted = np.array(predicted_all)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted != 1)).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "reference_known_queries": len(actual),
        "unknown_predictions_on_known_queries": int((predicted == -1).sum()),
        "coverage": float(np.mean(predicted >= 0)) if len(predicted) else None,
        "auprc_known_supported_queries": float(
            average_precision_score(ranked_actual, ranked_probability)
        )
        if any(ranked_actual)
        else None,
        "threshold": 0.5,
        "scope": "Observed-event presence within central14 query; not exact fault-day localization",
    }


def calibrate(data, encoding, out):
    if read_json(encoding / "protocol.json")["data_manifest_sha256"] != sha(
        data / "data-freeze.json"
    ):
        raise ValueError("Encoding belongs to a different data freeze")
    for relative, expected in read_json(data / "data-freeze.json").items():
        if sha(data / relative) != expected:
            raise ValueError("Frozen data changed: " + relative)
    for relative, expected in read_json(encoding / "encoding-freeze.json").items():
        if sha(encoding / relative) != expected:
            raise ValueError("Encoding changed: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    protocol = read_json(encoding / "protocol.json")
    split = protocol["split"]
    write_json(
        out / "training-plan.json",
        {
            "parent_protocol_sha256": sha(encoding / "protocol.json"),
            "data_manifest_sha256": sha(data / "data-freeze.json"),
            "arms": ARMS,
            "seed": SEED,
            "epochs": 12,
            "thresholds": THRESHOLDS,
            "widths": [16, 32],
            "stage": "Refit original numerical phase head on these train48; same head and original stage support in every learned arm",
            "postprocessing": "Same existing minimum14 original-date run rule in all new learned arms; fast8..13d events may be missed, report them rather than delete their labels",
            "trees": "Two depths/19 thresholds; raw84 or raw84+exact chart numeric descriptors; all complete supported training days (stronger data coverage than neural stride3), report the distinction",
            "quality_evaluation": "Central14-day observed artifact presence, threshold.5 fixed, auxiliary coverage retained; single arms have no trained quality prediction",
            "test_access": "No reserved test reference or prediction until registered validation gate passes",
        },
    )
    sources = [
        Path(__file__),
        ROOT / "src/gnss_sim/landslide_deformation_quality.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "src/gnss_sim/landslide_baselines.py",
        ROOT / "src/gnss_sim/landslide_local_experiment.py",
        ROOT / "src/gnss_sim/landslide_evaluation.py",
        ROOT / "scripts/run_landslide_comparison.py",
        ROOT / "scripts/score_landslide_claim_review.py",
        ROOT / "tests/test_landslide_deformation_quality_scoring.py",
    ]
    for source in sources:
        target = out / "source" / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
    write_json(
        out / "source-bindings.json", {str(path.relative_to(ROOT)): sha(path) for path in sources}
    )
    write_json(
        out / "pre-fit-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    banks = {}
    features = {}
    for group, ids in split.items():
        for cid in ids:
            with np.load(encoding / group / (cid + ".npz")) as archive:
                banks[cid] = {key: archive[key] for key in archive.files}
            _, features[cid] = load_case_features(data, cid)
    labels = {cid: [] for cid in banks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            labels[row["case_id"]].append(row)
    legacy = out / "legacy-baselines"
    legacy.mkdir()
    choices, activity_model, stage_model = fit_baselines(features, labels, split, legacy)
    activity_model.save_model(legacy / "activity-xgboost.json")
    stage_model.save_model(out / "stage-xgboost.json")
    codes = {
        cid: stage_model.predict(features[cid].matrix).astype(int) + 1
        for cid in split["validation"]
    }
    scaling = read_json(encoding / "numeric-scaling.json")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    trials = []
    selections = {}
    predictions = {}
    saved_probabilities = {}
    training_costs = []

    def evaluate(name, parameter, probability):
        candidates = {}
        for threshold in THRESHOLDS:
            output = {
                cid: prediction(probability[cid], banks[cid], codes[cid], threshold)
                for cid in split["validation"]
            }
            score = aggregate_scores([evaluate_case(labels[cid], output[cid]) for cid in output])
            row = {
                "method": name,
                "parameter": parameter,
                "threshold": threshold,
                "objective": calibration_objective(score),
                "validation": score,
            }
            trials.append(row)
            candidates[threshold] = {cid: value.to_dict() for cid, value in output.items()}
        return candidates

    activity_train = np.concatenate(
        [banks[cid]["activity"][banks[cid]["selected"]] for cid in split["train"]]
    )
    quality_train = np.concatenate(
        [banks[cid]["quality"][banks[cid]["selected"]] for cid in split["train"]]
    )
    for name, arm in ARMS.items():
        inputs = {cid: modality_inputs(bank, arm, scaling) for cid, bank in banks.items()}
        training = np.concatenate([inputs[cid][banks[cid]["selected"]] for cid in split["train"]])
        candidates = {}
        probability_bank = {}
        for width in (16, 32):
            model, history, seconds = train_head(
                training, activity_train, quality_train, width, arm["quality"]
            )
            torch.save(model.state_dict(), out / (name + f"-width{width}.pt"))
            write_json(out / (name + f"-width{width}-loss.json"), history)
            pair = {cid: probabilities(model, inputs[cid]) for cid in split["validation"]}
            candidates[width] = evaluate(
                name, width, {cid: values[0] for cid, values in pair.items()}
            )
            probability_bank[width] = pair
            training_costs.append({"method": name, "parameter": width, "training_seconds": seconds})
            print("trained", name, width, f"{seconds:.2f}s", flush=True)
        choice = max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        )
        selections[name] = choice
        predictions[name] = candidates[choice["parameter"]][choice["threshold"]]
        saved_probabilities[name] = {
            cid: {
                "activity": pair[0].tolist(),
                "quality": pair[1].tolist() if arm["quality"] else None,
            }
            for cid, pair in probability_bank[choice["parameter"]].items()
        }
    for name, chart in (("xgb_raw84", False), ("xgb_chart384", True)):
        matrices = {
            cid: np.column_stack((bank["numbers"], bank["chart_descriptors"]))
            if chart
            else bank["numbers"]
            for cid, bank in banks.items()
        }
        training = np.concatenate([matrices[cid] for cid in split["train"]])
        actual = np.concatenate([banks[cid]["activity"] for cid in split["train"]])
        support = np.concatenate([banks[cid]["supported"] for cid in split["train"]]) & (
            actual >= 0
        )
        candidates = {}
        probability_bank = {}
        for depth in (2, 4):
            started = time.perf_counter()
            model = XGBClassifier(
                n_estimators=120,
                max_depth=depth,
                learning_rate=0.05,
                random_state=20261007,
                n_jobs=2,
            )
            model.fit(training[support], actual[support])
            model.save_model(out / (name + f"-depth{depth}.json"))
            probability = {
                cid: model.predict_proba(matrices[cid])[:, 1] for cid in split["validation"]
            }
            candidates[depth] = evaluate(name, depth, probability)
            probability_bank[depth] = probability
            training_costs.append(
                {
                    "method": name,
                    "parameter": depth,
                    "training_seconds": time.perf_counter() - started,
                }
            )
        choice = max(
            (row for row in trials if row["method"] == name), key=lambda row: row["objective"]
        )
        selections[name] = choice
        predictions[name] = candidates[choice["parameter"]][choice["threshold"]]
        saved_probabilities[name] = {
            cid: {"activity": value.tolist(), "quality": None}
            for cid, value in probability_bank[choice["parameter"]].items()
        }
    original = predict_baselines(
        features, split["validation"], choices, activity_model, stage_model
    )
    predictions.update(
        {
            "original_" + name: {cid: value.to_dict() for cid, value in output.items()}
            for name, output in original.items()
        }
    )
    write_json(out / "selection.json", selections)
    write_json(out / "validation-trials.json", trials)
    write_json(out / "probabilities.json", saved_probabilities)
    write_json(out / "predictions.json", predictions)
    write_csv(out / "training-costs.csv", training_costs)
    write_json(
        out / "prediction-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    validation_labels = {cid: labels[cid] for cid in split["validation"]}
    observed = {cid: banks[cid]["observed"] for cid in validation_labels}
    weak = {
        cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]])
        for cid in validation_labels
    }
    results, summary, per_case = score_target(
        validation_labels, observed, weak, predictions, {}, {}
    )
    write_json(out / "results.json", results)
    write_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", per_case)
    quality_summary = {
        name: quality_query_scores(
            banks, {cid: pair["quality"] for cid, pair in saved_probabilities[name].items()}
        )
        for name, arm in ARMS.items()
        if arm["quality"]
    }
    write_json(out / "quality-query-summary.json", quality_summary)
    table = {row["method"]: row for row in summary}
    main, single, numeric = [
        table[name] for name in ("visual_numeric_dual", "visual_numeric_single", "numeric_dual")
    ]
    ledger = read_json(data / "training-generation-ledger.json")
    families = {
        row["case_id"]: row["truth"]["family"] for row in ledger if row["group"] == "validation"
    }
    negative = [
        cid for cid, family in families.items() if family in {"quiet", "step_only", "spike_only"}
    ]
    mixed = [
        cid
        for cid, family in families.items()
        if family in {"deformation_step", "deformation_spike"}
    ]

    def subset(name, ids):
        return aggregate_scores([results[name]["per_case"][cid] for cid in ids])["daily"]

    negative_fp = {
        name: subset(name, negative)["fp"]
        for name in ("visual_numeric_dual", "visual_numeric_single")
    }
    mixed_recall = {
        name: subset(name, mixed)["recall"]
        for name in ("visual_numeric_dual", "visual_numeric_single")
    }
    net_correct = single["fp"] + single["fn"] - main["fp"] - main["fn"]
    checks = {
        "activity_f1_gain": main["f1"] > single["f1"],
        "event_f1_nondecrease": main["event_f1_iou03"] >= single["event_f1_iou03"],
        "net_correction_positive": net_correct > 0,
        "negative_fp_nondecrease_avoided": negative_fp["visual_numeric_dual"]
        <= negative_fp["visual_numeric_single"],
        "mixed_recall_nondecrease": mixed_recall["visual_numeric_dual"]
        >= mixed_recall["visual_numeric_single"],
        "visual_value": main["f1"] > numeric["f1"]
        or (
            main["event_f1_iou03"] > numeric["event_f1_iou03"]
            and main["f1"] >= numeric["f1"] - 0.01
        ),
    }
    decision = {
        "decision": "PROCEED_DQVN_RESERVED_CONFIRMATION"
        if all(checks.values())
        else "STOP_DQVN_CALIBRATION",
        "checks": checks,
        "negative_fp": negative_fp,
        "mixed_recall": mixed_recall,
        "net_correct": net_correct,
        "test_truth_opened": False,
        "new_api_calls": 0,
        "interpretation": "Validation selection/development only, not an independent contribution claim",
    }
    write_json(out / "decision.json", decision)
    for source in sources:
        if sha(source) != sha(out / "source" / source.name):
            raise ValueError("Training source changed during execution: " + str(source))
    write_json(
        out / "calibration-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--encoding", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    calibrate(args.data.resolve(), args.encoding.resolve(), args.out.resolve())
