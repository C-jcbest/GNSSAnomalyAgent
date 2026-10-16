"""Freeze an observation-only stage-conflict probe before loading development labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
from run_landslide_comparison import ROOT, STAGES, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import precision_recall_fscore_support

from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_stage_raw_consistency import (
    StageContrastPolicy,
    abstain_on_stage_conflicts,
    raw_stage_contrasts,
    stage_conflict_mask,
)


def serial_contrasts(values):
    return [[float(item) if np.isfinite(item) else None for item in row] for row in values]


def execute(native, out):
    config = read_json(native / "protocol.json")
    source = Path(config["activity_run"])
    if sha(native / "predictions.json") != read_json(native / "prediction-freeze.json")["sha256"]:
        raise ValueError("Frozen native predictions changed")
    out.mkdir(parents=True, exist_ok=False)
    policy = StageContrastPolicy()
    save_json(out / "protocol.json", {
        "scope": "Previously seen 12-record controlled development, not independent confirmation",
        "native": str(native.resolve()), "source": str(source), "policy": asdict(policy),
        "primary": "Both-scale raw contradiction; single scales are mechanism ablations",
        "no_api": True, "no_training": True, "no_threshold_search": True,
        "review_gate": {
            "each_repeat_removed_error_fraction_minimum": 0.75,
            "each_repeat_correct_stage_loss_fraction_maximum": 0.05,
            "each_repeat_error_records_minimum": 2,
            "applies_to": "direct_visual_native_stage only; not choose a better activity arm",
        },
        "meaning": "Noise margin is a heuristic; rejection is unknown, never a replacement class",
    })
    dependencies = [Path(__file__), ROOT / "src/gnss_sim/landslide_stage_raw_consistency.py",
                    ROOT / "src/gnss_sim/landslide_evaluation.py",
                    native / "predictions.json", source / "numeric-predictions.json"]
    for cid in config["case_ids"]:
        dependencies.extend((source / "inputs" / cid).glob("*.json"))
    save_json(out / "input-freeze.json", {str(path.resolve()): sha(path) for path in dependencies})
    native_predictions = read_json(native / "predictions.json")
    numeric_predictions = read_json(source / "numeric-predictions.json")
    baselines = dict(native_predictions)
    for method, cases in numeric_predictions.items():
        baselines[method] = {cid: cases[cid] for cid in config["case_ids"]}
    predictions, conflicts, evidence = {}, {}, {}
    cases = {}
    for cid in config["case_ids"]:
        case, features = load_case_features(source, cid)
        cases[cid] = case
        cache = {}
        for method, values in baselines.items():
            value = values[cid]
            activity = np.asarray(value["activity"])
            prediction = ActivityPrediction(activity, np.asarray(value["stage"]),
                                            stage_status=value["stage_status"])
            key = tuple(activity)
            if key not in cache:
                cache[key] = {scale: raw_stage_contrasts(case, activity, features.noise, scale)
                              for scale in policy.separations}
            contrasts = cache[key]
            evidence.setdefault(method, {})[cid] = {
                str(scale): serial_contrasts(array) for scale, array in contrasts.items()
            }
            predictions.setdefault(method, {})[cid] = prediction.to_dict()
            for variant, scales in (("raw21", (21,)), ("raw42", (42,)), ("raw_both", (21, 42))):
                applied = StageContrastPolicy(separations=scales)
                mask = stage_conflict_mask(prediction.stage, contrasts, applied)
                name = f"{method}__{variant}"
                predictions.setdefault(name, {})[cid] = abstain_on_stage_conflicts(
                    prediction, mask).to_dict()
                conflicts.setdefault(name, {})[cid] = np.flatnonzero(mask).tolist()
    save_json(out / "evidence.json", evidence)
    save_json(out / "predictions.json", predictions)
    save_json(out / "conflicts.json", conflicts)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in (
        "predictions.json", "conflicts.json", "evidence.json")})
    # Development references are deliberately loaded only after predictions freeze.
    # They were previously exposed, so this order does not confer independence.
    labels = load_labels(source / "reference-subset.csv", cases)
    save_json(out / "reference-binding.json", {
        "path": str(source / "reference-subset.csv"),
        "sha256": sha(source / "reference-subset.csv"),
        "prior_exposure": True,
    })
    results, removals = {}, {}
    for method, values in predictions.items():
        per_case = {}
        actual, predicted = [], []
        for cid, value in values.items():
            prediction = ActivityPrediction(np.asarray(value["activity"]),
                                            np.asarray(value["stage"]),
                                            stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            for day, row in enumerate(labels[cid]):
                if int(row["activity_label"]) == 1 and row["feature"] in STAGES:
                    actual.append(STAGES[row["feature"]])
                    predicted.append(int(prediction.stage[day]))
        p, r, f, n = precision_recall_fscore_support(actual, predicted, labels=[1, 2, 3],
                                                   zero_division=0)
        results[method] = {
            "aggregate": aggregate_scores(list(per_case.values())), "per_case": per_case,
            "true_activity_internal": {
                "precision_ASD": p.tolist(), "recall_ASD": r.tolist(),
                "f1_ASD": f.tolist(), "support_ASD": n.tolist(), "macro_f1": float(f.mean()),
                "coverage": float(np.mean(np.asarray(predicted) > 0)),
                "meaning": "Includes missed activity and unknown, conditional on known analytic labels",
            },
        }
        if "__" not in method:
            continue
        original_method = method.split("__")[0]
        rows = {}
        for cid, days in conflicts[method].items():
            conflict_days = set(days)
            counts = {"correct_to_unknown": 0, "wrong_to_unknown": 0,
                      "nonactive_to_unknown": 0, "unscored_to_unknown": 0,
                      "original_correct_days": 0}
            original = baselines[original_method][cid]["stage"]
            for day, row in enumerate(labels[cid]):
                expected = STAGES.get(row["feature"], -1)
                active = int(row["activity_label"])
                if active == 1 and expected > 0 and original[day] == expected:
                    counts["original_correct_days"] += 1
                if day not in conflict_days:
                    continue
                if active < 0 or (active == 1 and expected < 0):
                    counts["unscored_to_unknown"] += 1
                elif active == 0:
                    counts["nonactive_to_unknown"] += 1
                elif original[day] == expected:
                    counts["correct_to_unknown"] += 1
                else:
                    counts["wrong_to_unknown"] += 1
            rows[cid] = counts
        total = {key: sum(row[key] for row in rows.values()) for key in next(iter(rows.values()))}
        errors = total["wrong_to_unknown"] + total["nonactive_to_unknown"]
        assessed = errors + total["correct_to_unknown"]
        total["removed_error_fraction"] = errors / assessed if assessed else None
        total["correct_stage_loss_fraction"] = (
            total["correct_to_unknown"] / total["original_correct_days"]
            if total["original_correct_days"] else None
        )
        total["error_records"] = sum(
            row["wrong_to_unknown"] + row["nonactive_to_unknown"] > 0 for row in rows.values())
        removals[method] = {"total": total, "per_case": rows}
    save_json(out / "results.json", results)
    save_json(out / "removals.json", removals)
    decisions = []
    for repeat in range(config["repetitions"]):
        method = f"direct_visual_native_stage_r{repeat}__raw_both"
        row = removals[method]["total"]
        passed = (row["removed_error_fraction"] is not None
                  and row["removed_error_fraction"] >= 0.75
                  and row["correct_stage_loss_fraction"] <= 0.05
                  and row["error_records"] >= 2)
        decisions.append({"method": method, "passed": passed, **row})
    decision = {"allows_visual_review": all(row["passed"] for row in decisions),
                "repeats": decisions, "new_api_requests": 0}
    save_json(out / "decision.json", decision)
    print(decision, flush=True)
    for path, expected in read_json(out / "input-freeze.json").items():
        if sha(Path(path)) != expected:
            raise ValueError("Frozen source or observation changed")
    save_json(out / "state.json", {"status": "complete", "configs": len(results)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    execute(args.native, args.out)


if __name__ == "__main__":
    main()
