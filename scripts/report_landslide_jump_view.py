"""Replay the frozen bounded jump-view validation and retain all comparisons."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import export
from probe_landslide_jump_view import (
    ARMS,
    PRIOR,
    TRAINING,
    inputs,
    numerical_chart_information,
    paired_probabilities,
    verify,
)
from probe_landslide_visual_invariance import calibration_labels, probabilities
from report_landslide_core_lockbox import full_table
from report_landslide_visual_invariance_calibration import (
    choose_diagnostics,
    diagnostic_figure,
)
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_core_lockbox import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case, runs
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual_invariance import activity_model, numeric_input


def load_models(run, target, selection):
    models = {}
    for name, selected in selection.items():
        if name == "xgboost_paired_information":
            model = XGBClassifier()
            model.load_model(run / target / f"xgboost-paired-depth{selected['depth']}.json")
        else:
            if name in ARMS:
                numeric = ARMS[name]["numeric"]
                image = True
                source = run
            else:
                numeric = True
                image = name == "chart_plain"
                source = PRIOR
            model = activity_model(selected["width"], 36, image, numeric)
            weights = torch.load(source / target / f"{name}-width{selected['width']}.pt", weights_only=True)
            model.load_state_dict(weights)
        models[name] = model
    return models


def paired_corrections(labels, predictions):
    """Compare actual method identities without counting reference unknown."""
    rows = []
    for control, cases in predictions.items():
        if control == "adaptive_jump_pair":
            continue
        for cid, value in cases.items():
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            known = actual >= 0
            before = np.array(value["activity"])
            after = np.array(predictions["adaptive_jump_pair"][cid]["activity"])
            gained = known & (before != actual) & (after == actual)
            lost = known & (before == actual) & (after != actual)
            rows.append({"control": control, "case_id": cid, "gained": int(gained.sum()),
                         "lost": int(lost.sum()), "net": int(gained.sum() - lost.sum()),
                         "gained_spans": runs(gained), "lost_spans": runs(lost)})
    return rows


def report(run, out):
    verify(run)
    verify_bindings(run, "calibration-freeze.json")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = read_json(run / "protocol.json")
    selections = read_json(run / "selection.json")
    cost, all_corrections = [], []
    checked_days = 0
    text = ["# 单主导跳变的有限替代观察：冻结校准重放", "",
            "决定：" + read_json(run / "calibration-decision.json")["decision"], "",
            "全部过程12/作者6验证记录；原24/6训练未改变。校准参考已开发暴露，非独立现场确认。", ""]
    for target, source in TRAINING.items():
        folder = out / target
        ids = protocol["calibration_splits"][target]["validation"]
        pairs = {cid: load_case_features(source, cid) for cid in ids}
        labels = calibration_labels(source, pairs)
        selected = selections[target]
        frozen = read_json(run / target / "validation-predictions.json")
        models = load_models(run, target, selected)
        scaling = read_json(run / target / "numeric-scaling.json")
        stage = XGBClassifier()
        stage.load_model(source / "stage-xgboost.json")
        predictions = {name: {} for name in selected}
        probability = {name: {} for name in selected}
        case_scores = {name: [] for name in selected}
        pair_difference = {name: [] for name in ARMS}
        for cid, (case, features) in pairs.items():
            raw, adaptive, fixed, evidence, matrix, support = inputs(case, features)
            numbers = numeric_input(matrix, scaling)
            codes = stage.predict(features.matrix).astype(int) + 1
            for name, model in models.items():
                started = time.perf_counter()
                if name in ARMS:
                    arm = ARMS[name]
                    requested = support if arm["fixed"] else support & evidence["dominant"]
                    alternative = fixed if arm["fixed"] else adaptive
                    value, left, right = paired_probabilities(model, raw, alternative, numbers, requested)
                    common = support & evidence["dominant"]
                    pair_difference[name].extend(np.abs(left[common] - right[common]).tolist())
                elif name == "xgboost_paired_information":
                    value = model.predict_proba(numerical_chart_information(raw, adaptive, evidence, matrix))[:, 1]
                    requested = np.zeros(len(raw), dtype=bool)
                else:
                    value = probabilities(model, raw, numbers)
                    requested = np.zeros(len(raw), dtype=bool)
                cost.append({"target": target, "arm": name, "case_id": cid,
                             "original_days": len(raw), "supported_days": int(support.sum()),
                             "second_views": int(requested.sum()),
                             "probability_wall_seconds": time.perf_counter() - started})
                activity = activity_from_mask(value >= selected[name]["threshold"], support)
                prediction = learned_stages(codes, activity, features.supported)
                prediction.validate(len(raw))
                if np.any(prediction.activity[~support] != -1):
                    raise ValueError("Original activity support changed")
                if np.any(prediction.activity[~features.observed] != -1) or np.any(prediction.stage[~features.observed] != -1):
                    raise ValueError("Missing observation gained a decision")
                checked_days += len(raw)
                predictions[name][cid] = prediction.to_dict()
                case_scores[name].append(evaluate_case(labels[cid], prediction))
                probability[name][cid] = value.tolist()
                if predictions[name][cid] != frozen[name][cid]:
                    raise ValueError("Frozen prediction differs from loaded weights")
        for name, scores in case_scores.items():
            if aggregate_scores(scores) != selected[name]["scores"]:
                raise ValueError("Frozen validation aggregate differs")
        for name, values in pair_difference.items():
            actual = float(np.mean(values)) if values else None
            if actual != read_json(run / "paired-disagreement.json")[target][name]:
                raise ValueError("Frozen paired disagreement differs")
        if target == "primary_process":
            weak = {cid: np.array([int(row["activity_label"]) == 1 and int(row["weak_signal"]) == 1
                                   for row in rows]) for cid, rows in labels.items()}
        else:
            weak = {cid: np.zeros(len(rows), dtype=bool) for cid, rows in labels.items()}
        results, summary, per_record = score_target(
            labels, {cid: feature.observed for cid, (_, feature) in pairs.items()},
            weak, predictions, {}, {},
        )
        export(folder / "predictions.json", predictions)
        export(folder / "probabilities.json", probability)
        export(folder / "results.json", results)
        write_csv(folder / "summary.csv", summary)
        write_csv(folder / "per-record.csv", per_record)
        corrected = paired_corrections(labels, predictions)
        all_corrections.extend({"target": target, **row} for row in corrected)
        chosen = choose_diagnostics(corrected, 4 if target == "primary_process" else 2)
        export(folder / "diagnostic-selection.json",
               {"post_score_selection": "Net correctness extrema vs raw chart and numeric MLP",
                "case_ids": chosen, "used_for_gate_or_reference": False})
        for cid in chosen:
            case, _ = pairs[cid]
            diagnostic_figure(
                case, labels[cid], {name: cases[cid] for name, cases in predictions.items()},
                {name: cases[cid] for name, cases in probability.items()},
                folder / f"{cid}-diagnostic.png", target,
            )
            values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
            blocks = []
            for start in range(0, len(values), 14):
                sample = values[start : start + 14]
                complete = sample[np.isfinite(sample).all(axis=1)]
                median = np.median(complete, axis=0).tolist() if len(complete) >= 8 else None
                blocks.append({"start": start, "stop": min(start + 14, len(values)),
                               "complete_days": len(complete), "median_neu_mm": median})
            export(folder / f"{cid}-raw-blocks.json", blocks)
        text.extend(["## " + target, "", full_table(summary), ""])
        print("Replayed and scored " + target, flush=True)
    write_csv(out / "probability-inference-cost.csv", cost)
    write_csv(out / "per-record-corrections.csv", all_corrections)
    export(out / "verification.json", {
        "weight_prediction_exact_match": True, "frozen_aggregate_exact_match": True,
        "paired_disagreement_exact_match": True, "missing_support_stage_contract": True,
        "checked_calendar_days": checked_days, "retraining": False, "new_api_calls": 0,
        "process_weak_source": "Original weak_signal intersect known activity",
        "author_weak_reference": "Unavailable; never inferred from predictions",
    })
    (out / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    verify(run)
    verify_bindings(run, "calibration-freeze.json")
    export(out / "manifest.json", {str(path.relative_to(out)): sha(path)
                                  for path in out.rglob("*") if path.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.out.resolve())
