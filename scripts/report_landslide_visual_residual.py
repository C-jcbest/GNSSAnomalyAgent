"""Replay frozen visual residual calibration and export unified diagnostics."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import export
from probe_landslide_visual_invariance import TRAINING, calibration_labels, probabilities
from probe_landslide_visual_residual import ARMS, PRIOR, REUSED, residual_probabilities, verify
from report_landslide_core_lockbox import full_table
from report_landslide_visual_invariance_calibration import diagnostic_figure
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_core_lockbox import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case, runs
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual_invariance import (
    activity_model,
    chart_numeric_descriptors,
    invariance_inputs,
    numeric_input,
)
from gnss_sim.landslide_visual_residual import anchor_logits, residual_model


def corrections(labels, predictions):
    rows = []
    for control, cases in predictions.items():
        if control == "bounded_visual_oof":
            continue
        for cid, value in cases.items():
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            known = actual >= 0
            before = np.array(value["activity"])
            after = np.array(predictions["bounded_visual_oof"][cid]["activity"])
            gained = known & (before != actual) & (after == actual)
            lost = known & (before == actual) & (after != actual)
            rows.append({"control": control, "case_id": cid, "gained": int(gained.sum()),
                         "lost": int(lost.sum()), "net": int(gained.sum() - lost.sum()),
                         "gained_spans": runs(gained), "lost_spans": runs(lost)})
    return rows


def choose_records(rows, count):
    chosen = []
    for control in ("numeric_anchor", "bounded_numeric_oof"):
        records = [row for row in rows if row["control"] == control]
        for reverse in (False, True):
            for row in sorted(records, key=lambda row: (row["net"], row["case_id"]), reverse=reverse):
                if row["case_id"] not in chosen:
                    chosen.append(row["case_id"])
                    break
            if len(chosen) >= count:
                return chosen
    return chosen


def replay_training_anchors(run, target, source, protocol):
    """Verify each held-out prediction against the weights that never saw its record."""
    split = protocol["calibration_splits"][target]
    pairs = {cid: load_case_features(source, cid) for cid in split["train"]}
    labels = calibration_labels(source, pairs)
    matrices, records, actual = [], [], []
    for cid, pair in pairs.items():
        _, matrix, support = invariance_inputs(*pair)
        truth = np.array([int(row["activity_label"]) for row in labels[cid]])
        eligible = support & (truth >= 0) & (np.arange(len(truth)) % 2 == 0)
        matrices.append(matrix[eligible])
        records.append(np.full(eligible.sum(), cid))
        actual.append(truth[eligible])
    matrix = np.concatenate(matrices)
    record = np.concatenate(records)
    bank = np.load(run / target / "training-anchor-predictions.npz")
    if not np.array_equal(record, bank["record"]) or not np.array_equal(np.concatenate(actual), bank["labels"]):
        raise ValueError("Training anchor bank dates or labels changed")
    audit = read_json(run / target / "cross-fitting-audit.json")
    replay = np.full(len(record), np.nan)
    for index, held_out in enumerate(protocol["folds"][target]):
        selected = np.isin(record, held_out)
        fitted_records = sorted(set(record[~selected].tolist()))
        if audit[index]["fit_records"] != fitted_records or set(fitted_records) & set(held_out):
            raise ValueError("A held-out record entered the numerical anchor fit")
        model = XGBClassifier()
        model.load_model(run / target / f"anchor-oof-fold{index}.json")
        replay[selected] = model.predict_proba(matrix[selected])[:, 1]
    if not np.array_equal(replay, bank["oof_probability"]):
        raise ValueError("OOF bank differs from frozen fold weights")
    model.load_model(run / target / "anchor-depth4.json")
    if not np.array_equal(model.predict_proba(matrix)[:, 1], bank["full_probability"]):
        raise ValueError("In-sample bank differs from frozen anchor weights")
    return len(record)


def report(run, out):
    verify(run)
    verify_bindings(run, "calibration-freeze.json")
    if read_json(run / "state.json")["status"] != "calibration_complete":
        raise ValueError("Only completed frozen calibration may be reported")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = read_json(run / "protocol.json")
    selections = read_json(run / "selection.json")
    cost, all_corrections, families = [], [], []
    checked_days = training_days = 0
    text = ["# 固定数值初判与有界视觉修正：校准与权重重放", "",
            "决定：" + read_json(run / "calibration-decision.json")["decision"], "",
            "已开发暴露的过程12验证/作者6验证；原训练24/6。所有臂共用19阈值、38选择机会。", ""]
    for target, source in TRAINING.items():
        training_days += replay_training_anchors(run, target, source, protocol)
        folder = out / target
        folder.mkdir()
        ids = protocol["calibration_splits"][target]["validation"]
        pairs = {cid: load_case_features(source, cid) for cid in ids}
        labels = calibration_labels(source, pairs)
        inputs = {cid: invariance_inputs(*pairs[cid]) for cid in ids}
        scaling = read_json(run / target / "numeric-scaling.json")
        stage = XGBClassifier()
        stage.load_model(source / "stage-xgboost.json")
        codes = {cid: stage.predict(pairs[cid][1].matrix).astype(int) + 1 for cid in ids}
        anchor = XGBClassifier()
        anchor.load_model(run / target / "anchor-depth4.json")
        started = time.perf_counter()
        logits = {cid: anchor_logits(anchor.predict_proba(inputs[cid][1])[:, 1]) for cid in ids}
        cost.append({"target": target, "arm": "shared_depth4_anchor", "case_id": "all_validation",
                     "wall_seconds": time.perf_counter() - started})
        frozen = read_json(run / target / "validation-predictions.json")
        frozen_probability = read_json(run / target / "validation-probabilities.json")
        predictions, probability_bank = {}, {}
        for name, selected in selections[target].items():
            if name in ARMS:
                arm = ARMS[name]
                model = residual_model(selected["width"], 36, arm["image"], arm["bounded"])
                model.load_state_dict(torch.load(run / target / f"{name}-width{selected['width']}.pt", weights_only=True))
            elif name in REUSED:
                image, numeric = REUSED[name]
                model = activity_model(selected["width"], 36, image, numeric)
                model.load_state_dict(torch.load(PRIOR / target / f"{name}-width{selected['width']}.pt", weights_only=True))
            else:
                model = XGBClassifier()
                filename = f"anchor-depth{selected['depth']}.json" if name == "numeric_anchor" else f"chart-information-depth{selected['depth']}.json"
                model.load_model(run / target / filename)
            predictions[name], probability_bank[name] = {}, {}
            scores = []
            for cid in ids:
                images, matrix, support = inputs[cid]
                numbers = numeric_input(matrix, scaling)
                started = time.perf_counter()
                if name in ARMS:
                    probability = residual_probabilities(model, images, numbers, logits[cid])
                    if ARMS[name]["bounded"]:
                        with torch.no_grad():
                            final = model(torch.from_numpy(images), torch.from_numpy(numbers), torch.from_numpy(logits[cid])).numpy()
                        if np.any(np.abs(final - logits[cid]) > math.log(4) + 2e-6):
                            raise ValueError("Frozen bounded model exceeded its allowed odds correction")
                elif name in REUSED:
                    probability = probabilities(model, images, numbers)
                else:
                    observations = matrix if name == "numeric_anchor" else np.column_stack((matrix, chart_numeric_descriptors(images)))
                    probability = model.predict_proba(observations)[:, 1]
                cost.append({"target": target, "arm": name, "case_id": cid,
                             "wall_seconds": time.perf_counter() - started})
                if probability.tolist() != frozen_probability[name][cid]:
                    raise ValueError("Frozen probability differs from selected weights")
                activity = activity_from_mask(probability >= selected["threshold"], support)
                prediction = learned_stages(codes[cid], activity, pairs[cid][1].supported)
                prediction.validate(len(images))
                if np.any(prediction.activity[~support] != -1):
                    raise ValueError("Activity escaped original support")
                if np.any(prediction.activity[~pairs[cid][1].observed] != -1) or np.any(prediction.stage[~pairs[cid][1].observed] != -1):
                    raise ValueError("Missing observation gained a decision")
                if prediction.to_dict() != frozen[name][cid]:
                    raise ValueError("Frozen decision differs from selected weights")
                checked_days += len(images)
                scores.append(evaluate_case(labels[cid], prediction))
                predictions[name][cid] = prediction.to_dict()
                probability_bank[name][cid] = probability.tolist()
            if aggregate_scores(scores) != selected["scores"]:
                raise ValueError("Frozen aggregate score changed")
        weak = {cid: np.array([int(row.get("weak_signal", 0)) == 1 and int(row["activity_label"]) == 1 for row in rows])
                for cid, rows in labels.items()}
        results, summary, per_record = score_target(labels, {cid: features.observed for cid, (_, features) in pairs.items()},
                                                     weak, predictions, {}, {})
        export(folder / "results.json", results)
        export(folder / "predictions.json", predictions)
        export(folder / "probabilities.json", probability_bank)
        write_csv(folder / "summary.csv", summary)
        write_csv(folder / "per-record.csv", per_record)
        corrected = corrections(labels, predictions)
        all_corrections.extend({"target": target, **row} for row in corrected)
        chosen = choose_records(corrected, 4 if target == "primary_process" else 2)
        export(folder / "diagnostic-selection.json", {"case_ids": chosen, "post_score_selection": "Net correction extrema vs numerical anchor and nonvisual residual", "used_for_gate_or_reference": False})
        for cid in chosen:
            case, _ = pairs[cid]
            diagnostic_figure(case, labels[cid], {name: cases[cid] for name, cases in predictions.items()},
                              {name: cases[cid] for name, cases in probability_bank.items()},
                              folder / f"{cid}-diagnostic.png", target)
            values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
            blocks = []
            for start in range(0, len(values), 14):
                sample = values[start : start + 14]
                complete = sample[np.isfinite(sample).all(axis=1)]
                blocks.append({"start": start, "stop": min(start + 14, len(values)),
                               "complete_days": len(complete),
                               "median_neu_mm": np.median(complete, axis=0).tolist() if len(complete) >= 8 else None})
            export(folder / f"{cid}-raw-blocks.json", blocks)
        if target == "primary_process":
            names = read_json(source / "protocol.json")["families"]
            for name, result in results.items():
                for family in names:
                    members = [cid for cid in ids if names[(int(cid.split('_')[-1]) - 1) % 6] == family]
                    score = aggregate_scores([result["per_case"][cid] for cid in members])
                    families.append({"method": name, "family": family, "f1": score["daily"]["f1"],
                                     "fp": score["daily"]["fp"], "fn": score["daily"]["fn"]})
        text.extend(["## " + target, "", full_table(summary), ""])
        print("Replayed all selected weights and training anchors: " + target, flush=True)
    write_csv(out / "per-record-corrections.csv", all_corrections)
    write_csv(out / "family-scores.csv", families)
    write_csv(out / "inference-verification-cost.csv", cost)
    export(out / "verification.json", {"selected_weight_probability_prediction_exact_match": True,
                                       "selected_aggregate_exact_match": True, "oof_and_full_anchor_exact_match": True,
                                       "no_record_overlap": True, "odds_bound_contract": True,
                                       "missing_support_stage_contract": True, "checked_calendar_days": checked_days,
                                       "checked_training_anchor_days": training_days, "retraining": False,
                                       "new_api_calls": 0, "cost_scope": "Single local replay with contract checks; not pure inference benchmark"})
    (out / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    export(out / "manifest.json", {"files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.out.resolve())
