"""Replay date-aligned visual models, verify label alignment and score all arms."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import export
from probe_landslide_temporal_readout import ARMS, temporal_probabilities, verify
from probe_landslide_visual_invariance import TRAINING, calibration_labels
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
from gnss_sim.landslide_temporal_readout import (
    column_numeric_input,
    column_supervision,
    temporal_activity_model,
)
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors, invariance_inputs


def corrections(labels, predictions):
    rows = []
    for control, cases in predictions.items():
        if control == "aligned_dense_fusion":
            continue
        for cid, value in cases.items():
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            known = actual >= 0
            before = np.array(value["activity"])
            after = np.array(predictions["aligned_dense_fusion"][cid]["activity"])
            gained = known & (before != actual) & (after == actual)
            lost = known & (before == actual) & (after != actual)
            rows.append({"control": control, "case_id": cid, "gained": int(gained.sum()),
                         "lost": int(lost.sum()), "net": int(gained.sum() - lost.sum()),
                         "gained_spans": runs(gained), "lost_spans": runs(lost)})
    return rows


def choose_records(rows, count):
    chosen = []
    for control in ("numeric_dense", "xgboost_raw84"):
        records = [row for row in rows if row["control"] == control]
        for reverse in (False, True):
            for row in sorted(records, key=lambda row: (row["net"], row["case_id"]), reverse=reverse):
                if row["case_id"] not in chosen:
                    chosen.append(row["case_id"])
                    break
            if len(chosen) >= count:
                return chosen
    return chosen


def verify_original_labels(run, target, source, protocol):
    """Check every actual supervised slot against its original date and mask."""
    ids = protocol["calibration_splits"][target]["train"]
    pairs = {cid: load_case_features(source, cid) for cid in ids}
    labels = calibration_labels(source, pairs)
    total_centers = total_auxiliary = total_original = 0
    recorded = {row["case_id"]: row for row in read_json(run / target / "label-exposure.json")}
    for cid, pair in pairs.items():
        _, _, support = invariance_inputs(*pair)
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        target_values, mask = column_supervision(actual, support)
        eligible = support & (actual >= 0)
        centers = np.flatnonzero(eligible & (np.arange(len(actual)) % 2 == 0))
        dates = centers[:, None] + np.arange(-90, 91, 3)[None, :]
        calendar = (dates >= 0) & (dates < len(actual))
        indices = np.clip(dates, 0, len(actual) - 1)
        expected = calendar & support[indices] & (actual[indices] >= 0)
        if not np.array_equal(mask[centers], expected):
            raise ValueError("Dense supervision changed the original date support")
        if not np.array_equal(target_values[centers][expected], actual[indices][expected]):
            raise ValueError("A chart column was supervised by another day's activity")
        if not np.array_equal(target_values[centers, 30], actual[centers]):
            raise ValueError("The original center-day target changed")
        expected[:, 30] = False
        row = recorded[cid]
        if row["training_centers"] != len(centers) or row["auxiliary_label_occurrences"] != int(expected.sum()):
            raise ValueError("Recorded training exposure differs")
        total_centers += len(centers)
        total_auxiliary += int(expected.sum())
        total_original += int(eligible.sum())
    return {"target": target, "training_centers": total_centers, "auxiliary_label_occurrences": total_auxiliary,
            "all_original_training_dates": total_original, "original_date_values_and_masks_match": True,
            "outside_unknown_unsupported_slots_supervised": 0}


def report(run, out):
    verify(run)
    verify_bindings(run, "calibration-freeze.json")
    if read_json(run / "state.json")["status"] != "calibration_complete":
        raise ValueError("Only completed calibration may be replayed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = read_json(run / "protocol.json")
    selections = read_json(run / "selection.json")
    cost, all_corrections, families, label_contracts = [], [], [], []
    checked_days = 0
    text = ["# 逐列原日期视觉活动判断：冻结校准重放", "",
            "决定：" + read_json(run / "calibration-decision.json")["decision"], "",
            "过程12验证、作者6验证；相同24/6训练，全部7臂各38机会。开发暴露材料，非现场独立确认。", ""]
    for target, source in TRAINING.items():
        label_contracts.append(verify_original_labels(run, target, source, protocol))
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
        frozen = read_json(run / target / "validation-predictions.json")
        frozen_probability = read_json(run / target / "validation-probabilities.json")
        predictions, probability_bank = {}, {}
        for name, selected in selections[target].items():
            if name in ARMS:
                arm = ARMS[name]
                model = temporal_activity_model(selected["width"], 36, arm["readout"], arm["numeric"])
                model.load_state_dict(torch.load(run / target / f"{name}-width{selected['width']}.pt", weights_only=True))
            else:
                model = XGBClassifier()
                model.load_model(run / target / f"{name}-depth{selected['depth']}.json")
            predictions[name], probability_bank[name] = {}, {}
            scores = []
            for cid in ids:
                images, matrix, support = inputs[cid]
                started = time.perf_counter()
                if name in ARMS:
                    probability = temporal_probabilities(model, images, column_numeric_input(matrix, scaling))
                else:
                    observations = matrix
                    if name == "xgboost_chart_information":
                        observations = np.column_stack((matrix, chart_numeric_descriptors(images)))
                    probability = model.predict_proba(observations)[:, 1]
                cost.append({"target": target, "arm": name, "case_id": cid, "calendar_days": len(images),
                             "wall_seconds": time.perf_counter() - started})
                if probability.tolist() != frozen_probability[name][cid]:
                    raise ValueError("Loaded weights do not reproduce frozen probabilities")
                activity = activity_from_mask(probability >= selected["threshold"], support)
                prediction = learned_stages(codes[cid], activity, pairs[cid][1].supported)
                prediction.validate(len(images))
                if np.any(prediction.activity[~support] != -1):
                    raise ValueError("Activity escaped its original support")
                if np.any(prediction.activity[~pairs[cid][1].observed] != -1) or np.any(prediction.stage[~pairs[cid][1].observed] != -1):
                    raise ValueError("Missing observations gained a decision")
                if prediction.to_dict() != frozen[name][cid]:
                    raise ValueError("Frozen activity/stage prediction changed")
                checked_days += len(images)
                scores.append(evaluate_case(labels[cid], prediction))
                predictions[name][cid] = prediction.to_dict()
                probability_bank[name][cid] = probability.tolist()
            if aggregate_scores(scores) != selected["scores"]:
                raise ValueError("Selected frozen aggregate changed")
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
        export(folder / "diagnostic-selection.json", {"case_ids": chosen, "post_score_selection": "Net correction extrema vs numerical dense and strong rawXGB", "used_for_reference_or_gate": False})
        for cid in chosen:
            case, _ = pairs[cid]
            diagnostic_figure(case, labels[cid], {name: cases[cid] for name, cases in predictions.items()},
                              {name: cases[cid] for name, cases in probability_bank.items()},
                              folder / f"{cid}-diagnostic.png", target)
            values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
            blocks = []
            for start in range(0, len(values), 14):
                complete = values[start : start + 14]
                complete = complete[np.isfinite(complete).all(axis=1)]
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
        print("Replayed selected weights and checked original-date supervision: " + target, flush=True)
    write_csv(out / "per-record-corrections.csv", all_corrections)
    write_csv(out / "family-scores.csv", families)
    write_csv(out / "inference-cost.csv", cost)
    export(out / "training-label-contracts.json", label_contracts)
    export(out / "verification.json", {"selected_probability_prediction_exact_match": True,
                                       "selected_aggregate_exact_match": True,
                                       "training_date_labels_and_support_match": True,
                                       "outside_unknown_unsupported_training_labels": 0,
                                       "missing_support_stage_contract": True,
                                       "checked_calendar_days": checked_days,
                                       "retraining": False, "new_api_calls": 0})
    (out / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    export(out / "manifest.json", {"files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.out.resolve())
