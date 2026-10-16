"""Replay frozen validation weights and explain the registered calibration gate."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import csv_rows, export
from probe_landslide_visual_invariance import (
    ARMS,
    TRAINING,
    calibration_labels,
    probabilities,
    verify,
)
from report_landslide_core_lockbox import full_table
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
    reflect_chart_axes,
)


def corrections(labels, predictions):
    """Count definite correctness; reference unknown never earns credit."""
    records = []
    for control in predictions:
        if control == "chart_consistency":
            continue
        for cid, value in predictions[control].items():
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            known = actual >= 0
            before = np.array(value["activity"])
            after = np.array(predictions["chart_consistency"][cid]["activity"])
            gained = known & (before != actual) & (after == actual)
            lost = known & (before == actual) & (after != actual)
            records.append(
                {"control": control, "case_id": cid, "gained": int(gained.sum()),
                 "lost": int(lost.sum()), "net": int(gained.sum() - lost.sum()),
                 "gained_spans": runs(gained), "lost_spans": runs(lost)}
            )
    return records


def diagnostic_figure(case, labels, predictions, probability, path, target):
    """Show original observations before the already scored output alignment."""
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    days = np.arange(len(values))
    figure, panels = plt.subplots(
        5, 1, figsize=(14, 11), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 0.9, 1.4]},
    )
    for axis, component in enumerate(("N", "E", "U")):
        panels[axis].plot(days, values[:, axis], linewidth=0.7)
        panels[axis].set_ylabel(component + " (mm)")
        panels[axis].grid(alpha=0.2)
    names = list(predictions)
    for name in names:
        panels[3].plot(days, probability[name], linewidth=0.8, label=name)
    panels[3].set_ylabel("Activity probability")
    panels[3].legend(fontsize=7, ncol=3)
    matrix = [[int(row["activity_label"]) for row in labels]]
    matrix.extend(predictions[name]["activity"] for name in names)
    panels[4].imshow(
        matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
        cmap=ListedColormap(["#bcbcbc", "#fafafa", "#277e8c"]),
        extent=(-0.5, len(values) - 0.5, len(matrix) - 0.5, -0.5),
    )
    panels[4].set_yticks(range(len(matrix)), ["Reference"] + names, fontsize=8)
    panels[4].set_xlabel("Original day | teal=activity,white=stationary,gray=unknown")
    panels[4].set_xlim(0, len(values) - 1)
    figure.suptitle(case.case_id + " | " + target + " | post-calibration diagnostic")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def choose_diagnostics(records, count):
    """Disclose extrema selection without changing the population or gate."""
    chosen = []
    for control in ("chart_plain", "numeric_mlp", "chart_augmentation"):
        candidates = [row for row in records if row["control"] == control]
        for reverse in (False, True):
            for row in sorted(candidates, key=lambda row: (row["net"], row["case_id"]), reverse=reverse):
                if row["case_id"] not in chosen:
                    chosen.append(row["case_id"])
                    break
            if len(chosen) >= count:
                return chosen
    return chosen


def report(run, out):
    verify(run)
    verify_bindings(run, "calibration-freeze.json")
    if read_json(run / "state.json")["status"] != "calibration_complete":
        raise ValueError("Only terminal frozen calibration weights may be replayed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol = read_json(run / "protocol.json")
    selection = read_json(run / "selection.json")
    reflection_rows = csv_rows(run / "validation-reflection.csv")
    all_corrections, costs, reflection = [], [], []
    verification = {"frozen_weight_prediction_match": True, "frozen_aggregate_match": True,
                    "missing_and_stage_contract": True, "checked_calendar_days": 0,
                    "new_api_calls": 0, "retraining": False}
    text = ["# 坐标反射训练一致性：冻结校准与权重重放", "",
            "决定：" + read_json(run / "calibration-decision.json")["decision"], "",
            "过程12条验证、作者6条验证使用原划分；仅校准/开发证据，非独立现场确认。"
            "本脚本不训练、不改变已冻结参数。全部臂均报告，未知参考不计正确静稳。", ""]
    for target, source in TRAINING.items():
        folder = out / target
        folder.mkdir()
        ids = protocol["calibration_splits"][target]["validation"]
        pairs = {cid: load_case_features(source, cid) for cid in ids}
        labels = calibration_labels(source, pairs)
        inputs = {cid: invariance_inputs(*pairs[cid]) for cid in ids}
        scaling = read_json(run / target / "numeric-scaling.json")
        head = XGBClassifier()
        head.load_model(source / "stage-xgboost.json")
        codes = {cid: head.predict(pairs[cid][1].matrix).astype(int) + 1 for cid in ids}
        predictions, all_probability = {}, {}
        for name, selected in selection[target].items():
            started = time.perf_counter()
            if name == "xgboost_chart_information":
                model = XGBClassifier()
                model.load_model(run / target / f"xgboost-information-depth{selected['depth']}.json")
            else:
                arm = ARMS[name]
                model = activity_model(selected["width"], 36, arm["use_image"], arm["use_numeric"])
                weights = torch.load(run / target / f"{name}-width{selected['width']}.pt", weights_only=True)
                model.load_state_dict(weights)
            cases, probability = {}, {}
            case_scores = []
            for cid in ids:
                images, matrix, support = inputs[cid]
                numbers = numeric_input(matrix, scaling)
                if name == "xgboost_chart_information":
                    original = model.predict_proba(np.column_stack((matrix, chart_numeric_descriptors(images))))[:, 1]
                else:
                    original = probabilities(model, images, numbers)
                probability[cid] = original.tolist()
                activity = activity_from_mask(original >= selected["threshold"], support)
                prediction = learned_stages(codes[cid], activity, pairs[cid][1].supported)
                prediction.validate(len(labels[cid]))
                observed = pairs[cid][1].observed
                if np.any(prediction.activity[~observed] != -1) or np.any(prediction.stage[~observed] != -1):
                    raise ValueError("Missing observation gained a decision")
                if np.any(prediction.activity[~support] != -1):
                    raise ValueError("Original numerical support changed")
                verification["checked_calendar_days"] += len(labels[cid])
                cases[cid] = prediction.to_dict()
                case_scores.append(evaluate_case(labels[cid], prediction))
                differences = []
                for sign_code in range(1, 8):
                    signs = np.tile([(-1 if sign_code & (1 << axis) else 1) for axis in range(3)], (len(images), 1))
                    transformed = reflect_chart_axes(images, signs)
                    if name == "xgboost_chart_information":
                        other = model.predict_proba(np.column_stack((matrix, chart_numeric_descriptors(transformed))))[:, 1]
                    else:
                        other = probabilities(model, transformed, numbers)
                    differences.extend(np.abs(other[support] - original[support]).tolist())
                disagreement = float(np.mean(differences))
                if name != "xgboost_chart_information":
                    frozen = next(row for row in reflection_rows
                                  if row["target"] == target and row["arm"] == name
                                  and int(row["width"]) == selected["width"] and row["case_id"] == cid)
                    if disagreement != float(frozen["mean_probability_disagreement"]):
                        raise ValueError("Frozen reflection probability statistic differs")
                reflection.append({"target": target, "arm": name, "case_id": cid,
                                   "supported_days": int(support.sum()),
                                   "mean_probability_disagreement": disagreement})
            if aggregate_scores(case_scores) != selected["scores"]:
                raise ValueError("Frozen selected validation metrics differ")
            if name != "xgboost_chart_information":
                if cases != read_json(run / target / f"{name}-validation-predictions.json"):
                    raise ValueError("Frozen validation predictions differ")
            predictions[name] = cases
            all_probability[name] = probability
            costs.append({"target": target, "arm": name, "records": len(ids),
                          "original_plus_7_reflections_wall_seconds": time.perf_counter() - started})
            print(f"Replayed {target} {name}", flush=True)
        weak = {cid: np.array([row["feature"] == "slow_displacement" for row in rows]) for cid, rows in labels.items()}
        scores, summary, per_case = score_target(
            labels, {cid: pair[1].observed for cid, pair in pairs.items()}, weak, predictions, {}, {},
        )
        export(folder / "predictions.json", predictions)
        export(folder / "probabilities.json", all_probability)
        export(folder / "results.json", scores)
        write_csv(folder / "summary.csv", summary)
        write_csv(folder / "per-record.csv", per_case)
        corrected = corrections(labels, predictions)
        all_corrections.extend({"target": target, **row} for row in corrected)
        chosen = choose_diagnostics(corrected, 4 if target == "primary_process" else 2)
        export(folder / "diagnostic-selection.json",
               {"post_score_selection": "Extrema of net correctness vs plain, numeric-only and augmentation",
                "case_ids": chosen, "used_for_gate_or_labels": False})
        for cid in chosen:
            diagnostic_figure(
                pairs[cid][0], labels[cid], {name: cases[cid] for name, cases in predictions.items()},
                {name: cases[cid] for name, cases in all_probability.items()},
                folder / f"{cid}-diagnostic.png", target,
            )
            values = np.array([row if row is not None else [np.nan] * 3 for row in pairs[cid][0].displacement_mm])
            # Original, disjoint blocks provide numerical context for visual review.
            blocks = []
            for start in range(0, len(values), 14):
                sample = values[start : start + 14]
                complete = sample[np.isfinite(sample).all(axis=1)]
                median = np.median(complete, axis=0).tolist() if len(complete) >= 8 else None
                blocks.append({"start": start, "stop": min(start + 14, len(values)),
                               "complete_days": len(complete), "median_neu_mm": median})
            export(folder / f"{cid}-raw-blocks.json", blocks)
        text.extend(["## " + target, "", full_table(summary), ""])
    write_csv(out / "per-record-corrections.csv", all_corrections)
    write_csv(out / "weight-replay-cost.csv", costs)
    write_csv(out / "reflection-replay.csv", reflection)
    export(out / "verification.json", verification)
    (out / "results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    verify(run)
    verify_bindings(run, "calibration-freeze.json")
    export(out / "manifest.json", {str(path.relative_to(out)): sha(path)
                                  for path in out.rglob("*") if path.is_file()})
    print("Frozen validation replay and common scoring passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.run.resolve(), arguments.out.resolve())
