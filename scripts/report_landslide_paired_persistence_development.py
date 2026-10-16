"""Replay complete development and retain original VLM deployment accounting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from analyze_landslide_adaptive_review import paired_record_interval
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import ROOT, csv_rows, export
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_paired_persistence_development import DATA, OLD
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_paired_persistence import (
    ARMS,
    calibrated_prediction,
    modality_inputs,
    probabilities,
)
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_paired_persistence import pretrained_activity_head
from gnss_sim.landslide_visual_invariance import numeric_input


def report(run, out):
    verify_bindings(run, "result-freeze.json")
    for name, expected in read_json(run / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen development dependency changed: " + name)
    protocol = read_json(run / "protocol.json")
    calibration = ROOT / protocol["calibration"]
    verify_bindings(calibration, "calibration-freeze.json")
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    scaling = read_json(calibration / "numeric-scaling.json")
    new_names = list(ARMS) + ["xgb_raw18", "xgb_expert24", "xgb_chart384"]
    models = {}
    for name in new_names:
        chosen = protocol["selection"][name]
        if name in ARMS:
            arm = ARMS[name]
            model = pretrained_activity_head(512 if arm["image"] else 0, 36 if arm["numeric"] else 0, chosen["width"])
            model.load_state_dict(torch.load(calibration / "models" / f"{name}-width{chosen['width']}.pt", weights_only=True))
        else:
            model = XGBClassifier()
            model.load_model(calibration / "models" / f"{name}-depth{chosen['depth']}.json")
        models[name] = model
    predictions = {name: {} for name in new_names}
    observed = {}
    checked = 0
    for cid in protocol["case_ids"]:
        stored_probability = read_json(run / (cid + "-probabilities.json"))
        with np.load(run / (cid + "-inputs.npz")) as data:
            observed[cid] = data["observed"]
            normalized = numeric_input(data["numbers"], scaling)
            for name, model in models.items():
                if name in ARMS:
                    probability = probabilities(model, modality_inputs(data["embeddings"], normalized, ARMS[name]))
                else:
                    matrix = data["numbers"]
                    if name == "xgb_expert24":
                        matrix = np.column_stack((matrix, data["expert_features"]))
                    elif name == "xgb_chart384":
                        matrix = np.column_stack((matrix, data["chart_descriptors"]))
                    probability = model.predict_proba(matrix)[:, 1]
                np.testing.assert_array_equal(probability, stored_probability[name])
                prediction = calibrated_prediction(probability, data["supported"], data["codes"], data["stage_support"],
                                                     protocol["selection"][name]["threshold"])
                if np.any(prediction.activity[~data["observed"]] != -1) or np.any(prediction.stage[~data["observed"]] != -1):
                    raise ValueError("Weight replay filled missing dates")
                predictions[name][cid] = prediction.to_dict()
                checked += len(probability)
    if predictions != read_json(run / "new-predictions.json"):
        raise ValueError("Development weight prediction replay differs")
    predictions.update(read_json(OLD / "predictions.json"))
    labels = read_json(OLD / "reference-labels.json")
    weak = {cid: np.array([int(row["activity_label"]) == 1 and int(row["weak_signal"]) == 1 for row in rows])
            for cid, rows in labels.items()}
    results, summary, records = score_target(labels, observed, weak, predictions, {}, {})
    if results != read_json(run / "results.json") or summary != read_json(run / "summary.json"):
        raise ValueError("Complete frozen development scores differ")
    old_results = read_json(OLD / "results.json")
    for name, result in old_results.items():
        if results[name] != result:
            raise ValueError("Existing baseline task metrics changed: " + name)
    # Score replay consumes no API. Public deployment accounting comes from the
    # original response-backed experiment, not from the empty replay receipts.
    old_summary = {row["method"]: row for row in csv_rows(OLD / "summary.csv")}
    for row in summary:
        name = row["method"]
        if name in old_summary:
            row["requests"] = int(old_summary[name]["requests"])
            row["reported_tokens"] = int(old_summary[name]["reported_tokens"])
        row["replay_new_requests"] = 0
    export(out / "summary.json", summary)
    export(out / "results.json", results)
    export(out / "predictions.json", predictions)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", records)
    comparisons = ["visual_numeric_plain", "visual_numeric_augmentation", "numeric_pair", "xgb_expert24"]
    intervals = {name: paired_record_interval(results, name, "visual_numeric_pair") for name in comparisons}
    export(out / "descriptive-record-bootstrap.json", intervals)
    export(out / "verification.json", {"selected_probabilities": "exact", "new_weight_predictions": "exact",
        "common_scores": "exact", "original37_baseline_scores": "exact", "checked_original_days": checked,
        "retraining": False, "new_api_calls": 0, "original_deployment_costs_retained": True,
        "independent_confirmation": False})
    corrections = read_json(run / "corrections.json")
    cases = []
    for control in ("visual_numeric_augmentation", "numeric_pair"):
        relevant = [row for row in corrections if row["control"] == control]
        for reverse in (False, True):
            for row in sorted(relevant, key=lambda row: (row["net"], row["case_id"]), reverse=reverse):
                if row["case_id"] not in cases:
                    cases.append(row["case_id"])
                    break
    export(out / "diagnostic-selection.json", {"case_ids": cases, "rule": "Post-score extrema, complete population unchanged"})
    names = ["visual_numeric_pair", "visual_numeric_augmentation", "visual_numeric_plain", "numeric_pair", "xgb_expert24",
             "direct_visual_r0", "fixed_image_review_r0__whole_target"]
    for cid in cases:
        case = load_case_features(DATA, cid)[0]
        values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        figure, panels = plt.subplots(4, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1.6]})
        for axis in range(3):
            panels[axis].plot(values[:, axis], linewidth=.65)
            panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
            panels[axis].grid(alpha=.2)
        matrix = [[int(row["activity_label"]) for row in labels[cid]]]
        matrix.extend(predictions[name][cid]["activity"] for name in names)
        panels[3].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
            cmap=ListedColormap(["#bbbbbb", "#fafafa", "#227e8b"]), extent=(-.5, len(values) - .5, len(matrix) - .5, -.5))
        panels[3].set_yticks(range(len(matrix)), ["Controlled reference"] + names, fontsize=7)
        panels[3].set_xlabel("Original day | gray unknown, white stationary, teal continuing activity")
        figure.suptitle(cid + " | Complete exposed development diagnostics")
        figure.tight_layout()
        figure.savefig(out / (cid + "-diagnostic.png"), dpi=120)
        plt.close(figure)
    columns = ("method", "precision", "recall", "f1", "event_f1_iou03", "weak_recall",
               "onset_core_mae_days", "offset_core_mae_days", "requests", "reported_tokens")
    text = ["# 持续性配对完整24条开发：冻结权重重放及统一对照", "",
            "决定：" + read_json(run / "decision.json")["decision"], "",
            "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in summary:
        text.append("| " + " | ".join(f"{row[key]:.6f}" if isinstance(row[key], float) else str(row[key]) for key in columns) + " |")
    text += ["", "37原方法任务分数逐值复现，API部署成本保留原回执支持的summary，重放新调用0不冒充旧VLM免费。",
             "完整过程参考已参与历史方法探索，因此仅开发。bootstrap按完整记录描述，不合并相关日期/两遍为独立样本。", ""]
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")
    export(out / "manifest.json", {"files": {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()}})
    print("Replayed", len(predictions), "complete-population methods and retained original deployment costs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.run.resolve(), arguments.out.resolve())
