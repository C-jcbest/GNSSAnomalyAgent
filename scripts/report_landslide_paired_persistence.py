"""Replay terminal frozen weights, common scoring and diagnostic figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from encode_landslide_paired_persistence import verify_inputs
from matplotlib.colors import ListedColormap
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_contrastive_exemplars import SOURCE, raw_values
from probe_landslide_endpoint_fit import ROOT, export
from probe_landslide_feature_discovery import expert_features
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import predict_baselines
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
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_paired_persistence import pretrained_activity_head
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors, numeric_input


def report(run, out):
    verify_inputs(run)
    verify_bindings(run, "encoding-freeze.json")
    verify_bindings(run, "calibration-freeze.json")
    for name, expected in read_json(run / "learning-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Learning dependency changed: " + name)
    if read_json(run / "state.json")["status"] != "calibration_complete":
        raise ValueError("Replay only terminal frozen calibration")
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    labels = read_json(run / "validation-labels.json")
    pairs = {cid: load_case_features(SOURCE, cid) for cid in labels}
    selection = read_json(run / "selection.json")
    scaling = read_json(run / "numeric-scaling.json")
    stage_head = XGBClassifier()
    stage_head.load_model(SOURCE / "stage-xgboost.json")
    codes = {cid: stage_head.predict(value[1].matrix).astype(int) + 1 for cid, value in pairs.items()}
    banks = {}
    for cid, (case, features) in pairs.items():
        with np.load(run / "original/validation" / (cid + ".npz")) as data:
            bank = {key: data[key] for key in ("numbers", "supported", "observed", "weak")}
            bank["chart"] = chart_numeric_descriptors(data["images"])
        with np.load(run / "encoded/validation" / (cid + ".npz")) as data:
            bank["embeddings"] = data["embeddings"]
        bank["expert"] = expert_features(discovery_block_matrix(raw_values(case), features.noise)[0])
        banks[cid] = bank
    predictions = {}
    checked_probabilities = checked_days = 0
    for name in list(ARMS) + ["xgb_raw18", "xgb_expert24", "xgb_chart384"]:
        chosen = selection[name]
        if name in ARMS:
            arm = ARMS[name]
            visual_width = 512 if arm["image"] else 0
            numeric_width = 36 if arm["numeric"] else 0
            model = pretrained_activity_head(visual_width, numeric_width, chosen["width"])
            model.load_state_dict(torch.load(run / "models" / f"{name}-width{chosen['width']}.pt", weights_only=True))
            probability = {cid: probabilities(model, modality_inputs(bank["embeddings"],
                            numeric_input(bank["numbers"], scaling), arm)) for cid, bank in banks.items()}
            parameter = "width"
        else:
            model = XGBClassifier()
            model.load_model(run / "models" / f"{name}-depth{chosen['depth']}.json")
            fields = ["numbers"]
            if name == "xgb_expert24":
                fields.append("expert")
            elif name == "xgb_chart384":
                fields.append("chart")
            probability = {cid: model.predict_proba(np.column_stack([bank[field] for field in fields]))[:, 1]
                           for cid, bank in banks.items()}
            parameter = "depth"
        stored = read_json(run / "probabilities" / f"{name}-{parameter}{chosen[parameter]}.json")
        for cid, values in probability.items():
            np.testing.assert_array_equal(values, stored[cid])
            checked_probabilities += len(values)
        predictions[name] = {cid: calibrated_prediction(probability[cid], bank["supported"], codes[cid],
                             pairs[cid][1].supported, chosen["threshold"]).to_dict() for cid, bank in banks.items()}
    original_head = XGBClassifier()
    original_head.load_model(SOURCE / "activity-xgboost.json")
    original = predict_baselines({cid: value[1] for cid, value in pairs.items()}, list(labels),
                                read_json(SOURCE / "frozen-selection.json"), original_head, stage_head)
    for name in ("robust_rule", "pelt_rule", "xgboost"):
        predictions["original_" + name] = {cid: value.to_dict() for cid, value in original[name].items()}
    frozen = read_json(run / "predictions.json")
    if predictions != frozen:
        raise ValueError("Frozen weight replay predictions differ")
    for cases in predictions.values():
        for cid, value in cases.items():
            activity, stage = np.asarray(value["activity"]), np.asarray(value["stage"])
            if np.any(activity[~banks[cid]["observed"]] != -1) or np.any(stage[~banks[cid]["observed"]] != -1):
                raise ValueError("Replay filled raw missing dates")
            if np.any(stage[(activity == -1)] != -1) or np.any(stage[activity == 0] != 0):
                raise ValueError("Replay broke activity-first stage gate")
            checked_days += len(activity)
    results, summary, records = score_target(labels, {cid: bank["observed"] for cid, bank in banks.items()},
        {cid: bank["weak"] for cid, bank in banks.items()}, predictions, {}, {})
    if results != read_json(run / "results.json") or summary != read_json(run / "validation-summary.json"):
        raise ValueError("Common scoring differs on frozen replay")
    export(out / "predictions.json", predictions)
    export(out / "results.json", results)
    export(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", records)
    export(out / "verification.json", {"weight_predictions": "exact", "selected_probabilities": "exact",
        "common_scores": "exact", "checked_probabilities": checked_probabilities,
        "checked_original_days": checked_days, "missing_and_activity_stage_gate": "passed",
        "retraining": False, "new_api_calls": 0, "independent_confirmation": False})
    corrections = read_json(run / "corrections.json")
    chosen_cases = []
    for control in ("visual_numeric_plain", "visual_numeric_augmentation", "numeric_pair"):
        relevant = [row for row in corrections if row["control"] == control]
        for reverse in (False, True):
            for row in sorted(relevant, key=lambda row: (row["net"], row["case_id"]), reverse=reverse):
                if row["case_id"] not in chosen_cases:
                    chosen_cases.append(row["case_id"])
                    break
        if len(chosen_cases) >= 4:
            break
    chosen_cases = chosen_cases[:4]
    export(out / "diagnostic-selection.json", {"rule": "Post-calibration extrema of net corrections, not an evaluation subset",
                                               "case_ids": chosen_cases})
    names = ["visual_numeric_pair", "visual_numeric_augmentation", "visual_numeric_plain", "numeric_pair", "xgb_expert24"]
    for cid in chosen_cases:
        figure, panels = plt.subplots(4, 1, figsize=(12, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1, 1, 1.4]})
        values = raw_values(pairs[cid][0])
        for axis in range(3):
            panels[axis].plot(values[:, axis], linewidth=.65)
            panels[axis].set_ylabel(("N", "E", "U")[axis] + " (mm)")
            panels[axis].grid(alpha=.2)
        matrix = [[int(row["activity_label"]) for row in labels[cid]]]
        matrix.extend(predictions[name][cid]["activity"] for name in names)
        panels[3].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1,
                         cmap=ListedColormap(["#bbbbbb", "#fafafa", "#227e8b"]),
                         extent=(-.5, len(values) - .5, len(matrix) - .5, -.5))
        panels[3].set_yticks(range(len(matrix)), ["Controlled reference"] + names, fontsize=7)
        panels[3].set_xlabel("Original day | gray unknown, white stationary, teal continuing activity")
        figure.suptitle(cid + " | Original observations and post-calibration diagnostics")
        figure.tight_layout()
        figure.savefig(out / (cid + "-diagnostic.png"), dpi=120)
        plt.close(figure)
    decision = read_json(run / "calibration-decision.json")
    columns = ("method", "precision", "recall", "f1", "event_f1_iou03", "weak_recall",
               "onset_core_mae_days", "offset_core_mae_days", "stage_internal_macro_f1", "coverage")
    text = ["# 同净位移持续运动／单次偏移配对：冻结校准与重放", "",
            "决定：" + decision["decision"] + "。仅原12条验证校准，非独立观测结论。", "",
            "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in summary:
        text.append("| " + " | ".join(str(row[key]) if not isinstance(row[key], float)
                                         else f"{row[key]:.6f}" for key in columns) + " |")
    text += ["", "所有学习臂共享原活动/阶段门控；原规则/PELT保留各自既有支撑，覆盖单独报告。",
             "配对只用于原训练parent，新增137负中心不是137独立记录。数值强基线同样获得这些负中心。",
             "阶段来自既有共同数值头，无原生视觉阶段创新。全部校准机会、成本和失败决定保存在原run。",
             "诊断图按净纠错极值选择，只作错误解释，不替换完整评分总体。", ""]
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")
    export(out / "manifest.json", {"files": {path.relative_to(out).as_posix(): sha(path)
        for path in sorted(out.rglob("*")) if path.is_file()}})
    print("Replayed", len(predictions), "methods; exact predictions, probabilities and scores", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.run.resolve(), arguments.out.resolve())
