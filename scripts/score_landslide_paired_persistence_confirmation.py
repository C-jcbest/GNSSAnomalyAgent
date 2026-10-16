"""Replay frozen forecasts, then score one registered primary and secondary trial."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from analyze_landslide_adaptive_review import paired_record_interval
from analyze_landslide_codex_observation_reference import scoring_inputs
from analyze_landslide_core_lockbox import developer_predictions
from analyze_landslide_review_utility_v2 import verify_receipts
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import ROOT, export
from run_landslide_adaptive_review import verify_freeze
from run_landslide_ai_reference_holdout import replay, verify_bindings
from run_landslide_comparison import load_labels
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


def replay_heads(run):
    verify_bindings(run, "result-freeze.json")
    for name, expected in read_json(run / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen inference dependency changed: " + name)
    protocol = read_json(run / "protocol.json")
    calibration = ROOT / protocol["calibration"]
    verify_bindings(calibration, "calibration-freeze.json")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    models = {}
    for name, chosen in protocol["selection"].items():
        if name in ARMS:
            arm = ARMS[name]
            model = pretrained_activity_head(512 if arm["image"] else 0,
                                            36 if arm["numeric"] else 0, chosen["width"])
            model.load_state_dict(torch.load(calibration / "models" / f"{name}-width{chosen['width']}.pt",
                                             weights_only=True))
        else:
            model = XGBClassifier()
            model.load_model(calibration / "models" / f"{name}-depth{chosen['depth']}.json")
        models[name] = model
    scaling = read_json(calibration / "numeric-scaling.json")
    predictions = {name: {} for name in models}
    for cid in protocol["case_ids"]:
        saved = read_json(run / (cid + "-probabilities.json"))
        with np.load(run / (cid + "-inputs.npz")) as inputs:
            normalized = numeric_input(inputs["numbers"], scaling)
            for name, model in models.items():
                if name in ARMS:
                    probability = probabilities(model, modality_inputs(inputs["embeddings"], normalized, ARMS[name]))
                else:
                    matrix = inputs["numbers"]
                    if name == "xgb_expert24":
                        matrix = np.column_stack((matrix, inputs["expert_features"]))
                    elif name == "xgb_chart384":
                        matrix = np.column_stack((matrix, inputs["chart_descriptors"]))
                    probability = model.predict_proba(matrix)[:, 1]
                np.testing.assert_array_equal(probability, saved[name])
                predictions[name][cid] = calibrated_prediction(
                    probability, inputs["supported"], inputs["codes"], inputs["stage_support"],
                    protocol["selection"][name]["threshold"]).to_dict()
    if predictions != read_json(run / "new-predictions.json"):
        raise ValueError("Frozen ten-head forecast replay differs")
    return predictions


def corrections(labels, predictions):
    rows = []
    for name, cases in predictions.items():
        if name == "visual_numeric_pair":
            continue
        for cid, value in cases.items():
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            before = np.array(value["activity"])
            after = np.array(predictions["visual_numeric_pair"][cid]["activity"])
            known = actual >= 0
            gained = int((known & (before != actual) & (after == actual)).sum())
            lost = int((known & (before == actual) & (after != actual)).sum())
            rows.append({"control": name, "case_id": cid, "gained": gained, "lost": lost, "net": gained - lost})
    return rows


def decision(summary, correction_rows):
    table = {row["method"]: row for row in summary}
    main = table["visual_numeric_pair"]
    plain, augment, shuffled, numeric = [table[name] for name in (
        "visual_numeric_plain", "visual_numeric_augmentation", "visual_numeric_shuffled_pairs", "numeric_pair")]
    checks = {
        "f1_nondecrease_augmentation": main["f1"] >= augment["f1"],
        "f1_nondecrease_shuffled": main["f1"] >= shuffled["f1"],
        "f1_gain_plain_005": main["f1"] - plain["f1"] >= .005,
        "strict_pairing_gain": main["f1"] > augment["f1"] or main["event_f1_iou03"] > augment["event_f1_iou03"],
        "weak_nondecrease": main["weak_recall"] >= max(plain["weak_recall"], augment["weak_recall"]),
        "visual_value_vs_numeric": main["f1"] >= numeric["f1"] or (
            main["event_f1_iou03"] > numeric["event_f1_iou03"] and main["f1"] >= numeric["f1"] - .01),
    }
    for control in ("visual_numeric_plain", "visual_numeric_augmentation"):
        rows = [row for row in correction_rows if row["control"] == control]
        checks["net_" + control] = sum(row["net"] for row in rows) > 0
        checks["two_records_" + control] = sum(row["net"] > 0 for row in rows) >= 2
    return {"decision": "CONFIRMED_CONTROLLED_PAIR_MECHANISM" if all(checks.values()) else
            "NOT_CONFIRMED_PAIR_MECHANISM", "checks": checks, "passed": sum(checks.values()),
            "total": len(checks), "field_accuracy_verified": False}


def score(data, heads, vlm, out):
    verify_bindings(data, "data-freeze.json")
    verify_freeze(vlm)
    verify_bindings(vlm, "prediction-freeze.json")
    if read_json(vlm / "state.json")["status"] != "predictions_complete":
        raise ValueError("Freeze all registered VLM forecasts before opening process labels")
    reference = data / "codex-review/compiled/reference-freeze.json"
    for name, expected in read_json(reference)["files"].items():
        if sha(reference.parent / name) != expected:
            raise ValueError("Pre-inference observation reference changed")
    if sha(reference) != read_json(heads / "protocol.json")["review_binding"]:
        raise ValueError("Neural inference reference binding changed")
    if sha(reference) != read_json(vlm / "developer-reference-binding.json")["sha256"]:
        raise ValueError("VLM inference reference binding changed")
    predictions = replay_heads(heads)
    receipts = verify_receipts(vlm)
    old, used, scopes = replay(vlm, receipts)
    predictions.update(old)
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    export(out / "pre-score-prediction-freeze.json", {
        "new_head_prediction_sha256": sha(heads / "new-predictions.json"),
        "vlm_prediction_freeze_sha256": sha(vlm / "prediction-freeze.json"),
        "reference_freeze_sha256": sha(reference), "source_sha256": sha(Path(__file__))})
    export(out / "predictions.json", predictions)
    export(out / "used-requests.json", used)
    export(out / "scopes.json", scopes)
    protocol = read_json(data / "protocol.json")
    features = {cid: load_case_features(data, cid) for cid in protocol["case_ids"]}
    primary = load_labels(data / "mechanism-sealed/reference-labels.csv",
                          {cid: pair[0] for cid, pair in features.items()})
    observed = {cid: pair[1].observed for cid, pair in features.items()}
    weak = {cid: np.array([int(row["activity_label"]) == 1 and int(row["weak_signal"]) == 1 for row in rows])
            for cid, rows in primary.items()}
    ai, ai_observed, _ = scoring_inputs(data / "codex-review")
    ai_weak = {cid: np.zeros(len(rows), dtype=bool) for cid, rows in ai.items()}
    spans = read_json(data / "codex-review/activity-decisions.json")["weak_interiors_authored_before_predictions"]
    for cid, rows in ai.items():
        for start, stop in spans.get(cid, []):
            ai_weak[cid][start:stop] = True
        ai_weak[cid] &= np.array([int(row["activity_label"]) for row in rows]) == 1
    secondary = {name: {cid: values[cid] for cid in ai} for name, values in predictions.items()}
    matched = developer_predictions(data, {cid: features[cid] for cid in ai}, {})
    secondary.update({"observable_target__" + name: values for name, values in matched.items()})
    final_decision = None
    for target, labels, masks, weak_masks, values in (
        ("primary_process", primary, observed, weak, predictions),
        ("secondary_observation", ai, ai_observed, ai_weak, secondary),
    ):
        folder = out / target
        folder.mkdir()
        target_used = {name: {cid: keys for cid, keys in cases.items() if cid in labels}
                       for name, cases in used.items()}
        results, summary, records = score_target(labels, masks, weak_masks, values, target_used, receipts)
        correction_rows = corrections(labels, values)
        intervals = {name: paired_record_interval(results, name, "visual_numeric_pair")
                     for name in ("visual_numeric_plain", "visual_numeric_augmentation", "numeric_pair", "xgb_expert24")}
        for name, value in (("results.json", results), ("summary.json", summary),
                            ("reference-labels.json", labels), ("predictions.json", values),
                            ("corrections.json", correction_rows), ("descriptive-bootstrap.json", intervals)):
            export(folder / name, value)
        write_csv(folder / "summary.csv", summary)
        write_csv(folder / "per-record.csv", records)
        if target == "primary_process":
            final_decision = decision(summary, correction_rows)
        columns = ("method", "precision", "recall", "f1", "weak_recall", "event_f1_iou03",
                   "event_f1_iou05", "onset_core_mae_days", "offset_core_mae_days",
                   "stage_internal_macro_f1", "stage_end_to_end_macro_f1", "coverage", "requests", "reported_tokens")
        text = ["# " + target + "：完整冻结方法对照", "",
                "受控过程恢复与开发者观测敏感性分别报告；当前复核非独立专家，过程头对观测任务仅作迁移敏感性。", "",
                "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in summary:
            text.append("| " + " | ".join(f"{row[key]:.6f}" if isinstance(row[key], float) else str(row[key])
                                           for key in columns) + " |")
        (folder / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
        selected = [row for row in summary if row["method"] in (
            "visual_numeric_pair", "visual_numeric_augmentation", "visual_numeric_plain", "numeric_pair",
            "xgb_expert24", "direct_visual_r0", "fixed_image_review_r0__whole_target")]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        names = [row["method"] for row in selected]
        for axis, metric in zip(axes, ("f1", "event_f1_iou03"), strict=True):
            axis.barh(names, [row[metric] for row in selected])
            axis.set_xlim(0, 1)
            axis.set_title(metric)
            axis.tick_params(axis="y", labelsize=7)
        figure.suptitle(target + " | Pre-frozen new records; all outcomes retained")
        figure.tight_layout()
        figure.savefig(folder / "comparison.png", dpi=140)
        plt.close(figure)
    export(out / "decision.json", final_decision)
    generation = read_json(data / "mechanism-sealed/generation-ledger.json")
    family_rows = []
    for family in protocol["generation"]["families"]:
        ids = [row["case_id"] for row in generation if row["truth"]["family"] == family]
        if len(ids) != 4:
            raise ValueError("Registered family population changed")
        family_used = {name: {cid: keys for cid, keys in cases.items() if cid in ids}
                       for name, cases in used.items()}
        _, family_summary, _ = score_target(
            {cid: primary[cid] for cid in ids}, {cid: observed[cid] for cid in ids},
            {cid: weak[cid] for cid in ids},
            {name: {cid: cases[cid] for cid in ids} for name, cases in predictions.items()},
            family_used, receipts)
        family_rows.extend(dict(row, family=family, records=4) for row in family_summary)
    write_csv(out / "primary-family-summary.csv", family_rows)
    export(out / "audit.json", {"full_primary_records": 24, "secondary_records": 12,
        "primary_methods": len(predictions), "secondary_methods": len(secondary),
        "new_head_forecast_replay": "exact", "api_responses": len(receipts),
        "api_starts": len(list((vlm / "requests").glob("*.started.json"))),
        "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts.values()),
        "reference_before_all_inference": True, "field_accuracy_verified": False})
    export(out / "manifest.json", {"files": {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()}})
    print(final_decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "heads", "vlm", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    score(arguments.data.resolve(), arguments.heads.resolve(), arguments.vlm.resolve(), arguments.out.resolve())
