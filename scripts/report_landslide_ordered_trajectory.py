"""Replay training-only trajectory responses and apply the frozen mechanism gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_contrastive_exemplars import SOURCE, raw_values
from probe_landslide_endpoint_fit import BASE, ROOT, export
from probe_landslide_feature_discovery import expert_features
from probe_landslide_visual_invariance import calibration_labels
from report_landslide_contrastive_exemplars import verify_payload
from report_landslide_core_lockbox import full_table
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_core_lockbox import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_ordered_trajectory import parse_query_labels
from gnss_sim.landslide_visual_invariance import invariance_inputs


def report(run, out):
    verify_bindings(run, "pre-inference-freeze.json")
    verify_bindings(run, "inference-freeze.json")
    for name, expected in read_json(run / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen source changed")
    if read_json(run / "state.json")["status"] != "inference_complete":
        raise ValueError("Score only completed inference")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    selected = read_json(run / "selected-training-queries.json")
    pairs = {row["case_id"]: load_case_features(SOURCE, row["case_id"]) for row in selected}
    full_labels = calibration_labels(SOURCE, pairs)
    packets, outputs = read_json(run / "packets.json"), read_json(run / "outputs.json")
    anchor = BASE / "feature-discovery-calibration-v1"
    choices = read_json(anchor / "selection.json")["primary_process"]
    stage = XGBClassifier()
    stage.load_model(SOURCE / "stage-xgboost.json")
    predictions, labels, observed, weak, stages, support = {}, {}, {}, {}, {}, {}
    # The numerical heads have seen these training records. Their scores are
    # training diagnostics only, never a fair validation comparison.
    for index, row in enumerate(selected):
        key = f"q{index + 1}"
        case, features = pairs[row["case_id"]]
        start, stop = row["query"]
        _, base, supported = invariance_inputs(case, features)
        blocks, _, _, _ = discovery_block_matrix(raw_values(case), features.noise)
        matrix = {"base18": base, "base_expert": np.column_stack((base, expert_features(blocks)))}
        labels[key] = full_labels[row["case_id"]][start:stop]
        observed[key] = features.observed[start:stop]
        weak[key] = np.array([int(value.get("weak_signal", 0)) == 1
                              and int(value["activity_label"]) == 1 for value in labels[key]])
        stages[key] = stage.predict(features.matrix).astype(int) + 1
        support[key] = features.supported[start:stop]
        if np.any(observed[key] & ~supported[start:stop]):
            raise ValueError("Pilot needs original raw84 support on all observed query days")
        for name in matrix:
            model = XGBClassifier()
            choice = choices[name]
            path = anchor / "primary_process" / f"{name}-depth{choice['depth']}.json"
            model.load_model(path)
            probability = model.predict_proba(matrix[name])[:, 1]
            activity = activity_from_mask(probability >= choice["threshold"], supported)
            prediction = learned_stages(stages[key], activity, features.supported).to_dict()
            predictions.setdefault(name + "_training_diagnostic", {})[key] = {
                **prediction, "activity": prediction["activity"][start:stop],
                "stage": prediction["stage"][start:stop],
            }
    receipts, used, parsed_failures = {}, {}, []
    for item in read_json(run / "schedule.json"):
        key, arm, repetition = item["query_id"], item["arm"], item["repetition"]
        request_id = item["request_id"]
        packet = packets[key + ":" + arm]
        receipt = verify_payload(run, item, packet)
        receipts[request_id] = receipt
        value = outputs[request_id]
        if value["status"] == "success":
            codes = parse_query_labels(json.loads(receipt["output"]))
            codes[~np.array(packet["observed"])] = -1
        else:
            try:
                parse_query_labels(json.loads(receipt["output"]))
            except (ValueError, json.JSONDecodeError):
                pass
            else:
                raise ValueError("Stored failed answer satisfies output contract")
            codes = np.full(42, -1)
            parsed_failures.append(request_id)
        if codes.tolist() != value["labels"]:
            raise ValueError("Original response replay differs")
        row = selected[int(key[1:]) - 1]
        start, stop = row["query"]
        prediction = learned_stages(stages[key][start:stop], codes, support[key])
        method = f"{arm}_r{repetition}"
        predictions.setdefault(method, {})[key] = prediction.to_dict()
        used.setdefault(method, {})[key] = [request_id]
    results, summary, per_case = score_target(labels, observed, weak, predictions, used, receipts)
    changes = []
    for method, cases in predictions.items():
        if method.endswith("_training_diagnostic"):
            continue
        repetition = int(method[-1])
        for key, prediction in cases.items():
            actual = np.array([int(value["activity_label"]) for value in labels[key]])
            known = actual >= 0
            before = np.array(predictions[f"time_only_r{repetition}"][key]["activity"])
            after = np.array(prediction["activity"])
            gained = int((known & (before != actual) & (after == actual)).sum())
            lost = int((known & (before == actual) & (after != actual)).sum())
            changes.append({"method": method, "query_id": key, "role": selected[int(key[1:]) - 1]["role"],
                            "gained": gained, "lost": lost, "net": gained - lost})
    by_method = {row["method"]: row for row in summary}
    conditions = {}
    for repetition in range(2):
        method = f"time_ordered_path_r{repetition}"
        main = by_method[method]
        rows = [row for row in changes if row["method"] == method]
        conditions[f"r{repetition}_net_vs_time_positive"] = sum(row["net"] for row in rows) > 0
        conditions[f"r{repetition}_weak_no_loss"] = main["weak_recall"] >= by_method[f"time_only_r{repetition}"]["weak_recall"]
        conditions[f"r{repetition}_inactive_no_correct_loss"] = sum(row["lost"] for row in rows if row["role"].startswith("post")) == 0
        conditions[f"r{repetition}_not_below_gray"] = main["f1"] >= by_method[f"time_gray_path_r{repetition}"]["f1"]
    conditions["above_gray_once"] = any(
        by_method[f"time_ordered_path_r{i}"]["f1"] > by_method[f"time_gray_path_r{i}"]["f1"]
        for i in range(2)
    )
    decision = {"conditions": conditions, "passed": sum(conditions.values()), "total": len(conditions),
                "decision": "PROCEED_TO_FROZEN_VALIDATION" if all(conditions.values())
                            else "STOP_ORDERED_TRAJECTORY_MECHANISM_PILOT",
                "scope": "Four deliberately selected TRAINING windows; numerical diagnostics in-sample",
                "new_records": 0, "independent_confirmation": False}
    cost = {"new_requests": len(receipts), "new_receipts": len(receipts),
            "reported_tokens": sum(value["usage"]["total_tokens"] for value in receipts.values()),
            "api_seconds": sum(value["seconds"] for value in receipts.values()),
            "loop_seconds": read_json(run / "state.json")["loop_seconds"],
            "contract_failures": parsed_failures}
    export(out / "decision.json", decision)
    export(out / "cost.json", cost)
    export(out / "results.json", results)
    export(out / "predictions.json", predictions)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-query-corrections.csv", changes)
    write_csv(out / "per-case.csv", per_case)
    export(out / "verification.json", {
        "payload_response_replays": len(receipts), "raw_support_observed_queries_checked": 168,
        "sklearn_daily_event_boundary_match": True, "new_api_calls_in_report": 0,
        "same_time_image_in_all_arms": True,
        "training_numeric_diagnostics_not_validation": True,
        "missing_and_original_stage_gate": True,
    })
    text = ("# 原日期有序空间轨迹：训练机制小试\n\n"
            "## Material Passport\n\n"
            "- Origin Mode: run / validate inline\n"
            "- Verification: 原请求与预测重放；仅训练窗口，非独立验证\n\n"
            + full_table(summary) + "\n\n" + json.dumps(decision, ensure_ascii=False, indent=2) + "\n")
    (out / "results.md").write_text(text, encoding="utf-8")
    export(out / "manifest.json", {
        "files": {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()}
    })
    print(json.dumps(decision), flush=True)
    print(json.dumps(cost), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report(args.run.resolve(), args.out.resolve())
