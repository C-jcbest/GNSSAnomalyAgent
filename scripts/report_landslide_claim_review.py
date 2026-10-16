"""Replay factual-review requests, original-date predictions and the fixed gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_contrastive_exemplars import ANCHOR, SOURCE, block_packet, raw_values
from probe_landslide_endpoint_fit import ROOT, export
from probe_landslide_feature_discovery import expert_features
from probe_landslide_visual_invariance import calibration_labels
from report_landslide_contrastive_exemplars import verify_payload
from report_landslide_core_lockbox import full_table
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_claim_review import check_block_claims, parse_block_claims
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual_invariance import invariance_inputs


def report(run, out):
    verify_bindings(run, "pre-inference-freeze.json")
    verify_bindings(run, "inference-freeze.json")
    for name, expected in read_json(run / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen source changed")
    if read_json(run / "state.json")["status"] != "inference_complete":
        raise ValueError("Score a terminal pilot only")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    selected = read_json(run / "selected-training-queries.json")
    pairs = {row["case_id"]: load_case_features(SOURCE, row["case_id"]) for row in selected}
    reference = calibration_labels(SOURCE, pairs)
    packets = read_json(run / "packets.json")
    outputs = read_json(run / "outputs.json")
    prompts = read_json(run / "prompts.json")
    choices = read_json(ANCHOR / "selection.json")["primary_process"]
    stage = XGBClassifier()
    stage.load_model(SOURCE / "stage-xgboost.json")
    labels, observed, weak, phases, support = {}, {}, {}, {}, {}
    predictions, used, receipts = {}, {}, {}
    claims_report = []
    failed = []
    for index, row in enumerate(selected):
        key = f"q{index + 1}"
        case, features = pairs[row["case_id"]]
        if case.model_dump(mode="json") != read_json(run / "inputs" / (row["case_id"] + ".json")):
            raise ValueError("Original input changed")
        if packets[key]["table"] != block_packet(case, features, row["day"], "TARGET", None):
            raise ValueError("Original block facts differ")
        start, stop = row["query"]
        labels[key] = reference[row["case_id"]][start:stop]
        observed[key] = features.observed[start:stop]
        if observed[key].tolist() != packets[key]["observed"]:
            raise ValueError("Original missing mask differs")
        weak[key] = np.array([int(value.get("weak_signal", 0)) == 1
                              and int(value["activity_label"]) == 1 for value in labels[key]])
        support[key] = features.supported[start:stop]
        phases[key] = stage.predict(features.matrix[start:stop]).astype(int) + 1
        _, base, raw_support = invariance_inputs(case, features)
        if np.any(observed[key] & ~raw_support[start:stop]):
            raise ValueError("Training query lost original raw support")
        blocks, _, _, _ = discovery_block_matrix(raw_values(case), features.noise)
        matrices = {"base18": base, "base_expert": np.column_stack((base, expert_features(blocks)))}
        for name, matrix in matrices.items():
            choice = choices[name]
            model = XGBClassifier()
            model.load_model(ANCHOR / "primary_process" / f"{name}-depth{choice['depth']}.json")
            probability = model.predict_proba(matrix)[:, 1]
            activity = activity_from_mask(probability >= choice["threshold"], raw_support)[start:stop]
            predictions.setdefault(name + "_training_diagnostic", {})[key] = learned_stages(
                phases[key], activity, support[key]).to_dict()

    def replay(request_id, prompt, key):
        packet = packets[key]
        receipt = verify_payload(run, {"request_id": request_id},
                                 {"prompt": prompt, "images": [packet["image"]]})
        receipts[request_id] = receipt
        stored = outputs[request_id]
        if stored["status"] == "success":
            payload = json.loads(receipt["output"])
            codes, claims = parse_block_claims(payload)
            codes[~observed[key]] = -1
            facts, conflicts = check_block_claims(packet["table"]["normalized_neu_blocks"], claims)
            if (payload != stored["payload"] or facts != stored["facts"]
                    or conflicts != stored["conflicts"]):
                raise ValueError("Original factual claims do not replay")
            known_pairs = sum(value is not None for value in claims)
            claims_report.append({"request_id": request_id, "concrete_pair_claims": known_pairs,
                                  "false_pair_claims": len(conflicts), "unknown_pair_claims": 12 - known_pairs})
        else:
            if "output" in receipt:
                try:
                    parse_block_claims(json.loads(receipt["output"]))
                except (ValueError, json.JSONDecodeError):
                    pass
                else:
                    raise ValueError("Stored contract failure is a valid answer")
            codes = np.full(42, -1)
            failed.append(request_id)
        if codes.tolist() != stored["labels"]:
            raise ValueError("Original-date prediction does not replay")
        return codes

    triggered = []
    for row in read_json(run / "schedule.json"):
        key, repetition = row["query_id"], row["repetition"]
        prefix = f"{key}-r{repetition}"
        packet = packets[key]
        initial_id = prefix + "-image_initial"
        initial = outputs[initial_id]
        initial_codes = replay(initial_id, prompts["initial"], key)
        codes = {"image_initial": initial_codes}
        requests = {"image_initial": [initial_id]}
        numeric_id = prefix + "-numeric_first"
        codes["numeric_first"] = replay(numeric_id, prompts["initial"] + "\nRaw block table:"
                                        + json.dumps(packet["table"]), key)
        requests["numeric_first"] = [numeric_id]
        context = (prompts["initial"] + "\n" + prompts["followup"] + "\nPreceding answer:"
                   + json.dumps(initial.get("payload", {"status": "failed"}), separators=(",", ":")))
        trigger = initial["status"] == "success" and bool(initial["conflicts"])
        triggered.append({**row, "triggered": trigger,
                          "conflicts": len(initial.get("conflicts", [])),
                          "initial_failed": initial["status"] != "success"})
        for arm in ("conditional_fact_review", "matched_repeat"):
            if trigger:
                suffix = ("\nChecked factual conflicts only:" + json.dumps(initial["conflicts"], separators=(",", ":"))
                          if arm == "conditional_fact_review" else "\nNo additional numerical observations.")
                request_id = prefix + "-" + arm
                codes[arm] = replay(request_id, context + suffix, key)
                requests[arm] = [initial_id, request_id]
            else:
                if prefix + "-" + arm in outputs:
                    raise ValueError("Unexpected untriggered followup")
                codes[arm] = initial_codes
                requests[arm] = [initial_id]
        facts, _ = check_block_claims(packet["table"]["normalized_neu_blocks"], [None] * 12)
        fixed_id = prefix + "-fixed_all_facts"
        codes["fixed_all_facts"] = replay(fixed_id, context + "\nAll adjacent raw block differences:"
                                          + json.dumps(facts, separators=(",", ":")), key)
        requests["fixed_all_facts"] = [initial_id, fixed_id]
        for arm, values in codes.items():
            method = f"{arm}_r{repetition}"
            predictions.setdefault(method, {})[key] = learned_stages(phases[key], values, support[key]).to_dict()
            used.setdefault(method, {})[key] = requests[arm]
    if set(receipts) != set(outputs):
        raise ValueError("Unreplayed request exists")
    results, summary, per_case = score_target(labels, observed, weak, predictions, used, receipts)
    by_method = {row["method"]: row for row in summary}
    changes = []
    conditions = {}
    for repetition in range(2):
        main = f"conditional_fact_review_r{repetition}"
        for control in ("image_initial", "matched_repeat", "fixed_all_facts", "numeric_first"):
            for index, row in enumerate(selected):
                key = f"q{index + 1}"
                actual = np.array([int(value["activity_label"]) for value in labels[key]])
                known = observed[key] & (actual >= 0)
                before = np.array(predictions[f"{control}_r{repetition}"][key]["activity"])
                after = np.array(predictions[main][key]["activity"])
                gained = int((known & (before != actual) & (after == actual)).sum())
                lost = int((known & (before == actual) & (after != actual)).sum())
                changes.append({"repetition": repetition, "control": control, "query_id": key,
                                "role": row["role"], "gained": gained, "lost": lost, "net": gained - lost})
        for control in ("image_initial", "matched_repeat"):
            rows = [row for row in changes if row["repetition"] == repetition and row["control"] == control]
            conditions[f"r{repetition}_net_vs_{control}"] = sum(row["net"] for row in rows) > 0
            conditions[f"r{repetition}_weak_vs_{control}"] = by_method[main]["weak_recall"] >= by_method[f"{control}_r{repetition}"]["weak_recall"]
            conditions[f"r{repetition}_correct_controls_vs_{control}"] = sum(
                row["lost"] for row in rows if row["role"].startswith("correct")) == 0
        conditions[f"r{repetition}_f1_vs_fixed"] = by_method[main]["f1"] >= by_method[f"fixed_all_facts_r{repetition}"]["f1"]
    conditions["real_conflict"] = any(row["triggered"] for row in triggered)
    conditions["corrected_activity_error"] = any(row["gained"] > 0 for row in changes if row["control"] == "image_initial")
    conditions["fewer_followups_once"] = any(
        sum(row["triggered"] for row in triggered if row["repetition"] == repetition) < 4 for repetition in range(2))
    decision = {"conditions": conditions, "passed": sum(conditions.values()), "total": len(conditions),
                "decision": "PROCEED_TO_FROZEN_VALIDATION" if all(conditions.values()) else "STOP_FACTUAL_CLAIM_REVIEW_PILOT",
                "training_only": True, "independent_confirmation": False}
    cost = {"new_requests": len(receipts), "new_receipts": len(receipts),
            "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts.values()),
            "api_seconds": sum(row["seconds"] for row in receipts.values()),
            "wall_seconds": read_json(run / "state.json")["wall_seconds"], "contract_failures": failed}
    export(out / "decision.json", decision)
    export(out / "cost.json", cost)
    export(out / "results.json", results)
    export(out / "predictions.json", predictions)
    export(out / "triggers.json", triggered)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", per_case)
    write_csv(out / "corrections.csv", changes)
    write_csv(out / "claims.csv", claims_report)
    export(out / "verification.json", {"payload_prediction_claim_replays": len(receipts),
        "original_calendar_queries_checked": 168, "missing_and_shared_stage_gate": True,
        "original_inputs_and_block_tables": True, "sklearn_day_event_boundary_match": True,
        "training_diagnostics_not_fair_validation": True, "new_api_requests_in_report": 0})
    (out / "results.md").write_text("# 有界原始块事实复核小试\n\n## Material Passport\n\n"
        "- Origin Skill: academic-research-suite / experiment-agent\n- Mode: run / validate inline\n"
        "- Verification: 原请求、活动、事实重放；仅训练窗，非独立验证\n\n"
        + full_table(summary) + "\n\n" + json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    export(out / "manifest.json", {"files": {path.relative_to(out).as_posix(): sha(path)
                                              for path in out.rglob("*") if path.is_file()}})
    print(json.dumps(decision), flush=True)
    print(json.dumps(cost), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve(), args.out.resolve())
