"""Test one numerical follow-up after a raw-motion/visual-denial conflict.

Prepare uses only frozen observations and answers. Score opens references only
after all candidate predictions, schedules and receipt costs are frozen. Cached
numeric tasks do not mention the first local answer: this is a prospective
schedule replay, not a new sequential-dialogue or independent experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from analyze_landslide_codex_observation_reference import scoring_inputs, stage_reference_arrays
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence, review_prompt
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_window_experiment import parse_window_activity

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
BATCHES = ("pilot-v3", "mechanism-v1", "confirmation-v1", "background-review-v1")
ARMS = ("direct", "fixed", "pure", "numeric_first", "conditional", "matched_early", "always_followup")
PROTOCOL = {
    "schema": "conditional-numeric-followup-development-v2",
    "scope": "Already exposed cache development; AI and mechanism references reported separately",
    "basis": "0057 was observed but raw-supported weak movement was denied by pure local answers",
    "unchanged": "raw21/42, .03mm/day, noise ratio2, original14-day selection, two disjoint180-day targets, original49-day revision scope and common stage head",
    "candidate": "Among originally acquired successful image targets, count definite-negative local days with raw motion support inside original permission; require14 cumulative days, choose largest count then earliest start; at most ONE additional numeric task per record/repeat",
    "unknowns": "Unknown or failed first local decisions cannot contribute definite denial; numeric failure stays unknown in the acquired permitted target, no retry or restored confident prediction",
    "revision": "Replace pure local decisions by the same-window numeric answer only inside original permission; do not force raw numerical evidence to labels or infer labels from prose",
    "controls": "Pure; original numeric-first at all selected targets; matched-early one follow-up whenever candidate has one; numeric follow-up at all targets after pure; direct, fixed and all same-record saved numerical baselines",
    "cost": "Candidate retains pure calls plus one extra when triggered. Numeric-first can be cheaper. Shared receipts never summed across arms; no new calls in this cache replay",
    "selection": "No reference, prediction correctness or response prose in trigger; no threshold or rank sweep",
    "continue_conditions": {
        "AI_reference": "Both repeats: activityF1 strictly above pure, weakR nondecreasing, net definite correction positive; at least two records have positive net correction in each repeat",
        "controlled_mechanism_and_confirmation": "Both repeats of BOTH batches: F1 and weakR nondecreasing versus pure, net definite correction nonnegative",
        "background": "Both repeats: F1 and weakR nondecreasing versus pure; does not establish success versus direct or field accuracy",
        "feedback_selection": "AI reference and both controlled batches: F1 at least numeric-first in BOTH repeats, with fewer numeric acquisitions than always-followup in at least one repetition",
        "failure": "If any condition fails, stop this version; no more thresholds, hints or new seed to rescue it",
    },
    "new_remote_model_requests": 0,
    "preflight_correction": "v1 prepared but unscored; controlled baseline file contained only four relevant controls plus ungated diagnostic. v2 explicitly binds existing same-record strong scoped controls; mechanism, thresholds and continuation conditions unchanged",
}


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_bindings(directory, filename):
    for name, expected in read_json(directory / filename).items():
        if sha(directory / name) != expected:
            raise ValueError("Frozen source differs: " + str(directory / name))


def choose_followup(candidates, minimum_denied_days=14):
    eligible = [row for row in candidates if row["status"] == "success"
                and row["denied_raw_motion_days"] >= minimum_denied_days]
    if not eligible:
        return []
    ranked = sorted(eligible, key=lambda row: (-row["denied_raw_motion_days"], row["start"]))
    return [ranked[0]["index"]]


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", PROTOCOL)
    shutil.copyfile(Path(__file__), out / "source.py")
    source_bindings = {}
    predictions, schedules, costs = {}, {}, {}
    for batch in BATCHES:
        source = BASE / batch
        check_bindings(source, "freeze.json")
        check_bindings(source, "prediction-freeze.json")
        config = read_json(source / "protocol.json")
        cached = read_json(source / "predictions.json")
        choices = read_json(source / "selections.json")
        failures = read_json(source / "failures.json")
        policy = ReviewPolicy(**config["policy"])
        model = XGBClassifier()
        model.load_model(source / "stage-xgboost.json")
        predictions[batch], costs[batch] = {}, {}
        if batch in {"mechanism-v1", "confirmation-v1"}:
            baseline_path = BASE / "revision-scope-probe-v1" / batch / "predictions.json"
            baseline_names = ("robust_rule", "pelt_rule", "xgboost", "robust_xgboost_stage",
                              "raw_multiscale_rule", "xgboost_raw_persistence")
        else:
            baseline_path = source / "numeric-predictions.json"
            baseline_names = ("robust_rule", "pelt_rule", "xgboost", "robust_xgboost_stage")
            if batch == "background-review-v1":
                baseline_names += ("raw_multiscale_rule", "raw_rule_84", "xgboost_raw_persistence",
                                   "xgboost_42_matched_support", "xgboost_84")
        numeric_baselines = read_json(baseline_path)
        source_bindings[str(baseline_path.relative_to(ROOT))] = sha(baseline_path)
        for method in baseline_names:
            cases = numeric_baselines[method]
            if not set(config["case_ids"]).issubset(cases):
                raise ValueError("Numerical baseline is missing an evaluated record")
            predictions[batch][method] = {cid: cases[cid] for cid in config["case_ids"]}
        for name in ("freeze.json", "prediction-freeze.json", "protocol.json", "numeric-predictions.json"):
            source_bindings[str((source / name).relative_to(ROOT))] = sha(source / name)
        for case_id in config["case_ids"]:
            case, features = load_case_features(source, case_id)
            long = raw_motion_evidence(case, features)
            neutral = np.where(features.observed, 0, -1)
            raw_motion, _ = conflict_neighbourhood(neutral, features, long, policy, margin=0)
            codes = model.predict(features.matrix).astype(int) + 1
            for repeat in (0, 1):
                initial = np.array(cached[f"direct_visual_r{repeat}"][case_id]["activity"])
                image = np.array(cached[f"adaptive_no_feedback_r{repeat}"][case_id]["activity"])
                numeric = np.array(cached[f"adaptive_numeric_feedback_r{repeat}"][case_id]["activity"])
                choice = choices[f"{case_id}-r{repeat}"]
                targets = choice["selected"]
                _, neighbourhood = conflict_neighbourhood(initial, features, long, policy)
                permission, _, _ = target_scopes(neighbourhood, features.observed, targets)
                candidates, receipts = [], {}
                for target in targets:
                    index, start, stop = target["index"], target["start"], target["stop"]
                    status = "success"
                    for representation in ("image", "numeric"):
                        request_id = f"{case_id}-window-{index:02d}-{representation}-r{repeat}"
                        receipt_path = source / "requests" / f"{request_id}.response.json"
                        receipt = read_json(receipt_path)
                        expected_prompt = review_prompt(features, long, start, stop, representation == "numeric")
                        if receipt["prompt"] != expected_prompt:
                            raise ValueError("Cached follow-up is a different task")
                        for name, expected in receipt["image_sha256"].items():
                            if sha(source / "images" / name) != expected:
                                raise ValueError("Cached follow-up has different observed images")
                        source_bindings[str(receipt_path.relative_to(ROOT))] = sha(receipt_path)
                        receipts[request_id] = receipt
                        if request_id in failures:
                            parsed = np.full(len(initial), -1)
                            if representation == "image":
                                status = "failed"
                        else:
                            parsed = parse_window_activity(json.loads(receipt["output"]), features.observed, start, stop)
                        expected = image if representation == "image" else numeric
                        if not np.array_equal(parsed[start:stop], expected[start:stop]):
                            raise ValueError("Cached local answer does not reconstruct the frozen decision")
                    denied = raw_motion[start:stop] & (image[start:stop] == 0) & permission[start:stop]
                    candidates.append({**target, "status": status, "denied_raw_motion_days": int(denied.sum())})
                selected = choose_followup(candidates, policy.minimum_conflict_days)
                early = [min(targets, key=lambda row: row["start"])["index"]] if selected else []
                indices = [row["index"] for row in targets]
                direct_id = f"{case_id}-direct-r{repeat}"
                direct_path = source / "requests" / f"{direct_id}.response.json"
                receipts[direct_id] = read_json(direct_path)
                source_bindings[str(direct_path.relative_to(ROOT))] = sha(direct_path)
                image_ids = [f"{case_id}-window-{index:02d}-image-r{repeat}" for index in indices]
                numeric_ids = [f"{case_id}-window-{index:02d}-numeric-r{repeat}" for index in indices]
                arm_requests = {"direct": [direct_id], "pure": [direct_id, *image_ids],
                                "numeric_first": [direct_id, *numeric_ids],
                                "always_followup": [direct_id, *image_ids, *numeric_ids]}
                activities = {"direct": initial, "pure": scoped_activity(initial, image, permission, features.observed),
                              "numeric_first": scoped_activity(initial, numeric, permission, features.observed)}
                activities["always_followup"] = activities["numeric_first"].copy()
                for arm, escalation in (("conditional", selected), ("matched_early", early)):
                    revised = image.copy()
                    for target in targets:
                        if target["index"] in escalation:
                            revised[target["start"]:target["stop"]] = numeric[target["start"]:target["stop"]]
                    activities[arm] = scoped_activity(initial, revised, permission, features.observed)
                    arm_requests[arm] = [direct_id, *image_ids,
                                         *[f"{case_id}-window-{index:02d}-numeric-r{repeat}" for index in escalation]]
                fixed_targets = [choice["priorities"][index] for index in choice["fixed_indices"]]
                fixed_permission, _, _ = target_scopes(neighbourhood, features.observed, fixed_targets)
                fixed_activity = np.array(cached[f"fixed_image_review_r{repeat}"][case_id]["activity"])
                activities["fixed"] = scoped_activity(initial, fixed_activity, fixed_permission, features.observed)
                arm_requests["fixed"] = [direct_id, *[f"{case_id}-window-{index:02d}-image-r{repeat}"
                                                      for index in choice["fixed_indices"]]]
                for request_id in arm_requests["fixed"]:
                    if request_id not in receipts:
                        path = source / "requests" / f"{request_id}.response.json"
                        receipts[request_id] = read_json(path)
                        source_bindings[str(path.relative_to(ROOT))] = sha(path)
                for arm in ARMS:
                    method = f"{arm}_r{repeat}"
                    activity = activities[arm]
                    if np.any(activity[~permission & features.observed] != initial[~permission & features.observed]) and arm != "fixed":
                        raise ValueError("Candidate exceeded the original revision permission")
                    prediction = learned_stages(codes, activity, features.supported)
                    predictions[batch].setdefault(method, {})[case_id] = prediction.to_dict()
                    usage = arm_requests[arm]
                    if len(set(usage)) != len(usage):
                        raise ValueError("Deployment counts duplicate shared request")
                    costs[batch].setdefault(method, []).append({
                        "case_id": case_id, "request_ids": usage, "requests": len(usage),
                        "tokens": sum(receipts[name]["usage"]["total_tokens"] for name in usage),
                        "seconds_sum": sum(receipts[name]["seconds"] for name in usage),
                        "numeric_acquisitions": sum("-numeric-" in name for name in usage),
                    })
                schedules[f"{batch}/{case_id}/r{repeat}"] = {
                    "candidates": candidates, "conditional_numeric_indices": selected,
                    "matched_early_indices": early, "original_editable_days": int(permission.sum()),
                    "maximum_total_candidate_calls": 1 + len(targets) + len(selected),
                }
    save_json(out / "source-bindings.json", source_bindings)
    save_json(out / "predictions.json", predictions)
    save_json(out / "schedules.json", schedules)
    save_json(out / "deployment-costs.json", costs)
    save_json(out / "prediction-freeze.json", {path.name: sha(path) for path in out.iterdir() if path.is_file()})
    print("Frozen schedules/predictions/costs for 54 exposed records; references not opened in prepare")


def score(out):
    check_bindings(out, "prediction-freeze.json")
    for name, expected in read_json(out / "source-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Cached source changed after candidate freeze")
    predictions = read_json(out / "predictions.json")
    costs = read_json(out / "deployment-costs.json")
    ai_labels, _, ai_weak = scoring_inputs(BASE / "codex-observation-review-v1")
    populations = {}
    for batch in BATCHES:
        source = BASE / batch
        cases = {cid: load_case_features(source, cid)[0] for cid in read_json(source / "protocol.json")["case_ids"]}
        reference = source / "reference-subset.csv" if batch != "background-review-v1" else source / "analysis-v1/reference-subset.csv"
        labels = load_labels(reference, cases)
        weak = {cid: np.array([int(row["weak_signal"]) == 1 for row in rows])
                if batch != "pilot-v3" else np.array([row["feature"] == "slow_displacement" for row in rows])
                for cid, rows in labels.items()}
        populations[batch] = (batch, labels, weak)
    populations["AI-reference"] = ("confirmation-v1", ai_labels, ai_weak)
    summary, results, corrections, decisions = [], {}, [], {}
    for population, (batch, labels, weak) in populations.items():
        results[population] = {}
        for method, cases in predictions[batch].items():
            per_case, boundaries, parsed_predictions = {}, {}, {}
            internal_actual, internal_prediction = [], []
            weak_count = sum(int(mask.sum()) for mask in weak.values())
            weak_detected = 0
            for case_id, rows in labels.items():
                serialized = cases[case_id]
                prediction = ActivityPrediction(**{key: np.array(value) if key in {"activity", "stage"} else value
                                                   for key, value in serialized.items()})
                parsed_predictions[case_id] = prediction
                per_case[case_id] = evaluate_case(rows, prediction)
                boundaries[case_id] = boundary_errors(rows, prediction.activity, .3)
                stages = stage_reference_arrays(rows, prediction)
                internal_actual.extend(stages["actual_stage"][stages["stage_interior"]].tolist())
                internal_prediction.extend(prediction.stage[stages["stage_interior"]].tolist())
                weak_detected += int(np.count_nonzero(weak[case_id] & (prediction.activity == 1)))
            aggregate = aggregate_scores(list(per_case.values()))
            event05 = aggregate_scores([evaluate_case(labels[cid], parsed_predictions[cid], .5)
                                        for cid in labels])["events"]
            _, _, stage_f1, _ = precision_recall_fscore_support(internal_actual, internal_prediction, labels=[1, 2, 3], zero_division=0)
            matches = [pair for row in boundaries.values() for pair in row["matched_pairs"]]
            if len(matches) != aggregate["events"]["tp"]:
                raise ValueError("Event and boundary matches disagree")
            usage = costs[batch].get(method, [])
            row = {"population": population, "method": method, **aggregate["daily"],
                   "weak_recall": weak_detected / weak_count if weak_count else None,
                   "weak_days": weak_count, "events_f1_iou03": aggregate["events"]["f1"],
                   "events_f1_iou05": event05["f1"], "coverage": aggregate["coverage"],
                   "stage_internal_f1": float(np.mean(stage_f1)), "stage_end_to_end_f1": aggregate["stage_macro_f1"],
                   "onset_mae": float(np.mean([abs(pair["onset_error_days"]) for pair in matches])) if matches else None,
                   "offset_mae": float(np.mean([abs(pair["offset_error_days"]) for pair in matches])) if matches else None,
                   "archived_deployment_calls": sum(item["requests"] for item in usage),
                   "reported_tokens": sum(item["tokens"] for item in usage),
                   "numeric_acquisitions": sum(item["numeric_acquisitions"] for item in usage)}
            summary.append(row)
            results[population][method] = {"summary": row, "per_case": per_case, "boundaries": boundaries}
        population_decisions = []
        for repeat in (0, 1):
            candidate = f"conditional_r{repeat}"
            changes = []
            for control in ("pure", "numeric_first", "matched_early", "direct", "fixed"):
                control_method = f"{control}_r{repeat}"
                for case_id, rows in labels.items():
                    actual = np.array([int(row["activity_label"]) for row in rows])
                    before = np.array(predictions[batch][control_method][case_id]["activity"])
                    after = np.array(predictions[batch][candidate][case_id]["activity"])
                    changes.append({"population": population, "case_id": case_id, "control": control_method,
                                    "treatment": candidate, **correction_utility(actual, before, after, actual >= 0),
                                    "lost_tp": int(np.count_nonzero((actual == 1) & (before == 1) & (after != 1))),
                                    "gained_tp": int(np.count_nonzero((actual == 1) & (before != 1) & (after == 1))),
                                    "removed_fp": int(np.count_nonzero((actual == 0) & (before == 1) & (after != 1))),
                                    "gained_fp": int(np.count_nonzero((actual == 0) & (before != 1) & (after == 1)))})
            corrections.extend(changes)
            indexed = results[population]
            new = indexed[candidate]["summary"]
            pure = indexed[f"pure_r{repeat}"]["summary"]
            numeric = indexed[f"numeric_first_r{repeat}"]["summary"]
            selected_changes = [row for row in changes if row["control"] == f"pure_r{repeat}"]
            net = sum(row["net_correct"] for row in selected_changes)
            beneficiaries = sum(row["net_correct"] > 0 for row in selected_changes)
            condition = new["f1"] >= pure["f1"] and new["weak_recall"] >= pure["weak_recall"] and net >= 0
            if population == "AI-reference":
                condition = condition and new["f1"] > pure["f1"] and net > 0 and beneficiaries >= 2
            if population in {"AI-reference", "mechanism-v1", "confirmation-v1"}:
                condition = condition and new["f1"] >= numeric["f1"]
            population_decisions.append({"repeat": repeat, "f1_difference_vs_pure": new["f1"] - pure["f1"],
                                         "weak_recall_difference_vs_pure": new["weak_recall"] - pure["weak_recall"],
                                         "f1_difference_vs_numeric_first": new["f1"] - numeric["f1"],
                                         "net_definite_correction": net, "benefited_records": beneficiaries,
                                         "passed": bool(condition), "registered_gate": population != "pilot-v3"})
        decisions[population] = population_decisions
    selection_saved = {}
    for population in ("AI-reference", "mechanism-v1", "confirmation-v1"):
        indexed = results[population]
        selection_saved[population] = any(
            indexed[f"conditional_r{repeat}"]["summary"]["numeric_acquisitions"]
            < indexed[f"always_followup_r{repeat}"]["summary"]["numeric_acquisitions"]
            for repeat in (0, 1))
    passed = all(item["passed"] for rows in decisions.values() for item in rows if item["registered_gate"])
    passed = passed and all(selection_saved.values())
    save_json(out / "results.json", results)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-record-corrections.csv", corrections)
    save_json(out / "decision.json", {"conditions": decisions, "fewer_numeric_acquisitions": selection_saved,
                                      "continue": bool(passed),
                                      "action": "Eligible for separately frozen new-record trial" if passed else "Stop this version; no threshold sweep or default change",
                                      "new_model_requests": 0, "independent_confirmation": False})
    save_json(out / "manifest.json", {"files": {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()}})
    print(read_json(out / "decision.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "score"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out.resolve())
    else:
        score(args.out.resolve())
