"""Freeze and test endpoint-directed review on two exposed author-AI batches."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from analyze_landslide_codex_observation_reference import scoring_inputs
from analyze_landslide_dual_reference_confirmation import score_target
from analyze_landslide_review_utility_v2 import verify_receipts
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import ROOT, save_json
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_prompt,
)
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_roles import (
    repair_role_support,
    role_neighbourhood,
    select_role_targets,
)
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import parse_window_activity

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
BATCHES = {
    "ai_0133_0144": BASE / "ai-observable-holdout-comparison-v1",
    "ai_0145_0156": BASE / "dual-reference-confirmation-comparison-v2",
}
METHODS = {
    "repair_roles": ("roles", "roles"),
    "roles_placement_only": ("roles", "original"),
    "roles_authority_only": ("original", "roles"),
    "roles_center_authority": ("roles", "center"),
    "untyped_joint": ("untyped_joint", "original"),
    "untyped_joint_roles_authority": ("untyped_joint", "roles"),
}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", {
        "date": "2026-10-11", "scope": "Development on both complete exposed author-AI batches; not unseen confirmation",
        "change": "Retain missed-motion conflicts; restrict below-threshold negative conflicts to within49 days of initial visual event endpoints. Select a same-count disjoint pair by this repair support and dilate it by the existing49-day raw footprint for authority.",
        "interpretation": "Raw cue deficit is not a plateau label. Effective endpoint authority can reach98 days before target clipping. Interior plateaus in overmerged initial events may remain unrepaired.",
        "unchanged": "All original raw images, prompts,21/42 motion evidence,14-day native admission, per-record/repeat original0/1/2 call budget, common frozen stage head and task scoring",
        "controls": METHODS, "repetitions": 2, "retries": 0, "model": "qwen3.8-flash",
        "maximum_new_requests": 128, "hard_timeout_seconds": 1800,
        "registration": "On BOTH author-AI batches and both repetitions, mainF1>original, weakR>=original,FP<=original,net definite correction>0. Also mainF1>=untyped joint,weakR>=joint,net>=0. Failures retain stop status; no parameter/seed/repeat rescue.",
        "cache": "Byte-identical historical inputs and responses reused; missing target replies acquired once per repeat. New and cached response dates differ; no contemporaneous-control or causal model-drift claim.",
        "data": "Both12-record author-AI references and all34 historical methods retained; new12 process target additionally reported with previously matched baselines/head",
        "structural_screen": "One-each conflict-type quota would change only0133 and0154, cannot affect0148 with only negative cues; not advanced to remote experiment",
    })
    (out / "images").mkdir()
    sources = [Path(__file__), ROOT / "src/gnss_sim/landslide_review_roles.py",
        ROOT / "scripts/analyze_landslide_dual_reference_confirmation.py",
        ROOT / "scripts/probe_landslide_revision_scope.py",
        ROOT / "src/gnss_sim/landslide_adaptive_review.py",
        ROOT / "src/gnss_sim/landslide_visual.py"]
    (out / "source").mkdir()
    for path in sources:
        shutil.copyfile(path, out / "source" / path.name)
    write_json(out / "source-bindings.json", {str(path.relative_to(ROOT)): sha(path) for path in sources})
    packets, schedules, bindings = {}, {}, {}
    for batch, comparison in BATCHES.items():
        verify_manifest(comparison / "analysis")
        verify_bindings(comparison / "run", "freeze.json")
        verify_bindings(comparison / "run", "prediction-freeze.json")
        verify_receipts(comparison / "run")
        binding = read_json(comparison / "source-bindings.json")
        verify_manifest(Path(binding["data"]) / "codex-review", "delivery-manifest.json")
        bindings[batch] = {"comparison": str(comparison), "data": binding["data"],
            "analysis_manifest_sha256": sha(comparison / "analysis/manifest.json"),
            "run_freeze_sha256": sha(comparison / "run/freeze.json"),
            "prediction_freeze_sha256": sha(comparison / "run/prediction-freeze.json"),
            "reference_delivery_sha256": sha(Path(binding["data"]) / "codex-review/delivery-manifest.json")}
        config = read_json(comparison / "run/protocol.json")
        policy = ReviewPolicy(**config["policy"])
        original_predictions = read_json(comparison / "analysis/predictions.json")
        saved_choices = read_json(comparison / "run/selections.json")
        schedules[batch] = {}
        for cid in config["case_ids"]:
            case, feature = load_case_features(comparison / "run", cid)
            long = raw_motion_evidence(case, feature)
            for rep in (0, 1):
                initial = np.array(original_predictions[f"direct_visual_r{rep}"][cid]["activity"])
                choice = saved_choices[f"{cid}-r{rep}"]
                conflicts, original_neighbourhood = conflict_neighbourhood(initial, feature, long, policy)
                roles = repair_role_support(initial, conflicts, feature.observed)
                _, selected = select_role_targets(choice["priorities"], choice["selected"],
                    roles["repair_support"], policy.minimum_conflict_days, "repair_roles")
                _, joint = select_role_targets(choice["priorities"], choice["selected"],
                    roles["repair_support"], policy.minimum_conflict_days, "untyped_joint")
                locations = {"original": choice["selected"], "roles": selected, "untyped_joint": joint}
                record = {"locations": locations, "original_neighbourhood": original_neighbourhood.tolist(),
                    "role_neighbourhood": role_neighbourhood(roles["repair_support"], feature.observed).tolist(),
                    "role_support": roles["repair_support"].tolist(),
                    "missed_motion_cues": int(roles["missed_motion"].sum()),
                    "boundary_negative_cues": int(roles["boundary_negative"].sum()),
                    "original_call_budget": len(choice["selected"])}
                schedules[batch][f"{cid}-r{rep}"] = record
                for targets in locations.values():
                    for target in targets:
                        index = target["index"]
                        key = f"{cid}-window-{index:02d}-image-r{rep}"
                        if key in packets:
                            continue
                        prompt = review_prompt(feature, long, target["start"], target["stop"], False)
                        names = [f"{cid}-raw.png", f"{cid}-window-{index:02d}.png"]
                        image_hashes = {}
                        for name in names:
                            source = comparison / "run/images" / name
                            image_hashes[name] = sha(source)
                            destination = out / "images" / name
                            if not destination.exists():
                                shutil.copyfile(source, destination)
                        cached = comparison / "run/requests" / (key + ".response.json")
                        packet = {"request_id": key, "batch": batch, "case_id": cid, "repeat": rep,
                            "start": target["start"], "stop": target["stop"], "prompt": prompt,
                            "images": names, "image_sha256": image_hashes, "cached": cached.exists()}
                        if cached.exists():
                            receipt = read_json(cached)
                            if receipt["prompt"] != prompt or receipt["image_sha256"] != image_hashes:
                                raise ValueError("Do not reuse a different historical visual question")
                            packet["cache_path"] = str(cached)
                            packet["cache_sha256"] = sha(cached)
                        packets[key] = packet
    write_json(out / "schedules.json", schedules)
    write_json(out / "packets.json", packets)
    write_json(out / "parent-bindings.json", bindings)
    new_count = sum(not row["cached"] for row in packets.values())
    if new_count > read_json(out / "protocol.json")["maximum_new_requests"]:
        raise ValueError("Prepared experiment exceeds registered bounded requests")
    write_json(out / "freeze.json", {str(path.relative_to(out)): sha(path)
        for path in sorted(out.rglob("*")) if path.is_file()})
    write_json(out / "state.json", {"status": "prepared", "new_requests_planned": new_count,
        "cached_local_questions": len(packets) - new_count})
    print(read_json(out / "state.json"), flush=True)


def verify(out):
    verify_bindings(out, "freeze.json")
    for name, digest in read_json(out / "source-bindings.json").items():
        if sha(ROOT / name) != digest:
            raise ValueError("Executing mechanism source changed: " + name)
    for row in read_json(out / "packets.json").values():
        if row["cached"] and sha(Path(row["cache_path"])) != row["cache_sha256"]:
            raise ValueError("Historical response changed")


def execute(out):
    verify(out)
    state = read_json(out / "state.json")
    if state["status"] != "prepared":
        raise ValueError("Only a never-started experiment may execute")
    config = read_json(out / "protocol.json")
    save_json(out / "state.json", {**state, "status": "running"})
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], config["maximum_new_requests"])
    outputs, failures = {}, {}
    started = time.monotonic()
    packets = read_json(out / "packets.json")
    observation_masks = {}
    for ordinal, (key, row) in enumerate(packets.items()):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise RuntimeError("Registered wall-time budget exceeded")
        case_key = (row["batch"], row["case_id"])
        if case_key not in observation_masks:
            observation_masks[case_key] = load_case_features(
                BATCHES[row["batch"]] / "run", row["case_id"])[1].observed
        observed = observation_masks[case_key]
        try:
            if row["cached"]:
                payload = json.loads(read_json(Path(row["cache_path"]))["output"])
            else:
                payload = transport.ask(key, row["prompt"], [out / "images" / name for name in row["images"]])
            activity = parse_window_activity(payload, observed, row["start"], row["stop"])
            status = "success"
        except (RuntimeError, ValueError, KeyError) as error:
            failures[key] = str(error)
            activity = np.full(len(observed), -1, dtype=int)
            status = "failed"
        outputs[key] = {"activity": activity.tolist(), "status": status}
        save_json(out / "outputs.json", outputs)
        save_json(out / "failures.json", failures)
        if not row["cached"]:
            print(f"{ordinal + 1}/{len(packets)} {key}: {status}", flush=True)
    write_json(out / "prediction-freeze.json", {name: sha(out / name) for name in ("outputs.json", "failures.json")})
    save_json(out / "state.json", {**state, "status": "predictions_complete", "seconds": time.monotonic() - started})


def predictions_for_batch(out, batch):
    comparison = BATCHES[batch]
    predictions = read_json(comparison / "analysis/predictions.json")
    used = read_json(comparison / "analysis/used-requests.json")
    schedules = read_json(out / "schedules.json")[batch]
    outputs = read_json(out / "outputs.json")
    config = read_json(comparison / "run/protocol.json")
    head = XGBClassifier()
    head.load_model(comparison / "run/stage-xgboost.json")
    for cid in config["case_ids"]:
        _, feature = load_case_features(comparison / "run", cid)
        codes = head.predict(feature.matrix).astype(int) + 1
        for rep in (0, 1):
            initial = np.array(predictions[f"direct_visual_r{rep}"][cid]["activity"])
            schedule = schedules[f"{cid}-r{rep}"]
            for stem, (placement, authority) in METHODS.items():
                targets = schedule["locations"][placement]
                keys = [f"{cid}-direct-r{rep}"]
                reviews = []
                for target in targets:
                    key = f"{cid}-window-{target['index']:02d}-image-r{rep}"
                    keys.append(key)
                    reviews.append({"start": target["start"], "stop": target["stop"], **outputs[key]})
                reviewed = combine_reviews(initial, reviews, feature.observed)
                neighbourhood = np.array(schedule["original_neighbourhood"] if authority == "original"
                    else schedule["role_neighbourhood"])
                scope, center, _ = target_scopes(neighbourhood, feature.observed, targets)
                if authority == "center":
                    scope = center
                activity = scoped_activity(initial, reviewed.activity, scope, feature.observed)
                method = f"{stem}_r{rep}"
                predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                used.setdefault(method, {})[cid] = keys
    return predictions, used


def reference_for_batch(batch):
    comparison = BATCHES[batch]
    data = Path(read_json(comparison / "source-bindings.json")["data"])
    labels, observed, _ = scoring_inputs(data / "codex-review")
    spans = read_json(data / "codex-review/activity-decisions.json")["weak_interiors_authored_before_predictions"]
    weak = {}
    for cid, rows in labels.items():
        mask = np.zeros(len(rows), dtype=bool)
        for start, stop in spans.get(cid, []):
            mask[start:stop] = True
        weak[cid] = mask & (np.array([row["activity_label"] for row in rows]) == 1)
    return labels, observed, weak


def save_target(folder, labels, observed, weak, predictions, used, receipts):
    folder.mkdir()
    results, summaries, records = score_target(labels, observed, weak, predictions, used, receipts)
    write_json(folder / "predictions.json", predictions)
    write_json(folder / "results.json", results)
    for name, rows in (("summary.csv", summaries), ("per-record.csv", records)):
        write_csv(folder / name, rows)
    corrections, registration = [], []
    table = {row["method"]: row for row in summaries}
    for rep in (0, 1):
        main = f"repair_roles_r{rep}"
        controls = [f"adaptive_no_feedback_r{rep}__conflict_scope", f"untyped_joint_r{rep}",
            f"roles_placement_only_r{rep}", f"roles_authority_only_r{rep}", f"roles_center_authority_r{rep}",
            f"direct_visual_r{rep}", f"fixed_image_review_r{rep}__whole_target"]
        for control in controls:
            counts = [correction_utility(np.array([row["activity_label"] for row in labels[cid]]),
                np.array(predictions[control][cid]["activity"]), np.array(predictions[main][cid]["activity"]),
                observed[cid]) for cid in labels]
            total = {key: sum(row[key] for row in counts)
                for key in ("gained_correct", "lost_correct", "net_correct", "eligible_days")}
            corrections.append({"control": control, "treatment": main, **total})
            if control.startswith("adaptive_no_feedback") or control.startswith("untyped_joint_r"):
                original = control.startswith("adaptive_no_feedback")
                conditions = {"f1": table[main]["f1"] > table[control]["f1"] if original
                    else table[main]["f1"] >= table[control]["f1"],
                    "weak_recall": table[main]["weak_recall"] >= table[control]["weak_recall"],
                    "net_correct": total["net_correct"] > 0 if original else total["net_correct"] >= 0}
                if original:
                    conditions["fp_not_higher"] = table[main]["fp"] <= table[control]["fp"]
                registration.append({"control": control, "treatment": main, **conditions,
                    "passed": all(conditions.values())})
    write_csv(folder / "corrections.csv", corrections)
    write_json(folder / "registration.json", registration)
    return results, summaries, registration


def analyze(out):
    verify(out)
    verify_bindings(out, "prediction-freeze.json")
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("Score only all frozen scheduled responses")
    target = out / "analysis"
    target.mkdir(exist_ok=False)
    receipts = {}
    for comparison in BATCHES.values():
        receipts.update(verify_receipts(comparison / "run"))
    if (out / "requests").exists():
        receipts.update(verify_receipts(out))
    registrations, replayed = {}, 0
    for batch in BATCHES:
        labels, observed, weak = reference_for_batch(batch)
        predictions, used = predictions_for_batch(out, batch)
        _, summaries, registration = save_target(target / batch, labels, observed, weak, predictions, used, receipts)
        registrations[batch] = registration
        with (BATCHES[batch] / "analysis/summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            original_rows = {row["method"]: row for row in csv.DictReader(stream)}
        table = {row["method"]: row for row in summaries}
        for method, archived in original_rows.items():
            for metric in ("f1", "precision", "recall", "weak_recall", "fp", "fn",
                    "stage_end_to_end_macro_f1", "stage_internal_macro_f1"):
                if not np.isclose(float(archived[metric]), table[method][metric]):
                    raise ValueError("Original frozen score differs")
            replayed += 1
        if batch == "ai_0145_0156":
            process = BASE / "dual-reference-confirmation-analysis-v1/secondary_process"
            process_predictions = read_json(process / "predictions.json")
            codes = read_json(BASE / "dual-reference-process-predictions-v1/stage-codes.json")
            process_labels = read_json(process / "reference-labels.json")
            weak_process = {cid: np.array([row["weak_signal"] == 1 and row["activity_label"] == 1
                for row in rows]) for cid, rows in process_labels.items()}
            features = {cid: load_case_features(BATCHES[batch] / "run", cid)[1]
                for cid in process_labels}
            for method, records in predictions.items():
                if method in process_predictions:
                    continue
                process_predictions[method] = {}
                for cid, value in records.items():
                    feature = features[cid]
                    process_predictions[method][cid] = learned_stages(np.array(codes[cid]),
                        np.array(value["activity"]), feature.supported).to_dict()
            save_target(target / "process_0145_0156", process_labels, observed, weak_process,
                process_predictions, used, receipts)
    passed = all(row["passed"] for rows in registrations.values() for row in rows)
    write_json(target / "decision.json", {"continue_to_unseen_confirmation": passed,
        "status": "PROMISING_DEVELOPMENT" if passed else "STOPPED",
        "registered_author_ai_comparisons": registrations,
        "unseen_confirmation": False, "default_policy_changed": False})
    actual = [read_json(path) for path in (out / "requests").glob("*.response.json")]
    write_json(target / "audit.json", {"status": "passed", "author_ai_records": 24,
        "methods_per_target": 46, "targets": 3, "original_scores_replayed": replayed,
        "new_requests": len(list((out / "requests").glob("*.started.json"))),
        "new_receipts": len(actual), "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in actual),
        "stage_heads_frozen_and_target_matched": True, "new_training": False,
        "new_records": 0, "primary_original_call_budget_retained": True,
        "scope": "Previously exposed development; no contemporaneous response or unseen accuracy claim"})
    write_json(target / "manifest.json", {"files": {
        path.relative_to(target).as_posix(): sha(path) for path in sorted(target.rglob("*")) if path.is_file()}})
    print(read_json(target / "audit.json"), flush=True)
    print("Development decision:", "PROMISING" if passed else "STOPPED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": execute, "analyze": analyze}[args.action](args.out.resolve())
