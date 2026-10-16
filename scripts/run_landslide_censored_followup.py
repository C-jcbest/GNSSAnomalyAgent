"""Freeze and evaluate a two-observation continuation agent on exposed records."""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from analyze_landslide_dual_reference_confirmation import score_target
from analyze_landslide_review_utility_v2 import verify_receipts
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import ROOT, save_json
from run_landslide_repair_role_probe import BASE, BATCHES, reference_for_batch
from run_landslide_window_optimization import load_case_features, read_json, render_window
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_prompt,
)
from gnss_sim.landslide_censored_followup import continuation_choice, followup_targets
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import parse_window_activity

ARMS = ("censored", "fixed_short", "wide", "no_numeric", "censored_latest")
PLOT = {"single_plot_inches": [10.24, 6.3], "single_plot_dpi": 150}


def cached_activity(path, observed, start, stop):
    try:
        value = parse_window_activity(json.loads(read_json(path)["output"]), observed, start, stop)
        return {"activity": value.tolist(), "status": "success"}
    except (ValueError, KeyError):
        return {"activity": np.full(len(observed), -1, dtype=int).tolist(), "status": "failed"}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", {
        "date": "2026-10-11", "scope": "Both complete exposed author-AI batches, not unseen confirmation",
        "mechanism": "First observe the original highest-conflict selected180-day target. If activity touches its right edge and at least14 original conflicts lie in the following49 days, observe a98-day raw view centered at the censored edge. Otherwise use the original other target. Maximum two reviews; retain each original0/1/2 budget.",
        "limits": "A visual interval truncated by target is not a physical cessation. Raw conflict may be missed motion or unsupported visual activity, never a plateau label. Left edges and uncensored local misses are not addressed.",
        "controls": "Matched-trigger fixed98-day crop at the other original target center;180-day centered continuation; no numerical gate; latest-answer overwrite instead of original strict-majority overlap aggregation; all34 original methods",
        "unchanged": "Initial answer, original21/42 motion cues,14 admission, images rendered by existing raw renderer, same WINDOW_TASK and common frozen numerical stage head. Original49-day revision authority remains, intersected with actual observed targets.",
        "registration": "Across all24 AI records, mean pooled activity F1 over both repeats must exceed original and matched fixed short, mean pooled weak recall must not fall below original, total net definite correction versus both controls must be positive. All per-batch/repeat FP,weak losses,events,bounds,unknowns and stage effects reported. No requirement to win every metric; no post-result threshold/repeat/seed rescue. Passing only permits new frozen confirmation.",
        "primary": "censored", "repetitions": 2, "maximum_new_requests": 128,
        "model": "qwen3.8-flash", "retries": 0, "hard_timeout_seconds": 1800,
        "cache": "First answers and matching existing questions reused byte-exactly. Sequential choice computed from frozen cached first response; no contemporaneous-control or model-drift causal claim. Fresh independent full sequential inference required for confirmation.",
        "selection_before_inference": "Forward-only continuation and fixed98 width derived from original49-day evidence footprint; developer inspected old errors and structural triggers, not an independent selection test.",
    })
    sources = [Path(__file__), ROOT / "src/gnss_sim/landslide_censored_followup.py",
        ROOT / "scripts/run_landslide_repair_role_probe.py",
        ROOT / "scripts/analyze_landslide_dual_reference_confirmation.py",
        ROOT / "scripts/probe_landslide_revision_scope.py",
        ROOT / "scripts/run_landslide_window_optimization.py",
        ROOT / "src/gnss_sim/landslide_adaptive_review.py", ROOT / "src/gnss_sim/landslide_visual.py"]
    (out / "source").mkdir()
    for path in sources:
        shutil.copyfile(path, out / "source" / path.name)
    write_json(out / "source-bindings.json", {str(path.relative_to(ROOT)): sha(path) for path in sources})
    (out / "images").mkdir()
    schedules, packets, parents = {}, {}, {}
    for batch, comparison in BATCHES.items():
        verify_manifest(comparison / "analysis")
        verify_bindings(comparison / "run", "prediction-freeze.json")
        verify_receipts(comparison / "run")
        config = read_json(comparison / "run/protocol.json")
        policy = ReviewPolicy(**config["policy"])
        predictions = read_json(comparison / "analysis/predictions.json")
        selections = read_json(comparison / "run/selections.json")
        parents[batch] = {"comparison": str(comparison),
            "analysis_manifest_sha256": sha(comparison / "analysis/manifest.json")}
        schedules[batch] = {}
        for cid in config["case_ids"]:
            case, feature = load_case_features(comparison / "run", cid)
            long = raw_motion_evidence(case, feature)
            for rep in (0, 1):
                choice = selections[f"{cid}-r{rep}"]
                initial = np.array(predictions[f"direct_visual_r{rep}"][cid]["activity"])
                conflicts, authority = conflict_neighbourhood(initial, feature, long, policy)
                selected = choice["selected"]
                record = {"budget": len(selected), "authority": authority.tolist(), "targets": {}}
                schedules[batch][f"{cid}-r{rep}"] = record
                if not selected:
                    continue
                first = min(selected, key=lambda row: (-row["conflict_days"], row["start"]))
                first_key = f"{cid}-window-{first['index']:02d}-image-r{rep}"
                first_path = comparison / "run/requests" / (first_key + ".response.json")
                first_result = cached_activity(first_path, feature.observed, first["start"], first["stop"])
                remaining = next((row for row in selected if row["index"] != first["index"]), None)
                continuation = continuation_choice(first, np.array(first_result["activity"]),
                    conflicts, feature.observed, policy.minimum_conflict_days)
                second_targets = followup_targets(first, remaining, continuation, len(feature.observed))
                record.update({"first": first, "first_key": first_key, "first_result": first_result,
                    "first_path": str(first_path), "first_sha256": sha(first_path), "continuation": continuation})
                for arm in ARMS:
                    target = second_targets["censored" if arm == "censored_latest" else arm]
                    if target is None:
                        record["targets"][arm] = None
                        continue
                    start, stop = target["start"], target["stop"]
                    original = next((row for row in choice["priorities"]
                        if (row["start"], row["stop"]) == (start, stop)), None)
                    local_name = f"{cid}-window-{original['index']:02d}.png" if original else f"{cid}-followup-{start}-{stop}.png"
                    key = f"{cid}-window-{original['index']:02d}-image-r{rep}" if original else f"{cid}-followup-{start}-{stop}-image-r{rep}"
                    record["targets"][arm] = {"start": start, "stop": stop, "request_id": key}
                    if key in packets:
                        continue
                    names = [f"{cid}-raw.png", local_name]
                    shutil.copyfile(comparison / "run/images" / names[0], out / "images" / names[0])
                    if original:
                        shutil.copyfile(comparison / "run/images" / local_name, out / "images" / local_name)
                    elif not (out / "images" / local_name).exists():
                        render_window(case, feature.noise, start, stop, out / "images" / local_name, PLOT)
                    prompt = review_prompt(feature, long, start, stop, False)
                    cached = comparison / "run/requests" / (key + ".response.json")
                    packet = {"case_id": cid, "batch": batch, "start": start, "stop": stop,
                        "prompt": prompt, "images": names, "image_sha256": {name: sha(out / "images" / name) for name in names},
                        "cached": cached.exists()}
                    if cached.exists():
                        receipt = read_json(cached)
                        if receipt["prompt"] != prompt or receipt["image_sha256"] != packet["image_sha256"]:
                            raise ValueError("Only byte-identical existing questions may be reused")
                        packet.update({"cache_path": str(cached), "cache_sha256": sha(cached)})
                    packets[key] = packet
    write_json(out / "schedules.json", schedules)
    write_json(out / "packets.json", packets)
    write_json(out / "parent-bindings.json", parents)
    new = sum(not row["cached"] for row in packets.values())
    if new > 128:
        raise ValueError("Prepared requests exceed the registered budget")
    write_json(out / "freeze.json", {str(path.relative_to(out)): sha(path)
        for path in sorted(out.rglob("*")) if path.is_file()})
    write_json(out / "state.json", {"status": "prepared", "new_requests": new,
        "cached_second_questions": len(packets) - new})
    print(read_json(out / "state.json"), flush=True)


def verify(out):
    verify_bindings(out, "freeze.json")
    for name, digest in read_json(out / "source-bindings.json").items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen executing source changed: " + name)
    for rows in read_json(out / "schedules.json").values():
        for row in rows.values():
            if row["budget"] and sha(Path(row["first_path"])) != row["first_sha256"]:
                raise ValueError("Cached first answer changed")
    for row in read_json(out / "packets.json").values():
        if row["cached"] and sha(Path(row["cache_path"])) != row["cache_sha256"]:
            raise ValueError("Cached second answer changed")


def execute(out):
    verify(out)
    state = read_json(out / "state.json")
    if state["status"] != "prepared":
        raise ValueError("Execute only a never-started prepared run")
    config = read_json(out / "protocol.json")
    save_json(out / "state.json", {**state, "status": "running"})
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], config["maximum_new_requests"])
    outputs, failures, masks = {}, {}, {}
    started = time.monotonic()
    packets = read_json(out / "packets.json")
    for ordinal, (key, row) in enumerate(packets.items()):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            raise RuntimeError("Registered wall-time exceeded")
        cid = row["case_id"]
        if cid not in masks:
            masks[cid] = load_case_features(BATCHES[row["batch"]] / "run", cid)[1].observed
        observed = masks[cid]
        try:
            if row["cached"]:
                payload = json.loads(read_json(Path(row["cache_path"]))["output"])
            else:
                payload = transport.ask(key, row["prompt"], [out / "images" / name for name in row["images"]])
            activity = parse_window_activity(payload, observed, row["start"], row["stop"])
            status = "success"
        except (RuntimeError, ValueError, KeyError) as error:
            activity = np.full(len(observed), -1, dtype=int)
            failures[key] = str(error)
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
    head = XGBClassifier()
    head.load_model(comparison / "run/stage-xgboost.json")
    for cid in read_json(comparison / "run/protocol.json")["case_ids"]:
        feature = load_case_features(comparison / "run", cid)[1]
        codes = head.predict(feature.matrix).astype(int) + 1
        for rep in (0, 1):
            initial = np.array(predictions[f"direct_visual_r{rep}"][cid]["activity"])
            row = schedules[f"{cid}-r{rep}"]
            for arm in ARMS:
                keys = [f"{cid}-direct-r{rep}"]
                activity = initial.copy()
                if row["budget"]:
                    keys.append(row["first_key"])
                    first = {**row["first"], **row["first_result"]}
                    reviews = [first]
                    second = row["targets"][arm]
                    if second is not None:
                        keys.append(second["request_id"])
                        reviews.append({**second, **outputs[second["request_id"]]})
                    reviewed = combine_reviews(initial, reviews, feature.observed).activity
                    if arm == "censored_latest" and second is not None:
                        value = np.array(outputs[second["request_id"]]["activity"])
                        reviewed[second["start"]:second["stop"]] = value[second["start"]:second["stop"]]
                    scope = target_scopes(np.array(row["authority"]), feature.observed, reviews)[0]
                    activity = scoped_activity(initial, reviewed, scope, feature.observed)
                method = f"followup_{arm}_r{rep}"
                predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                used.setdefault(method, {})[cid] = keys
    return predictions, used


def save_target(folder, labels, observed, weak, predictions, used, receipts):
    folder.mkdir()
    results, summary, records = score_target(labels, observed, weak, predictions, used, receipts)
    write_json(folder / "predictions.json", predictions)
    write_json(folder / "results.json", results)
    write_csv(folder / "summary.csv", summary)
    write_csv(folder / "per-record.csv", records)
    corrections = []
    for rep in (0, 1):
        main = f"followup_censored_r{rep}"
        for control in (f"adaptive_no_feedback_r{rep}__conflict_scope", f"followup_fixed_short_r{rep}",
                        f"followup_wide_r{rep}", f"followup_no_numeric_r{rep}", f"followup_censored_latest_r{rep}"):
            for cid in labels:
                actual = np.array([row["activity_label"] for row in labels[cid]])
                utility = correction_utility(actual, np.array(predictions[control][cid]["activity"]),
                    np.array(predictions[main][cid]["activity"]), observed[cid])
                corrections.append({"case_id": cid, "control": control, "treatment": main, **utility})
    write_csv(folder / "corrections.csv", corrections)
    return {row["method"]: row for row in summary}, corrections


def analyze(out):
    verify(out)
    verify_bindings(out, "prediction-freeze.json")
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("Freeze all answers before scoring")
    folder = out / "analysis"
    folder.mkdir(exist_ok=False)
    receipts = {}
    for comparison in BATCHES.values():
        receipts.update(verify_receipts(comparison / "run"))
    receipts.update(verify_receipts(out))
    tables, all_corrections, replayed = {}, [], 0
    for batch in BATCHES:
        labels, observed, weak = reference_for_batch(batch)
        predictions, used = predictions_for_batch(out, batch)
        table, corrections = save_target(folder / batch, labels, observed, weak, predictions, used, receipts)
        tables[batch] = table
        all_corrections.extend(corrections)
        with (BATCHES[batch] / "analysis/summary.csv").open(encoding="utf-8-sig") as stream:
            for original in csv.DictReader(stream):
                for key in ("f1", "precision", "recall", "fp", "fn", "weak_recall", "stage_end_to_end_macro_f1"):
                    if not np.isclose(float(original[key]), table[original["method"]][key]):
                        raise ValueError("An original score changed")
                replayed += 1
        if batch == "ai_0145_0156":
            process = BASE / "dual-reference-confirmation-analysis-v1/secondary_process"
            labels_process = read_json(process / "reference-labels.json")
            predictions_process = read_json(process / "predictions.json")
            codes = read_json(BASE / "dual-reference-process-predictions-v1/stage-codes.json")
            weak_process = {cid: np.array([row["weak_signal"] == 1 and row["activity_label"] == 1
                for row in rows]) for cid, rows in labels_process.items()}
            features = {cid: load_case_features(BATCHES[batch] / "run", cid)[1] for cid in labels_process}
            for method, records in predictions.items():
                if method in predictions_process:
                    continue
                predictions_process[method] = {cid: learned_stages(np.array(codes[cid]),
                    np.array(value["activity"]), features[cid].supported).to_dict() for cid, value in records.items()}
            save_target(folder / "process_0145_0156", labels_process, observed, weak_process,
                predictions_process, used, receipts)
    pooled = {}
    for method in next(iter(tables.values())):
        sums = {key: sum(table[method][key] for table in tables.values())
            for key in ("tp", "fp", "fn", "weak_days", "weak_detected_days")}
        pooled[method] = {**sums, "f1": 2 * sums["tp"] / max(1, 2 * sums["tp"] + sums["fp"] + sums["fn"]),
            "weak_recall": sums["weak_detected_days"] / max(1, sums["weak_days"])}
    write_json(folder / "pooled-activity.json", pooled)
    conditions, comparisons = [], []
    for stem in ("adaptive_no_feedback", "followup_fixed_short"):
        controls = [f"{stem}_r{rep}" + ("__conflict_scope" if stem == "adaptive_no_feedback" else "") for rep in (0, 1)]
        main = [f"followup_censored_r{rep}" for rep in (0, 1)]
        net = sum(row["net_correct"] for row in all_corrections if row["control"] in controls)
        difference = float(np.mean([pooled[a]["f1"] - pooled[b]["f1"] for a, b in zip(main, controls, strict=True)]))
        weak_difference = float(np.mean([pooled[a]["weak_recall"] - pooled[b]["weak_recall"] for a, b in zip(main, controls, strict=True)]))
        passed = difference > 0 and net > 0
        if stem == "adaptive_no_feedback":
            passed = passed and weak_difference >= 0
        conditions.append(passed)
        comparisons.append({"control": stem, "mean_pooled_f1_difference": difference,
            "mean_pooled_weak_recall_difference": weak_difference, "net_correct_both_repeats": net,
            "passed": passed})
    passed = all(conditions)
    write_json(folder / "decision.json", {"status": "PROMISING_DEVELOPMENT" if passed else "STOPPED",
        "continue_to_frozen_confirmation": passed, "comparisons": comparisons,
        "default_changed": False, "unseen_confirmation": False})
    actual = [read_json(path) for path in (out / "requests").glob("*.response.json")]
    write_json(folder / "audit.json", {"status": "passed", "original_scores_replayed": replayed,
        "methods_per_target": 44, "targets": 3, "new_records": 0, "new_training": False,
        "new_requests": len(list((out / "requests").glob("*.started.json"))), "new_receipts": len(actual),
        "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in actual)})
    write_json(folder / "manifest.json", {"files": {path.relative_to(folder).as_posix(): sha(path)
        for path in folder.rglob("*") if path.is_file()}})
    print(read_json(folder / "audit.json"), flush=True)
    print(read_json(folder / "decision.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": execute, "analyze": analyze}[args.action](args.out.resolve())
