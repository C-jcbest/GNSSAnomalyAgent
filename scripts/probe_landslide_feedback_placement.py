"""Complete fixed-placement controls with the identical existing numerical feedback."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from analyze_landslide_core_lockbox import developer_predictions
from analyze_landslide_review_utility_v2 import verify_receipts
from diagnose_landslide_feedback_contribution import compare_cases
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from report_landslide_core_lockbox import full_table
from run_landslide_adaptive_review import verify_freeze
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import save_json
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_core_lockbox import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_prompt,
)
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import parse_window_activity

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
PARENT = BASE / "core-lockbox-confirmation-v1"
DATA = BASE / "core-lockbox-data-v1"
PRIOR = BASE / "core-lockbox-analysis-v1"
SOURCES = (
    Path(__file__),
    ROOT / "scripts/diagnose_landslide_feedback_contribution.py",
    ROOT / "scripts/score_landslide_core_lockbox.py",
    ROOT / "scripts/probe_landslide_revision_scope.py",
    ROOT / "src/gnss_sim/landslide_adaptive_review.py",
    ROOT / "src/gnss_sim/landslide_visual.py",
    ROOT / "src/gnss_sim/landslide_window_experiment.py",
)


def prepare(out):
    verify_freeze(PARENT)
    verify_bindings(PARENT, "prediction-freeze.json")
    verify_manifest(PRIOR)
    receipts = verify_receipts(PARENT)
    config = read_json(PARENT / "protocol.json")
    selections = read_json(PARENT / "selections.json")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source").mkdir()
    shutil.copytree(PARENT / "inputs", out / "inputs")
    source_paths = list(SOURCES)
    source_paths.extend(path for path in PRIOR.rglob("*") if path.is_file())
    source_paths.extend(path for path in (PARENT / "requests").iterdir() if path.is_file())
    source_paths.extend([PARENT / "selections.json", PARENT / "protocol.json"])
    write_json(out / "source-bindings.json", {str(path): sha(path) for path in source_paths})
    for path in SOURCES:
        shutil.copyfile(path, out / "source" / path.name)
    (out / "images").mkdir()
    plan = []
    for cid in config["case_ids"]:
        case, feature = load_case_features(out, cid)
        long = raw_motion_evidence(case, feature)
        for rep in (0, 1):
            choice = selections[f"{cid}-r{rep}"]
            for index in choice["fixed_indices"]:
                target = choice["priorities"][index]
                images = [f"{cid}-raw.png", f"{cid}-window-{index:02d}.png"]
                for image in images:
                    destination = out / "images" / image
                    if not destination.exists():
                        shutil.copyfile(PARENT / "images" / image, destination)
                old_key = f"{cid}-window-{index:02d}-numeric-r{rep}"
                prompt = review_prompt(feature, long, target["start"], target["stop"], True)
                cached = old_key in receipts
                if cached:
                    receipt = receipts[old_key]
                    if receipt["prompt"] != prompt or receipt["image_sha256"] != {
                        image: sha(out / "images" / image) for image in images
                    }:
                        raise ValueError("Cached numerical review is not the same query")
                plan.append(
                    {
                        "case_id": cid,
                        "repeat": rep,
                        "index": index,
                        "start": target["start"],
                        "stop": target["stop"],
                        "request_id": old_key if cached else "fixed-feedback-" + old_key,
                        "cached": cached,
                        "images": images,
                        "prompt": prompt,
                    }
                )
    write_json(out / "query-plan.json", plan)
    new_count = sum(not row["cached"] for row in plan)
    write_json(
        out / "protocol.json",
        {
            "scope": "Post-score complete24 development control, not independent confirmation or rescue of the old NOT_CONFIRMED trial",
            "change": "Only provide the unchanged numerical summary to original fixed locations; no new threshold, prompt, scope, model, reference or fitting",
            "model": config["model"],
            "case_ids": config["case_ids"],
            "policy": config["policy"],
            "maximum_new_requests": new_count,
            "cached_queries": len(plan) - new_count,
            "hard_timeout_seconds": 1800,
            "retries": 0,
            "controls": "All original37/34 plus full fixed numerical feedback whole/conflict and matched-K early/late numerical feedback conflict",
            "next_action_rule": "Allow separate future confirmation of the existing feedback candidate only if both repeats on complete and valid-initial process records improve F1 and net definite corrections versus direct, full fixed numerical feedback and matched-K early numerical feedback, with fewer calls than full fixed and mean weak recall no lower than matched-K early",
            "failure_rule": "Stop this promotion on failure; no new seed, prompt or best control substitution to salvage it",
            "service_timing_limit": "New missing fixed answers and cached adaptive answers were acquired at different times; this is development evidence only. Any future confirmation must interleave same-feedback positions in one fresh run",
            "reference": "Unchanged process truth and developer visual/numerical labels; no independent expert claim",
        },
    )
    write_json(
        out / "freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    save_json(out / "state.json", {"status": "prepared", "new_requests": new_count})
    print(
        "Prepared identical-feedback controls:",
        new_count,
        "new queries,",
        len(plan) - new_count,
        "cached",
    )


def verify(out):
    verify_bindings(out, "freeze.json")
    for filename, expected in read_json(out / "source-bindings.json").items():
        if sha(Path(filename)) != expected:
            raise ValueError("Frozen source or parent evidence changed: " + filename)


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Execute only a never-started plan; no retries or restarts")
    config = read_json(out / "protocol.json")
    plan = read_json(out / "query-plan.json")
    transport = VisualRequests(
        out / "requests", ROOT / ".env", config["model"], config["maximum_new_requests"]
    )
    receipts = verify_receipts(PARENT)
    features = {cid: load_case_features(out, cid)[1] for cid in config["case_ids"]}
    reviews, failures = {}, {}
    started = time.monotonic()
    save_json(out / "state.json", {"status": "running", "completed": 0, "total": len(plan)})
    for ordinal, row in enumerate(plan):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            save_json(out / "state.json", {"status": "timeout", "completed": ordinal})
            raise RuntimeError("Frozen timeout reached; do not automatically restart")
        feature = features[row["case_id"]]
        activity = np.full(len(feature.observed), -1, dtype=int)
        status = "success"
        try:
            if row["cached"]:
                receipt = receipts[row["request_id"]]
                if receipt["status"] != "returned_json" or receipt["finish_reason"] != "stop":
                    raise ValueError("Cached answer failed")
                answer = json.loads(receipt["output"])
            else:
                answer = transport.ask(
                    row["request_id"],
                    row["prompt"],
                    [out / "images" / image for image in row["images"]],
                )
            activity = parse_window_activity(answer, feature.observed, row["start"], row["stop"])
        except (ValueError, RuntimeError) as error:
            status = "failed"
            failures[row["request_id"]] = str(error)
        reviews[row["request_id"]] = {
            "start": row["start"],
            "stop": row["stop"],
            "activity": activity.tolist(),
            "status": status,
        }
        save_json(out / "reviews.json", reviews)
        save_json(out / "failures.json", failures)
        save_json(
            out / "state.json", {"status": "running", "completed": ordinal + 1, "total": len(plan)}
        )
        print(f"{ordinal + 1}/{len(plan)} {row['request_id']} {status}", flush=True)
    verify(out)
    verify_receipts(out)
    write_json(
        out / "prediction-freeze.json",
        {name: sha(out / name) for name in ("reviews.json", "failures.json")},
    )
    save_json(
        out / "state.json",
        {"status": "predictions_complete", "seconds": time.monotonic() - started},
    )


def fixed_predictions(out, features):
    selections = read_json(PARENT / "selections.json")
    initial_predictions = read_json(PRIOR / "primary_process/predictions.json")
    reviews = read_json(out / "reviews.json")
    lookup = {
        (row["case_id"], row["repeat"], row["index"]): row["request_id"]
        for row in read_json(out / "query-plan.json")
    }
    head = XGBClassifier()
    head.load_model(DATA / "stage-xgboost.json")
    config = read_json(out / "protocol.json")
    predictions, used = {}, {}
    for cid, (case, feature) in features.items():
        codes = head.predict(feature.matrix).astype(int) + 1
        long = raw_motion_evidence(case, feature)
        for rep in (0, 1):
            initial = np.array(initial_predictions[f"direct_visual_r{rep}"][cid]["activity"])
            choice = selections[f"{cid}-r{rep}"]
            fixed = [choice["priorities"][index] for index in choice["fixed_indices"]]
            count = len(choice["selected"])
            authority = conflict_neighbourhood(
                initial, feature, long, ReviewPolicy(**config["policy"])
            )[1]
            for stem, targets in (
                ("fixed_numeric_feedback", fixed),
                ("same_count_fixed_early_numeric", fixed[:count]),
                ("same_count_fixed_late_numeric", fixed[len(fixed) - count :] if count else []),
            ):
                keys = [lookup[(cid, rep, target["index"])] for target in targets]
                local = [reviews[key] for key in keys]
                reviewed = combine_reviews(initial, local, feature.observed).activity
                scope = target_scopes(authority, feature.observed, targets)[0]
                variants = {
                    "conflict_scope": scoped_activity(initial, reviewed, scope, feature.observed)
                }
                if stem == "fixed_numeric_feedback":
                    variants["whole_target"] = reviewed
                for variant, activity in variants.items():
                    method = f"{stem}_r{rep}__{variant}"
                    predictions.setdefault(method, {})[cid] = learned_stages(
                        codes, activity, feature.supported
                    ).to_dict()
                    used.setdefault(method, {})[cid] = [f"{cid}-direct-r{rep}"] + keys
    return predictions, used


def score(out, destination):
    verify(out)
    verify_bindings(out, "prediction-freeze.json")
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("Score only terminal frozen answers")
    receipts = verify_receipts(PARENT)
    receipts.update(verify_receipts(out))
    destination.mkdir(parents=True, exist_ok=False)
    config = read_json(out / "protocol.json")
    features = {cid: load_case_features(out, cid) for cid in config["case_ids"]}
    additions, additional_used = fixed_predictions(out, features)
    conditions, comparisons = {}, []
    for target in ("primary_process", "secondary_developer"):
        labels = read_json(PRIOR / target / "reference-labels.json")
        predictions = read_json(PRIOR / target / "predictions.json")
        observed = {cid: features[cid][1].observed for cid in labels}
        if target == "primary_process":
            weak = {
                cid: np.array(
                    [
                        int(row["activity_label"]) == 1 and int(row["weak_signal"]) == 1
                        for row in rows
                    ]
                )
                for cid, rows in labels.items()
            }
            predictions.update(additions)
        else:
            spans = read_json(DATA / "codex-review/activity-decisions.json")[
                "weak_interiors_authored_before_predictions"
            ]
            weak = {cid: np.zeros(len(rows), dtype=bool) for cid, rows in labels.items()}
            for cid, rows in labels.items():
                for start, stop in spans.get(cid, []):
                    weak[cid][start:stop] = True
                weak[cid] &= np.array([int(row["activity_label"]) for row in rows]) == 1
            converted = developer_predictions(
                DATA, {cid: features[cid] for cid in labels}, additions
            )
            predictions.update({name: converted[name] for name in additions})
        used = read_json(PRIOR / "used-requests.json")
        used.update(additional_used)
        used = {
            method: {cid: keys for cid, keys in cases.items() if cid in labels}
            for method, cases in used.items()
        }
        results, summaries, records = score_target(
            labels, observed, weak, predictions, used, receipts
        )
        previous_results = read_json(PRIOR / target / "results.json")
        if any(results[name] != value for name, value in previous_results.items()):
            raise ValueError("Adding controls changed an original result")
        folder = destination / target
        folder.mkdir()
        write_json(folder / "results.json", results)
        write_json(folder / "predictions.json", predictions)
        write_csv(folder / "summary.csv", summaries)
        write_csv(folder / "per-record.csv", records)
        (folder / "all-methods.md").write_text(full_table(summaries), encoding="utf-8")
        record_table = {(row["method"], row["case_id"]): row for row in records}
        for rep in (0, 1):
            main = f"adaptive_numeric_feedback_r{rep}__conflict_scope"
            for population in ("complete", "valid_initial_repeat"):
                ids = [
                    cid
                    for cid in labels
                    if population == "complete"
                    or predictions[f"direct_visual_r{rep}"][cid]["status"] != "failed"
                ]
                controls = [
                    f"direct_visual_r{rep}",
                    f"adaptive_no_feedback_r{rep}__conflict_scope",
                    f"fixed_numeric_feedback_r{rep}__whole_target",
                    f"same_count_fixed_early_numeric_r{rep}__conflict_scope",
                    f"same_count_fixed_late_numeric_r{rep}__conflict_scope",
                ]
                for control in controls:
                    row = {
                        "target": target,
                        "repeat": rep,
                        "population": population,
                        "records": len(ids),
                        "control": control,
                        "treatment": main,
                        **compare_cases(
                            labels, predictions, results, record_table, ids, control, main
                        ),
                    }
                    comparisons.append(row)
                    if target == "primary_process" and control.startswith(
                        (
                            "direct_visual",
                            "fixed_numeric_feedback",
                            "same_count_fixed_early_numeric",
                        )
                    ):
                        conditions[f"{population}:r{rep}:{control}"] = (
                            row["f1_difference"] > 0 and row["net_correct"] > 0
                        )
        table = {row["method"]: row for row in summaries}
        if target == "primary_process":
            conditions["fewer_calls"] = all(
                table[f"adaptive_numeric_feedback_r{rep}__conflict_scope"]["requests"]
                < table[f"fixed_numeric_feedback_r{rep}__whole_target"]["requests"]
                for rep in (0, 1)
            )
            conditions["mean_weak_not_lower"] = bool(
                np.mean(
                    [
                        table[f"adaptive_numeric_feedback_r{rep}__conflict_scope"]["weak_recall"]
                        for rep in (0, 1)
                    ]
                )
                >= np.mean(
                    [
                        table[f"same_count_fixed_early_numeric_r{rep}__conflict_scope"][
                            "weak_recall"
                        ]
                        for rep in (0, 1)
                    ]
                )
            )
    write_csv(destination / "placement-comparisons.csv", comparisons)
    write_json(
        destination / "decision.json",
        {
            "conditions": conditions,
            "passed": all(conditions.values()),
            "decision": "ALLOW_SEPARATE_FEEDBACK_CONFIRMATION"
            if all(conditions.values())
            else "STOP_FEEDBACK_PROMOTION",
            "scope": "Development action only; no new independent confirmation",
            "old_core_decision": "NOT_CONFIRMED; unchanged",
        },
    )
    new_receipts = verify_receipts(out)
    write_json(
        destination / "cost.json",
        {
            "new_requests": len(new_receipts),
            "responses": len(new_receipts),
            "reported_tokens": sum(
                row.get("usage", {}).get("total_tokens", 0) for row in new_receipts.values()
            ),
            "wall_seconds": read_json(out / "state.json")["seconds"],
            "task_failures": len(read_json(out / "failures.json")),
            "retries": 0,
            "cached_fixed_queries": config["cached_queries"],
        },
    )
    shutil.copyfile(Path(__file__), destination / "source.py")
    write_json(
        destination / "manifest.json",
        {
            "files": {
                path.relative_to(destination).as_posix(): sha(path)
                for path in destination.rglob("*")
                if path.is_file()
            }
        },
    )
    print(read_json(destination / "decision.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out)
    elif args.mode == "run":
        execute(args.out)
    else:
        if args.analysis is None:
            parser.error("score requires --analysis")
        score(args.out, args.analysis)
