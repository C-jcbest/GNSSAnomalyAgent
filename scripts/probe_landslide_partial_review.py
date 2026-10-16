"""Freeze an abstention-update probe on complete exposed batches, then score it."""

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_landslide_dual_reference_confirmation import score_target
from analyze_landslide_review_utility_v2 import verify_receipts
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_ai_reference_holdout import replay, verify_bindings
from run_landslide_window_optimization import load_case_features, read_json
from write_landslide_dual_reference_report import full_table
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import ReviewPolicy, combine_reviews, raw_motion_evidence
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_partial_review import combine_partial_reviews
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_window_experiment import parse_window_activity

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
BATCHES = {
    "ai_0133_0144": ("ai-observable-holdout-comparison-v1", None),
    "ai_0145_0156": ("dual-reference-confirmation-comparison-v2", "dual-reference-confirmation-analysis-v1"),
    "ai_0157_0168": ("sequential-followup-confirmation-comparison-v1", "sequential-followup-confirmation-analysis-v1"),
}


def prepare(out):
    out.mkdir(exist_ok=False)
    write_json(out / "protocol.json", {
        "date": "2026-10-11", "scope": "All36 exposed author-AI records; two existing latent-process targets. Development only.",
        "mechanism": "Successful local abstention is no new decision only when every covering review abstains. Retain the global state at those dates. Original strict-majority definite changes, failures, contradictory evidence and missing dates stay unchanged.",
        "primary": "partial_adaptive_no_feedback__conflict_scope",
        "controls": "Original34 methods; fixed and same-count positions with identical partial-update semantics; whole-target authority; protect every initial positive; cheaper first-only with strict and partial updates. Sequential old candidates retained but not reused as main.",
        "unchanged": "Original21/42 motion evidence,14 admission,two original disjoint180-day targets,49-day authority,raw images,prompts,initial and local actual answers,target-matched numerical stage heads and six controls.",
        "gate": "Across36 complete AI records: mean pooled F1 over two repeats greater than original, fixed same partial-update whole-target, and cheaper partial first-only; net definite correction positive against each; mean pooled weak recall not lower than original. Latent-process targets reported but cannot override primary failure. Any failure stops candidate, no threshold,seed,replicate or subset rescue.",
        "new_remote_calls": 0, "new_training": False,
        "limits": "Keeping an erroneous global state can preserve a false positive or false negative; this is an inertia policy, not calibrated uncertainty or positive GNSS evidence. Unknown reference dates are never relabelled.",
    })
    sources = [Path(__file__), ROOT / "src/gnss_sim/landslide_partial_review.py",
               ROOT / "src/gnss_sim/landslide_adaptive_review.py",
               ROOT / "scripts/run_landslide_ai_reference_holdout.py",
               ROOT / "scripts/analyze_landslide_dual_reference_confirmation.py"]
    (out / "source").mkdir()
    for path in sources:
        (out / "source" / path.name).write_bytes(path.read_bytes())
    write_json(out / "source-bindings.json", {path.relative_to(ROOT).as_posix(): sha(path) for path in sources})
    receipts_count = 0
    for batch, (name, process_analysis) in BATCHES.items():
        comparison = BASE / name
        run = comparison / "run"
        verify_bindings(run, "prediction-freeze.json")
        receipts = verify_receipts(run)
        receipts_count += len(receipts)
        predictions, used, _ = replay(run, receipts)
        if (run / "followup-predictions.json").exists():
            predictions.update(read_json(run / "followup-predictions.json"))
            used.update(read_json(run / "followup-used-requests.json"))
        schedules = read_json(run / "selections.json")
        config = read_json(run / "protocol.json")
        policy = ReviewPolicy(**config["policy"])
        head = XGBClassifier()
        head.load_model(run / "stage-xgboost.json")
        features = {}
        audit = []
        for cid in config["case_ids"]:
            case, feature = load_case_features(run, cid)
            features[cid] = feature
            codes = head.predict(feature.matrix).astype(int) + 1
            long = raw_motion_evidence(case, feature)
            for rep in (0, 1):
                initial = np.array(predictions[f"direct_visual_r{rep}"][cid]["activity"])
                choice = schedules[f"{cid}-r{rep}"]
                selected = choice["selected"]
                fixed = [choice["priorities"][index] for index in choice["fixed_indices"]]
                count = len(selected)
                targets_by_stem = {
                    "adaptive_no_feedback": selected,
                    "adaptive_numeric_feedback": selected,
                    "fixed_image_review": fixed,
                    "same_count_fixed_early": fixed[:count],
                    "same_count_fixed_late": fixed[len(fixed) - count:] if count else [],
                    "first_only": [min(selected, key=lambda value: (-value["conflict_days"], value["start"]))]
                                  if selected else [],
                }
                authority = conflict_neighbourhood(initial, feature, long, policy)[1]
                for stem, targets in targets_by_stem.items():
                    reviews, keys = [], [f"{cid}-direct-r{rep}"]
                    for target in targets:
                        arm = "numeric" if stem == "adaptive_numeric_feedback" else "image"
                        key = f"{cid}-window-{target['index']:02d}-{arm}-r{rep}"
                        receipt = receipts[key]
                        try:
                            if receipt["status"] != "returned_json":
                                raise ValueError("Failed visual response")
                            activity = parse_window_activity(json.loads(receipt["output"]), feature.observed,
                                                             target["start"], target["stop"])
                            status = "success"
                        except (ValueError, KeyError):
                            activity = np.full(len(initial), -1)
                            status = "failed"
                        reviews.append({"start": target["start"], "stop": target["stop"],
                                        "activity": activity, "status": status})
                        keys.append(key)
                    partial, no_new_decision = combine_partial_reviews(initial, reviews, feature.observed)
                    strict = combine_reviews(initial, reviews, feature.observed)
                    scope = target_scopes(authority, feature.observed, targets)[0]
                    variants = {"conflict_scope": scoped_activity(initial, partial.activity, scope, feature.observed)}
                    if stem in ("fixed_image_review", "adaptive_no_feedback"):
                        variants["whole_target"] = partial.activity
                    if stem == "adaptive_no_feedback":
                        all_positive = strict.activity.copy()
                        all_positive[initial == 1] = 1
                        variants["preserve_all_initial_positive"] = scoped_activity(initial, all_positive, scope, feature.observed)
                    if stem == "first_only":
                        method = f"first_only_r{rep}"
                        first = learned_stages(codes, scoped_activity(initial, strict.activity, scope, feature.observed),
                                               feature.supported).to_dict()
                        if method in predictions and cid in predictions[method] and predictions[method][cid] != first:
                            raise ValueError("Original first-only control differs from its actual first receipt")
                        predictions.setdefault(method, {})[cid] = first
                        used.setdefault(method, {})[cid] = keys
                    for variant, activity in variants.items():
                        method = f"partial_{stem}_r{rep}__{variant}"
                        predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                        used.setdefault(method, {})[cid] = keys
                        baseline = strict.activity
                        permission = feature.observed
                        if variant != "whole_target":
                            baseline = scoped_activity(initial, baseline, scope, feature.observed)
                            permission = scope
                        changed = feature.observed & (activity != baseline)
                        audit.append({"method": method, "case_id": cid, "no_new_decision_days": int((no_new_decision & permission).sum()),
                            "restored_prior_positive_days": int((changed & (initial == 1) & (activity == 1)).sum()),
                            "restored_prior_negative_days": int((changed & (initial == 0) & (activity == 0)).sum()),
                            "used_requests": keys})
        folder = out / batch
        folder.mkdir()
        write_json(folder / "predictions.json", predictions)
        write_json(folder / "used-requests.json", used)
        write_csv(folder / "update-audit.csv", audit)
        write_json(folder / "parent.json", {"comparison": name, "process_analysis": process_analysis})
        if process_analysis is not None:
            parent = BASE / process_analysis / "secondary_process"
            base_process = read_json(parent / "predictions.json")
            codes = read_json(BASE / ("dual-reference-process-predictions-v1" if batch == "ai_0145_0156"
                                      else "sequential-followup-process-predictions-v1") / "stage-codes.json")
            for method, records in predictions.items():
                if method in base_process:
                    continue
                base_process[method] = {cid: learned_stages(np.array(codes[cid]), np.array(value["activity"]),
                    features[cid].supported).to_dict() for cid, value in records.items()}
            write_json(folder / "process-predictions.json", base_process)
    write_json(out / "prediction-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()})
    print({"status": "predictions_frozen", "historical_receipts_verified": receipts_count, "new_remote_calls": 0})


def score(out):
    verify_bindings(out, "prediction-freeze.json")
    for name, digest in read_json(out / "source-bindings.json").items():
        if sha(ROOT / name) != digest:
            raise ValueError("Probe source changed after prediction freeze")
    analysis = out / "analysis"
    analysis.mkdir(exist_ok=False)
    all_tables, labels_by_batch, observed_by_batch, predictions_by_batch = {}, {}, {}, {}
    text = "# 成功弃判的部分更新语义：完整开发小试\n\n"
    text += "36条已暴露AI记录和两批过程目标；0新远端请求，不是未见确认。保留旧确定判断可能保留旧错误；失败和明确相反判断仍未知或按原多数修订。\n\n"
    for batch, (name, process_analysis) in BATCHES.items():
        comparison = BASE / name
        data = Path(read_json(comparison / "source-bindings.json")["data"])
        if batch == "ai_0133_0144":
            from analyze_landslide_codex_observation_reference import scoring_inputs
            labels, observed, _ = scoring_inputs(data / "codex-review")
            authored = read_json(data / "codex-review/activity-decisions.json")["weak_interiors_authored_before_predictions"]
            weak = {}
            for cid, rows in labels.items():
                mask = np.zeros(len(rows), dtype=bool)
                for start, stop in authored.get(cid, []):
                    mask[start:stop] = True
                weak[cid] = mask & (np.array([row["activity_label"] for row in rows]) == 1)
            targets = [(batch, labels, weak, read_json(out / batch / "predictions.json"))]
        else:
            parent = BASE / process_analysis
            verify_manifest(parent)
            labels = read_json(parent / "primary_ai/reference-labels.json")
            observed = {cid: np.array([value is not None for value in read_json(
                data / "inputs" / cid / "input.json")["displacement_mm"]]) for cid in labels}
            authored = read_json(data / "codex-review/activity-decisions.json")["weak_interiors_authored_before_predictions"]
            weak = {}
            for cid, rows in labels.items():
                mask = np.zeros(len(rows), dtype=bool)
                for start, stop in authored.get(cid, []):
                    mask[start:stop] = True
                weak[cid] = mask & (np.array([row["activity_label"] for row in rows]) == 1)
            process = read_json(parent / "secondary_process/reference-labels.json")
            process_weak = {cid: np.array([row["weak_signal"] == 1 and row["activity_label"] == 1 for row in rows])
                            for cid, rows in process.items()}
            targets = [(batch, labels, weak, read_json(out / batch / "predictions.json")),
                       (batch.replace("ai_", "process_"), process, process_weak,
                        read_json(out / batch / "process-predictions.json"))]
        receipts = verify_receipts(comparison / "run")
        used = read_json(out / batch / "used-requests.json")
        for target, reference, weak_mask, predictions in targets:
            results, summaries, per_record = score_target(reference, observed, weak_mask, predictions, used, receipts)
            folder = analysis / target
            folder.mkdir()
            write_json(folder / "results.json", results)
            write_csv(folder / "summary.csv", summaries)
            write_csv(folder / "per-record.csv", per_record)
            all_tables[target] = {row["method"]: row for row in summaries}
            text += f"## {target}\n\n" + full_table(summaries) + "\n\n"
            if target == batch:
                labels_by_batch[batch] = reference
                observed_by_batch[batch] = observed
                predictions_by_batch[batch] = predictions
    comparisons = []
    for control_stem in ("adaptive_no_feedback", "partial_fixed_image_review", "partial_first_only"):
        differences, weak_differences, net = [], [], 0
        for rep in (0, 1):
            main = f"partial_adaptive_no_feedback_r{rep}__conflict_scope"
            control = f"{control_stem}_r{rep}" + ("__whole_target" if control_stem == "partial_fixed_image_review"
                                                 else "__conflict_scope")
            totals = {}
            for method in (main, control):
                rows = [all_tables[batch][method] for batch in BATCHES]
                tp, fp, fn = [sum(row[key] for row in rows) for key in ("tp", "fp", "fn")]
                totals[method] = {"f1": 2 * tp / (2 * tp + fp + fn),
                    "weak_recall": sum(row["weak_detected_days"] for row in rows) / sum(row["weak_days"] for row in rows)}
            differences.append(totals[main]["f1"] - totals[control]["f1"])
            weak_differences.append(totals[main]["weak_recall"] - totals[control]["weak_recall"])
            for batch, reference in labels_by_batch.items():
                predictions = predictions_by_batch[batch]
                for cid, rows in reference.items():
                    actual = np.array([row["activity_label"] for row in rows])
                    net += correction_utility(actual, np.array(predictions[control][cid]["activity"]),
                        np.array(predictions[main][cid]["activity"]), observed_by_batch[batch][cid])["net_correct"]
        passed = np.mean(differences) > 0 and net > 0
        if control_stem == "adaptive_no_feedback":
            passed = passed and np.mean(weak_differences) >= 0
        comparisons.append({"control": control_stem, "f1_difference_by_repeat": differences,
            "mean_f1_difference": float(np.mean(differences)), "mean_weak_recall_difference": float(np.mean(weak_differences)),
            "net_correct_both_repeats": net, "passed": bool(passed)})
    decision = {"passed": all(row["passed"] for row in comparisons), "comparisons": comparisons,
                "default_changed": False, "scope": "Exposed development only; passing permits frozen confirmation"}
    write_json(analysis / "decision.json", decision)
    (analysis / "report.md").write_text(text, encoding="utf-8")
    write_json(analysis / "manifest.json", {"files": {path.relative_to(analysis).as_posix(): sha(path)
        for path in analysis.rglob("*") if path.is_file()}})
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "score"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out.resolve())
    else:
        score(args.out.resolve())
