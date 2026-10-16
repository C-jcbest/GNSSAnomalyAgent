"""Score frozen continuation views with the registered union of overlapping targets.

The first scorer called a disjoint-target helper. Preserve that failed scorer and
all inference; this implementation applies the protocol's existing union rule.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from analyze_landslide_review_utility_v2 import verify_receipts
from probe_landslide_revision_scope import scoped_activity
from run_landslide_censored_followup import (
    ARMS,
    BASE,
    BATCHES,
    reference_for_batch,
    save_target,
    verify,
)
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import combine_reviews
from gnss_sim.landslide_local_experiment import learned_stages


def overlapping_scope(authority, observed, reviews):
    scope = np.zeros(len(observed), dtype=bool)
    for review in reviews:
        start, stop = review["start"], review["stop"]
        if not 0 <= start < stop <= len(observed):
            raise ValueError("Target lies outside the original observation calendar")
        scope[start:stop] = True
    return scope & authority & observed


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
                    reviews = [{**row["first"], **row["first_result"]}]
                    second = row["targets"][arm]
                    if second is not None:
                        keys.append(second["request_id"])
                        reviews.append({**second, **outputs[second["request_id"]]})
                    reviewed = combine_reviews(initial, reviews, feature.observed).activity
                    if arm == "censored_latest" and second is not None:
                        value = np.array(outputs[second["request_id"]]["activity"])
                        reviewed[second["start"]:second["stop"]] = value[second["start"]:second["stop"]]
                    scope = overlapping_scope(np.array(row["authority"]), feature.observed, reviews)
                    activity = scoped_activity(initial, reviewed, scope, feature.observed)
                method = f"followup_{arm}_r{rep}"
                predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                used.setdefault(method, {})[cid] = keys
    return predictions, used


def analyze(out, folder):
    from run_landslide_ai_reference_holdout import verify_bindings

    verify(out)
    verify_bindings(out, "prediction-freeze.json")
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("Freeze all answers before scoring")
    folder.mkdir(exist_ok=False)
    (folder / "source.py").write_bytes(Path(__file__).read_bytes())
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
    comparisons = []
    for stem in ("adaptive_no_feedback", "followup_fixed_short"):
        controls = [f"{stem}_r{rep}" + ("__conflict_scope" if stem == "adaptive_no_feedback" else "") for rep in (0, 1)]
        main = [f"followup_censored_r{rep}" for rep in (0, 1)]
        net = sum(row["net_correct"] for row in all_corrections if row["control"] in controls)
        difference = float(np.mean([pooled[a]["f1"] - pooled[b]["f1"] for a, b in zip(main, controls, strict=True)]))
        weak_difference = float(np.mean([pooled[a]["weak_recall"] - pooled[b]["weak_recall"] for a, b in zip(main, controls, strict=True)]))
        passed = difference > 0 and net > 0
        if stem == "adaptive_no_feedback":
            passed = passed and weak_difference >= 0
        comparisons.append({"control": stem, "mean_pooled_f1_difference": difference,
            "mean_pooled_weak_recall_difference": weak_difference, "net_correct_both_repeats": net,
            "passed": passed})
    passed = all(row["passed"] for row in comparisons)
    write_json(folder / "decision.json", {"status": "PROMISING_DEVELOPMENT" if passed else "STOPPED",
        "continue_to_frozen_confirmation": passed, "comparisons": comparisons,
        "default_changed": False, "unseen_confirmation": False})
    actual = [read_json(path) for path in (out / "requests").glob("*.response.json")]
    write_json(folder / "audit.json", {"status": "passed", "original_scores_replayed": replayed,
        "methods_per_target": 44, "targets": 3, "new_records": 0, "new_training": False,
        "new_requests": len(list((out / "requests").glob("*.started.json"))), "new_receipts": len(actual),
        "reported_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in actual),
        "scorer_correction": "v1disjoint helper rejected overlap before first score; v2applies registered observed target union; all original inference and code retained"})
    write_json(folder / "manifest.json", {"files": {path.relative_to(folder).as_posix(): sha(path)
        for path in folder.rglob("*") if path.is_file()}})
    print(read_json(folder / "audit.json"), flush=True)
    print(read_json(folder / "decision.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run.resolve(), args.out.resolve())
