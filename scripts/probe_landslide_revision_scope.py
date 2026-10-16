"""Replay bounded revision authority on cached local judgments; no new VLM calls.

The numerical evidence chooses which dates can be revised, never assigns their
activity state. A matched-size central scope controls for merely changing fewer
dates. This is development evidence, including previously inspected confirmations.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence, review_priorities
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case, runs
from gnss_sim.landslide_local_experiment import learned_stages

PROTOCOL = {
    "schema": "landslide-revision-scope-probe-v1",
    "scope": "Cached development mechanism replay on three already-viewed controlled "
             "batches and unlabelled field transfer. No independent confirmation claim.",
    "policy": {"maximum_reviews": 2, "minimum_conflict_days": 14,
               "minimum_rate": .03, "minimum_signal_to_noise": 2.0},
    "revision_neighbourhood_days": 49,
    "neighbourhood_basis": "Largest raw evidence half-footprint: 42-day centre separation "
                           "plus seven days for a median block; no threshold sweep",
    "rules": {
        "whole_target": "Original cached review overwrites every selected target day",
        "conflict_scope": "Keep the original global prediction outside the original raw "
                          "conflict dates and their +/-49-day neighbourhood, within the "
                          "selected targets; otherwise accept the same cached local judgment",
        "matched_center_scope": "Same number of observed editable dates in each target "
                                "as conflict_scope, chosen by distance from target centre",
    },
    "stage": "Same frozen train24 stage head; gate again on final activity",
    "fairness": "All four visual arms are retained. Original six numerical baselines "
                "are reported without pretending scope selection retrains their models.",
    "failures": "Initial failed prediction remains unknown outside editable dates; "
                "a failed local decision remains unknown inside its revision scope",
    "denominator": "Missing and unknown days retain the common evaluation protocol",
    "selection": "Reject an unhelpful scope rather than force it into the method; "
                 "any selected change requires separately frozen new evaluation data",
}


def conflict_neighbourhood(initial, features, long_evidence, policy, margin=49):
    """Expand the existing raw-conflict support by its observation half-footprint."""
    short_supported = np.isfinite(features.progressive_rate) & features.observed
    long_supported = np.isfinite(long_evidence["rate"]) & features.observed
    short_motion = ((features.progressive_rate >= policy.minimum_rate)
                    & (features.signal_to_noise >= policy.minimum_signal_to_noise))
    long_motion = ((long_evidence["rate"] >= policy.minimum_rate)
                   & (long_evidence["signal_to_noise"] >= policy.minimum_signal_to_noise))
    supported = short_supported | long_supported
    motion = (short_supported & short_motion) | (long_supported & long_motion)
    conflict = supported & (motion != (initial == 1))
    neighbourhood = np.zeros(len(initial), dtype=bool)
    for start, stop in runs(conflict):
        neighbourhood[max(0, start - margin):min(len(initial), stop + margin)] = True
    return conflict, neighbourhood & features.observed


def target_scopes(neighbourhood, observed, targets):
    """Compare evidence placement with a central scope at equal observed-day budget."""
    conflict_scope = np.zeros(len(observed), dtype=bool)
    center_scope = np.zeros(len(observed), dtype=bool)
    covered = np.zeros(len(observed), dtype=bool)
    details = []
    for target in targets:
        start, stop = target["start"], target["stop"]
        if not 0 <= start < stop <= len(observed):
            raise ValueError("Revision target lies outside the calendar")
        if np.any(covered[start:stop]):
            raise ValueError("Revision targets must be disjoint")
        covered[start:stop] = True
        conflict_scope[start:stop] = neighbourhood[start:stop]
        budget = int(np.count_nonzero(neighbourhood[start:stop]))
        available = np.flatnonzero(observed[start:stop]) + start
        middle = (start + stop - 1) / 2
        ordering = np.lexsort((available, np.abs(available - middle)))
        center_scope[available[ordering[:budget]]] = True
        details.append({"start": start, "stop": stop, "editable_observed_days": budget,
                        "target_observed_days": int(observed[start:stop].sum())})
    return conflict_scope, center_scope, details


def scoped_activity(initial, reviewed, scope, observed):
    """Restrict revision authority without converting raw evidence into predictions."""
    for values in (initial, reviewed):
        if values.shape != observed.shape or not np.isin(values, [-1, 0, 1]).all():
            raise ValueError("Prediction calendar or class differs")
        if np.any(values[~observed] != -1):
            raise ValueError("Missing observations must stay unknown")
    if scope.shape != observed.shape or scope.dtype != bool or np.any(scope & ~observed):
        raise ValueError("Revision scope must contain observed calendar days only")
    activity = initial.copy()
    activity[scope] = reviewed[scope]
    return activity


def run_probe(out, experiments, evaluation_scope):
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", {**PROTOCOL, "scope": evaluation_scope})
    shutil.copyfile(Path(__file__), out / "source.py")
    policy = ReviewPolicy(**PROTOCOL["policy"])
    summary, scope_rows, record_rows, results, bindings = [], [], [], {}, {}
    for source, extra in experiments:
        name = source.name
        config = read_json(source / "protocol.json")
        for file, expected in read_json(source / "freeze.json").items():
            if sha(source / file) != expected:
                raise ValueError("Frozen input/source differs")
        for file, expected in read_json(source / "prediction-freeze.json").items():
            if sha(source / file) != expected:
                raise ValueError("Frozen prediction differs")
        cached = read_json(source / "predictions.json")
        selections = read_json(source / "selections.json")
        stage_model_path = source / "stage-xgboost.json"
        stage_model = XGBClassifier()
        stage_model.load_model(stage_model_path)
        cases, features, stage_codes = {}, {}, {}
        scope_details, predictions = {}, {}
        for cid in config["case_ids"]:
            cases[cid], features[cid] = load_case_features(source, cid)
            stage_codes[cid] = stage_model.predict(features[cid].matrix).astype(int) + 1
            long = raw_motion_evidence(cases[cid], features[cid])
            for rep in range(2):
                initial = np.array(cached[f"direct_visual_r{rep}"][cid]["activity"])
                conflict, neighbourhood = conflict_neighbourhood(initial, features[cid], long, policy)
                choice = selections[f"{cid}-r{rep}"]
                windows = [(row["start"], row["stop"]) for row in choice["priorities"]]
                priorities, selected = review_priorities(initial, features[cid], long, windows, policy)
                if priorities != choice["priorities"] or selected != choice["selected"]:
                    raise ValueError("Existing selection policy no longer replays exactly")
                if any(int(conflict[row["start"]:row["stop"]].sum()) != row["conflict_days"]
                       for row in priorities):
                    raise ValueError("Daily conflict support differs from existing policy")
                for method in ("fixed_image_review", "adaptive_no_feedback", "adaptive_numeric_feedback"):
                    targets = [priorities[index] for index in choice["fixed_indices"]]
                    if method != "fixed_image_review":
                        targets = selected
                    conflict_scope, center_scope, detail = target_scopes(
                        neighbourhood, features[cid].observed, targets)
                    method_name = f"{method}_r{rep}"
                    reviewed = np.array(cached[method_name][cid]["activity"])
                    scope_details.setdefault(method_name, {})[cid] = {
                        "conflict_runs": runs(conflict), "conflict_scope_runs": runs(conflict_scope),
                        "matched_center_scope_runs": runs(center_scope), "targets": detail,
                    }
                    scope_rows.extend({"run": name, "method": method_name, "case_id": cid, **row}
                                      for row in detail)
                    for variant, scope in (("whole_target", None), ("conflict_scope", conflict_scope),
                                           ("matched_center_scope", center_scope)):
                        activity = reviewed.copy()
                        if scope is not None:
                            activity = scoped_activity(initial, reviewed, scope, features[cid].observed)
                        prediction = learned_stages(stage_codes[cid], activity, features[cid].supported)
                        predictions.setdefault(f"{method_name}__{variant}", {})[cid] = prediction.to_dict()
                predictions.setdefault(f"direct_visual_r{rep}", {})[cid] = cached[f"direct_visual_r{rep}"][cid]
        for method, records in read_json(source / "numeric-predictions.json").items():
            if method != "ungated_rule":
                predictions[method] = {cid: records[cid] for cid in config["case_ids"]}
        if extra is not None:
            predictions.update(read_json(extra))
        target = out / name
        target.mkdir()
        save_json(target / "predictions.json", predictions)
        save_json(target / "revision-scopes.json", scope_details)
        save_json(target / "prediction-freeze.json", {
            file: sha(target / file) for file in ("predictions.json", "revision-scopes.json")
        })
        labelled = name != "field-transfer-v1"
        labels = load_labels(source / "reference-subset.csv", cases) if labelled else None
        results[name] = {}
        for method, records in predictions.items():
            case_scores, matched = {}, []
            total_weak, found_weak, changed_tp, removed_tp, changed_fp, removed_fp = 0, 0, 0, 0, 0, 0
            activity_days, unknown_days = 0, 0
            for cid, value in records.items():
                activity, stage = np.array(value["activity"]), np.array(value["stage"])
                prediction = ActivityPrediction(activity, stage, status=value["status"],
                                                gated=value["gated"], stage_status=value["stage_status"])
                prediction.validate(len(cases[cid].dates))
                activity_days += int(np.count_nonzero(activity == 1))
                unknown_days += int(np.count_nonzero((activity == -1) & features[cid].observed))
                if not labelled:
                    record_rows.append({"run": name, "method": method, "case_id": cid,
                                        "activity_days": int(np.count_nonzero(activity == 1))})
                    continue
                case_scores[cid] = evaluate_case(labels[cid], prediction)
                weak = np.array([int(row["weak_signal"]) == 1 for row in labels[cid]])
                actual = np.array([int(row["activity_label"]) for row in labels[cid]])
                total_weak += int(weak.sum())
                found_weak += int(np.count_nonzero(weak & (activity == 1)))
                if "__" in method:
                    original = np.array(cached[method.split("__")[0]][cid]["activity"])
                    gained = (activity == 1) & (original != 1)
                    removed = (activity != 1) & (original == 1)
                    changed_tp += int(np.count_nonzero(gained & (actual == 1)))
                    removed_tp += int(np.count_nonzero(removed & (actual == 1)))
                    changed_fp += int(np.count_nonzero(gained & (actual == 0)))
                    removed_fp += int(np.count_nonzero(removed & (actual == 0)))
                matched.extend(boundary_errors(labels[cid], activity, .3)["matched_pairs"])
                record_rows.append({"run": name, "method": method, "case_id": cid,
                                    **case_scores[cid]["daily"],
                                    "weak_detected_days": int(np.count_nonzero(weak & (activity == 1)))})
            row = {"run": name, "method": method, "labelled": labelled,
                   "activity_days": activity_days, "unknown_observed_days": unknown_days}
            if labelled:
                aggregate = aggregate_scores(list(case_scores.values()))
                if len(matched) != aggregate["events"]["tp"]:
                    raise ValueError("Common boundary and event scoring differ")
                row.update(**aggregate["daily"], event_f1=aggregate["events"]["f1"],
                           coverage=aggregate["coverage"], stage_macro_f1=aggregate["stage_macro_f1"],
                           weak_recall=found_weak / total_weak,
                           gained_tp_vs_whole=changed_tp, lost_tp_vs_whole=removed_tp,
                           gained_fp_vs_whole=changed_fp, removed_fp_vs_whole=removed_fp)
                row["onset_mae_matched_days"] = float(np.mean([
                    abs(pair["onset_error_days"]) for pair in matched
                ])) if matched else None
                row["offset_mae_matched_days"] = float(np.mean([
                    abs(pair["offset_error_days"]) for pair in matched
                ])) if matched else None
                results[name][method] = {"aggregate": aggregate, "per_case": case_scores}
            summary.append(row)
        bindings[name] = {file: sha(source / file) for file in (
            "freeze.json", "prediction-freeze.json", "selections.json", "stage-xgboost.json"
        )}
        if extra is not None:
            bindings[name][str(extra)] = sha(extra)
        print(f"Revision-scope replay completed: {name}, {len(predictions)} methods", flush=True)
    save_json(out / "results.json", results)
    save_json(out / "source-bindings.json", bindings)
    for file, rows in (("summary.csv", summary), ("revision-sizes.csv", scope_rows),
                       ("per-record.csv", record_rows)):
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (out / file).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    save_json(out / "manifest.json", {
        "status": "complete", "new_vlm_requests": 0,
        "files": {str(path.relative_to(out)): sha(path)
                  for path in out.rglob("*") if path.is_file()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--additional-baselines", type=Path)
    args = parser.parse_args()
    base = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
    if args.run is None:
        if args.additional_baselines is not None:
            parser.error("--additional-baselines requires --run")
        experiments = (
            (base / "mechanism-v1", base / "raw-baselines-v1/predictions.json"),
            (base / "confirmation-v1", base / "confirmation-data-v1/additional-baselines/predictions.json"),
            (base / "interval-confirmation-v1", base / "interval-confirmation-data-v1/additional-baselines/predictions.json"),
            (base / "field-transfer-v1", None),
        )
        scope = PROTOCOL["scope"]
    else:
        experiments = ((args.run, args.additional_baselines),)
        scope = "Single frozen source evaluation; independence standing comes from "
        scope += "the source generation and candidate pre-freeze protocol, never from this CLI"
    run_probe(args.out, experiments, scope)


if __name__ == "__main__":
    main()
