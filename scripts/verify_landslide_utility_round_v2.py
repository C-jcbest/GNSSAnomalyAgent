"""Independently export event, daily and calibration checks for the utility round."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_proposal_utility_v2 import raw_prior_matrix
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_missed_motion_repair import write_rows
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import score_counts
from gnss_sim.landslide_proposal_utility import (
    expected_proposal_utility,
    fit_proposal_model,
    proposal_features,
)
from gnss_sim.landslide_review_utility import correction_utility


def verify_round(run, out):
    config = read_json(run / "protocol.json")
    pilot, data = Path(config["pilot"]), Path(config["data"])
    train, validation = config["split"]["train"], config["split"]["validation"]
    records = {cid: load_case_features(pilot, cid) for cid in train + validation}
    probabilities = read_json(run / "numeric-probabilities.json")
    crossfit, full = XGBClassifier(), XGBClassifier()
    crossfit.load_model(run / "cross-fitted-prior.json")
    full.load_model(data / "xgboost_84.json")
    for cid in train + validation:
        model = crossfit if cid in train else full
        reproduced = model.predict_proba(raw_prior_matrix(*records[cid])[0])[:, 1]
        np.testing.assert_array_equal(reproduced.astype(float), probabilities[cid])
    models = read_json(run / "models.json")
    training = read_json(run / "training-utilities.json")
    matrix = np.array([row["features"] for row in training])
    target = np.array([row["utility_fraction"] for row in training])
    for name, indices in (("proposal_guard", list(range(5))), ("visual_proposal_guard", [0, 1, 2])):
        if fit_proposal_model(matrix, target, indices) != models[name]:
            raise ValueError("Proposal acceptance model does not refit exactly")
    initial_saved = read_json(pilot / "initial-predictions.json")
    local_saved = read_json(pilot / "local-predictions.json")
    predictions = read_json(run / "predictions.json")
    labels = load_labels(run / "reference-subset.csv", {cid: records[cid][0] for cid in validation})
    changes = []
    for proposal in read_json(run / "proposals.json"):
        method, cid, index = proposal["method"], proposal["case_id"], proposal["index"]
        rep = int(method[-1])
        case, features = records[cid]
        initial = np.array(initial_saved[f"{cid}-direct-r{rep}"]["activity"])
        reviewed = np.array(local_saved[f"{cid}-window-{index:02d}-image-r{rep}"]["activity"])
        start, stop = regular_windows(len(initial), 180, 90)[index]
        long = raw_motion_evidence(case, features)
        neighbourhood = conflict_neighbourhood(initial, features, long, ReviewPolicy())[1]
        scope = target_scopes(neighbourhood, features.observed, [{"start": start, "stop": stop}])[0]
        row, changed = proposal_features(initial, reviewed, scope, features, long, probabilities[cid], ReviewPolicy())
        np.testing.assert_array_equal(row, proposal["features"])
        if method.startswith("numeric_vote"):
            expected = float(row[4])
        else:
            name = "visual_proposal_guard" if method.startswith("visual_proposal") else "proposal_guard"
            expected = expected_proposal_utility(models[name], row)
        if expected != proposal["expected_utility"] or bool(expected > 0 and changed.any()) != proposal["accepted"]:
            raise ValueError("Recorded proposal approval does not replay")
        actual = np.array([int(item["activity_label"]) for item in labels[cid]])
        changes.append({"method": method, "case_id": cid, "index": index, "accepted": proposal["accepted"],
                        **correction_utility(actual, initial, reviewed, changed)})
    events, daily = [], []
    for method, values in predictions.items():
        for threshold in (.3, .5):
            counts = np.zeros(3, dtype=int)
            for cid, value in values.items():
                detail = boundary_errors(labels[cid], np.array(value["activity"]), threshold)
                counts += [len(detail["matched_pairs"]), detail["unmatched_prediction_events"], detail["missed_events"]]
            events.append({"method": method, "iou": threshold, **score_counts(*counts)})
        for cid, value in values.items():
            for index, row in enumerate(labels[cid]):
                daily.append({"method": method, "case_id": cid, "day_index": index, "date": row["date"],
                              "actual_activity": row["activity_label"], "actual_stage": row["feature"],
                              "predicted_activity": value["activity"][index], "predicted_stage": value["stage"][index]})
    with (run / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    for row in summary:
        measured = next(item for item in events if item["method"] == row["method"] and item["iou"] == .3)
        if measured["f1"] != float(row["event_f1_iou03"]):
            raise ValueError("Independent event recount differs")
        row["event_f1_iou05"] = next(item["f1"] for item in events if item["method"] == row["method"] and item["iou"] == .5)
    previous = run.parent / "proposal-utility-development-v1"
    identical = {name: sha(run / name) == sha(previous / name) for name in
                 ("cross-fitted-prior.json", "numeric-probabilities.json", "models.json", "training-utilities.json",
                  "predictions.json", "proposals.json", "permissions.json")}
    if not all(identical.values()):
        raise ValueError("The scoring repair changed a frozen model or prediction")
    same_as_numeric = {str(rep): predictions[f"proposal_guard_r{rep}"] == predictions[f"numeric_vote_guard_r{rep}"]
                       for rep in (0, 1)}
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out / "summary-with-events.csv", summary)
    write_rows(out / "events.csv", events)
    write_rows(out / "daily-predictions.csv", daily)
    write_rows(out / "actual-proposal-utilities.csv", changes)
    save_json(out / "checks.json", {"status": "complete", "prior_probabilities_recomputed": len(train) + len(validation),
                                   "models_refit_exactly": len(models), "proposal_approvals_replayed": len(changes),
                                   "event_recounts": len(events), "daily_rows": len(daily),
                                   "v1_v2_frozen_artifacts_identical": identical,
                                   "main_equals_uncalibrated_numeric_vote": same_as_numeric,
                                   "new_requests": 0, "new_records": 0,
                                   "interpretation": "No independent benefit established for learned approval over numeric vote"})
    save_json(out / "source-bindings.json", {str(Path(__file__).resolve()): sha(Path(__file__))})
    print(read_json(out / "checks.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify_round(args.run, args.out)
