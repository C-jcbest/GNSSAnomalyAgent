"""Check actual train labels, background exclusions and vote-relative changes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import load_labels, save_json, sha
from run_landslide_grouped_proposal import verify
from run_landslide_proposal_utility_v2 import raw_prior_matrix
from run_landslide_review_utility import training_labels
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_missed_motion_repair import write_rows
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import ReviewPolicy
from gnss_sim.landslide_proposal_utility import fit_proposal_model, proposal_features
from gnss_sim.landslide_review_utility import correction_utility


def check(calibration, complete, out):
    verify(calibration)
    config = read_json(calibration / "protocol.json")
    data = Path(config["data"])
    train, validation = config["split"]["train"], config["split"]["validation"]
    records = {cid: load_case_features(calibration, cid) for cid in train + validation}
    probabilities = read_json(calibration / "numeric-probabilities.json")
    actual = training_labels(data, train)
    initial_saved = read_json(calibration / "initial-predictions.json")
    local_saved = read_json(calibration / "local-predictions.json")
    selections = read_json(calibration / "training-selections.json")
    rebuilt = []
    for cid in train:
        case, features = records[cid]
        _, long = raw_prior_matrix(case, features)
        for rep in (0, 1):
            initial = np.array(initial_saved[f"{cid}-direct-r{rep}"]["activity"])
            neighbourhood = conflict_neighbourhood(initial, features, long, ReviewPolicy())[1]
            for selected in selections[f"{cid}-r{rep}"]["raw"]:
                reviewed = np.array(local_saved[f"{cid}-window-{selected['index']:02d}-image-r{rep}"]["activity"])
                permission = target_scopes(neighbourhood, features.observed, [selected])[0]
                row, changed = proposal_features(initial, reviewed, permission, features, long, probabilities[cid], ReviewPolicy())
                utility = correction_utility(actual[cid], initial, reviewed, changed)
                rebuilt.append({"case_id": cid, "repeat": rep, "index": selected["index"],
                                "features": row.tolist(), **utility})
    if rebuilt != read_json(calibration / "training-utilities.json"):
        raise ValueError("Deployment training utilities differ from real cached/new judgments")
    models = read_json(calibration / "models.json")
    matrix = np.array([row["features"] for row in rebuilt])
    target = [row["utility_fraction"] for row in rebuilt]
    for name, indices in (("grouped_proposal_guard", list(range(5))), ("visual_guard", [0, 1, 2])):
        if fit_proposal_model(matrix, target, indices) != models[name]:
            raise ValueError("Actual training targets do not refit the frozen models")
    exclusions = read_json(calibration / "prior-exclusions.json")
    full_prior = XGBClassifier()
    full_prior.load_model(data / "xgboost_84.json")
    for group, fitting in exclusions.items():
        if len(fitting) != 18 or set(fitting) != {cid for cid in train if config["training_groups"][cid] != int(group)}:
            raise ValueError("Background cross-fit used an incorrect population")
        model = XGBClassifier()
        model.load_model(calibration / f"prior-excluding-background-{group}.json")
        if model.get_booster().num_boosted_rounds() != 120:
            raise ValueError("A cross-fit prior changed the fixed number of trees")
        for cid in train:
            if config["training_groups"][cid] == int(group):
                probability = model.predict_proba(raw_prior_matrix(*records[cid])[0])[:, 1].astype(float)
                np.testing.assert_array_equal(probability, probabilities[cid])
    for cid in validation:
        probability = full_prior.predict_proba(raw_prior_matrix(*records[cid])[0])[:, 1].astype(float)
        np.testing.assert_array_equal(probability, probabilities[cid])
    # The complete prototype must use the calibration bytes without refitting.
    if sha(complete / "models.json") != sha(calibration / "models.json"):
        raise ValueError("The complete prototype refitted the acceptance model")
    changes = []
    for name, predictions_path, references in (
        ("validation", calibration / "predictions.json", calibration / "analysis-v1/reference-subset.csv"),
        ("complete", complete / "predictions.json", Path(read_json(complete / "protocol.json")["complete_cache"])),
    ):
        if name == "complete":
            source = Path(read_json(references / "protocol.json")["source"])
            references = source / "analysis-v1/reference-subset.csv"
        predictions = read_json(predictions_path)
        ids = list(predictions["grouped_proposal_guard_r0"])
        if name == "complete":
            cache = Path(read_json(complete / "protocol.json")["complete_cache"])
            cases = {cid: load_case_features(cache, cid)[0] for cid in ids}
        else:
            cases = {cid: records[cid][0] for cid in ids}
        labels = load_labels(references, cases)
        for rep in (0, 1):
            for cid in ids:
                actual = np.array([int(row["activity_label"]) for row in labels[cid]])
                main = np.array(predictions[f"grouped_proposal_guard_r{rep}"][cid]["activity"])
                vote = np.array(predictions[f"numeric_vote_r{rep}"][cid]["activity"])
                eligible = actual >= 0
                utility = correction_utility(actual, vote, main, eligible)
                changes.append({"scope": name, "repeat": rep, "case_id": cid, **utility,
                                "vote_positive_to_main_unknown": int((eligible & (vote == 1) & (main == -1)).sum()),
                                "vote_unknown_to_main_positive": int((eligible & (vote == -1) & (main == 1)).sum())})
    request_dir = calibration / "requests"
    new_validation = [path for path in request_dir.glob("*.started.json")
                      if path.name.startswith(tuple(validation))]
    # File metadata supplements execution ordering; it is not an external timestamp attestation.
    first_validation_ns = min(path.stat().st_mtime_ns for path in request_dir.glob("*.started.json")
                              if path.name.startswith(tuple(validation)))
    models_before_validation = (calibration / "model-freeze.json").stat().st_mtime_ns < first_validation_ns
    if not models_before_validation:
        raise ValueError("The model freeze did not precede new validation requests")
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out / "vote-relative-definite-corrections.csv", changes)
    save_json(out / "checks.json", {"status": "complete", "actual_training_proposals_rebuilt": len(rebuilt),
                                   "models_refit_exactly": len(models), "background_exclusions": len(exclusions),
                                   "probabilities_recomputed": len(train) + len(validation),
                                   "vote_relative_record_checks": len(changes), "complete_model_bytes_unchanged": True,
                                   "filesystem_models_before_new_validation": models_before_validation,
                                   "new_validation_started_receipts": len(new_validation),
                                   "ordering_scope": "Frozen execution logic plus local file metadata, not external attestation",
                                   "new_requests": 0, "new_records": 0})
    save_json(out / "source-bindings.json", {str(Path(__file__).resolve()): sha(Path(__file__))})
    print(read_json(out / "checks.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--complete", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check(args.calibration, args.complete, args.out)
