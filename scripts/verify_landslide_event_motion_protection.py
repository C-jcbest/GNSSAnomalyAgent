"""Independently verify the ordered-event protection replay and all scores."""

import argparse
from pathlib import Path

import numpy as np
from probe_landslide_event_motion_protection import BASE, CACHE, HOLDOUT, ROOT, populations
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import precision_recall_fscore_support

from gnss_sim.artifacts import sha, write_json


def verify(out, replay, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for filename in ("protocol.json", "predictions.json", "protections.json", "event-evidence.json", "source-bindings.json"):
        if sha(out / filename) != sha(replay / filename):
            raise ValueError("Deterministic preparation replay differs: " + filename)
    for filename, digest in read_json(out / "manifest.json")["files"].items():
        if sha(out / filename) != digest:
            raise ValueError("Trial results changed")
    reference_bindings = read_json(CACHE / "reference-bindings.json")
    for filename, digest in {**reference_bindings, **read_json(out / "source-bindings.json")}.items():
        if sha(ROOT / filename) != digest:
            raise ValueError("Source or reference differs from frozen comparison")
    predictions = read_json(out / "predictions.json")
    results = read_json(out / "results.json")
    protections = read_json(out / "protections.json")
    rows, scores, contracts = 0, 0, 0
    for population, (batch, labels, _) in populations(predictions).items():
        source = HOLDOUT / "run" if batch == "new-AI" else BASE / batch
        features = {cid: load_case_features(source, cid)[1] for cid in labels}
        for method, records in predictions[batch].items():
            actual_all, predicted_all, stage_all, stage_prediction, internal_actual, internal_prediction = [], [], [], [], [], []
            for cid, reference in labels.items():
                activity = np.array(records[cid]["activity"])
                stage = np.array(records[cid]["stage"])
                actual = np.array([int(row["activity_label"]) for row in reference])
                known = actual >= 0
                actual_all.extend(actual[known])
                predicted_all.extend(activity[known])
                expected_stage = np.array([{"acceleration": 1, "steady_motion": 2, "deceleration": 3}.get(row["feature"], -1)
                                           for row in reference])
                expected_stage[actual == 0] = 0
                stage_known, interior = expected_stage >= 0, expected_stage > 0
                stage_all.extend(expected_stage[stage_known])
                stage_prediction.extend(stage[stage_known])
                internal_actual.extend(expected_stage[interior])
                internal_prediction.extend(stage[interior])
                if np.any(activity[~features[cid].observed] != -1):
                    raise ValueError("Missing date has a definite activity")
                if np.any((stage > 0) & ((activity != 1) | ~features[cid].supported)):
                    raise ValueError("Stage exceeds activity/support")
                rows += len(reference)
            expected = results[population][method]["summary"]
            precision, recall, f1, _ = precision_recall_fscore_support(actual_all, predicted_all, labels=[1], zero_division=0)
            for name, value in (("precision", precision[0]), ("recall", recall[0]), ("f1", f1[0])):
                if not np.isclose(expected[name], value, rtol=0, atol=1e-12):
                    raise ValueError("Independent activity score differs")
            for name, actual, predicted in (("internal_stage_f1", internal_actual, internal_prediction),
                                             ("end_to_end_stage_f1", stage_all, stage_prediction)):
                value = precision_recall_fscore_support(actual, predicted, labels=[1, 2, 3], zero_division=0)[2].mean()
                if not np.isclose(expected[name], value, rtol=0, atol=1e-12):
                    raise ValueError("Independent stage score differs")
            scores += 1
    for batch, methods in protections.items():
        for method, records in methods.items():
            repeat = int(method[-1])
            for cid, details in records.items():
                initial = np.array(predictions[batch][f"direct_r{repeat}"][cid]["activity"])
                pure = np.array(predictions[batch][f"pure_r{repeat}"][cid]["activity"])
                final = np.array(predictions[batch][method][cid]["activity"])
                protected = np.zeros(len(initial), dtype=bool)
                protected[details["protected_days"]] = True
                if np.any(final[protected] != initial[protected]) or np.any(final[~protected] != pure[~protected]):
                    raise ValueError("Protection creates a label or changes an unrelated date")
                if np.any(protected & ((initial != 1) | (pure != 0))):
                    raise ValueError("Protection is not a definite activity denial")
                if not set(details["protected_days"]).issubset(details["eligible_days"]):
                    raise ValueError("Protection exceeds original permission")
                if method.startswith("matched_"):
                    candidate = protections[batch][f"event_protection_r{repeat}"][cid]["protected_days"]
                    for target in details["targets"]:
                        start, stop = target["start"], target["stop"]
                        if sum(start <= day < stop for day in candidate) != int(protected[start:stop].sum()):
                            raise ValueError("Matched target budget differs")
                contracts += 1
    write_json(destination / "reference-bindings.json", reference_bindings)
    audit = {"status": "passed", "exact_preparation_replay": True, "hashes": "passed",
             "sklearn_activity_internal_and_end_to_end_stages": "passed", "protection_contracts": contracts,
             "configuration_scores": scores, "daily_evaluation_rows": rows, "new_remote_calls": 0}
    write_json(destination / "audit.json", audit)
    print(audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    verify(args.out.resolve(), args.replay.resolve(), args.destination.resolve())
