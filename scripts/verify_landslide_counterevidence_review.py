"""Check independent scoring, veto contracts and exact unscored replay agreement."""

import argparse
from pathlib import Path

import numpy as np
from analyze_landslide_codex_observation_reference import scoring_inputs, stage_reference_arrays
from probe_landslide_counterevidence_review import BASE, BATCHES, HOLDOUT, ROOT
from run_landslide_comparison import load_labels
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import ActivityPrediction


def verify(out, earlier, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for filename in ("protocol.json", "predictions.json", "vetoes.json", "evidence.json", "source-bindings.json"):
        if sha(out / filename) != sha(earlier / filename):
            raise ValueError("Deterministic prepare replay differs: " + filename)
    for name, digest in read_json(out / "manifest.json")["files"].items():
        if sha(out / name) != digest:
            raise ValueError("Delivered results changed")
    for manifest in ("source-bindings.json", "reference-bindings.json"):
        for filename, digest in read_json(out / manifest).items():
            if sha(ROOT / filename) != digest:
                raise ValueError("Bound source or reference changed")
    predictions = read_json(out / "predictions.json")
    vetoes = read_json(out / "vetoes.json")
    results = read_json(out / "results.json")
    sources, populations = {}, {}
    for batch in (*BATCHES, "new-AI"):
        source = HOLDOUT / "run" if batch == "new-AI" else BASE / batch
        sources[batch] = {cid: load_case_features(source, cid)
                          for cid in predictions[batch]["direct_r0"]}
        if batch != "new-AI":
            reference = source / ("analysis-v1/reference-subset.csv" if batch == "background-review-v1" else "reference-subset.csv")
            labels = load_labels(reference, {cid: pair[0] for cid, pair in sources[batch].items()})
            populations[batch] = (batch, labels)
    for population, reference, batch in (("old-AI", BASE / "codex-observation-review-v1", "confirmation-v1"),
                                          ("new-AI", BASE / "ai-observable-holdout-data-v1/codex-review", "new-AI")):
        populations[population] = (batch, scoring_inputs(reference)[0])
    daily_rows, contracts, score_checks = 0, 0, 0
    for population, (batch, labels) in populations.items():
        for method, records in predictions[batch].items():
            actual_all, predicted_all, actual_stage, predicted_stage, internal_a, internal_p = [], [], [], [], [], []
            for cid, rows in labels.items():
                case, feature = sources[batch][cid]
                prediction = ActivityPrediction(**{key: np.array(value) if key in {"activity", "stage"} else value
                                                   for key, value in records[cid].items()})
                actual = np.array([int(row["activity_label"]) for row in rows])
                known = actual >= 0
                actual_all.extend(actual[known])
                predicted_all.extend(prediction.activity[known])
                daily_rows += len(actual)
                if np.any(prediction.activity[~feature.observed] != -1):
                    raise ValueError("Missing activity lost unknown state")
                if np.any((prediction.stage > 0) & ((prediction.activity != 1) | ~feature.supported)):
                    raise ValueError("Stages exceed activity/support")
                # CSV controls contain string codes; the AI JSON helper expects
                # integer codes when assigning the stationary stage reference.
                stage_rows = [dict(row, activity_label=int(row["activity_label"])) for row in rows]
                stage = stage_reference_arrays(stage_rows, prediction)
                actual_stage.extend(stage["actual_stage"][stage["stage_known"]])
                predicted_stage.extend(prediction.stage[stage["stage_known"]])
                internal_a.extend(stage["actual_stage"][stage["stage_interior"]])
                internal_p.extend(prediction.stage[stage["stage_interior"]])
            expected = results[population][method]["summary"]
            precision, recall, f1, _ = precision_recall_fscore_support(actual_all, predicted_all, labels=[1], zero_division=0)
            for key, number in (("precision", precision[0]), ("recall", recall[0]), ("f1", f1[0])):
                if not np.isclose(expected[key], number, rtol=0, atol=1e-12):
                    raise ValueError("Independent daily score differs: " + key)
            matrix = confusion_matrix(actual_all, predicted_all, labels=[-1, 0, 1])
            if (expected["tp"], expected["fp"], expected["fn"]) != (matrix[2, 2], matrix[1, 2], matrix[2, :2].sum()):
                raise ValueError("Independent confusion counts differ")
            for key, actual, predicted, codes in (("end_to_end_stage_f1", actual_stage, predicted_stage, [1, 2, 3]),
                                                  ("internal_stage_f1", internal_a, internal_p, [1, 2, 3])):
                value = precision_recall_fscore_support(actual, predicted, labels=codes, zero_division=0)[2].mean()
                if not np.isclose(expected[key], value, rtol=0, atol=1e-12):
                    raise ValueError("Independent stage score differs")
            score_checks += 1
    for batch, methods in vetoes.items():
        for method, records in methods.items():
            repeat = int(method[-1])
            for cid, details in records.items():
                initial = np.array(predictions[batch][f"direct_r{repeat}"][cid]["activity"])
                pure = np.array(predictions[batch][f"pure_r{repeat}"][cid]["activity"])
                final = np.array(predictions[batch][method][cid]["activity"])
                rejected = np.zeros(len(final), dtype=bool)
                rejected[details["rejected_days"]] = True
                if np.any(final[rejected] != initial[rejected]) or np.any(final[~rejected] != pure[~rejected]):
                    raise ValueError("Veto created a new class or changed unrelated dates")
                if np.any(rejected & ((initial < 0) | (pure < 0) | (initial == pure))):
                    raise ValueError("Unknown or unchanged decision vetoed")
                if method.startswith(("matched_uniform", "matched_center")):
                    candidate = vetoes[batch][f"counterevidence_r{repeat}"][cid]["rejected_days"]
                    for target in details["targets"]:
                        start, stop = target["start"], target["stop"]
                        if sum(start <= day < stop for day in candidate) != int(rejected[start:stop].sum()):
                            raise ValueError("Matched per-target veto budget differs")
                contracts += 1
    audit = {"status": "passed", "exact_prepare_replay": True, "unchanged_predictions_after_csv_type_fix": True,
             "source_and_reference_hashes": "passed", "sklearn_activity_and_stages": "passed",
             "veto_unknown_missing_stage_and_matched_budget_contracts": "passed", "configuration_scores": score_checks,
             "daily_evaluation_rows": daily_rows, "veto_contracts": contracts, "new_remote_calls": 0}
    write_json(destination / "audit.json", audit)
    print(audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--earlier", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    verify(args.out.resolve(), args.earlier.resolve(), args.destination.resolve())
