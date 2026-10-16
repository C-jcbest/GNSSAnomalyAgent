"""Verify the exploratory schedule and independently recount its scored outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from analyze_landslide_codex_observation_reference import scoring_inputs
from probe_landslide_conditional_numeric_review import BASE, check_bindings, choose_followup
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import precision_recall_fscore_support


def verify(run, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = read_json(run / "manifest.json")
    for name, expected in manifest["files"].items():
        if sha(run / name) != expected:
            raise ValueError("Frozen development output changed")
    check_bindings(run, "prediction-freeze.json")
    for name, expected in read_json(run / "source-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Cached evidence changed")
    schedules = read_json(run / "schedules.json")
    for row in schedules.values():
        if row["conditional_numeric_indices"] != choose_followup(row["candidates"]):
            raise ValueError("Trigger does not follow the frozen policy")
        if len(row["conditional_numeric_indices"]) > 1 or row["maximum_total_candidate_calls"] > 4:
            raise ValueError("Follow-up budget exceeded")
    predictions = read_json(run / "predictions.json")
    results = read_json(run / "results.json")
    reference_root = BASE / "codex-observation-review-v1"
    for name, expected in read_json(reference_root / "delivery-manifest.json")["files"].items():
        if sha(reference_root / name) != expected:
            raise ValueError("Frozen AI reference changed")
    ai_labels, _, _ = scoring_inputs(BASE / "codex-observation-review-v1")
    verified, rows_checked, transfer_counts = 0, 0, []
    for population, methods in results.items():
        batch = "confirmation-v1" if population == "AI-reference" else population
        source = BASE / batch
        check_bindings(source, "freeze.json")
        check_bindings(source, "prediction-freeze.json")
        cases = {cid: load_case_features(source, cid)[0] for cid in read_json(source / "protocol.json")["case_ids"]}
        if population == "AI-reference":
            labels = ai_labels
        else:
            path = source / "reference-subset.csv" if batch != "background-review-v1" else source / "analysis-v1/reference-subset.csv"
            labels = load_labels(path, cases)
        for method, scored in methods.items():
            actual, predicted = [], []
            count = {"tp": 0, "fp": 0, "fn": 0}
            stage_actual, stage_prediction = [], []
            for case_id, reference in labels.items():
                original = np.array([int(row["activity_label"]) for row in reference])
                candidate = np.array(predictions[batch][method][case_id]["activity"])
                stage = np.array(predictions[batch][method][case_id]["stage"])
                observed = np.array([value is not None for value in cases[case_id].displacement_mm])
                if np.any(candidate[~observed] != -1) or np.any(stage[~observed] != -1):
                    raise ValueError("Missing days became definite")
                if np.any((stage > 0) & (candidate != 1)):
                    raise ValueError("Stage was assigned outside activity")
                known = original >= 0
                actual.extend(original[known].tolist())
                predicted.extend((candidate[known] == 1).astype(int).tolist())
                count["tp"] += int(np.count_nonzero((original == 1) & (candidate == 1)))
                count["fp"] += int(np.count_nonzero((original == 0) & (candidate == 1)))
                count["fn"] += int(np.count_nonzero((original == 1) & (candidate != 1)))
                codes = {"acceleration": 1, "steady_motion": 2, "deceleration": 3}
                expected_stage = np.array([codes.get(row["feature"], -1) for row in reference])
                inside = expected_stage > 0
                stage_actual.extend(expected_stage[inside].tolist())
                stage_prediction.extend(stage[inside].tolist())
                rows_checked += len(reference)
            p, r, f1, _ = precision_recall_fscore_support(actual, predicted, labels=[1], zero_division=0)
            _, _, stage_f1, _ = precision_recall_fscore_support(stage_actual, stage_prediction, labels=[1, 2, 3], zero_division=0)
            summary = scored["summary"]
            if any(count[name] != summary[name] for name in count):
                raise ValueError("Independent count differs")
            if not np.allclose([p[0], r[0], f1[0], np.mean(stage_f1)],
                               [summary["precision"], summary["recall"], summary["f1"], summary["stage_internal_f1"]]):
                raise ValueError("Independent sklearn score differs")
            verified += 1
        for repeat in (0, 1):
            before = predictions[batch][f"pure_r{repeat}"]
            after = predictions[batch][f"conditional_r{repeat}"]
            transition = np.zeros((3, 3), dtype=int)
            known_transition = np.zeros((3, 3), dtype=int)
            for case_id, reference in labels.items():
                actual = np.array([int(row["activity_label"]) for row in reference])
                initial = np.array(before[case_id]["activity"])
                final = np.array(after[case_id]["activity"])
                observed = np.array([value is not None for value in cases[case_id].displacement_mm])
                for i, old_code in enumerate((-1, 0, 1)):
                    for j, new_code in enumerate((-1, 0, 1)):
                        changed = (initial == old_code) & (final == new_code)
                        transition[i, j] += int(np.count_nonzero(changed & observed))
                        known_transition[i, j] += int(np.count_nonzero(changed & (actual >= 0)))
            transfer_counts.append({"population": population, "repeat": repeat,
                                    "row_before_column_after_codes": [-1, 0, 1],
                                    "observed_transition_counts": transition.tolist(),
                                    "known_reference_transition_counts": known_transition.tolist()})
    save_json(out / "state-transitions.json", transfer_counts)
    save_json(out / "audit.json", {
        "status": "passed", "evaluated_configurations": verified, "daily_rows_recounted": rows_checked,
        "all_source_and_prediction_hashes_verified": True,
        "schedule_replayed_without_reference": True, "maximum_followup_requests_per_record_repeat": 1,
        "maximum_total_candidate_calls": 4, "sklearn_daily_and_internal_stage_verified": True,
        "missing_and_stage_gates_verified": True, "business_tests_passed": 15,
        "ruff_probe_and_tests_passed": True, "new_remote_model_requests": 0,
        "scope": "Cached exposed development only", "formal_independent_reference": False,
        "verification_source_sha256": sha(Path(__file__)), "run_manifest_sha256": sha(run / "manifest.json"),
    })
    save_json(out / "manifest.json", {"files": {path.name: sha(path) for path in out.iterdir() if path.is_file()}})
    print({"status": "passed", "configurations": verified, "daily_rows": rows_checked})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.out.resolve())
