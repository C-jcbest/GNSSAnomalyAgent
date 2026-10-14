"""Retain run provenance and distinguish classifier changes from response variation."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.schema import Prediction, event_masks, observed_values


def signature(result):
    return sorted((event["kind"], event["start"], event["end"], event["status"])
                  for event in result["prediction"]["events"])


def audit_run(baseline, numeric, visual, report, provenance):
    registration = read_json(baseline / "registration.json")
    dataset = Path(registration["dataset"])
    manifest = read_json(dataset / "manifest.json")
    records = {record["station_id"]: record for record in manifest["records"]}
    current_source = source_hashes()
    assert registration["source_hashes"] == current_source
    assert registration["manifest_sha256"] == sha256(dataset / "manifest.json")
    for folder in (numeric, visual):
        assert read_json(folder / "registration.json")["source_hashes"] == current_source
    checked = 0
    numeric_changes = []
    visual_changes = []
    request_comparison = []
    for station in registration["station_ids"]:
        observation = read_json(dataset / records[station]["observation"])
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        for parent in ("original", "binary_rf_default"):
            work = visual / "predictions" / parent / station
            response = read_json(work / "result.json")
            if (work / "request.json").exists():
                receipt = read_json(work / "input-receipt.json")
                assert receipt["request_sha256"] == sha256(work / "request.json")
                for name, digest in receipt["images"].items():
                    assert digest == sha256(work / name)
            else:
                assert response["request_count"] == 0
                assert response["review_mode"] == "not_requested_no_ambiguity"
        original_numeric = read_json(baseline / "predictions/segment_numeric" / station / "result.json")
        candidate_numeric = read_json(numeric / "predictions/binary_rf_default_numeric" / station / "result.json")
        original_visual = read_json(report / "predictions/pair_focus_repair" / station / "result.json")
        candidate_visual = read_json(report / "predictions/binary_rf_default_repair" / station / "result.json")
        if signature(original_numeric) != signature(candidate_numeric):
            numeric_changes.append({"station": station, "original": signature(original_numeric),
                                    "candidate": signature(candidate_numeric)})
        if signature(original_visual) != signature(candidate_visual):
            visual_changes.append({"station": station, "original": signature(original_visual),
                                   "candidate": signature(candidate_visual)})
        first = visual / "predictions/original" / station / "request.json"
        second = visual / "predictions/binary_rf_default" / station / "request.json"
        if first.exists() and second.exists():
            request_comparison.append({"station": station, "identical_request": sha256(first) == sha256(second),
                                       "different_final_events": signature(original_visual) != signature(candidate_visual)})
        for result in (original_numeric, candidate_numeric, original_visual, candidate_visual):
            masks = event_masks(Prediction.model_validate(result["prediction"]), observed)
            union = np.any(np.column_stack(list(masks.values())), axis=1)
            assert np.array_equal(union, result["generic_mask"])
            assert not union[~observed].any()
            checked += 1
    for role, folder in (("baseline", baseline), ("numeric", numeric), ("visual", visual)):
        destination = provenance / role
        destination.mkdir(parents=True, exist_ok=False)
        # Original data, plots and model responses remain in the immutable local run.
        for name in ("registration.json", "training.json", "experiment-source.py",
                     "pilot-started.json", "pilot-completed.json", "remaining-started.json",
                     "remaining-completed.json", "execution.json"):
            source = folder / name
            if source.exists():
                shutil.copyfile(source, destination / name)
        snapshot = folder / "source"
        if snapshot.exists():
            shutil.copytree(snapshot, destination / "source")
    shutil.copyfile(dataset / "manifest.json", provenance / "dataset-manifest.json")
    return {"seed": manifest["seed"], "checked_predictions": checked,
            "numeric_changed_stations": numeric_changes, "final_changed_stations": visual_changes,
            "paired_request_comparison": request_comparison,
            "original_run": str(baseline.resolve()), "numeric_run": str(numeric.resolve()),
            "visual_run": str(visual.resolve())}


def main():
    output = Path("results/exp08-main-v1")
    provenance = output / "provenance"
    provenance.mkdir(exist_ok=False)
    runs = []
    for seed in (2026101412, 2026101413):
        runs.append(audit_run(Path(f"runs/exp08-seed{seed}-baseline-v1"),
                              Path(f"results/exp08-seed{seed}-numeric-v1"),
                              Path(f"runs/exp08-seed{seed}-visual-v1"),
                              Path(f"results/exp08-seed{seed}-test-v1"),
                              provenance / str(seed)))
    identical = [comparison for run in runs for comparison in run["paired_request_comparison"]
                 if comparison["identical_request"]]
    changed = [comparison for comparison in identical if comparison["different_final_events"]]
    receipt = {"source_and_missing_masks": "passed", "runs": runs,
               "identical_paired_requests": len(identical),
               "different_final_events_after_identical_requests": len(changed),
               "interpretation": "Identical requests with different responses measure service variation, not classifier effect."}
    write_json(output / "run-audit.json", receipt)
    shutil.copyfile(__file__, output / "audit-source.py")
    print({key: value for key, value in receipt.items() if key != "runs"})


if __name__ == "__main__":
    main()
