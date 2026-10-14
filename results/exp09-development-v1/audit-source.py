"""Check paired membership and keep compact provenance for the completed study."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.schema import Prediction, event_masks, observed_values


def main():
    baseline = Path("runs/exp09-development-baseline-v1")
    visual = Path("runs/exp09-development-visual-v1")
    report = Path("results/exp09-development-v1")
    registered = read_json(visual / "registration.json")
    original = read_json(baseline / "registration.json")
    assert registered["source_hashes"] == original["source_hashes"] == source_hashes()
    assert registered["script_sha256"] == sha256(Path("scripts/experiment_exp09.py"))
    assert registered["protocol_sha256"] == sha256(Path("configs/experiment-exp09.json"))
    dataset = Path(registered["dataset"])
    manifest = read_json(dataset / "manifest.json")
    assert sha256(dataset / "manifest.json") == original["manifest_sha256"]
    records = {record["station_id"]: record for record in manifest["records"]}
    checked_predictions = 0
    identical_triplets = []
    actual_requests = 0
    for station in registered["station_ids"]:
        observation = read_json(dataset / records[station]["observation"])
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        request = visual / "inputs" / station / "request.json"
        if request.exists():
            receipt = read_json(visual / "inputs" / station / "input-receipt.json")
            assert receipt["request_sha256"] == sha256(request)
            for name, digest in receipt["images"].items():
                assert digest == sha256(request.parent / name)
            for member in range(1, 4):
                assert sha256(visual / f"member{member}" / station / "request.json") == sha256(request)
            identical_triplets.append(station)
        for member in range(1, 4):
            response = read_json(visual / f"member{member}" / station / "result.json")
            actual_requests += response["request_count"]
            if not request.exists():
                assert response["request_count"] == 0
                assert response["review_mode"] == "not_requested_no_ambiguity"
        for method in ("consensus_repair", "member1_repair", "member2_repair", "member3_repair"):
            result = read_json(report / "predictions" / method / station / "result.json")
            masks = event_masks(Prediction.model_validate(result["prediction"]), observed)
            union = np.any(np.column_stack(list(masks.values())), axis=1)
            assert np.array_equal(union, result["generic_mask"])
            assert not union[~observed].any()
            checked_predictions += 1
    assert actual_requests == len(registered["pending"]) == 3 * len(identical_triplets)
    provenance = report / "provenance"
    provenance.mkdir(exist_ok=False)
    for role, folder in (("baseline", baseline), ("visual", visual)):
        destination = provenance / role
        destination.mkdir()
        for name in ("registration.json", "training.json", "execution-started.json",
                     "execution-complete.json", "experiment-source.py", "pilot-started.json",
                     "pilot-completed.json", "remaining-started.json", "remaining-completed.json"):
            source = folder / name
            if source.exists():
                shutil.copyfile(source, destination / name)
        if (folder / "source").exists():
            shutil.copytree(folder / "source", destination / "source")
    shutil.copyfile(dataset / "manifest.json", provenance / "dataset-manifest.json")
    shutil.copyfile(__file__, report / "audit-source.py")
    write_json(report / "run-audit.json", {
        "source_and_missing_masks": "passed", "checked_predictions": checked_predictions,
        "identical_three_member_requests": identical_triplets, "actual_requests": actual_requests,
        "not_requested_stations": len(registered["station_ids"]) - len(identical_triplets),
        "note": "Separate planned requests, not statistically independent errors; all raw responses retained locally."})
    print({"checked_predictions": checked_predictions, "identical_triplets": len(identical_triplets),
           "actual_requests": actual_requests})


if __name__ == "__main__":
    main()
