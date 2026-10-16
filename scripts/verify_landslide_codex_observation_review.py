"""Replay reference contracts and verify source, daily exports and plotted label masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_diagnostics import LandslideDiagnostics
from gnss_sim.landslide_reference_interface import partition_sha256
from gnss_sim.landslide_stage_reference import compile_reference


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(review_directory, pack):
    out = review_directory / "compiled"
    freeze = read_json(out / "reference-freeze.json")
    for name, digest in freeze["files"].items():
        if sha(out / name) != digest:
            raise ValueError("Compiled artifact changed: " + name)
    for manifest_path in (pack / "manifest.json", review_directory / "preparation-manifest.json"):
        for name, digest in read_json(manifest_path)["files"].items():
            if sha(manifest_path.parent / name) != digest:
                raise ValueError("Original input or prepared evidence changed: " + name)
    for filename, key in (("activity-decisions.json", "activity_decisions_sha256"),
                          ("activity-freeze.json", "activity_freeze_sha256"),
                          ("stage-decisions.json", "stage_decisions_sha256")):
        if sha(review_directory / filename) != freeze[key]:
            raise ValueError("Authored decision or pre-stage freeze changed")
    activity_freeze = read_json(review_directory / "activity-freeze.json")
    if (activity_freeze["auxiliary_images_inspected_before_this_freeze"]
            or activity_freeze["frozen_at"] > freeze["reviewed_at"]):
        raise ValueError("Activity-first chronology invalid")
    daily = read_json(out / "daily-reference.json")
    contexts = []
    plot_data = read_json(out / "plot-data.json")
    feature_codes = {"unknown": 0, "none": 1, "acceleration": 2,
                     "steady_motion": 3, "deceleration": 4}
    activity_codes = {-1: 0, 0: 1, 1: 2}
    for review in read_json(out / "activity-reviews.json"):
        case_id = review["case_id"]
        folder = review_directory / case_id
        case = LandslideInput.model_validate_json((folder / "input.json").read_bytes())
        if sha(folder / "input.json") != sha(pack / "observations" / case_id / "input.json"):
            raise ValueError("Input copy changed")
        rows = [row for row in daily if row["case_id"] == case_id]
        for day, row in enumerate(rows):
            missing = case.displacement_mm[day] is None
            if missing and (row["activity_label"] != -1 or row["feature"] != "unknown"):
                raise ValueError("Missing day has definite activity or stage")
            if row["stage_evaluable"] and (row["activity_label"] != 1 or missing):
                raise ValueError("Stage escapes observed confirmed activity")
            expected_activity = 3 if missing else activity_codes[row["activity_label"]]
            expected_stage = 5 if missing else feature_codes[row["feature"]]
            if (plot_data[case_id]["activity_codes"][day] != expected_activity
                    or plot_data[case_id]["stage_codes"][day] != expected_stage):
                raise ValueError("Plot and export masks differ")
        contexts.append({"case": case, "review": review, "rows": rows,
                         "activity_sha256": partition_sha256(review), "motions": {
                             window: LandslideDiagnostics.model_validate_json(
                                 (folder / f"motion-{window}.json").read_bytes())
                             for window in (31, 61, 91)}})
    payload = read_json(out / "review.json")
    packet, replay, summary, support = compile_reference(payload, contexts, freeze["source_manifest_sha256"])
    if packet != payload or replay != daily or support != read_json(out / "support-inventory.json"):
        raise ValueError("Shared reference compiler replay differs")
    saved_summary = read_json(out / "summary.json")
    if any(saved_summary[key] != value for key, value in summary.items()):
        raise ValueError("Reference coverage summary differs")
    with (out / "daily-reference.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    expected_csv = [{key: str(value) for key, value in row.items()} for row in daily]
    if csv_rows != expected_csv:
        raise ValueError("CSV and JSON differ")
    ledger = read_json(out / "inspection-ledger.json")
    for name, digest in ledger["images_actually_inspected_sha256"].items():
        if sha(review_directory / name) != digest:
            raise ValueError("Inspected image changed")
    audit = {"status": "passed", "rows": len(daily), "cases": len(contexts),
             "inspected_images": len(ledger["images_actually_inspected_sha256"]),
             "original_pack_unchanged": True, "shared_compiler_replay_equal": True,
             "csv_json_plot_masks_equal": True, "missing_and_stage_gates_checked": True,
             "reference_use": "development_only", "formal_reference_verified": False,
             "new_remote_model_requests": 0, "verifier_sha256": sha(Path(__file__))}
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_directory", type=Path)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    result = verify(args.review_directory.resolve(), args.pack.resolve())
    if args.audit is not None:
        write_json(args.audit, result)
