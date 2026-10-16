"""Verify generated data without opening reserved-test labels for selection."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide import LandslideInput, LandslideRequest
from gnss_sim.landslide_deformation_diversity import diverse_case
from gnss_sim.landslide_diagnostics import LandslideDiagnostics


def verify(data, out):
    manifest = read_json(data / "data-freeze.json")
    for relative, expected in manifest.items():
        if sha(data / relative) != expected:
            raise ValueError("Frozen data changed: " + relative)
    protocol = read_json(data / "protocol.json")
    cases = {}
    for group, ids in protocol["split"].items():
        for cid in ids:
            folder = data / "inputs" / cid
            case = LandslideInput.model_validate_json(
                (folder / "input.json").read_text(encoding="utf-8")
            )
            canonical = sha_bytes(case.model_dump_json().encode())
            for window in (31, 61, 91):
                motion = LandslideDiagnostics.model_validate_json(
                    (folder / f"motion-{window}.json").read_text(encoding="utf-8")
                )
                if motion.input_sha256 != canonical or motion.window_days != window:
                    raise ValueError("Motion diagnostics do not match original observations")
            if group != "mechanism_test":
                cases[cid] = case
    labels = {cid: [] for cid in cases}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            labels[row["case_id"]].append(row)
    ledger = read_json(data / "training-generation-ledger.json")
    counts = Counter()
    simultaneous = 0
    for entry in ledger:
        cid = entry["case_id"]
        truth = entry["truth"]
        case = cases[cid]
        rows = labels[cid]
        if len(rows) != len(case.dates):
            raise ValueError("Incomplete training/validation reference")
        counts[(entry["group"], truth["family"])] += 1
        expected_quality = set(truth["quality_event_indices"])
        for day, (row, raw) in enumerate(zip(rows, case.displacement_mm, strict=True)):
            if int(row["day_index"]) != day or row["date"] != str(case.dates[day]):
                raise ValueError("Reference calendar changed")
            activity, quality = int(row["activity_label"]), int(row["quality_event_label"])
            if raw is None:
                if activity != -1 or quality != -1 or row["feature"]:
                    raise ValueError("Missing observation acquired a label")
                continue
            expected = sum(
                np.array(truth[key][day])
                for key in ("latent_displacement_mm", "measurement_noise_mm", "artifact_mm")
            )
            np.testing.assert_array_equal(raw, expected)
            if activity != int(truth["speed_mm_day"][day] > 0) or quality != int(
                day in expected_quality
            ):
                raise ValueError("Motion and artifact references were conflated")
            if row["feature"] and activity != 1:
                raise ValueError("Stage outside activity")
            simultaneous += int(activity == quality == 1)
    request = LandslideRequest(seed=protocol["seed"], count=96, days=730)
    replayed = []
    for group in ("train", "validation"):
        for cid in protocol["split"][group][:2]:
            entry = next(row for row in ledger if row["case_id"] == cid)
            original = entry["truth"]
            case, _, truth = diverse_case(
                request, original["stream_index"], original["family"], cid
            )
            if case.model_dump_json() != cases[cid].model_dump_json() or truth != original:
                raise ValueError("Generation replay differs")
            replayed.append(cid)
    if len({entry["truth"]["noise_seed"] for entry in ledger}) != len(ledger):
        raise ValueError("Training/validation background seed reused")
    write_json(
        out,
        {
            "status": "VERIFIED",
            "frozen_files": len(manifest),
            "original_inputs_and_diagnostics_checked": 96,
            "training_validation_reference_days_checked": sum(
                len(rows) for rows in labels.values()
            ),
            "observed_motion_quality_coincidences": simultaneous,
            "family_counts": {
                group + "/" + family: count for (group, family), count in counts.items()
            },
            "replayed_records": replayed,
            "reserved_test_labels_opened": False,
            "test_label_validation": "Hash binding only until model freeze; no test reference content imported",
        },
    )
    print(
        "Verified96 input calendars/diagnostics,72 reference records,4 generation replays; test labels unopened"
    )


def sha_bytes(value):
    import hashlib

    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.data.resolve(), args.out.resolve())
