"""Validate the expanded dataset after the observational reference is frozen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from prepare_landslide_motion_probe import FAMILIES, controlled_case

from gnss_sim.artifacts import sha
from gnss_sim.landslide import LandslideInput, LandslideRequest


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_bindings(directory: Path, filename: str, nested: bool = False) -> int:
    bindings = read_json(directory / filename)
    if nested:
        bindings = bindings["files"]
    for name, expected in bindings.items():
        if sha(directory / name) != expected:
            raise ValueError("Frozen artifact changed: " + name)
    return len(bindings)


def observation_digest(case: LandslideInput) -> str:
    # Ignore the opaque record ID, so renaming cannot hide duplicate observations.
    content = case.model_dump(mode="json", exclude={"case_id"})
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def validate(data: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError("Keep prior validation results; use a new directory")
    protocol = read_json(data / "protocol.json")
    review = data / "codex-review"
    compiled = review / "compiled"
    bound = {
        "data_files": check_bindings(data, "data-freeze.json"),
        "observational_reference_files": check_bindings(compiled, "reference-freeze.json", True),
    }
    activity_freeze = read_json(review / "activity-freeze.json")
    if sha(review / "activity-decisions.json") != activity_freeze["activity_decisions_sha256"]:
        raise ValueError("Activity was revised after the first-layer freeze")
    if datetime.fromisoformat(read_json(compiled / "reference-freeze.json")["reviewed_at"]) <= datetime.fromisoformat(activity_freeze["frozen_at"]):
        raise ValueError("Stage review must follow activity freeze")
    ids = protocol["case_ids"]
    cases = {cid: LandslideInput.model_validate_json((data / "inputs" / cid / "input.json").read_bytes())
             for cid in ids}
    digests = [observation_digest(case) for case in cases.values()]
    if len(set(digests)) != len(ids):
        raise ValueError("Duplicate raw observations in the new batch")
    earlier = data.parent / "paired-persistence-confirmation-data-v1"
    earlier_cases = [LandslideInput.model_validate_json(path.read_bytes())
                     for path in sorted((earlier / "inputs").glob("*/input.json"))]
    if len(earlier_cases) != 24:
        raise ValueError("Earlier confirmation inventory could not be checked")
    if set(digests) & {observation_digest(case) for case in earlier_cases}:
        raise ValueError("New batch reuses an earlier confirmation observation")
    # Opening process references occurs only after frozen observational review.
    opened_at = datetime.now(timezone.utc).isoformat()
    with (data / "mechanism-sealed/reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        process_rows = list(csv.DictReader(stream))
    ledger = read_json(data / "mechanism-sealed/generation-ledger.json")
    if [row["case_id"] for row in ledger] != ids or len(process_rows) != 120 * 730:
        raise ValueError("Process references dropped or reordered a record")
    family_counts = Counter(row["truth"]["family"] for row in ledger)
    if family_counts != Counter({family: 20 for family in FAMILIES}):
        raise ValueError("Wrong family balance")
    review_families = Counter(row["truth"]["family"] for row in ledger
                             if row["case_id"] in protocol["secondary_review_ids"])
    if review_families != Counter({family: 4 for family in FAMILIES}):
        raise ValueError("Review population changed from the pre-specified sample")
    grouped = defaultdict(list)
    for row in process_rows:
        grouped[row["case_id"]].append(row)
    missing = active = 0
    for entry in ledger:
        cid = entry["case_id"]
        case, rows, truth = cases[cid], grouped[cid], entry["truth"]
        if len(rows) != 730 or [int(row["day_index"]) for row in rows] != list(range(730)):
            raise ValueError("Invalid original calendar")
        for day, row in enumerate(rows):
            unavailable = case.displacement_mm[day] is None
            expected = -1 if unavailable else int(truth["speed_mm_day"][day] > 0)
            if row["date"] != str(case.dates[day]) or int(row["activity_label"]) != expected:
                raise ValueError("Process activity/calendar mismatch")
            if expected != 1 and (row["feature"] or row["episode_id"] or int(row["weak_signal"])):
                raise ValueError("Definite process stage outside observed activity")
            missing += int(unavailable)
            active += int(expected == 1)
    reproduced = []
    specification = protocol["generation"]
    request = LandslideRequest(seed=specification["seed"], count=120, days=730)
    for position in (0, 60, 119):
        entry = ledger[position]
        regenerated, labels, _ = controlled_case(request, entry["stream_index"])
        regenerated = LandslideInput.model_validate(dict(regenerated.model_dump(), case_id=entry["case_id"]))
        if regenerated != cases[entry["case_id"]]:
            raise ValueError("Raw observation regeneration differs")
        for saved, original in zip(grouped[entry["case_id"]], labels, strict=True):
            for key in original:
                if key != "case_id" and saved[key] != str(original[key]):
                    raise ValueError("Process label regeneration differs")
        reproduced.append(entry["case_id"])
    secondary = read_json(compiled / "daily-reference.json")
    plot_data = read_json(compiled / "plot-data.json")
    secondary_grouped = defaultdict(list)
    for row in secondary:
        secondary_grouped[row["case_id"]].append(row)
    if list(secondary_grouped) != protocol["secondary_review_ids"]:
        raise ValueError("Review removed a pre-specified record")
    activity_codes = {-1: 0, 0: 1, 1: 2}
    feature_codes = {"unknown": 0, "none": 1, "acceleration": 2,
                     "steady_motion": 3, "deceleration": 4}
    for cid, rows in secondary_grouped.items():
        if len(rows) != 730:
            raise ValueError("Review calendar incomplete")
        for day, row in enumerate(rows):
            unavailable = cases[cid].displacement_mm[day] is None
            if unavailable and (row["activity_label"] != -1 or row["feature"] != "unknown" or row["stage_evaluable"]):
                raise ValueError("Review assigned a label on a missing observation")
            if row["stage_evaluable"] and row["activity_label"] != 1:
                raise ValueError("Stage extends beyond confirmed activity")
            expected_activity = 3 if unavailable else activity_codes[row["activity_label"]]
            expected_stage = 5 if unavailable else feature_codes[row["feature"]]
            if (plot_data[cid]["activity_codes"][day] != expected_activity
                    or plot_data[cid]["stage_codes"][day] != expected_stage):
                raise ValueError("Plot bands disagree with exported daily labels")
    with (compiled / "daily-reference.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(secondary):
        raise ValueError("CSV and JSON lengths disagree")
    for csv_row, row in zip(csv_rows, secondary, strict=True):
        if any(csv_row[key] != ("" if value is None else str(value)) for key, value in row.items()):
            raise ValueError("CSV and JSON labels disagree")
    result = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "validator_process_reference_opened_at": opened_at,
        "status": "passed", "records": len(cases), "unique_observation_hashes": len(set(digests)),
        "prior_confirmation_records_checked": len(earlier_cases), "prior_observation_overlap": 0,
        "process_calendar_rows": len(process_rows), "process_observed_days": len(process_rows) - missing,
        "process_missing_days": missing, "process_active_observed_days": active,
        "families": dict(family_counts), "review_families": dict(review_families),
        "secondary_calendar_rows": len(secondary), "regenerated_exact_matches": reproduced,
        "stage_only_within_frozen_activity": True, "missing_masks_equal": True,
        "plot_csv_json_equal": True, "bindings_checked": bound,
        "observational_reference_is_independent_expert": False,
        "field_distribution_validated": False,
        "new_model_performance_measured": False,
        "secondary_summary": read_json(compiled / "summary.json"),
        "observation_sha256_excluding_ids": dict(zip(ids, digests, strict=True)),
    }
    out.mkdir()
    (out / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(Path(__file__), out / "source.py")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("secondary_summary", "observation_sha256_excluding_ids")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validate(args.data.resolve(), args.out.resolve())
