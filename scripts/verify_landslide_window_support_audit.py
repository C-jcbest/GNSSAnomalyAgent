"""Independent CSV/calendar arithmetic and frozen-byte checks for the support audit."""

from __future__ import annotations

import csv
import json
from collections import Counter
from itertools import combinations

import numpy as np
from audit_landslide_window_support import ROOT, RUN
from landslide_stage_evidence import load_case
from landslide_stage_grounding import RUN as SOURCE_RUN
from landslide_stage_grounding import verify_prepared

from gnss_sim.artifacts import sha, write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value in ("True", "False"):
                row[key] = value == "True"
            elif value == "":
                row[key] = None
            elif key.endswith("count") or key in ("day_index", "window_days", "first_window", "second_window",
                                                   "interior_start", "interior_stop", "window_start", "window_stop", "longest_gap"):
                row[key] = int(value)
            elif key == "velocity_mm_day":
                row[key] = json.loads(value)
            elif key in ("speed_mm_day", "tangential_mm_day2", "velocity_cosine", "speed_absolute_difference",
                         "speed_relative_difference", "tangential_absolute_difference",
                         "opposite_min_abs_tangential", "opposite_max_abs_tangential"):
                row[key] = float(value)
    return rows


def verify_summary(summary, rows, pairs):
    activity = [row for row in rows if row["activity_state"] == "activity"]
    selected = [row for row in activity if row["observed"]]
    assert summary["activity_calendar_days"] == len(activity) // 3
    assert summary["activity_observed_days"] == len(selected) // 3
    assert summary["activity_missing_days"] == (len(activity) - len(selected)) // 3
    sets = {field: {} for field in ("velocity_available", "tangential_available")}
    contained = {}
    for window in (31, 61, 91):
        current = [row for row in selected if row["window_days"] == window]
        data = summary["windows"][str(window)]
        assert sum(data["primary_reasons"].values()) == len(current)
        assert data["primary_reasons"] == dict(Counter(row["primary_reason"] for row in current))
        assert data["overlapping_reasons"] == dict(Counter(reason for row in current
            for reason in (row["overlapping_reasons"] or "").split("|") if reason))
        for field, label in (("velocity_available", "velocity"), ("tangential_available", "tangential")):
            available = [row for row in current if row[field]]
            sets[field][window] = {(row["case_id"], row["day_index"]) for row in available}
            assert data[f"{label}_days"] == len(available)
            assert data[f"contained_{label}_days"] == sum(row["contained_in_same_interior"] for row in available)
            assert data[f"cross_boundary_{label}_days"] == sum(not row["contained_in_same_interior"] for row in available)
        assert data["display_gate_failed_days"] == data["velocity_days"] - data["tangential_days"]
        contained[window] = {(row["case_id"], row["day_index"]) for row in current if row["contained_in_same_interior"]}
    for field, support in sets.items():
        expected = summary["availability_sets"][field]
        assert expected["union_days"] == len(support[31] | support[61] | support[91])
        assert expected["intersection_days"] == len(support[31] & support[61] & support[91])
        for window in (31, 91):
            assert expected[f"additional_{window}_over_61_days"] == len(support[window] - support[61])
            assert expected[f"lost_{window}_vs_61_days"] == len(support[61] - support[window])
        assert expected["additional_31_over_61_contained_days"] == len((support[31]-support[61]) & contained[31])
    for first, second in combinations((31, 61, 91), 2):
        current = [row for row in pairs if row["observed"] and row["activity_state"] == "activity"
                   and (row["first_window"], row["second_window"]) == (first, second)]
        data = summary["pairs"][f"{first}-{second}"]
        assert data["common_velocity_days"] == sum(row["common_velocity"] for row in current)
        assert data["common_tangential_days"] == sum(row["common_tangential"] for row in current)
        assert data["raw_opposite_sign_days"] == sum(row["tangential_opposite_sign"] for row in current)
        assert data["negative_velocity_cosine_days"] == sum(row["velocity_cosine"] is not None
            and row["velocity_cosine"] < 0 for row in current)
        for field in ("velocity_cosine", "speed_absolute_difference", "speed_relative_difference",
                      "tangential_absolute_difference", "opposite_min_abs_tangential", "opposite_max_abs_tangential"):
            values = [row[field] for row in current if row[field] is not None]
            if not values:
                assert data[field] is None
            else:
                expected = np.percentile(values, [0, 25, 50, 75, 100])
                assert np.allclose(list(data[field][key] for key in ("min", "q25", "median", "q75", "max")), expected)


def verify():
    _, source_public, records, entries = verify_prepared()
    public = RUN / "public"
    ledger = read_json(RUN / "run-ledger-start.json")
    freeze = read_json(RUN / "audit-freeze.json")
    assert freeze["ledger_start_sha256"] == sha(RUN / "run-ledger-start.json")
    assert ledger["source_inference_freeze_sha256"] == sha(SOURCE_RUN / "inference-freeze.json")
    assert ledger["source_ledger_end_sha256"] == sha(SOURCE_RUN / "run-ledger-end.json")
    for name, digest in ledger["source_sha256"].items():
        assert sha(ROOT / name) == sha(RUN / "source-code" / name) == digest
    for name, digest in ledger["input_sha256"].items():
        assert sha(source_public / name) == sha(public / name) == digest
    for name, digest in freeze["files_sha256"].items():
        assert sha(public / name) == digest
    rows = load_csv(public / "daily-window-support.csv")
    pairs = load_csv(public / "daily-window-pairs.csv")
    analysis = read_json(public / "analysis.json")
    assert len(rows) == len(pairs) == 12 * 1095 * 3
    keyed = {(row["case_id"], row["window_days"], row["day_index"]): row for row in rows}
    assert len(keyed) == len(rows)
    assert len({(row["case_id"], row["first_window"], row["second_window"], row["day_index"]) for row in pairs}) == len(pairs)
    for record in records:
        case, review, _, _, motions = load_case(source_public, record, entries)
        observed = [value is not None for value in case.displacement_mm]
        for window, motion in motions.items():
            half = window // 2
            for day in range(len(observed)):
                row = keyed[case.case_id, window, day]
                start, stop = day-half, day+half+1
                visible = range(max(0, start), min(len(observed), stop))
                valid = sum(observed[sample] for sample in visible)
                longest = run_length = 0
                for sample in visible:
                    run_length = 0 if observed[sample] else run_length + 1
                    longest = max(longest, run_length)
                left = sum(observed[sample] for sample in visible if sample < day)
                right = sum(observed[sample] for sample in visible if sample > day)
                reasons = []
                for reason, failed in (
                    ("record_edge", start < 0 or stop > len(observed)),
                    ("center_missing", not observed[day]), ("insufficient_count", valid < np.ceil(.75*window)),
                    ("insufficient_side", min(left, right) < .6*half), ("long_gap", longest > 7)):
                    if failed:
                        reasons.append(reason)
                assert row["date"] == case.dates[day].isoformat()
                assert row["observed"] == observed[day]
                assert row["valid_count"] == motion.valid_counts[day] == valid
                assert row["longest_gap"] == longest
                assert row["primary_reason"] == (reasons[0] if reasons else "eligible")
                assert (row["overlapping_reasons"] or "") == "|".join(reasons)
                assert row["velocity_available"] == (motion.velocity_mm_day[day] is not None) == (not reasons)
                assert row["tangential_available"] == (motion.tangential_acceleration_mm_day2[day] is not None)
                assert row["velocity_mm_day"] == (list(motion.velocity_mm_day[day]) if motion.velocity_mm_day[day] is not None else None)
                assert row["speed_mm_day"] == motion.speed_mm_day[day]
                assert row["tangential_mm_day2"] == motion.tangential_acceleration_mm_day2[day]
                span = next(span for span in review["spans"] if span["start"] <= day < span["stop"])
                assert row["activity_state"] == span["state"]
                if span["state"] == "activity":
                    assert (row["interior_start"], row["interior_stop"]) == (span["start"], span["stop"])
                    assert row["contained_in_same_interior"] == (start >= span["start"] and stop <= span["stop"])
                    same = sum(observed[sample] for sample in visible if span["start"] <= sample < span["stop"])
                    assert row["same_interior_observed_count"] == same
                    for state, field in (("stationary", "outside_stationary_observed_count"),
                                         ("unknown", "outside_unknown_observed_count"),
                                         ("activity", "outside_other_activity_observed_count")):
                        expected = 0
                        for other in review["spans"]:
                            if other is not span and other["state"] == state:
                                expected += sum(observed[sample] for sample in visible
                                    if other["start"] <= sample < other["stop"])
                        assert row[field] == expected
    for pair in pairs:
        first = keyed[pair["case_id"], pair["first_window"], pair["day_index"]]
        second = keyed[pair["case_id"], pair["second_window"], pair["day_index"]]
        assert pair["observed"] == first["observed"]
        assert pair["activity_state"] == first["activity_state"]
        assert pair["common_velocity"] == (first["velocity_available"] and second["velocity_available"])
        assert pair["common_tangential"] == (first["tangential_available"] and second["tangential_available"])
        opposite = pair["common_tangential"] and first["tangential_mm_day2"] * second["tangential_mm_day2"] < 0
        assert pair["tangential_opposite_sign"] == opposite
        if pair["common_velocity"]:
            a, b = first["speed_mm_day"], second["speed_mm_day"]
            assert pair["speed_absolute_difference"] == abs(a-b)
            assert pair["speed_relative_difference"] == (abs(a-b)/max(a,b) if max(a,b) > 1e-9 else None)
            if min(a,b) > 1e-9:
                cosine = np.dot(first["velocity_mm_day"], second["velocity_mm_day"]) / (a*b)
                assert np.isclose(pair["velocity_cosine"], cosine)
            else:
                assert pair["velocity_cosine"] is None
        if pair["common_tangential"]:
            a, b = first["tangential_mm_day2"], second["tangential_mm_day2"]
            assert pair["tangential_absolute_difference"] == abs(a-b)
            assert pair["opposite_min_abs_tangential"] == (min(abs(a),abs(b)) if opposite else None)
            assert pair["opposite_max_abs_tangential"] == (max(abs(a),abs(b)) if opposite else None)
    verify_summary(analysis["summary"], rows, pairs)
    for case in analysis["cases"]:
        current_rows = [row for row in rows if row["case_id"] == case["case_id"]]
        current_pairs = [row for row in pairs if row["case_id"] == case["case_id"]]
        verify_summary(case["summary"], current_rows, current_pairs)
        for interior in case["interiors"]:
            inside_rows = [row for row in current_rows if interior["start"] <= row["day_index"] < interior["stop"]]
            inside_pairs = [row for row in current_pairs if interior["start"] <= row["day_index"] < interior["stop"]]
            verify_summary(interior, inside_rows, inside_pairs)
            for field, range_key in (("velocity_available", "additional_31_over_61_velocity"),
                                     ("tangential_available", "additional_31_over_61_tangential")):
                expected = {row["day_index"] for row in inside_rows if row["window_days"] == 31
                    and row["observed"] and row[field] and not keyed[row["case_id"],61,row["day_index"]][field]}
                restored = {day for start, stop in interior["ranges"][range_key] for day in range(start,stop)}
                assert expected == restored
            for pair_key, ranges in interior["ranges"]["raw_opposite_sign_pairs"].items():
                first, second = map(int, pair_key.split("-"))
                expected = {row["day_index"] for row in inside_pairs if row["first_window"] == first
                    and row["second_window"] == second and row["tangential_opposite_sign"]}
                assert expected == {day for start,stop in ranges for day in range(start,stop)}
    result = {"status": "passed", "window_rows": len(rows), "pair_rows": len(pairs),
              "summary_groups": 34, "input_files": len(ledger["input_sha256"]),
              "frozen_files": len(freeze["files_sha256"]), "execution_sources": len(ledger["source_sha256"]),
              "stage_reference_access": "none", "model_calls": 0,
              "browser_rendering": "not_verified", "audit_freeze_sha256": sha(RUN / "audit-freeze.json")}
    write_json(RUN / "verification-audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
