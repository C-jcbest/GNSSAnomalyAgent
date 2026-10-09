"""Read-only observation support and window sensitivity; never assign stages."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np

from gnss_sim.landslide_stage_reference import cache_support_inventory

WINDOWS = (31, 61, 91)
REASONS = ("record_edge", "center_missing", "insufficient_count", "insufficient_side", "long_gap")


def observation_guard(observed, center, window):
    """Reconstruct diagnostic guards, retaining both first and overlapping failures."""
    half = window // 2
    start, stop = center - half, center + half + 1
    clipped_start, clipped_stop = max(0, start), min(len(observed), stop)
    support = np.asarray(observed[clipped_start:clipped_stop], dtype=bool)
    left = int(np.count_nonzero(observed[clipped_start:center]))
    right = int(np.count_nonzero(observed[center + 1:clipped_stop]))
    positions = np.flatnonzero(np.r_[True, support, True])
    longest_gap = int(np.max(np.diff(positions) - 1))
    conditions = (
        start < 0 or stop > len(observed),
        not observed[center],
        int(support.sum()) < int(np.ceil(0.75 * window)),
        min(left, right) < 0.6 * half,
        longest_gap > 7,
    )
    failures = [reason for reason, failed in zip(REASONS, conditions, strict=True) if failed]
    return {
        "window_start": start, "window_stop": stop,
        "valid_count": int(support.sum()), "left_count": left, "right_count": right,
        "longest_gap": longest_gap, "primary_reason": failures[0] if failures else "eligible",
        "overlapping_reasons": "|".join(failures),
    }


def finite(value):
    return value is not None and bool(np.all(np.isfinite(value)))


def compare_windows(first, second):
    """Continuous sensitivity plus raw opposite signs, with no classification threshold."""
    result = {
        "common_velocity": first["velocity_available"] and second["velocity_available"],
        "common_tangential": first["tangential_available"] and second["tangential_available"],
        "velocity_cosine": None, "speed_absolute_difference": None,
        "speed_relative_difference": None, "tangential_opposite_sign": False,
        "tangential_absolute_difference": None, "opposite_min_abs_tangential": None,
        "opposite_max_abs_tangential": None,
    }
    if result["common_velocity"]:
        first_speed, second_speed = first["speed_mm_day"], second["speed_mm_day"]
        difference = abs(first_speed - second_speed)
        result["speed_absolute_difference"] = difference
        maximum = max(first_speed, second_speed)
        if maximum > 1e-9:
            result["speed_relative_difference"] = difference / maximum
        if min(first_speed, second_speed) > 1e-9:
            result["velocity_cosine"] = float(np.clip(
                np.dot(first["velocity_mm_day"], second["velocity_mm_day"])
                / (first_speed * second_speed), -1, 1))
    if result["common_tangential"]:
        first_a, second_a = first["tangential_mm_day2"], second["tangential_mm_day2"]
        result["tangential_absolute_difference"] = abs(first_a - second_a)
        if first_a * second_a < 0:
            result["tangential_opposite_sign"] = True
            result["opposite_min_abs_tangential"] = min(abs(first_a), abs(second_a))
            result["opposite_max_abs_tangential"] = max(abs(first_a), abs(second_a))
    return result


def audit_record(case, review, motions):
    """Audit the whole calendar, preserving the frozen raw-displacement activity partition."""
    observed = np.array([finite(value) for value in case.displacement_mm])
    size = len(observed)
    state = [None] * size
    interior = [None] * size
    spans = [span for span in review["spans"] if span["state"] == "activity"]
    for span in review["spans"]:
        for day in range(span["start"], span["stop"]):
            if state[day] is not None:
                raise ValueError("Overlapping activity partition")
            state[day] = span["state"]
            if span["state"] == "activity":
                interior[day] = (span["start"], span["stop"])
    if any(value not in ("activity", "stationary", "unknown") for value in state):
        raise ValueError("Incomplete activity partition")
    window_rows = []
    by_window = {}
    for window in WINDOWS:
        motion = motions[window]
        fields = (motion.valid_counts, motion.fitted_displacement_mm, motion.velocity_mm_day,
                  motion.acceleration_mm_day2, motion.speed_mm_day,
                  motion.tangential_acceleration_mm_day2)
        if any(len(field) != size for field in fields):
            raise ValueError("Diagnostic calendar mismatch")
        if motion.source != "observations_only" or motion.edge_policy != "full_centered_window":
            raise ValueError("Unexpected diagnostic support contract")
        current_rows = []
        for day in range(size):
            guard = observation_guard(observed, day, window)
            velocity_available = finite(motion.velocity_mm_day[day])
            tangential_available = finite(motion.tangential_acceleration_mm_day2[day])
            eligible = guard["primary_reason"] == "eligible"
            if guard["valid_count"] != motion.valid_counts[day]:
                raise ValueError(f"{case.case_id}/{day}/{window}: valid count mismatch")
            if any(finite(field[day]) != eligible for field in fields[1:5]):
                raise ValueError(f"{case.case_id}/{day}/{window}: guard/fit mismatch")
            if tangential_available and not velocity_available:
                raise ValueError("Tangential estimate without velocity")
            velocity = motion.velocity_mm_day[day]
            speed = motion.speed_mm_day[day]
            tangential = motion.tangential_acceleration_mm_day2[day]
            if velocity_available and not np.isclose(np.linalg.norm(velocity), speed):
                raise ValueError("Cached speed disagrees with velocity magnitude")
            if tangential_available:
                expected = np.dot(velocity, motion.acceleration_mm_day2[day]) / speed
                if not np.isclose(expected, tangential, rtol=1e-10, atol=1e-12):
                    raise ValueError("Cached tangential acceleration mismatch")
            bounds = interior[day]
            outside_counts = Counter()
            same_count = 0
            contained = False
            if bounds is not None:
                contained = guard["window_start"] >= bounds[0] and guard["window_stop"] <= bounds[1]
                for sample in range(max(0, guard["window_start"]), min(size, guard["window_stop"])):
                    if not observed[sample]:
                        continue
                    if bounds[0] <= sample < bounds[1]:
                        same_count += 1
                    else:
                        outside_counts[state[sample]] += 1
            row = {
                "case_id": case.case_id, "day_index": day, "date": case.dates[day].isoformat(),
                "window_days": window, "observed": bool(observed[day]),
                "activity_state": state[day], "interior_start": bounds[0] if bounds else None,
                "interior_stop": bounds[1] if bounds else None, **guard,
                "velocity_available": velocity_available, "tangential_available": tangential_available,
                "tangential_display_gate_failed": velocity_available and not tangential_available,
                "velocity_mm_day": velocity, "speed_mm_day": speed,
                "tangential_mm_day2": tangential, "contained_in_same_interior": contained,
                "same_interior_observed_count": same_count,
                "outside_stationary_observed_count": outside_counts["stationary"],
                "outside_unknown_observed_count": outside_counts["unknown"],
                "outside_other_activity_observed_count": outside_counts["activity"],
            }
            window_rows.append(row)
            current_rows.append(row)
        by_window[window] = current_rows
    pair_rows = []
    for first, second in combinations(WINDOWS, 2):
        for day in range(size):
            row = by_window[first][day]
            pair_rows.append({
                "case_id": case.case_id, "day_index": day, "date": case.dates[day].isoformat(),
                "first_window": first, "second_window": second, "observed": row["observed"],
                "activity_state": row["activity_state"], "interior_start": row["interior_start"],
                "interior_stop": row["interior_stop"],
                **compare_windows(row, by_window[second][day]),
            })
    # Existing availability inventory independently cross-checks the new detailed rows.
    for span in spans:
        inventory = cache_support_inventory(case, motions, span["start"], span["stop"])
        for window in WINDOWS:
            selected = [row for row in by_window[window][span["start"]:span["stop"]] if row["observed"]]
            expected = inventory["windows"][str(window)]
            if sum(row["velocity_available"] for row in selected) != expected["velocity_days"]:
                raise ValueError("Availability inventory mismatch")
            if sum(row["tangential_available"] for row in selected) != expected["tangential_days"]:
                raise ValueError("Tangential inventory mismatch")
    return window_rows, pair_rows, spans


def quantiles(values):
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return dict(zip(("min", "q25", "median", "q75", "max"),
                    map(float, np.quantile(finite_values, [0, .25, .5, .75, 1])), strict=True))


def summarize(window_rows, pair_rows):
    """Every activity statistic uses observed activity days; calendar gaps remain explicit."""
    activity = [row for row in window_rows if row["activity_state"] == "activity"]
    selected = [row for row in activity if row["observed"]]
    result = {"activity_calendar_days": len(activity) // 3,
              "activity_observed_days": len(selected) // 3,
              "activity_missing_days": (len(activity) - len(selected)) // 3, "windows": {}}
    daily = {}
    for window in WINDOWS:
        rows = [row for row in selected if row["window_days"] == window]
        available = [row for row in rows if row["velocity_available"]]
        tangential = [row for row in rows if row["tangential_available"]]
        daily[window] = {(row["case_id"], row["day_index"]): row for row in rows}
        result["windows"][str(window)] = {
            "velocity_days": len(available), "tangential_days": len(tangential),
            "display_gate_failed_days": len(available) - len(tangential),
            "primary_reasons": dict(Counter(row["primary_reason"] for row in rows)),
            "overlapping_reasons": dict(Counter(reason for row in rows
                                                 for reason in row["overlapping_reasons"].split("|") if reason)),
            "contained_velocity_days": sum(row["contained_in_same_interior"] for row in available),
            "contained_tangential_days": sum(row["contained_in_same_interior"] for row in tangential),
            "cross_boundary_velocity_days": sum(not row["contained_in_same_interior"] for row in available),
            "cross_boundary_tangential_days": sum(not row["contained_in_same_interior"] for row in tangential),
        }
    result["availability_sets"] = {}
    for field in ("velocity_available", "tangential_available"):
        sets = {window: {key for key, row in daily[window].items() if row[field]} for window in WINDOWS}
        result["availability_sets"][field] = {
            "union_days": len(set.union(*sets.values())), "intersection_days": len(set.intersection(*sets.values())),
            "additional_31_over_61_days": len(sets[31] - sets[61]),
            "lost_31_vs_61_days": len(sets[61] - sets[31]),
            "additional_91_over_61_days": len(sets[91] - sets[61]),
            "lost_91_vs_61_days": len(sets[61] - sets[91]),
            "additional_31_over_61_contained_days": sum(daily[31][key]["contained_in_same_interior"]
                                                       for key in sets[31] - sets[61]),
        }
    result["pairs"] = {}
    for first, second in combinations(WINDOWS, 2):
        rows = [row for row in pair_rows if row["observed"] and row["activity_state"] == "activity"
                and row["first_window"] == first and row["second_window"] == second]
        result["pairs"][f"{first}-{second}"] = {
            "common_velocity_days": sum(row["common_velocity"] for row in rows),
            "common_tangential_days": sum(row["common_tangential"] for row in rows),
            "raw_opposite_sign_days": sum(row["tangential_opposite_sign"] for row in rows),
            "negative_velocity_cosine_days": sum(row["velocity_cosine"] is not None
                                                  and row["velocity_cosine"] < 0 for row in rows),
            **{field: quantiles([row[field] for row in rows]) for field in (
                "velocity_cosine", "speed_absolute_difference", "speed_relative_difference",
                "tangential_absolute_difference", "opposite_min_abs_tangential",
                "opposite_max_abs_tangential")},
        }
    return result
