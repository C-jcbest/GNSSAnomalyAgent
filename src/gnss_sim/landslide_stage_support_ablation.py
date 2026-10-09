"""Frozen-native support masks: suppress or retain, never infer a stage."""

from __future__ import annotations

from collections import Counter

from gnss_sim.landslide_frozen_stage_heads import MOTION_FEATURES, apply_stage_support
from gnss_sim.landslide_stage_agreement import validate_aligned_rows

CONDITIONS = ("fixed_61", "fixed_31", "fixed_91", "fallback_31_contained", "contained_61")


def build_support_variants(case, review, activity_rows, native_rows, motions):
    """Reuse the existing finite-support contract, then apply the registered policies."""
    supported = {window: apply_stage_support(native_rows, activity_rows, case, motions[window])[0]
                 for window in (31, 61, 91)}
    interiors = [None] * len(case.dates)
    for span in review["spans"]:
        if span["state"] == "activity":
            for day in range(span["start"], span["stop"]):
                interiors[day] = (span["start"], span["stop"])
    outputs = {condition: [] for condition in CONDITIONS}
    traces = {condition: [] for condition in CONDITIONS}
    for day, native in enumerate(native_rows):
        feature = native["feature"]
        bounds = interiors[day]
        if feature in MOTION_FEATURES and (bounds is None or case.displacement_mm[day] is None):
            raise ValueError("Native definite stage lacks observed frozen activity")
        for condition in CONDITIONS:
            selected_window = None
            selected_contained = None
            result = native["feature"]
            if feature not in MOTION_FEATURES:
                result = supported[61][day]["feature"]
                reason = "native_unknown" if result == "unknown" else "outside_activity_none"
            else:
                candidates = [61]
                if condition == "fixed_31":
                    candidates = [31]
                elif condition == "fixed_91":
                    candidates = [91]
                elif condition == "fallback_31_contained":
                    candidates = [61, 31]
                result = "unknown"
                reason = "insufficient_class_support"
                for window in candidates:
                    if supported[window][day]["feature"] not in MOTION_FEATURES:
                        continue
                    half = window // 2
                    contained = bounds[0] <= day-half and day+half+1 <= bounds[1]
                    require_contained = condition == "contained_61" or (
                        condition == "fallback_31_contained" and window == 31)
                    if require_contained and not contained:
                        reason = "window_crosses_activity_boundary"
                        continue
                    selected_window = window
                    selected_contained = contained
                    result = feature
                    reason = "retained_fallback_31" if condition == "fallback_31_contained" and window == 31 else "retained_primary"
                    break
            output = {**activity_rows[day], "feature": result}
            outputs[condition].append(output)
            traces[condition].append({
                "selected_window": selected_window, "selected_contained": selected_contained,
                "support_reason": reason, "native_feature": feature,
            })
    return outputs, traces


def coverage_inventory(native, output, trace):
    active = [(before, after, evidence) for before, after, evidence in zip(native, output, trace, strict=True)
              if before["activity_label"] == 1]
    return {
        "observed_activity_days": len(active),
        "native_definite_days": sum(before["feature"] in MOTION_FEATURES for before, _, _ in active),
        "native_unknown_days": sum(before["feature"] == "unknown" for before, _, _ in active),
        "final_definite_days": sum(after["feature"] in MOTION_FEATURES for _, after, _ in active),
        "suppressed_native_days": sum(before["feature"] in MOTION_FEATURES and after["feature"] == "unknown"
                                      for before, after, _ in active),
        "selected_window_days": dict(Counter(str(evidence["selected_window"]) for _, after, evidence in active
                                              if after["feature"] in MOTION_FEATURES)),
        "selected_cross_boundary_days": sum(evidence["selected_contained"] is False for _, _, evidence in active),
        "support_reasons": dict(Counter(evidence["support_reason"] for _, _, evidence in active)),
    }


def baseline_changes(reference, baseline, alternate):
    validate_aligned_rows(reference, baseline)
    validate_aligned_rows(reference, alternate)
    counts = Counter()
    spans = []
    for actual, before, after in zip(reference, baseline, alternate, strict=True):
        if before["feature"] == after["feature"]:
            continue
        if actual["activity_label"] != 1:
            raise ValueError("A support policy changed activity-external labels")
        restored = before["feature"] == "unknown" and after["feature"] in MOTION_FEATURES
        removed = before["feature"] in MOTION_FEATURES and after["feature"] == "unknown"
        if not restored and not removed:
            raise ValueError("Support policies cannot change definite stage classes")
        action = "restored" if restored else "removed"
        feature = after["feature"] if restored else before["feature"]
        if actual["stage_evaluable"]:
            relationship = "same" if feature == actual["feature"] else "different"
        else:
            relationship = "reference_unknown"
        counts[f"{action}_{relationship}_days"] += 1
        identity = (actual["case_id"], before["feature"], after["feature"], relationship)
        if spans and spans[-1]["stop"] == actual["day_index"] and spans[-1]["identity"] == identity:
            spans[-1]["stop"] += 1
        else:
            spans.append({"identity": identity, "case_id": actual["case_id"], "start": actual["day_index"],
                          "stop": actual["day_index"]+1, "baseline_feature": before["feature"],
                          "alternate_feature": after["feature"], "reference_relation": relationship})
    for span in spans:
        del span["identity"]
    return {"counts": {f"{action}_{relationship}_days": counts[f"{action}_{relationship}_days"]
                        for action in ("restored", "removed")
                        for relationship in ("same", "different", "reference_unknown")}, "spans": spans}
