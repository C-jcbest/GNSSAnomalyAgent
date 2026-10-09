"""Support-policy business boundaries, independent of authored stage references."""

from datetime import date, timedelta

import pytest

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_diagnostics import LandslideDiagnostics
from gnss_sim.landslide_frozen_stage_heads import apply_stage_support
from gnss_sim.landslide_stage_support_ablation import (
    baseline_changes,
    build_support_variants,
    coverage_inventory,
)


def support_fixture():
    case = LandslideInput(case_id="case_0001",
        dates=[date(2023,1,1)+timedelta(days=day) for day in range(730)],
        displacement_mm=[[float(day),0.,0.] for day in range(730)])
    review = {"spans": [{"start":0,"stop":100,"state":"stationary"},
                        {"start":100,"stop":200,"state":"activity"},
                        {"start":200,"stop":730,"state":"unknown"}]}
    rows = []
    for day in range(730):
        if day < 100:
            activity = 0
        elif day < 200:
            activity = 1
        else:
            activity = -1
        rows.append({"case_id":case.case_id,"day_index":day,"date":str(case.dates[day]),
                     "activity_label":activity,"feature":"none" if day < 100 else "unknown"})
    motions = {}
    for window in (31,61,91):
        half = window // 2
        valid = [half <= day < 730-half for day in range(730)]
        motions[window] = LandslideDiagnostics(case_id=case.case_id,input_sha256="fixture",
            window_days=window,valid_counts=[window]*730,
            fitted_displacement_mm=[(float(day),0.,0.) if ok else None for day,ok in enumerate(valid)],
            velocity_mm_day=[(1.,0.,0.) if ok else None for ok in valid],
            acceleration_mm_day2=[(0.,0.,0.) if ok else None for ok in valid],
            speed_mm_day=[1. if ok else None for ok in valid],
            tangential_acceleration_mm_day2=[0. if ok else None for ok in valid])
    return case, review, rows, motions


def test_baseline_exact_and_containment_half_open_boundaries():
    case, review, rows, motions = support_fixture()
    native = [{**row,"feature":"acceleration" if row["activity_label"] == 1 else row["feature"]} for row in rows]
    outputs, traces = build_support_variants(case,review,rows,native,motions)
    assert outputs["fixed_61"] == apply_stage_support(native,rows,case,motions[61])[0]
    assert outputs["contained_61"][129]["feature"] == "unknown"
    assert outputs["contained_61"][130]["feature"] == "acceleration"
    assert outputs["contained_61"][169]["feature"] == "acceleration"
    assert outputs["contained_61"][170]["feature"] == "unknown"
    assert traces["contained_61"][170]["support_reason"] == "window_crosses_activity_boundary"


def test_fallback_retains_baseline_and_only_adds_contained_short_support():
    case, review, rows, motions = support_fixture()
    native = [{**row,"feature":"deceleration" if row["activity_label"] == 1 else row["feature"]} for row in rows]
    motions[61].tangential_acceleration_mm_day2[114:186] = [None]*72
    outputs, traces = build_support_variants(case,review,rows,native,motions)
    for day in range(730):
        if outputs["fixed_61"][day]["feature"] == "deceleration":
            assert outputs["fallback_31_contained"][day]["feature"] == "deceleration"
    assert outputs["fallback_31_contained"][114]["feature"] == "unknown"
    assert outputs["fallback_31_contained"][115]["feature"] == "deceleration"
    assert outputs["fallback_31_contained"][184]["feature"] == "deceleration"
    assert outputs["fallback_31_contained"][185]["feature"] == "unknown"
    assert traces["fallback_31_contained"][115]["selected_window"] == 31
    assert traces["fallback_31_contained"][100]["selected_window"] == 61
    assert traces["fallback_31_contained"][100]["selected_contained"] is False


def test_native_unknown_never_recovers_and_steady_needs_velocity_only():
    case, review, rows, motions = support_fixture()
    native = [dict(row) for row in rows]
    native[150]["feature"] = "steady_motion"
    for motion in motions.values():
        motion.tangential_acceleration_mm_day2 = [None]*730
    outputs, traces = build_support_variants(case,review,rows,native,motions)
    for condition, output in outputs.items():
        assert output[151]["feature"] == "unknown"
        assert output[150]["feature"] == "steady_motion"
        assert traces[condition][151]["support_reason"] == "native_unknown"
    inventory = coverage_inventory(native,outputs["fixed_31"],traces["fixed_31"])
    assert inventory["native_unknown_days"] == 99
    assert inventory["final_definite_days"] == 1


def test_short_tail_with_no_contained_window_is_not_extended():
    case, _, rows, motions = support_fixture()
    review = {"spans":[{"start":0,"stop":700,"state":"unknown"},
                        {"start":700,"stop":730,"state":"activity"}]}
    rows = [{**row,"activity_label":1 if row["day_index"] >= 700 else -1,"feature":"unknown"} for row in rows]
    native = [{**row,"feature":"acceleration" if row["activity_label"] == 1 else "unknown"} for row in rows]
    outputs, traces = build_support_variants(case,review,rows,native,motions)
    assert any(row["feature"] == "acceleration" for row in outputs["fixed_31"][700:])
    assert all(row["feature"] == "unknown" for row in outputs["fallback_31_contained"][700:])
    assert traces["fallback_31_contained"][700]["support_reason"] == "window_crosses_activity_boundary"


def test_illegal_native_and_changed_calendar_are_rejected_not_clipped():
    case, review, rows, motions = support_fixture()
    native = [dict(row) for row in rows]
    native[99]["feature"] = "acceleration"
    with pytest.raises(ValueError,match="outside frozen activity"):
        build_support_variants(case,review,rows,native,motions)
    native[99]["feature"] = "none"
    native[150]["date"] = "2023-01-01"
    with pytest.raises(ValueError,match="changed the frozen"):
        build_support_variants(case,review,rows,native,motions)


def test_change_spans_preserve_missing_calendar_and_unknown_reference():
    reference = [{"case_id":"case_0001","day_index":day,"date":str(date(2023,1,1)+timedelta(days=day)),
                  "activity_label":-1 if day == 1 else 1,"feature":"unknown" if day == 1 else "deceleration",
                  "stage_evaluable":day != 1} for day in range(4)]
    reference[3].update(feature="unknown",stage_evaluable=False)
    baseline = [{**row,"feature":"unknown"} for row in reference]
    alternate = [{**row,"feature":"unknown" if row["activity_label"] != 1 else "acceleration"} for row in reference]
    result = baseline_changes(reference,baseline,alternate)
    assert result["counts"]["restored_different_days"] == 2
    assert result["counts"]["restored_reference_unknown_days"] == 1
    assert [(span["start"],span["stop"]) for span in result["spans"]] == [(0,1),(2,3),(3,4)]
