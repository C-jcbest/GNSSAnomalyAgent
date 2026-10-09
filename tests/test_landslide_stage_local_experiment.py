import hashlib
from datetime import date, timedelta

import pytest

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_diagnostics import LandslideDiagnostics
from gnss_sim.landslide_stage_local_experiment import (
    LOCAL_CONTEXT_NOTE,
    activity_windows,
    local_prompt,
    render_auxiliary_activity_window,
    request_order,
)


def test_context_uses_half_open_activity_without_expanding_uncertain_spans():
    review = {"spans": [{"start": 0, "stop": 20, "state": "activity"},
                        {"start": 20, "stop": 100, "state": "uncertain"},
                        {"start": 100, "stop": 1095, "state": "activity"}]}
    assert activity_windows(review, 1095, 45) == [
        {"start": 0, "stop": 20, "left": 0, "right": 64},
        {"start": 100, "stop": 1095, "left": 55, "right": 1094},
    ]
    assert review["spans"][0]["stop"] == 20
    with pytest.raises(ValueError, match="Context"):
        activity_windows(review, 1095, -1)
    with pytest.raises(ValueError, match="calendar"):
        activity_windows({"spans": [{"start": 0, "stop": 1096, "state": "activity"}]}, 1095, 45)


def test_prompt_is_shared_and_order_balances_both_repeats():
    previous = "Frozen activity and seven images\n"
    assert local_prompt(previous) == previous + LOCAL_CONTEXT_NOTE
    assert "PER-ACTIVITY OBSERVABLE EVIDENCE PROCEDURE" not in local_prompt(previous)
    assert request_order(1) == (("full", 1), ("local", 1), ("local", 2), ("full", 2))
    assert request_order(2) == (("local", 1), ("full", 1), ("full", 2), ("local", 2))
    with pytest.raises(ValueError):
        request_order(0)


def test_local_view_preserves_cached_gaps_values_y_scales_and_panel_geometry():
    days = 730
    positions = [(float(i), 0.1 * i, -0.05 * i) if i != 365 else None for i in range(days)]
    case = LandslideInput(case_id="case_0001",
                          dates=[date(2023, 1, 1) + timedelta(days=i) for i in range(days)],
                          displacement_mm=positions)
    supported = [30 <= i < days - 30 and i != 365 for i in range(days)]
    motion = LandslideDiagnostics(
        case_id=case.case_id,
        input_sha256=hashlib.sha256(case.model_dump_json().encode("utf-8")).hexdigest(),
        window_days=61, valid_counts=[60] * days,
        fitted_displacement_mm=[positions[i] if supported[i] else None for i in range(days)],
        velocity_mm_day=[(1, 0.1, -0.05) if ok else None for ok in supported],
        acceleration_mm_day2=[(0, 0, 0) if ok else None for ok in supported],
        speed_mm_day=[1.006 if ok else None for ok in supported],
        tangential_acceleration_mm_day2=[0 if ok else None for ok in supported],
    )
    before = motion.model_dump_json()
    png, audit = render_auxiliary_activity_window(
        case, motion, {"start": 340, "stop": 400, "left": 295, "right": 444},
        [0, 61, 122, 182, 243, 304, 364, 425, 486, 546, 607, 668, 729],
    )
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert audit["local"]["size_pixels"] == [2100, 1650]
    assert audit["local"]["ticks"][0] == 295
    assert audit["local"]["ticks"][-1] == 444
    for full, local in zip(audit["full"]["panels"], audit["local"]["panels"], strict=True):
        assert full["line_sha256"] == local["line_sha256"]
        assert full["ylim"] == local["ylim"]
        assert full["rect_pixels"] == local["rect_pixels"]
        assert local["xlim"] == [295, 444]
    assert motion.model_dump_json() == before
    motion.input_sha256 = "0" * 64
    with pytest.raises(ValueError, match="record"):
        render_auxiliary_activity_window(case, motion, audit["window"], [0, 729])
