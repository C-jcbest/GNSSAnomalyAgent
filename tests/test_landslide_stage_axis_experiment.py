from datetime import date, timedelta

import pytest

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_diagnostics import LandslideDiagnostics, render_diagnostics
from gnss_sim.landslide_stage_axis_experiment import (
    render_auxiliary_axis_pair,
    render_stage_axis_comparison,
)


def motion_case():
    case = LandslideInput(case_id="case_0001", dates=[date(2023, 1, 1) + timedelta(days=i) for i in range(730)],
                          displacement_mm=[(float(i), 0.0, 0.0) if i != 360 else None for i in range(730)])
    data = [(float(i), 0.0, 0.0) if i != 360 else None for i in range(730)]
    motion = LandslideDiagnostics(case_id=case.case_id, input_sha256="example", window_days=61,
                                 valid_counts=[61] * 730, fitted_displacement_mm=data,
                                 velocity_mm_day=[(1.0, 0.0, 0.0)] * 730,
                                 acceleration_mm_day2=[(0.0, 0.0, 0.0)] * 730,
                                 speed_mm_day=[1.0] * 730, tangential_acceleration_mm_day2=[0.0] * 730)
    return case, motion


def test_axis_intervention_replays_original_and_preserves_curves_and_geometry():
    case, motion = motion_case()
    ticks = [0, 90, 180, 270, 360, 450, 540, 630, 729]
    dates, days, audit = render_auxiliary_axis_pair(case, motion, ticks)
    assert dates == render_diagnostics(case, motion)
    assert days != dates
    assert audit["dates"]["panels"] == audit["day_index"]["panels"]
    assert audit["dates"]["size_pixels"] == audit["day_index"]["size_pixels"] == [2100, 1650]
    assert audit["day_index"]["ticks"] == ticks
    assert audit["day_index"]["tick_labels"] == list(map(str, ticks))
    assert audit["day_index"]["minimum_label_gap_pixels"] >= 8


@pytest.mark.parametrize("ticks", [[0, 90, 729, 729], [0, 729, 90], [10, 729], [0, 730], [0, 90.0, 729]])
def test_invalid_frozen_calendar_ticks_are_rejected(ticks):
    case, motion = motion_case()
    with pytest.raises(ValueError):
        render_auxiliary_axis_pair(case, motion, ticks)


def test_four_comparison_strips_keep_export_labels_missing_mask_and_raw_axes(tmp_path):
    case, _ = motion_case()
    labels = {}
    for arm, feature in zip(("dates-r1", "day_index-r1", "dates-r2", "day_index-r2"),
                            ("acceleration", "steady_motion", "deceleration", "unknown"), strict=True):
        labels[arm] = [{"day_index": i, "date": str(case.dates[i]),
                        "activity_label": 1 if i != 360 else -1,
                        "feature": feature if i != 360 else "unknown"} for i in range(730)]
    view = {"audit": {"panels": [{"xlim": [320, 409], "ylim": [-10, 750]}] * 3,
                      "tick_audit": {"ticks": [320, 350, 380, 409]}}}
    audit = render_stage_axis_comparison(case, view, labels, tmp_path / "paired.png",
                                        {"figure_inches": [12, 7], "dpi": 80})
    assert audit["xlim"] == [320, 409]
    assert audit["ylim_NEU"] == [[-10, 750]] * 3
    assert [row[350] for row in audit["label_colors"]] == [2, 2, 3, 4, 0]
    assert [row[360] for row in audit["label_colors"]] == [5] * 5
    labels["day_index-r1"][350]["activity_label"] = 0
    with pytest.raises(ValueError, match="calendar/activity"):
        render_stage_axis_comparison(case, view, labels, tmp_path / "bad.png",
                                     {"figure_inches": [12, 7], "dpi": 80})
