from datetime import date, timedelta

import pytest

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_stage_evidence_experiment import (
    ARMS,
    EVIDENCE_INSTRUCTION,
    evidence_prompt,
    render_evidence_comparison,
    request_order,
)


def test_control_is_byte_identical_and_structured_is_append_only():
    control = "Frozen input\n活动：[10,100)\n"
    assert evidence_prompt(control, "control").encode() == control.encode()
    assert evidence_prompt(control, "structured") == control + EVIDENCE_INSTRUCTION
    with pytest.raises(ValueError):
        evidence_prompt(control, "other")


def test_schedule_balances_condition_order_and_preserves_repeats():
    first = request_order(1)
    second = request_order(2)
    assert [repetition for _, repetition in first] == [1, 1, 2, 2]
    assert [repetition for _, repetition in second] == [1, 1, 2, 2]
    assert all(left[0] != right[0] for left, right in zip(first, second, strict=True))
    assert set(first) == set(second) == {("control", 1), ("control", 2), ("structured", 1), ("structured", 2)}
    assert first[0][0] != second[0][0]
    with pytest.raises(ValueError):
        request_order(0)


def test_strips_share_export_calendar_missing_mask_and_frozen_limits(tmp_path):
    case = LandslideInput(case_id="case_0001",
                          dates=[date(2023, 1, 1) + timedelta(days=i) for i in range(730)],
                          displacement_mm=[(float(i), 0, 0) if i != 360 else None for i in range(730)])
    rows_by_arm = {}
    for arm, feature in zip(ARMS, ("acceleration", "steady_motion", "deceleration", "unknown"), strict=True):
        rows_by_arm[arm] = [{"case_id": case.case_id, "day_index": i, "date": str(day),
                             "activity_label": 1 if i != 360 else -1,
                             "feature": feature if i != 360 else "unknown"} for i, day in enumerate(case.dates)]
    view = {"audit": {"panels": [{"xlim": [320, 409], "ylim": [-10, 750]}] * 3,
                      "tick_audit": {"ticks": [320, 350, 380, 409]}}}
    audit = render_evidence_comparison(case, view, rows_by_arm, tmp_path / "paired.png",
                                       {"figure_inches": [12, 7], "dpi": 80})
    assert audit["xlim"] == [320, 409]
    assert audit["ylim_NEU"] == [[-10, 750]] * 3
    assert [strip[350] for strip in audit["label_colors"]] == [2, 2, 3, 4, 0]
    assert [strip[360] for strip in audit["label_colors"]] == [5] * 5
    rows_by_arm["structured-r1"][350]["activity_label"] = 0
    with pytest.raises(ValueError, match="calendar/activity"):
        render_evidence_comparison(case, view, rows_by_arm, tmp_path / "bad.png",
                                   {"figure_inches": [12, 7], "dpi": 80})
