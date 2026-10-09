import hashlib
import json
from datetime import date, timedelta

import pytest

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_diagnostics import LandslideDiagnostics
from gnss_sim.landslide_stage_numeric_experiment import (
    NUMERIC_NOTE,
    numeric_prompt,
    numeric_summary,
    request_order,
)


def fixture_inputs():
    days = 730
    positions = [(float(i), -2.0 * i, 0.0) for i in range(days)]
    positions[31] = None
    case = LandslideInput(case_id="case_0001",
                          dates=[date(2023, 1, 1) + timedelta(days=i) for i in range(days)],
                          displacement_mm=positions)
    motions = {}
    for window in (31, 61, 91):
        motions[window] = LandslideDiagnostics(
            case_id=case.case_id, window_days=window,
            input_sha256=hashlib.sha256(case.model_dump_json().encode()).hexdigest(),
            valid_counts=[window - 1] * days,
            fitted_displacement_mm=positions,
            velocity_mm_day=[(3, -4, 0)] * days,
            acceleration_mm_day2=[(0, 0, 0)] * days,
            speed_mm_day=[5.0] * days,
            tangential_acceleration_mm_day2=[0.0] * days,
        )
    review = {"spans": [{"start": 29, "stop": 33, "state": "activity"},
                        {"start": 33, "stop": 700, "state": "uncertain"},
                        {"start": 700, "stop": 730, "state": "activity"}]}
    return case, review, motions


def test_bins_clip_to_activity_and_exclude_missing_cached_values():
    case, review, motions = fixture_inputs()
    # Cached values exist on the missing day; they must not enter any summary.
    for motion in motions.values():
        motion.speed_mm_day[31] = 9999
        motion.velocity_mm_day[31] = (9999, 9999, 9999)
        motion.velocity_mm_day[32] = None
        motion.tangential_acceleration_mm_day2[30] = None
    result = numeric_summary(case, review, motions)
    blocks = result["activities"][0]["blocks"]
    assert [b["bin"] for b in blocks] == [[29, 30], [30, 33]]
    assert blocks[1]["observed"] == 2
    assert blocks[1]["raw_NEU_mm"] == [31.0, -62.0, 0.0]
    assert blocks[1]["fits"]["61"] == {
        "n": [2, 1, 0], "v": [3.0, -4.0, 0.0], "q": [5.0, 5.0, 5.0],
        "t": None, "c": 60.0,
    }
    assert [b["bin"] for b in result["activities"][1]["blocks"]] == [[700, 720], [720, 730]]


def test_all_missing_and_truncated_fit_support_remain_null():
    case, review, motions = fixture_inputs()
    case.displacement_mm[29] = None
    digest = hashlib.sha256(case.model_dump_json().encode()).hexdigest()
    for motion in motions.values():
        motion.input_sha256 = digest
        for i in range(700, 730):
            motion.velocity_mm_day[i] = None
            motion.speed_mm_day[i] = None
            motion.tangential_acceleration_mm_day2[i] = None
    result = numeric_summary(case, review, motions)
    missing = result["activities"][0]["blocks"][0]
    assert missing["raw_NEU_mm"] is None
    assert missing["fits"]["61"] == {"n": [0, 0, 0], "v": None, "q": None, "t": None, "c": None}
    tail = result["activities"][1]["blocks"][1]["fits"]["91"]
    assert tail == {"n": [10, 0, 0], "v": None, "q": None, "t": None, "c": 90.0}
    json.dumps(result, allow_nan=False)


def test_direction_and_speed_are_distinct_and_summary_does_not_modify_inputs():
    case, review, motions = fixture_inputs()
    motion = motions[61]
    motion.velocity_mm_day[30] = (3, 0, 0)
    motion.velocity_mm_day[32] = (-3, 0, 0)
    motion.speed_mm_day[30] = 3
    motion.speed_mm_day[32] = 3
    before = {w: m.model_dump_json() for w, m in motions.items()}
    result = numeric_summary(case, review, motions)
    block = result["activities"][0]["blocks"][1]["fits"]["61"]
    assert block["v"] == [0.0, 0.0, 0.0]
    assert block["q"] == [3.0, 3.0, 3.0]
    assert numeric_summary(case, review, motions) == result
    assert before == {w: m.model_dump_json() for w, m in motions.items()}
    assert "stages" not in json.dumps(result)
    assert numeric_prompt("ORIGINAL", result).startswith("ORIGINAL" + NUMERIC_NOTE)
    motions[91].input_sha256 = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        numeric_summary(case, review, motions)


def test_request_order_balances_both_repetitions():
    assert request_order(1) == (("image", 1), ("numeric", 1), ("numeric", 2), ("image", 2))
    assert request_order(2) == (("numeric", 1), ("image", 1), ("image", 2), ("numeric", 2))
    with pytest.raises(ValueError):
        request_order(0)
