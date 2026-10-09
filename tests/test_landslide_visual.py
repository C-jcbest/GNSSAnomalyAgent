from __future__ import annotations

import numpy as np
import pytest

from gnss_sim.landslide_visual import interval_mask, parse_activity, parse_stages


@pytest.mark.parametrize("intervals", [[[0, 3]], [[-1, 1]], [[True, 1]], [[1.0, 2]], [[1, 0]], [[0, 1], [1, 2]]])
def test_invalid_visual_calendar_is_rejected_without_clipping(intervals):
    with pytest.raises(ValueError):
        interval_mask(intervals, 3)


def test_inclusive_last_day_and_internal_missing_preserve_calendar():
    observed = np.array([True, False, True])
    activity = parse_activity({"activity": [[0, 2]], "uncertain": []}, observed)
    assert activity.tolist() == [1, -1, 1]
    prediction = parse_stages({"stages": [[0, 2, "A"]]}, activity, observed)
    assert prediction.stage.tolist() == [1, -1, 1]


def test_stage_cannot_start_before_confirmed_displacement():
    with pytest.raises(ValueError, match="beyond confirmed displacement"):
        parse_stages({"stages": [[0, 2, "A"]]}, np.array([0, 1, 1]), np.ones(3, dtype=bool))


def test_uncertain_overlap_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        parse_activity({"activity": [[1, 2]], "uncertain": [[0, 1]]}, np.ones(3, dtype=bool))


def test_adjacent_stage_endpoints_are_inclusive_and_cannot_share_a_day():
    activity = np.ones(4, dtype=int)
    observed = np.ones(4, dtype=bool)
    valid = parse_stages({"stages": [[0, 1, "A"], [2, 3, "D"]]}, activity, observed)
    assert valid.stage.tolist() == [1, 1, 3, 3]
    with pytest.raises(ValueError, match="overlap"):
        parse_stages({"stages": [[0, 2, "A"], [2, 3, "D"]]}, activity, observed)
