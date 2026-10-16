from types import SimpleNamespace

import numpy as np
import pytest

from gnss_sim.landslide_signed_evidence import signed_block_facts


def test_decreasing_motion_retains_axis_sign_and_resists_an_isolated_outlier():
    values = [[2, -3, 0]] * 14 + [[4, -6, 0]] * 14
    values[1] = [1000, 1000, 1000]
    facts = signed_block_facts(SimpleNamespace(displacement_mm=values), [1, 1, 2], 0, 28)
    assert facts["blocks"][0]["median_NEU_mm"] == [2, -3, 0]
    assert facts["blocks"][1]["change_from_previous_block_NEU_mm"] == [2, -3, 0]


def test_unsupported_gap_breaks_deltas_and_partial_blocks_stay_unavailable():
    values = [[1, 2, 3]] * 14 + [None] * 7 + [[9, 10, 11]] * 7 + [[5, 6, 7]] * 14 + [[8, 9, 10]] * 5
    facts = signed_block_facts(SimpleNamespace(displacement_mm=values), [1, 1, 2], 0, 47)
    blocks = facts["blocks"]
    assert blocks[1]["complete_observed_days"] == 7
    assert blocks[1]["median_NEU_mm"] is None
    assert blocks[2]["median_NEU_mm"] == [5, 6, 7]
    assert blocks[2]["change_from_previous_block_NEU_mm"] is None
    assert blocks[3]["days_inclusive"] == [42, 46]
    assert blocks[3]["median_NEU_mm"] is None


def test_facts_use_only_the_requested_window_and_never_emit_activity_answers():
    values = [[100, 200, 300]] * 10 + [[0, 1, 2]] * 28 + [[999, 999, 999]] * 10
    facts = signed_block_facts(SimpleNamespace(displacement_mm=values), np.ones(3), 10, 38)
    assert [row["median_NEU_mm"] for row in facts["blocks"]] == [[0, 1, 2], [0, 1, 2]]
    assert facts["blocks"][0]["days_inclusive"] == [10, 23]
    assert facts["blocks"][1]["change_from_previous_block_NEU_mm"] == [0, 0, 0]
    assert set(facts) == {"axis_order", "daily_difference_noise_proxy_NEU_mm", "blocks"}


def test_invalid_calendar_or_noise_is_rejected():
    case = SimpleNamespace(displacement_mm=[[0, 0, 0]] * 14)
    with pytest.raises(ValueError, match="calendar"):
        signed_block_facts(case, [1, 1, 1], 0, 15)
    with pytest.raises(ValueError, match="noise"):
        signed_block_facts(case, [1, 0, 1], 0, 14)
