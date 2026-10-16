"""Retrieval authority and calendar constraints for observed normal examples."""

import numpy as np
import pytest

from gnss_sim.landslide_normal_contrast import (
    anchored_blocks,
    anchored_series,
    select_normal_examples,
)


def test_retrieval_uses_separate_backgrounds_and_does_not_assign_activity():
    bank = [
        {"case_id": "normal_a", "background": 0, "start": 0, "stop": 180, "descriptor": [0.] * 15},
        {"case_id": "normal_a", "background": 0, "start": 90, "stop": 270, "descriptor": [.01] * 15},
        {"case_id": "normal_b", "background": 1, "start": 0, "stop": 180, "descriptor": [1.] * 15},
        {"case_id": "normal_c", "background": 6, "start": 0, "stop": 180, "descriptor": [2.] * 15},
    ]
    selected, status = select_normal_examples(np.zeros(15), bank, np.ones(3))
    assert status == "available"
    assert [item["case_id"] for item in selected] == ["normal_a", "normal_b", "normal_c"]
    assert not any("activity" in item for item in selected)


def test_missing_calendar_support_never_creates_a_retrieved_normal_match():
    values = np.ones((180, 3))
    values[60:76] = np.nan
    assert anchored_blocks(values, 0, 180) is None
    bank = [{"background": group} for group in range(3)]
    selected, status = select_normal_examples(None, bank, np.ones(3))
    assert selected == []
    assert status == "query_block_support_unavailable"


def test_constant_anchoring_preserves_raw_amplitude_and_missing_locations():
    values = np.column_stack([np.arange(180), -np.arange(180), np.arange(180) * .1]).astype(float)
    values[100:104] = np.nan
    shifted = anchored_series(values, 0, 180)
    np.testing.assert_array_equal(np.isnan(shifted), np.isnan(values))
    np.testing.assert_allclose(shifted[130] - shifted[30], values[130] - values[30])
    with pytest.raises(ValueError):
        anchored_blocks(values, 0, 179)
