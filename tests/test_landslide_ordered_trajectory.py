import numpy as np
import pytest

from gnss_sim.landslide_ordered_trajectory import parse_query_labels, trajectory_segments


def test_original_gap_is_not_connected_to_next_available_block():
    blocks = np.arange(39, dtype=float).reshape(13, 3)
    blocks[6] = np.nan
    left, segments = trajectory_segments(blocks)
    assert left.tolist() == [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
    np.testing.assert_array_equal(segments[:, 0], blocks[left])
    np.testing.assert_array_equal(segments[:, 1], blocks[left + 1])


def test_unavailable_trajectory_contains_no_observation_segments():
    left, segments = trajectory_segments(np.full((13, 3), np.nan))
    assert left.size == 0
    assert segments.shape == (0, 2, 3)


@pytest.mark.parametrize("payload", [
    {"labels": [0] * 41}, {"labels": [0] * 43}, {"labels": [True] * 42},
    {"labels": [1.] * 42}, {"labels": [2] * 42}, {"labels": [0] * 42, "extra": 0},
])
def test_original_date_contract_rejects_short_or_noninteger_answer(payload):
    with pytest.raises(ValueError):
        parse_query_labels(payload)
