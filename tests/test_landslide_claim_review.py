import numpy as np
import pytest

from gnss_sim.landslide_claim_review import check_block_claims, parse_block_claims


def test_missing_slots_are_not_compressed_or_connected():
    blocks = np.column_stack((np.arange(13), -np.arange(13), np.zeros(13))).astype(float)
    blocks[6] = np.nan
    claims = [[1, -1, 0] for _ in range(12)]
    claims[5] = claims[6] = None
    facts, conflicts = check_block_claims(blocks, claims)
    assert not conflicts
    assert facts[5]["original_pair"] == [5, 6]
    assert facts[6]["actual_signs"] is None
    assert facts[7]["difference_neu_noise_units"] == [1, -1, 0]


def test_false_sign_and_unavailable_claim_produce_fact_conflicts():
    blocks = np.zeros((13, 3))
    blocks[1:] = [3, -2, 1]
    blocks[8] = np.nan
    claims = [[0, 0, 0] for _ in range(12)]
    claims[7] = claims[8] = None
    claims[3] = None
    _, conflicts = check_block_claims(blocks, claims)
    assert [item["original_pair"] for item in conflicts] == [[0, 1]]
    assert conflicts[0]["actual_signs"] == [1, -1, 1]
    claims[7] = [0, 0, 0]
    _, conflicts = check_block_claims(blocks, claims)
    assert conflicts[-1]["original_pair"] == [7, 8]
    assert conflicts[-1]["actual_signs"] is None


@pytest.mark.parametrize("replacement", [True, 1.0, 2])
def test_output_rejects_noninteger_or_invalid_states(replacement):
    payload = {"labels": [0] * 42, "changes": [[0, 0, 0] for _ in range(12)]}
    payload["labels"][0] = replacement
    with pytest.raises(ValueError):
        parse_block_claims(payload)


def test_schema_preserves_unknown_and_exact_original_slot_count():
    labels, claims = parse_block_claims({"labels": [-1] * 42, "changes": [None] * 12})
    assert labels.tolist() == [-1] * 42
    assert claims == [None] * 12
    with pytest.raises(ValueError):
        parse_block_claims({"labels": [0] * 42, "changes": [None] * 11})
