from __future__ import annotations

import numpy as np
import pytest

from gnss_sim.landslide_confirmation_prompt_experiment import (
    WORDING_CHANGES,
    confirmation_prompt,
    parse_confirmation,
    target_blocks,
)


def block_payload():
    return {"blocks": [{"start": 30, "end": 59, "state": "stationary", "observation": "raw plateau"},
                       {"start": 60, "end": 69, "state": "mixed", "observation": "motion begins inside block"}],
            "first_activity_day": 64, "activity": [[64, 68]], "uncertain": [[60, 63], [69, 69]],
            "evidence": "plateau followed by observed motion"}


def test_neutral_prompt_changes_only_frozen_wording_and_keeps_output_contract():
    original = confirmation_prompt("r0_candidate", 630, 990)
    expected = original
    for previous, replacement in WORDING_CHANGES:
        expected = expected.replace(previous, replacement)
    assert confirmation_prompt("r1_neutral", 630, 990) == expected
    assert "candidate" not in expected


def test_blocks_preserve_short_final_target_and_never_extend_context():
    assert target_blocks(30, 69) == [(30, 60), (60, 69)]
    assert target_blocks(30, 31) == [(30, 31)]
    with pytest.raises(ValueError, match="calendar"):
        target_blocks(30, 30)


def test_structured_observations_allow_internal_onset_and_preserve_missing_mask():
    observed = np.ones(80, dtype=bool)
    observed[65] = False
    activity = parse_confirmation(block_payload(), observed, 30, 70, "r2_blocks")
    assert activity[30:60].tolist() == [0] * 30
    assert activity[60:70].tolist() == [-1, -1, -1, -1, 1, -1, 1, 1, 1, -1]
    assert (activity[:30] == -1).all() and (activity[70:] == -1).all()


def test_structured_observation_rejects_wrong_first_day_and_block_calendar():
    payload = block_payload()
    payload["first_activity_day"] = 60
    with pytest.raises(ValueError, match="First activity"):
        parse_confirmation(payload, np.ones(80, dtype=bool), 30, 70, "r2_blocks")
    payload = block_payload()
    payload["blocks"][1]["end"] = 75
    with pytest.raises(ValueError, match="calendar"):
        parse_confirmation(payload, np.ones(80, dtype=bool), 30, 70, "r2_blocks")


def test_stationary_block_cannot_be_promoted_to_activity_or_repaired():
    payload = block_payload()
    payload["blocks"][1]["state"] = "stationary"
    with pytest.raises(ValueError, match="contradicts"):
        parse_confirmation(payload, np.ones(80, dtype=bool), 30, 70, "r2_blocks")
    payload = block_payload()
    payload["activity"] = [[64, 74]]
    payload["uncertain"] = [[60, 63]]
    with pytest.raises(ValueError, match="beyond target"):
        parse_confirmation(payload, np.ones(80, dtype=bool), 30, 70, "r2_blocks")
