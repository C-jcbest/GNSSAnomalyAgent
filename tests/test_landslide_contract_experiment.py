from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from gnss_sim.landslide_boundary_experiment import boundary_prompt
from gnss_sim.landslide_confirmation_prompt_experiment import target_blocks
from gnss_sim.landslide_contract_experiment import contract_prompt, parse_contract_response


def response(arm):
    blocks = [
        {"start": start, "end": stop - 1, "observation": "plateau followed by continuing motion"}
        for start, stop in target_blocks(450, 660)
    ]
    payload = {"blocks": blocks, "activity": [[520, 659]], "uncertain": [[500, 519]],
               "evidence": "Visible accumulation starts after an uncertain boundary."}
    if arm == "c0_redundant":
        payload["first_activity_day"] = 520
        for block in blocks:
            block["state"] = "mixed"
    return payload


def test_control_preserves_b1_and_treatment_changes_only_frozen_contract_anchors():
    control = contract_prompt("c0_redundant", 450, 660)
    assert control == boundary_prompt("b1_endpoints", 450, 660)
    treatment = contract_prompt("c1_intervals", 450, 660)
    assert "first_activity_day" not in treatment
    assert '"state"' not in treatment
    assert "maximum permitted endpoint is 659" in treatment
    assert "Do not carry a late movement backward" in treatment
    assert str([[left, right - 1] for left, right in target_blocks(450, 660)]) in treatment


@pytest.mark.parametrize("arm", ["c0_redundant", "c1_intervals"])
def test_shared_intervals_and_missing_masks_are_identical(arm):
    observed = np.ones(700, dtype=bool)
    observed[550] = False
    activity = parse_contract_response(response(arm), observed, 450, 660, arm)
    assert (activity[450:500] == 0).all()
    assert (activity[500:520] == -1).all()
    assert activity[520] == activity[659] == 1
    assert activity[550] == activity[660] == -1
    assert (activity[:450] == -1).all()


@pytest.mark.parametrize("arm", ["c0_redundant", "c1_intervals"])
@pytest.mark.parametrize("intervals", [[[520, 660]], [[449, 490]], [[True, 659]],
                                      [[530, 520]], [[520, 600], [600, 659]]])
def test_invalid_intervals_are_rejected_without_repair(arm, intervals):
    payload = response(arm)
    payload["activity"] = intervals
    original = deepcopy(payload)
    with pytest.raises(ValueError):
        parse_contract_response(payload, np.ones(700, dtype=bool), 450, 660, arm)
    assert payload == original


def test_uncertain_overlap_is_rejected_in_simplified_contract():
    payload = response("c1_intervals")
    payload["uncertain"] = [[500, 530]]
    with pytest.raises(ValueError):
        parse_contract_response(payload, np.ones(700, dtype=bool), 450, 660, "c1_intervals")


def test_redundant_control_still_rejects_stationary_contradiction():
    payload = response("c0_redundant")
    payload["blocks"][-1]["state"] = "stationary"
    with pytest.raises(ValueError, match="contradicts"):
        parse_contract_response(payload, np.ones(700, dtype=bool), 450, 660, "c0_redundant")


@pytest.mark.parametrize("mutation", ["legacy_state", "extra_first", "missing_block", "wrong_calendar", "empty_observation"])
def test_simplified_contract_has_no_legacy_fallback(mutation):
    payload = response("c1_intervals")
    if mutation == "legacy_state":
        payload["blocks"][0]["state"] = "mixed"
    elif mutation == "extra_first":
        payload["first_activity_day"] = 520
    elif mutation == "missing_block":
        payload["blocks"].pop()
    elif mutation == "wrong_calendar":
        payload["blocks"][0]["end"] = 480
    else:
        payload["blocks"][0]["observation"] = " "
    with pytest.raises(ValueError):
        parse_contract_response(payload, np.ones(700, dtype=bool), 450, 660, "c1_intervals")
