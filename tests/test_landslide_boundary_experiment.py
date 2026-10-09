from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from gnss_sim.landslide_boundary_experiment import (
    boundary_blocks,
    boundary_prompt,
    parse_boundary_response,
)
from gnss_sim.landslide_confirmation_prompt_experiment import confirmation_prompt


def example(arm):
    blocks = boundary_blocks(450, 660, arm)
    return {"blocks": [{"start": left, "end": right - 1, "state": "mixed",
                        "observation": "visible plateau followed by continuing displacement"}
                       for left, right in blocks],
            "first_activity_day": 520, "activity": [[520, 659]], "uncertain": [],
            "evidence": "continuing displacement reaches the target boundary"}


def test_control_is_exact_historical_prompt_and_boundary_check_is_only_appended():
    original = confirmation_prompt("r2_blocks", 450, 660)
    assert boundary_prompt("b0_blocks", 450, 660) == original
    treatment = boundary_prompt("b1_endpoints", 450, 660)
    assert treatment.startswith(original)
    assert "maximum permitted endpoint is 659" in treatment
    assert "number 660 is outside" in treatment


def test_grid_perturbation_only_replaces_list_with_same_target_coverage():
    left = boundary_prompt("b1_endpoints", 450, 660)
    old = str([[a, b - 1] for a, b in boundary_blocks(450, 660, "b1_endpoints")])
    new = str([[a, b - 1] for a, b in boundary_blocks(450, 660, "b2_phase15")])
    assert boundary_prompt("b2_phase15", 450, 660) == left.replace(old, new)
    assert boundary_blocks(450, 660, "b2_phase15") == [
        (450, 465), (465, 495), (495, 525), (525, 555),
        (555, 585), (585, 615), (615, 645), (645, 660),
    ]
    assert boundary_blocks(30, 31, "b2_phase15") == [(30, 31)]
    assert boundary_blocks(30, 50, "b2_phase15") == [(30, 45), (45, 50)]


@pytest.mark.parametrize("arm", ["b0_blocks", "b1_endpoints", "b2_phase15"])
def test_valid_last_day_and_internal_onset_preserve_target_and_missing_masks(arm):
    observed = np.ones(700, dtype=bool)
    observed[550] = False
    activity = parse_boundary_response(example(arm), observed, 450, 660, arm)
    assert (activity[450:520] == 0).all()
    assert activity[520] == 1 and activity[659] == 1
    assert activity[550] == -1 and activity[660] == -1
    assert (activity[:450] == -1).all()


@pytest.mark.parametrize("arm", ["b0_blocks", "b1_endpoints", "b2_phase15"])
def test_previous_one_day_overrun_still_fails_without_clipping(arm):
    payload = example(arm)
    payload["activity"] = [[520, 660]]
    original = deepcopy(payload)
    with pytest.raises(ValueError, match="beyond target"):
        parse_boundary_response(payload, np.ones(700, dtype=bool), 450, 660, arm)
    assert payload == original


def test_shifted_contract_rejects_old_grid_contradictions_and_wrong_first_day():
    observed = np.ones(700, dtype=bool)
    with pytest.raises(ValueError, match="every fixed|calendar"):
        parse_boundary_response(example("b1_endpoints"), observed, 450, 660, "b2_phase15")
    payload = example("b2_phase15")
    payload["blocks"][-1]["state"] = "stationary"
    with pytest.raises(ValueError, match="contradicts"):
        parse_boundary_response(payload, observed, 450, 660, "b2_phase15")
    payload = example("b2_phase15")
    payload["first_activity_day"] = 525
    with pytest.raises(ValueError, match="First activity"):
        parse_boundary_response(payload, observed, 450, 660, "b2_phase15")
