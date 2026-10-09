import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_stage_turning_experiment import turning_prompt, turning_summary


@pytest.fixture
def inputs():
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/landslide-stage-turning-v1.json").read_text())
    case = LandslideInput(case_id="case_9999", dates=[date(2023, 1, 1) + timedelta(days=i) for i in range(730)],
                          displacement_mm=[(float(i), 0.0, 0.0) for i in range(730)])
    digest = hashlib.sha256(case.model_dump_json().encode()).hexdigest()
    speed = [dict(zip(range(3, 10), [0.2, 1, 2, 4, 2, 1, 0.2])).get(day // 30, 0.2) for day in range(730)]
    motions = {window: SimpleNamespace(
        case_id=case.case_id, input_sha256=digest, window_days=window,
        velocity_mm_day=[(value, 0.0, 0.0) for value in speed], speed_mm_day=list(speed),
        tangential_acceleration_mm_day2=[0.0] * 730, valid_counts=[window] * 730,
        model_dump_json=lambda: '{"source":"software fixture only"}',
    ) for window in (31, 61, 91)}
    review = {"spans": [{"start": 90, "stop": 300, "state": "activity"}]}
    return case, review, motions, config["proposal_parameters"]


def test_shared_peak_is_coarse_candidate_not_stage_label(inputs):
    result = turning_summary(*inputs)
    candidates = result["activities"][0]["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["kind"] == "local_maximum"
    assert candidates[0]["coarse_block_range"] == [180, 210]
    assert candidates[0]["windows"] == [31, 61, 91]
    assert candidates[0]["fits"][0]["speed_medians_mm_day"] == [2.0, 4.0, 2.0]
    assert not {"feature", "stages", "reference"} & set(candidates[0])


def test_missing_cached_block_is_not_compressed_or_bridged(inputs):
    case, review, motions, parameters = inputs
    for motion in motions.values():
        motion.velocity_mm_day[180:210] = [None] * 30
        motion.speed_mm_day[180:210] = [None] * 30
    assert turning_summary(case, review, motions, parameters)["activities"][0]["candidates"] == []


def test_single_window_peak_does_not_become_multiscale_candidate(inputs):
    case, review, motions, parameters = inputs
    for window in (61, 91):
        motions[window].speed_mm_day = [1.0] * 730
        motions[window].velocity_mm_day = [(1.0, 0.0, 0.0)] * 730
    assert turning_summary(case, review, motions, parameters)["activities"][0]["candidates"] == []


def test_flat_rates_do_not_get_proposals_and_no_activity_stays_empty(inputs):
    case, review, motions, parameters = inputs
    for motion in motions.values():
        motion.speed_mm_day = [1.0] * 730
        motion.velocity_mm_day = [(1.0, 0.0, 0.0)] * 730
    assert turning_summary(case, review, motions, parameters)["activities"][0]["candidates"] == []
    assert turning_summary(case, {"spans": []}, motions, parameters)["activities"] == []


def test_candidate_does_not_use_extra_author_stage_fields(inputs):
    case, review, motions, parameters = inputs
    expected = turning_summary(case, review, motions, parameters)
    review["stages"] = [{"feature": "deceleration", "start": 90, "stop": 300}]
    review["spans"][0]["raw_evidence"] = "MUST NOT LEAK INTO THE PROMPT"
    assert turning_summary(case, review, motions, parameters) == expected
    prompt = turning_prompt("ORIGINAL PROMPT BYTES", expected)
    assert prompt.startswith("ORIGINAL PROMPT BYTES")
    assert "MUST NOT LEAK" not in prompt
    assert "No proposal does NOT imply a single stage" in prompt


def test_unmatched_provenance_rejected(inputs):
    case, review, motions, parameters = inputs
    motions[61].input_sha256 = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        turning_summary(case, review, motions, parameters)
