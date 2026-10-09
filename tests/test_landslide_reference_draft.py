from __future__ import annotations

import hashlib
from datetime import date, timedelta

import pytest

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_observation_review import activity_review_sha256
from gnss_sim.landslide_reference_draft import (
    activity_prompt,
    parse_activity_draft,
    parse_stage_draft,
    render_draft_overlay,
    stage_prompt,
    unknown_rows,
)


@pytest.fixture(scope="module")
def observations():
    values = [(day * 0.04, day * 0.02, -day * 0.01) for day in range(730)]
    values[250] = None
    case = LandslideInput(case_id="case_0001", dates=[date(2023, 1, 1) + timedelta(days=day)
                                                    for day in range(730)], displacement_mm=values)
    motion = derive_motion(case, 61)
    record = {"input_sha256": hashlib.sha256(case.model_dump_json().encode()).hexdigest(),
              "views": [{"image_sha256": "raw-image"}]}
    return case, motion, record


def payload():
    return {"episodes": [{"possible_start":180, "confirmed_start":200, "confirmed_end":400,
                           "possible_end":420, "raw_evidence":"raw cumulative movement"}],
            "stable_ranges":[{"start":0, "end":160, "raw_evidence":"stable raw curve"}],
            "activity_notes":"AI fixture for contract checks only"}


def test_unreviewed_and_uncertain_days_never_become_implicit_negative(observations):
    case, motion, record = observations
    review, rows = parse_activity_draft(payload(), case, record, motion, "test-model")
    assert rows[170]["activity_label"] == rows[190]["activity_label"] == -1
    assert rows[210]["activity_label"] == 1
    assert rows[100]["activity_label"] == 0
    assert rows[250]["activity_label"] == -1
    assert not review.stages and review.stage_activity_sha256 is None


@pytest.mark.parametrize("mistake", ["overlap", "boolean", "empty_string", "extra_key"])
def test_invalid_activity_answer_is_rejected_without_repair(observations, mistake):
    case, motion, record = observations
    answer = payload()
    if mistake == "overlap":
        answer["stable_ranges"][0]["end"] = 200
    elif mistake == "boolean":
        answer["episodes"][0]["confirmed_start"] = True
    elif mistake == "empty_string":
        answer["episodes"] = ""
    else:
        answer["stages"] = []
    with pytest.raises(ValueError):
        parse_activity_draft(answer, case, record, motion, "test-model")


def test_stage_keeps_activity_hash_and_masks_missing(observations):
    case, motion, record = observations
    review, activity_rows = parse_activity_draft(payload(), case, record, motion, "test-model")
    stage_payload = {"stages":[{"start":210, "end":390, "feature":"steady_motion", "evidence":"steady rate"}],
                     "stage_notes":"none"}
    staged, rows = parse_stage_draft(stage_payload, review, case, motion)
    assert activity_review_sha256(staged) == activity_review_sha256(review)
    assert staged.stage_activity_sha256 == activity_review_sha256(review)
    assert [row["activity_label"] for row in rows] == [row["activity_label"] for row in activity_rows]
    assert rows[250]["feature"] == "unknown"
    assert rows[260]["feature"] == "steady_motion"
    assert not review.stages


def test_shared_possible_boundary_is_rejected_even_when_confirmed_interior_is_disjoint(observations):
    case, motion, record = observations
    answer = payload()
    answer["stable_ranges"] = [{"start":420, "end":729, "raw_evidence":"plateau after activity"}]
    with pytest.raises(ValueError, match="overlap"):
        parse_activity_draft(answer, case, record, motion, "test-model")


@pytest.mark.parametrize("mistake", ["outside", "shared_endpoint", "empty_string"])
def test_invalid_stage_answer_cannot_broaden_parent_activity(observations, mistake):
    case, motion, record = observations
    review, _ = parse_activity_draft(payload(), case, record, motion, "test-model")
    answer = {"stages":[{"start":210, "end":300, "feature":"acceleration", "evidence":"rate change"}],
              "stage_notes":"none"}
    if mistake == "outside":
        answer["stages"][0]["start"] = 190
    elif mistake == "shared_endpoint":
        answer["stages"].append({"start":300, "end":380, "feature":"deceleration", "evidence":"decrease"})
    else:
        answer["stages"] = ""
    with pytest.raises(ValueError):
        parse_stage_draft(answer, review, case, motion)


def test_overlay_breaks_activity_at_the_same_missing_day_as_export(observations, tmp_path):
    case, motion, record = observations
    _, rows = parse_activity_draft(payload(), case, record, motion, "test-model")
    record = {**record, "views": [{"audit": {
        "panels": [{"xlim":[0,729], "ylim":[-20,40]} for _ in range(3)],
        "tick_audit":{"ticks":[0,180,360,540,729]},
    }}]}
    path = tmp_path / "overlay.png"
    spans = render_draft_overlay(case, record, rows, path, {"figure_inches":[12,7], "dpi":100})
    assert spans["activity"] == [[200,249],[251,400]]
    assert not spans["acceleration"]
    assert path.read_bytes().startswith(b"\x89PNG")
    assert all(row["activity_label"] == -1 for row in unknown_rows(case))


def test_prompts_preserve_raw_first_gate_and_image_order(observations):
    case, motion, record = observations
    review, _ = parse_activity_draft(payload(), case, record, motion, "test-model")
    raw_images = ["global.png", "local-1.png", "local-2.png", "local-3.png"]
    first = activity_prompt(case.case_id, 729, raw_images)
    second = stage_prompt(review, 729, raw_images + ["aux-31.png", "aux-61.png", "aux-91.png"])
    assert "aux-31.png" not in first
    assert activity_review_sha256(review) in second
    assert "[[200, 400]]" in second
    assert second.index("global.png") < second.index("local-3.png") < second.index("aux-31.png")
