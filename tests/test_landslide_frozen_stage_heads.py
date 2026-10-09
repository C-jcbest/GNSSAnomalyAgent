from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_frozen_stage_heads import (
    apply_stage_support,
    compile_stage_answer,
    render_stage_heads,
    stage_pair_counts,
    stage_prompt,
    unclassified_rows,
    validate_activity,
)
from gnss_sim.landslide_raw_adjudication import compile_author_activity, raw_block_statistics
from gnss_sim.landslide_reference_interface import partition_sha256


@pytest.fixture
def frozen_case():
    values = [(float(day), 0.0, 0.0) for day in range(730)]
    values[104] = None
    case = LandslideInput(case_id="case_0001", dates=[date(2023, 1, 1) + timedelta(days=day)
                                                    for day in range(730)], displacement_mm=values)
    record = {"input_sha256": raw_block_statistics(case)["input_sha256"],
              "views": [{"image": "raw.png", "image_sha256": "hash"}]}
    answer = {"spans": [
        {"start": 0, "stop": 100, "state": "stationary", "raw_evidence": "stationary"},
        {"start": 100, "stop": 110, "state": "activity", "raw_evidence": "raw movement"},
        {"start": 110, "stop": 730, "state": "unknown", "raw_evidence": "uncertain"},
    ], "activity_notes": "author review"}
    review, rows = compile_author_activity(answer, case, record)
    velocity = [(1.0, 0.0, 0.0)] * 730
    velocity[101] = None
    tangential = [0.01] * 730
    tangential[102] = None
    motion = SimpleNamespace(velocity_mm_day=velocity, tangential_acceleration_mm_day2=tangential)
    return case, review, rows, partition_sha256(review), motion


def answer():
    return {"stages": [
        {"start": 100, "stop": 105, "feature": "acceleration", "evidence": "raw curvature and rising rate"},
        {"start": 105, "stop": 110, "feature": "deceleration", "evidence": "declining nonzero rate"},
    ], "stage_notes": "uncertain edges omitted"}


def test_sparse_half_open_stages_keep_activity_gaps_and_support(frozen_case):
    case, review, rows, frozen_hash, motion = frozen_case
    stage_review, native, compiled, suppressed = compile_stage_answer(answer(), case, review, rows, frozen_hash, motion)
    assert stage_review["activity_sha256"] == frozen_hash
    assert native[100]["feature"] == "acceleration"
    assert native[104]["feature"] == "unknown"
    assert compiled[101]["feature"] == compiled[102]["feature"] == "unknown"
    assert compiled[105]["feature"] == "deceleration" and compiled[110]["feature"] == "unknown"
    assert compiled[99]["feature"] == "none"
    assert suppressed == {"no_velocity": 1, "no_tangential_acceleration": 1}
    assert [row["activity_label"] for row in compiled] == [row["activity_label"] for row in rows]


@pytest.mark.parametrize("mistake", ["outside", "overlap", "unsorted", "float", "boolean", "zero",
                                    "empty_evidence", "unsupported_class", "extra_key", "changed_hash", "changed_daily"])
def test_bad_stage_answer_is_rejected_without_clipping(frozen_case, mistake):
    case, review, rows, frozen_hash, motion = frozen_case
    payload = answer()
    if mistake == "outside":
        payload["stages"][0]["start"] = 99
    elif mistake == "overlap":
        payload["stages"][1]["start"] = 104
    elif mistake == "unsorted":
        payload["stages"].reverse()
    elif mistake == "float":
        payload["stages"][0]["start"] = 100.0
    elif mistake == "boolean":
        payload["stages"][0]["start"] = True
    elif mistake == "zero":
        payload["stages"][0]["stop"] = 100
    elif mistake == "empty_evidence":
        payload["stages"][0]["evidence"] = " "
    elif mistake == "unsupported_class":
        payload["stages"][0]["feature"] = "low_speed_deformation"
    elif mistake == "extra_key":
        payload["activity"] = []
    elif mistake == "changed_hash":
        frozen_hash = "different"
    else:
        rows[103]["activity_label"] = 0
    with pytest.raises(ValueError):
        compile_stage_answer(payload, case, review, rows, frozen_hash, motion)


def test_velocity_support_does_not_imply_acceleration_and_unknown_is_not_steady(frozen_case):
    case, review, rows, frozen_hash, motion = frozen_case
    payload = {"stages": [{"start": 102, "stop": 103, "feature": "steady_motion", "evidence": "constant nonzero rate"}],
               "stage_notes": "other movement unresolved"}
    _, _, compiled, suppressed = compile_stage_answer(payload, case, review, rows, frozen_hash, motion)
    assert compiled[102]["feature"] == "steady_motion" and suppressed == {}
    assert compiled[103]["feature"] == "unknown"
    invalid = unclassified_rows(rows)
    invalid[99]["feature"] = "acceleration"
    with pytest.raises(ValueError, match="outside"):
        apply_stage_support(invalid, rows, case, motion)


def test_pairing_counts_abstention_and_class_disagreement_only_inside_activity(frozen_case):
    case, review, rows, frozen_hash, motion = frozen_case
    _, _, first, _ = compile_stage_answer(answer(), case, review, rows, frozen_hash, motion)
    second = unclassified_rows(rows)
    second[100]["feature"] = "deceleration"
    pair = stage_pair_counts(first, second)
    assert pair["active_days"] == 9
    assert pair["common_classified_days"] == pair["common_class_disagreement_days"] == 1
    assert "none->none" not in pair["transitions"]
    assert pair["transitions"]["unknown->unknown"] == 2
    assert validate_activity(case, review, rows, frozen_hash)[104] == -1


def test_stage_prompt_supplies_bounds_without_author_evidence(frozen_case):
    case, review, rows, frozen_hash, motion = frozen_case
    prompt = stage_prompt(case.case_id, review, frozen_hash, ["raw.png", "aux.png"], 730)
    assert frozen_hash in prompt and "[[100, 110]]" in prompt and "HALF-OPEN" in prompt
    assert "raw movement" not in prompt and "author review" not in prompt
    assert "NONZERO" in prompt and "Weak activity may remain unclassified" in prompt


def test_stage_strip_uses_exported_masks_and_original_local_axes(frozen_case, tmp_path):
    case, review, rows, frozen_hash, motion = frozen_case
    _, _, compiled, _ = compile_stage_answer(answer(), case, review, rows, frozen_hash, motion)
    view = {"audit": {"panels": [{"xlim": [90, 120], "ylim": [-10, 150]}] * 3,
                      "tick_audit": {"ticks": [90, 105, 120]}}}
    audit = render_stage_heads(case, view, {head: compiled for head in ("rule", "xgboost", "visual")},
                               tmp_path / "stages.png", {"figure_inches": [12, 7], "dpi": 60})
    colors = np.full(730, 0)
    colors[:100] = 1
    colors[100:105] = 2
    colors[101:103] = 0
    colors[104] = 5
    colors[105:110] = 4
    assert audit["label_colors"][1:] == [colors.tolist()] * 3
    assert audit["xlim"] == [90, 120] and audit["ylim_NEU"] == [[-10, 150]] * 3
