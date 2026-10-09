from __future__ import annotations

import hashlib
from datetime import date, timedelta

import pytest

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_reference_draft import activity_prompt, unknown_rows
from gnss_sim.landslide_reference_interface import (
    paired_activity_counts,
    parse_partition_draft,
    partition_prompt,
    render_activity_pair,
)


@pytest.fixture
def observations():
    values = [(float(day), 0.0, 0.0) for day in range(730)]
    values[5] = None
    case = LandslideInput(case_id="case_0001", dates=[date(2023, 1, 1) + timedelta(days=day)
                                                    for day in range(730)], displacement_mm=values)
    record = {"input_sha256": hashlib.sha256(case.model_dump_json().encode()).hexdigest(),
              "views": [{"image_sha256": "raw-hash"}]}
    return case, record


def answer():
    return {"spans": [
        {"start": 0, "stop": 3, "state": "stationary", "raw_evidence": "raw stationary"},
        {"start": 3, "stop": 4, "state": "unknown", "raw_evidence": "start uncertain"},
        {"start": 4, "stop": 8, "state": "activity", "raw_evidence": "raw ongoing movement"},
        {"start": 8, "stop": 730, "state": "unknown", "raw_evidence": "not resolved"},
    ], "activity_notes": "AI contract fixture"}


def test_cut_point_belongs_only_to_next_span_and_missing_stays_unknown(observations):
    case, record = observations
    _, rows = parse_partition_draft(answer(), case, record, "fixture")
    assert [row["activity_label"] for row in rows[:12]] == [0, 0, 0, -1, 1, -1, 1, 1, -1, -1, -1, -1]
    assert all(row["activity_label"] == -1 for row in rows[8:])
    assert rows[5]["feature"] == "unknown"
    assert all(row["feature"] == "unknown" for row in rows if row["activity_label"] != 0)


@pytest.mark.parametrize("mistake", ["gap", "overlap", "unsorted", "past_end", "short",
                                    "empty", "zero_length", "boolean", "float", "evidence",
                                    "extra", "old_schema", "wrong_input"])
def test_invalid_partition_rejected_without_repair(observations, mistake):
    case, record = observations
    payload = answer()
    if mistake == "gap":
        payload["spans"][1]["start"] = 2
        payload["spans"][0]["stop"] = 1
    elif mistake == "overlap":
        payload["spans"][1]["start"] = 2
    elif mistake == "unsorted":
        payload["spans"].reverse()
    elif mistake == "past_end":
        payload["spans"][-1]["stop"] = 731
    elif mistake == "short":
        payload["spans"][-1]["stop"] = 729
    elif mistake == "empty":
        payload["spans"] = []
    elif mistake == "zero_length":
        payload["spans"][1]["stop"] = 3
    elif mistake == "boolean":
        payload["spans"][0]["start"] = False
    elif mistake == "float":
        payload["spans"][0]["start"] = 0.0
    elif mistake == "evidence":
        payload["spans"][0]["raw_evidence"] = " "
    elif mistake == "extra":
        payload["stages"] = []
    elif mistake == "old_schema":
        payload = {"episodes": [], "stable_ranges": [], "activity_notes": "old"}
    else:
        record = {**record, "input_sha256": "wrong"}
    with pytest.raises(ValueError):
        parse_partition_draft(payload, case, record, "fixture")


def test_full_unknown_is_valid_and_never_implicit_normal(observations):
    case, record = observations
    payload = {"spans": [{"start": 0, "stop": 730, "state": "unknown", "raw_evidence": "unresolved"}],
               "activity_notes": "no decision"}
    _, rows = parse_partition_draft(payload, case, record, "fixture")
    assert rows == unknown_rows(case)


def test_pair_transitions_exclude_missing_and_are_not_error_counts(observations):
    case, record = observations
    _, rows = parse_partition_draft(answer(), case, record, "fixture")
    observed = [value is not None for value in case.displacement_mm]
    counts = paired_activity_counts(unknown_rows(case), rows, observed)
    assert counts["observed_days"] == 729
    assert counts["state_disagreement_days"] == 6
    assert counts["transitions_episode_to_partition"]["-1->1"] == 3
    assert sum(counts["transitions_episode_to_partition"].values()) == 729
    with pytest.raises(ValueError, match="Missing"):
        incorrect = [dict(row) for row in rows]
        incorrect[5]["activity_label"] = 1
        paired_activity_counts(rows, incorrect, observed)


def test_prompt_preserves_scientific_task_before_output_contract():
    images = ["raw-global.png", "raw-1.png", "raw-2.png", "raw-3.png"]
    old = activity_prompt("case_0001", 1094, images).split("For each episode state", 1)[0]
    common = old.replace("All endpoints are inclusive INTEGER original day indices in 0..1094.",
                         "Observation day indices are INTEGER original indices in 0..1094.")
    prompt = partition_prompt("case_0001", 1094, images)
    assert prompt.startswith(common)
    assert "HALF-OPEN" in prompt and "final stop equals\n1095" in prompt
    assert "UNKNOWN is valid anywhere" in prompt


def test_comparison_strip_matches_export_and_separates_missing(observations, tmp_path):
    case, record = observations
    _, rows = parse_partition_draft(answer(), case, record, "fixture")
    record["views"][0]["audit"] = {
        "panels": [{"xlim": [0, 729], "ylim": [-10, 750]} for _ in range(3)],
        "tick_audit": {"ticks": [0, 180, 360, 540, 729]},
    }
    path = tmp_path / "comparison.png"
    audit = render_activity_pair(case, record, {"episode_v1": unknown_rows(case), "partition_v2": rows},
                                 path, {"figure_inches": [12, 7], "dpi": 100})
    assert audit["label_image"][1][:9] == [1, 1, 1, 0, 2, 3, 2, 2, 0]
    assert audit["label_image"][0][5] == 3
    assert path.read_bytes().startswith(b"\x89PNG")
