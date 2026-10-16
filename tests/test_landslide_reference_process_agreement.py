"""Reference disagreements must retain unknown/missing and activity gating."""

from check_landslide_reference_process_agreement import classify_day


def test_missing_and_unknown_are_not_negative_agreement():
    ai = {"activity_label": -1, "feature": "unknown"}
    process = {"activity_label": "1", "feature": "steady_motion"}
    assert classify_day(ai, process, False) == ("missing", "not_evaluable")
    assert classify_day(ai, process, True) == ("unknown_process_1", "not_evaluable")


def test_stage_disagreement_includes_process_inactive_day():
    ai = {"activity_label": 1, "feature": "acceleration"}
    process = {"activity_label": "0", "feature": ""}
    assert classify_day(ai, process, True) == ("ai_1_process_0", "disagree")


def test_stage_comparison_requires_ai_confirmed_activity():
    ai = {"activity_label": 0, "feature": "acceleration"}
    process = {"activity_label": "1", "feature": "acceleration"}
    assert classify_day(ai, process, True) == ("ai_0_process_1", "not_evaluable")
