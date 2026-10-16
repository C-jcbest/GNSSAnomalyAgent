"""The extra observation budget is spent only on an eligible acquired denial."""

from probe_landslide_conditional_numeric_review import choose_followup


def test_one_followup_targets_largest_supported_denial_and_breaks_ties_by_time():
    candidates = [
        {"index": 5, "start": 450, "status": "success", "denied_raw_motion_days": 40},
        {"index": 1, "start": 90, "status": "success", "denied_raw_motion_days": 40},
        {"index": 3, "start": 270, "status": "success", "denied_raw_motion_days": 14},
    ]
    assert choose_followup(candidates) == [1]


def test_failed_or_insufficient_first_reviews_do_not_authorize_extra_observation():
    candidates = [
        {"index": 1, "start": 90, "status": "failed", "denied_raw_motion_days": 180},
        {"index": 3, "start": 270, "status": "success", "denied_raw_motion_days": 13},
    ]
    assert choose_followup(candidates) == []
