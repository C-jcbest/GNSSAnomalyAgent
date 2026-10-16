import numpy as np
import pytest
from analyze_landslide_censored_followup_v2 import overlapping_scope

from gnss_sim.landslide_adaptive_review import combine_reviews
from gnss_sim.landslide_censored_followup import continuation_choice, followup_targets


def test_censored_activity_uses_continuation_view_without_more_calls():
    observed = np.ones(730, dtype=bool)
    activity = np.full(730, -1)
    activity[270:450] = 1
    conflicts = np.zeros(730, dtype=bool)
    conflicts[450:480] = True
    first = {"start": 270, "stop": 450}
    other = {"start": 90, "stop": 270}
    choice = continuation_choice(first, activity, conflicts, observed)
    targets = followup_targets(first, other, choice, 730)
    assert targets["censored"] == {"start": 401, "stop": 499}
    assert targets["fixed_short"] == {"start": 131, "stop": 229}
    assert targets["wide"] == {"start": 360, "stop": 540}
    assert all(target is not None for target in targets.values())


def test_interior_end_is_not_reinterpreted_as_a_censored_boundary():
    observed = np.ones(400, dtype=bool)
    activity = np.full(400, -1)
    activity[100:170] = 1
    conflicts = np.ones(400, dtype=bool)
    first = {"start": 90, "stop": 270}
    other = {"start": 0, "stop": 90}
    choice = continuation_choice(first, activity, conflicts, observed)
    assert not choice["right_censored"]
    assert all(target == other for target in followup_targets(first, other, choice, 400).values())


def test_numeric_gate_and_missing_observations_do_not_create_activity():
    observed = np.ones(400, dtype=bool)
    observed[180:195] = False
    activity = np.full(400, -1)
    activity[100:180] = 1
    conflicts = np.zeros(400, dtype=bool)
    conflicts[195:205] = True
    first = {"start": 0, "stop": 180}
    other = {"start": 180, "stop": 360}
    choice = continuation_choice(first, activity, conflicts, observed)
    targets = followup_targets(first, other, choice, 400)
    assert choice["right_censored"] and not choice["numeric_admitted"]
    assert targets["censored"] == other
    assert targets["no_numeric"] == {"start": 131, "stop": 229}
    conflicts[185] = True
    with pytest.raises(ValueError, match="observed days"):
        continuation_choice(first, activity, conflicts, observed)


def test_single_call_budget_has_no_followup_even_for_a_censored_answer():
    assert all(value is None for value in followup_targets(
        {"start": 0, "stop": 180}, None,
        {"numeric_admitted": True, "right_censored": True}, 400).values())


def test_overlapping_observations_keep_conflict_unknown_and_missing_unlabelled():
    observed = np.ones(300, dtype=bool)
    observed[170] = False
    initial = np.zeros(300, dtype=int)
    initial[170] = -1
    first = np.full(300, -1)
    first[100:200] = 1
    first[170] = -1
    second = np.full(300, -1)
    second[150:250] = 0
    second[170] = -1
    reviews = [{"start": 100, "stop": 200, "activity": first, "status": "success"},
               {"start": 150, "stop": 250, "activity": second, "status": "success"}]
    result = combine_reviews(initial, reviews, observed)
    assert (result.activity[100:150] == 1).all()
    assert (result.activity[150:200] == -1).all()
    assert (result.activity[200:250] == 0).all()
    authority = np.zeros(300, dtype=bool)
    authority[125:225] = True
    scope = overlapping_scope(authority, observed, reviews)
    assert scope.sum() == 99
    assert not scope[170] and not scope[:125].any() and not scope[225:].any()
