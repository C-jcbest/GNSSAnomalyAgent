import numpy as np
import pytest

from gnss_sim.landslide_review_roles import (
    repair_role_support,
    role_neighbourhood,
    select_role_targets,
)


def test_weak_event_interior_is_not_a_boundary_negative_cue():
    activity = np.zeros(400, dtype=int)
    activity[50:350] = 1
    activity[190:195] = -1
    observed = np.ones(400, dtype=bool)
    observed[190:195] = False
    conflicts = observed & (activity == 1)
    evidence = repair_role_support(activity, conflicts, observed)
    assert evidence["boundary_negative"][50:100].all()
    assert evidence["boundary_negative"][300:350].all()
    assert not evidence["repair_support"][150:250].any()
    authority = role_neighbourhood(evidence["repair_support"], observed)
    assert not authority[149:251].any()
    assert not authority[~observed].any()
    assert np.array_equal(activity[50:350], np.r_[np.ones(140), np.full(5, -1), np.ones(155)])


def test_missed_motion_outside_visual_events_remains_reviewable():
    activity = np.zeros(300, dtype=int)
    observed = np.ones(300, dtype=bool)
    conflicts = np.zeros(300, dtype=bool)
    conflicts[150:170] = True
    evidence = repair_role_support(activity, conflicts, observed)
    assert np.array_equal(evidence["repair_support"], conflicts)
    assert role_neighbourhood(conflicts, observed)[101:219].all()


def test_role_pair_covers_endpoint_cue_at_the_original_call_budget():
    support = np.zeros(400, dtype=bool)
    support[20:40] = True
    support[300:320] = True
    priorities = [
        {"index": 0, "start": 0, "stop": 150, "conflict_days": 20},
        {"index": 1, "start": 150, "stop": 300, "conflict_days": 100},
        {"index": 2, "start": 250, "stop": 400, "conflict_days": 30},
    ]
    original = priorities[:2]
    _, selected = select_role_targets(priorities, original, support, 14, "repair_roles")
    assert [row["index"] for row in selected] == [0, 2]
    assert len(selected) == len(original)
    _, untyped = select_role_targets(priorities, original, support, 14, "untyped_joint")
    assert [row["index"] for row in untyped] == [0, 1]


def test_unavailable_motion_cue_cannot_use_missing_observations():
    observed = np.ones(10, dtype=bool)
    observed[5] = False
    activity = np.zeros(10, dtype=int)
    activity[5] = -1
    conflicts = np.zeros(10, dtype=bool)
    conflicts[5] = True
    with pytest.raises(ValueError, match="observed days"):
        repair_role_support(activity, conflicts, observed)
