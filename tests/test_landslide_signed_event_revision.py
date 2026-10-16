"""Business contracts for directional revision authority."""

from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_signed_event_revision import (
    candidate_event_authority,
    revise_with_signed_authority,
)


def test_addition_permission_does_not_grant_withdrawal_or_unknown_confirmation():
    initial = np.array([1, 0, 0, -1, -1])
    reviewed = np.array([0, 1, -1, 1, -1])
    observed = np.array([True, True, True, True, False])
    withdrawals = np.zeros(5, dtype=bool)
    additions = np.array([True, True, True, True, False])
    revised, changed = revise_with_signed_authority(initial, reviewed, observed, withdrawals, additions)
    np.testing.assert_array_equal(revised, [1, 1, 0, 1, -1])
    np.testing.assert_array_equal(changed, [False, True, False, True, False])


def test_failed_permitted_local_review_remains_unknown():
    initial = np.array([1, 0, 1, -1])
    reviewed = np.array([-1, -1, 0, -1])
    observed = np.array([True, True, True, False])
    withdrawals = np.array([True, True, False, False])
    revised, _ = revise_with_signed_authority(initial, reviewed, observed, withdrawals, np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(revised, [-1, -1, 1, -1])


def test_local_proposal_needs_distributed_motion_and_a_conflict_seed():
    days = 98
    raw = np.column_stack((np.arange(days) * .1, np.zeros(days), np.zeros(days)))
    case = SimpleNamespace(displacement_mm=raw.tolist())
    observed = np.ones(days, dtype=bool)
    initial = np.zeros(days, dtype=int)
    local = np.ones(days, dtype=int)
    permission = np.zeros(days, dtype=bool)
    permission[49] = True
    targets = [dict(start=0, stop=days)]
    additions, _, centered, _, _ = candidate_event_authority(
        case, np.full(3, .25), initial, local, permission, observed, targets,
    )
    assert additions.all()
    assert additions.sum() == centered.sum()
    without_seed = candidate_event_authority(
        case, np.full(3, .25), initial, local, np.zeros(days, dtype=bool), observed, targets,
    )[0]
    assert not without_seed.any()


def test_single_offset_cannot_authorize_whole_local_candidate():
    days = 98
    raw = np.zeros((days, 3))
    raw[49:, 0] = 10
    case = SimpleNamespace(displacement_mm=raw.tolist())
    observed = np.ones(days, dtype=bool)
    permission = np.zeros(days, dtype=bool)
    permission[49] = True
    additions, unfiltered, _, _, _ = candidate_event_authority(
        case, np.full(3, .25), np.zeros(days, dtype=int), np.ones(days, dtype=int),
        permission, observed, [dict(start=0, stop=days)],
    )
    assert not additions.any()
    assert unfiltered.all()
