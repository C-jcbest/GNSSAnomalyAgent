"""Accumulated weak motion can support a denial check; a level shift cannot."""

from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_event_motion_protection import (
    event_motion_support,
    protect_supported_denials,
)


def test_accumulated_weak_motion_is_supported_but_plateau_and_step_are_not():
    initial = np.ones(196, dtype=int)
    raw = np.arange(196)[:, None] * np.array([[.06, -.02, 0]])
    supported, details = event_motion_support(SimpleNamespace(displacement_mm=raw.tolist()), np.ones(3), initial)
    assert supported.all() and details[0]["reason"] == "positive_motion"
    for values in (np.zeros((196, 3)), np.vstack((np.zeros((98, 3)), np.full((98, 3), 20)))):
        support, rows = event_motion_support(SimpleNamespace(displacement_mm=values.tolist()), np.ones(3), initial)
        assert not support.any() and not rows[0]["supported"]


def test_reversals_and_short_fragments_do_not_establish_motion_support():
    raw = np.zeros((196, 3))
    raw[:, 0] = np.r_[np.linspace(0, 20, 98), np.linspace(20, 0, 98)]
    support, _ = event_motion_support(SimpleNamespace(displacement_mm=raw.tolist()), np.ones(3), np.ones(196, dtype=int))
    assert not support.any()
    support, _ = event_motion_support(SimpleNamespace(displacement_mm=raw[:70].tolist()), np.ones(3), np.ones(70, dtype=int))
    assert not support.any()


def test_gap_and_unknowns_are_preserved_without_extending_an_event():
    raw = (np.arange(196)[:, None] * np.array([[.1, 0, 0]])).tolist()
    raw[90:100] = [None] * 10
    initial = np.ones(196, dtype=int)
    initial[90:100] = -1
    support, rows = event_motion_support(SimpleNamespace(displacement_mm=raw), np.ones(3), initial)
    assert not support[90:100].any()
    assert len(rows) == 2
    original = np.array([1, 1, -1, 0, 1, -1])
    local = np.array([0, -1, 1, 1, 0, -1])
    observed = np.array([True] * 5 + [False])
    permission = np.array([True] * 4 + [False] * 2)
    result, protected = protect_supported_denials(original, local, permission, observed,
                                                  np.array([True, True, False, False, True, False]))
    np.testing.assert_array_equal(result, [1, -1, 1, 1, 1, -1])
    np.testing.assert_array_equal(protected, [True, False, False, False, False, False])
