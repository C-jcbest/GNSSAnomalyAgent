"""Business limits of observation-based vetoes, including inconclusive weak motion."""

from types import SimpleNamespace

import numpy as np
import pytest

from gnss_sim.landslide_counterevidence_review import (
    raw_counterevidence,
    restrict_contradicted_edits,
)
from gnss_sim.landslide_evaluation import runs


def raw_case(rate=0):
    return SimpleNamespace(displacement_mm=(np.arange(240)[:, None] * np.array([[rate, 0, 0]])).tolist())


def test_plateau_and_continuous_motion_have_distinct_counterevidence():
    plateau = raw_counterevidence(raw_case(), np.ones(3))
    motion = raw_counterevidence(raw_case(.15), np.ones(3))
    assert plateau["plateau"][60:180].all()
    assert not plateau["motion"].any()
    assert motion["motion"][60:180].all()
    assert not motion["plateau"].any()


def test_weak_motion_and_step_transition_are_inconclusive():
    weak = raw_counterevidence(raw_case(.06), np.ones(3))
    assert not weak["plateau"][60:180].any()
    assert not weak["motion"][60:180].any()
    step = raw_case()
    step.displacement_mm[120:] = [[20, 0, 0]] * 120
    evidence = raw_counterevidence(step, np.ones(3))
    assert not evidence["plateau"][120]
    # A median exactly straddling a step can support a single date. It cannot
    # establish the sustained conflict required by the existing review selector.
    assert all(stop - start < 14 for start, stop in runs(evidence["motion"]))


def test_long_gap_prevents_evidence_without_interpolation():
    case = raw_case(.15)
    case.displacement_mm[105:115] = [None] * 10
    evidence = raw_counterevidence(case, np.ones(3))
    assert not evidence["supported"][100:140].any()
    assert not evidence["motion"][100:140].any()
    assert evidence["motion"][170]


def test_veto_retains_old_definite_labels_only_inside_permission():
    initial = np.array([0, 1, -1, 1, 0, -1])
    reviewed = np.array([1, 0, 1, -1, 1, -1])
    observed = np.array([True] * 5 + [False])
    permission = np.array([True] * 4 + [False] * 2)
    evidence = {"plateau": np.array([True, False, True, False, True, False]),
                "motion": np.array([False, True, False, True, False, False])}
    final, rejected = restrict_contradicted_edits(initial, reviewed, permission, observed, evidence)
    np.testing.assert_array_equal(final, [0, 1, 1, -1, 0, -1])
    np.testing.assert_array_equal(rejected, [True, True, False, False, False, False])
    plateau_only, _ = restrict_contradicted_edits(initial, reviewed, permission, observed, evidence, True, False)
    np.testing.assert_array_equal(plateau_only, [0, 0, 1, -1, 0, -1])


def test_missing_decision_and_invalid_noise_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        raw_counterevidence(raw_case(), np.array([1, 0, 1]))
    with pytest.raises(ValueError, match="Missing"):
        restrict_contradicted_edits(np.array([0]), np.array([-1]), np.array([False]),
                                   np.array([False]), {"plateau": np.array([False]), "motion": np.array([False])})
