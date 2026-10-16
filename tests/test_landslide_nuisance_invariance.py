"""Business contracts for motion-preserving additive nuisance interventions."""

import numpy as np
import pytest

from gnss_sim.landslide_nuisance_invariance import (
    nuisance_consistency_loss,
    nuisance_counterpart,
    shuffled_within_activity,
)


def test_intervention_preserves_motion_and_calendar_without_labels():
    motion = np.arange(730)[:, None] * np.array([[.1, -.2, .05]])
    observations = motion.copy()
    observations[300:312] = np.nan
    counterpart, artifact, events = nuisance_counterpart(observations, np.array([1, 1, 2]), 20261015, 337)
    np.testing.assert_allclose(counterpart - artifact, observations, atol=1e-13, equal_nan=True)
    np.testing.assert_array_equal(np.isnan(counterpart), np.isnan(observations))
    assert len(events) == 6
    assert sum(event["kind"] == "instrument_offset" for event in events) == 2
    assert all(np.isfinite(observations[event["day"]]).all() for event in events)
    reproduced = nuisance_counterpart(observations, np.array([1, 1, 2]), 20261015, 337)
    np.testing.assert_array_equal(counterpart, reproduced[0])
    assert events == reproduced[2]


def test_intervention_does_not_fill_partial_missing_or_silent_unknown_noise():
    values = np.zeros((730, 3))
    values[3, 0] = np.nan
    with pytest.raises(ValueError, match="fully observed"):
        nuisance_counterpart(values, np.ones(3), 1, 2)
    with pytest.raises(ValueError, match="positive"):
        nuisance_counterpart(np.zeros((730, 3)), np.array([1, 0, 1]), 1, 2)


def test_shuffled_control_preserves_activity_classes_and_population():
    labels = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    indices = shuffled_within_activity(labels, 2)
    np.testing.assert_array_equal(np.sort(indices), np.arange(len(labels)))
    np.testing.assert_array_equal(labels[indices], labels)
    assert not np.array_equal(indices, np.arange(len(labels)))


def test_consistency_changes_only_activity_and_has_finite_gradients():
    import torch

    original = torch.tensor([-2., 1., 4.], requires_grad=True)
    counterpart = torch.tensor([0., -1., 4.], requires_grad=True)
    loss = nuisance_consistency_loss(original, counterpart)
    assert 0 < loss.item() < 1
    loss.backward()
    assert original.grad[0] < 0 < counterpart.grad[0]
    assert original.grad[2] == counterpart.grad[2] == 0
