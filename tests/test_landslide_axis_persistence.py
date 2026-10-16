from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_axis_persistence import axis_persistence, persistence_review_plan
from gnss_sim.landslide_baselines import activity_from_mask


def test_single_step_is_not_sustained_but_slow_linear_displacement_is():
    days = np.arange(250)
    step = np.zeros((250, 3))
    step[125:, 0] = 100
    ramp = np.zeros((250, 3))
    ramp[:, 0] = days * 0.01
    step_evidence = axis_persistence(SimpleNamespace(displacement_mm=step.tolist()), np.ones(3))
    ramp_evidence = axis_persistence(SimpleNamespace(displacement_mm=ramp.tolist()), np.ones(3))
    # An even median exactly across the step can create one proxy day. It must
    # not establish a sustained interval under the shared 14-day rule.
    supported = np.isfinite(step_evidence["score"])
    step_activity = activity_from_mask(step_evidence["score"] >= 0.1, supported)
    assert not np.any(step_activity == 1)
    assert np.nanmin(ramp_evidence["score"]) > 0


def test_unrelated_noisy_axis_does_not_cancel_real_axis_progress():
    days = np.arange(250)
    values = np.column_stack((days * 0.01, np.zeros(250), np.sin(days / 10) * 100))
    evidence = axis_persistence(SimpleNamespace(displacement_mm=values.tolist()), np.ones(3))
    assert np.all(evidence["axis_score"][70:180, 0] > 0)
    reflected = axis_persistence(SimpleNamespace(displacement_mm=(-values).tolist()), np.ones(3))
    np.testing.assert_allclose(evidence["score"], reflected["score"], equal_nan=True)


def test_missing_block_cannot_create_trigger_or_revised_missing_date():
    values = np.column_stack((np.arange(250) * 0.01, np.zeros(250), np.zeros(250)))
    rows = values.tolist()
    rows[100:125] = [None] * 25
    observed = np.ones(250, dtype=bool)
    observed[100:125] = False
    initial = np.zeros(250, dtype=int)
    initial[~observed] = -1
    evidence = axis_persistence(SimpleNamespace(displacement_mm=rows), np.ones(3))
    assert np.isnan(evidence["score"][112])
    _, selected, scope = persistence_review_plan(
        initial, observed, evidence, [(0, 180), (70, 250)], 0.1
    )
    assert len(selected) <= 2
    assert not np.any(scope[~observed])
