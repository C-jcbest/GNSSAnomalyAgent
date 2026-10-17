"""Boundary evidence distinguishes smooth slope changes without becoming a hard gate."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp38 import result_from_events  # noqa: E402
from experiment_exp41 import boundary_evidence, visual_owned  # noqa: E402


def test_local_model_evidence_distinguishes_continuous_hinge_from_real_offset():
    time = np.arange(730)
    rng = np.random.default_rng(41)
    values = rng.normal(0, .1, (730, 3))
    values[:, 0] += np.maximum(time - 430, 0) * .5
    values[500:, 1] += 10
    values[600:, 1] -= 10
    candidate = {'id': 'R1', 'start': 430, 'end': 599}
    before = values.copy()
    result = boundary_evidence(values, candidate, [430, 500, 600])
    onset, returning = result
    assert onset['local_fits'][0]['selected'] == 'hinge'
    assert returning['local_fits'][0]['selected'] == 'step'
    assert returning['local_fits'][0]['local_median_change_NEU'][1] < -9
    assert np.array_equal(values, before)


def test_visual_rejection_removes_numeric_transient_but_preserves_other_event_owners():
    observation = {'dates': list(range(730)), 'displacement_mm': [[0., 0., 0.]] * 730}
    events = [{'kind': kind, 'start': 400, 'end': 400 if kind in ('spike', 'step') else 450,
        'status': 'confirmed', 'phases': [], 'evidence': 'observed'}
        for kind in ('spike', 'step', 'transient', 'deformation')]
    base = result_from_events(observation, events)
    before = copy.deepcopy(base)
    for status in ('success', 'failed'):
        result = visual_owned(observation, base, {'status': status, 'decision': {'intervals': []}})
        assert result['prediction']['events'] == [e for e in events if e['kind'] != 'transient']
    assert base == before
