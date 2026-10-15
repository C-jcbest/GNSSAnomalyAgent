import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp29 import compose, refine_ranges, transient_event  # noqa: E402


def observed_curve(values):
    return {'dates': list(range(len(values))), 'background_end': 89,
            'displacement_mm': values.tolist()}


def test_visual_range_can_recover_edges_absent_from_numerical_candidates():
    values = np.random.default_rng(11).normal(0, .1, (300, 3))
    values[150:210] += [4, -3, 2]
    obs = observed_curve(values)
    numeric = {'prediction': {'events': []}, 'jumps': []}
    _, verified, receipts = refine_ranges(obs, numeric,
        {'intervals': [{'start': 148, 'end': 211}], 'evidence': 'opposing shift and return'})
    assert [(e['start'], e['end']) for e in verified] == [(150, 209)]
    assert receipts[0]['accepted']


def test_smooth_displacement_does_not_become_a_verified_offset():
    time = np.arange(300)
    values = np.random.default_rng(21).normal(0, .1, (300, 3))
    ramp = np.clip((time - 150) / 60, 0, 1)
    values += ramp[:, None] * [4, -3, 2]
    _, verified, _ = refine_ranges(observed_curve(values), {'prediction': {'events': []}},
        {'intervals': [{'start': 150, 'end': 209}], 'evidence': 'proposed interval'})
    assert not verified


def test_independent_step_inside_a_temporary_offset_is_preserved_as_nuisance():
    values = np.random.default_rng(11).normal(0, .1, (300, 3))
    values[150:210] += [4, -3, 2]
    values[180:] += [1, 4, -2]
    numeric = {'prediction': {'events': []}, 'jumps': [{'index': 180}]}
    _, verified, receipts = refine_ranges(observed_curve(values), numeric,
        {'intervals': [{'start': 148, 'end': 211}], 'evidence': 'offset with interior step'})
    assert [(e['start'], e['end']) for e in verified] == [(150, 209)]
    assert receipts[0]['nuisance_step_days'] == [180]


def test_complement_preserves_existing_ranges_activity_and_missing_mask():
    values = np.random.default_rng(31).normal(size=(300, 3))
    obs = observed_curve(values)
    obs['displacement_mm'][205] = None
    existing = transient_event(130, 160, 'existing offset')
    activity = {**existing, 'kind': 'deformation', 'start': 170, 'end': 250}
    step = {**existing, 'kind': 'step', 'start': 200, 'end': 200}
    base = {'prediction': {'events': [existing, activity, step]}}
    result = compose(obs, base, [transient_event(140, 180, 'conflicting proposal'),
                                transient_event(200, 220, 'new independent interval')])
    assert result['prediction']['events'] == [existing, activity,
        transient_event(200, 220, 'new independent interval')]
    assert not result['generic_mask'][205]
