"""Numerical activity owns detection; visual alternatives may improve its fitted bounds."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp36 import result_from_events  # noqa: E402
from experiment_exp45 import (  # noqa: E402
    activity_context,
    candidate_gain,
    select_boundaries,
)
from report_exp45 import visual_events  # noqa: E402

from gnss_research.data import simulate  # noqa: E402
from gnss_research.segmentation import segment_proposals  # noqa: E402


def event(kind, start, end):
    return {'kind': kind, 'start': start, 'end': end, 'status': 'confirmed',
        'phases': [], 'evidence': 'observed displacement'}


def test_fixed_boundary_gain_matches_original_activity_fit():
    observation, _ = simulate(4501, 'gain-check', 'mixed', 730, 365)
    numeric = segment_proposals(observation, result_from_events(observation, []))
    activity = numeric['activity_fit']
    gain = candidate_gain(activity_context(observation, numeric), activity['start'], activity['end'])
    assert np.isclose(gain, activity['gain'], rtol=1e-10, atol=1e-10)


def test_visual_candidate_can_change_bounds_without_changing_other_types_or_inputs():
    observation, _ = simulate(4502, 'boundary-check', 'normal', 730, 365)
    values = np.random.default_rng(4502).normal(0, .2, (730, 3))
    fraction = np.clip((np.arange(730) - 440 + 1) / (580 - 440 + 1), 0, 1)
    values[:, 0] += 40 * fraction
    observation['displacement_mm'] = values.tolist()
    others = [event('spike', 400, 400), event('step', 420, 420), event('transient', 600, 620)]
    numeric = result_from_events(observation, [*others, event('deformation', 410, 650)])
    numeric['jumps'] = []
    alternatives = [event('deformation', 440, 580)]
    before = copy.deepcopy((observation, numeric, alternatives))
    result, audit = select_boundaries(observation, numeric, alternatives)
    activity = [item for item in result['prediction']['events'] if item['kind'] == 'deformation']
    assert [(item['start'], item['end']) for item in activity] == [(440, 580)]
    assert audit[0]['changed'] and audit[0]['selected_gain'] > audit[0]['numeric_gain']
    assert [item for item in result['prediction']['events'] if item['kind'] != 'deformation'] == others
    assert (observation, numeric, alternatives) == before
    preserved, decisions = select_boundaries(observation, numeric, [])
    assert preserved['prediction'] == numeric['prediction']
    assert not decisions[0]['changed']


def test_no_numerical_activity_cannot_be_added_by_visual_candidate():
    observation = {'dates': list(range(730)), 'displacement_mm': [[0., 0., 0.]] * 730}
    numeric = result_from_events(observation, [event('step', 450, 450)])
    result, audit = select_boundaries(observation, numeric, [event('deformation', 440, 580)])
    assert result['prediction'] == numeric['prediction']
    assert audit == []


def test_visual_bounds_respect_missing_observations_and_failed_requests():
    observation = {'dates': list(range(730)), 'displacement_mm': [[0., 0., 0.]] * 730}
    for day in (440, 450):
        observation['displacement_mm'][day] = None
    receipt = {'status': 'success', 'decision': {'intervals': [
        {'start': 440, 'end': 450, 'status': 'confirmed', 'observation': 'movement'},
        {'start': 460, 'end': 470, 'status': 'uncertain', 'observation': 'unknown'},
        {'start': 440, 'end': 440, 'status': 'confirmed', 'observation': 'gap'}]}}
    assert [(item['start'], item['end']) for item in visual_events(observation, receipt)] == [(441, 449)]
    assert visual_events(observation, {'status': 'failed'}) == []
