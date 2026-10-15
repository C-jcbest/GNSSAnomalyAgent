import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from exp29_joint_boundaries import joint_ranges  # noqa: E402


def test_short_temporary_offset_keeps_distinct_jointly_supported_boundaries():
    values = np.random.default_rng(11).normal(0, .1, (300, 3))
    values[150:160] += [4, -3, 2]
    observation = {'dates': list(range(300)), 'background_end': 89,
                   'displacement_mm': values.tolist()}
    accepted, receipts = joint_ranges(observation, {'prediction': {'events': []}, 'jumps': []},
        {'intervals': [{'start': 148, 'end': 158}], 'evidence': 'Short visible platform'})
    assert [(e['start'], e['end']) for e in accepted] == [(150, 159)]
    assert receipts[0]['best']['first'] < receipts[0]['best']['second']


def test_joint_search_does_not_convert_a_permanent_step_into_an_offset():
    values = np.random.default_rng(11).normal(0, .1, (300, 3))
    values[150:] += [4, -3, 2]
    observation = {'dates': list(range(300)), 'background_end': 89,
                   'displacement_mm': values.tolist()}
    accepted, _ = joint_ranges(observation, {'prediction': {'events': []}, 'jumps': []},
        {'intervals': [{'start': 148, 'end': 158}], 'evidence': 'Fallible visual proposal'})
    assert not accepted
