"""Only transient ranges may change under the frozen EXP40 ownership rule."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp38 import result_from_events  # noqa: E402
from experiment_exp40 import protect_other_types  # noqa: E402


def event(kind, start, end):
    return {'kind': kind, 'start': start, 'end': end, 'status': 'confirmed',
        'evidence': 'observed morphology', 'phases': []}


def test_visual_deformation_cannot_change_its_numerical_owner_or_remove_points():
    observation = {'dates': list(range(100)), 'displacement_mm': [[0., 0., 0.]] * 100}
    events = [event('step', 50, 50), event('spike', 55, 55),
        event('transient', 50, 57), event('deformation', 45, 75)]
    base = result_from_events(observation, events)
    addition = [event('transient', 55, 70), event('deformation', 40, 95)]
    before = copy.deepcopy((base, addition))
    result = protect_other_types(observation, base, addition)
    for kind in ('spike', 'step', 'deformation'):
        assert [e for e in result['prediction']['events'] if e['kind'] == kind] == [e for e in events if e['kind'] == kind]
    assert [(e['start'], e['end']) for e in result['prediction']['events'] if e['kind'] == 'transient'] == [(50, 70)]
    assert (base, addition) == before
