"""Fixed type ownership must preserve overlapping point and range events."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp35 import route_prediction  # noqa: E402


def event(kind, start, end=None):
    return {'kind': kind, 'start': start, 'end': start if end is None else end,
        'status': 'confirmed', 'evidence': 'test observation', 'phases': []}


def test_fixed_sources_preserve_step_at_transient_endpoints_and_do_not_mutate_inputs():
    numerical = {'events': [event('step', 50), event('step', 61), event('spike', 52),
        event('transient', 70, 80), event('deformation', 90, 110)]}
    visual = {'events': [event('transient', 50, 60), event('deformation', 85, 120),
        event('step', 55), event('spike', 57)]}
    before = copy.deepcopy((numerical, visual))
    result = route_prediction(numerical, visual, 150).model_dump()
    assert result['events'] == [event('step', 50), event('transient', 50, 60),
        event('spike', 52), event('step', 61), event('deformation', 85, 120)]
    assert (numerical, visual) == before
