"""Intersecting visual ranges can restore days while preserving point ownership."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp38 import result_from_events  # noqa: E402
from experiment_exp39 import union_ranges  # noqa: E402


def event(kind, start, end):
    return {'kind': kind, 'start': start, 'end': end, 'status': 'confirmed',
        'evidence': 'observed morphology', 'phases': []}


def test_overlapping_ranges_restore_cover_without_merging_types_or_removing_step():
    observation = {'dates': list(range(100)), 'displacement_mm': [[0., 0., 0.]] * 100}
    observation['displacement_mm'][52] = None
    base = result_from_events(observation, [event('step', 50, 50), event('transient', 50, 57)])
    addition = [event('transient', 55, 70), event('transient', 69, 75), event('deformation', 45, 80)]
    before = copy.deepcopy((base, addition))
    result = union_ranges(observation, base, addition)
    assert [(e['kind'], e['start'], e['end']) for e in result['prediction']['events']] == [
        ('step', 50, 50), ('transient', 50, 75), ('deformation', 45, 80)]
    assert result['generic_mask'][74]
    assert not result['generic_mask'][52]
    assert (base, addition) == before
