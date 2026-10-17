"""Visual additions may not remove point events or fabricate activity phases."""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp36 import fuse  # noqa: E402


def event(kind, start, end):
    return {'kind': kind, 'start': start, 'end': end, 'status': 'confirmed',
        'evidence': 'observed shape', 'phases': []}


def test_addition_preserves_step_onsets_existing_ranges_and_missing_mask():
    observation = {'dates': [str(i) for i in range(730)],
        'displacement_mm': [[0., 0., 0.]] * 730}
    observation['displacement_mm'][401] = None
    base = {'prediction': {'events': [event('step', 400, 400), event('step', 411, 411),
        event('transient', 450, 460), event('deformation', 500, 550)]}}
    before = copy.deepcopy(base)
    result = fuse(observation, base, [event('transient', 400, 410)])
    signature = {(item['kind'], item['start'], item['end']) for item in result['prediction']['events']}
    assert signature == {('step', 400, 400), ('step', 411, 411), ('transient', 400, 410),
        ('transient', 450, 460), ('deformation', 500, 550)}
    assert not np.asarray(result['generic_mask'])[401]
    assert base == before
