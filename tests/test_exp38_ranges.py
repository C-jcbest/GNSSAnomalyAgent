"""Two range types share observed masks, while point events retain their owner."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_exp38 import combine, result_from_events, typed_rows  # noqa: E402


def event(kind, start, end):
    return {'kind': kind, 'start': start, 'end': end, 'status': 'confirmed',
        'phases': [], 'evidence': 'observed morphology'}


def test_range_additions_preserve_points_and_cross_type_overlap_and_missing_mask():
    observation = {'dates': list(range(100)), 'displacement_mm': [[0., 0., 0.]] * 100}
    observation['displacement_mm'][52] = None
    base = result_from_events(observation, [event('step', 50, 50), event('spike', 55, 55),
        event('transient', 60, 70)])
    additions = [event('transient', 50, 54), event('transient', 65, 75), event('deformation', 48, 80)]
    before = copy.deepcopy((base, additions))
    result = combine(observation, base, additions)
    assert result['prediction']['events'] == base['prediction']['events'] + [additions[0], additions[2]]
    assert not result['generic_mask'][52]
    assert (base, additions) == before


def test_range_vision_scope_reports_deformation_and_keeps_point_scores_inapplicable():
    from experiment_exp35 import aggregate_types
    from report_exp25 import score_record

    observation = {'dates': list(range(100)), 'background_end': 29,
        'displacement_mm': [[0., 0., 0.]] * 100}
    truth = {'events': [event('deformation', 48, 80)]}
    prediction = result_from_events(observation, truth['events'])
    scores = aggregate_types([score_record(observation, truth, prediction)])
    table = typed_rows({'methods': {'nrv_image_vision': scores, 'owv_image_vision': scores}})
    lookup = {(row['method'], row['kind']): row for row in table}
    assert lookup['nrv_image_vision', 'deformation']['aff_f1'] == 1.0
    assert lookup['nrv_image_vision', 'spike']['f1'] is None
    assert lookup['owv_image_vision', 'deformation']['aff_f1'] is None
