from datetime import date, timedelta

import pytest
from affiliation.metrics import pr_from_events

from gnss_sim.schemas import PointResult, RangeResult


def test_native_labels_cross_task_fp_failure_and_range_dedup():
    from gnss_sim.metrics import evaluate
    from gnss_sim.schemas import CaseTruth

    def native(cid, kind=None, start=100, duration=1):
        labels = [[0, 0, 0] for _ in range(365)]
        events = []
        if kind:
            end = start + duration - 1
            for i in range(start, end + 1):
                labels[i][0] = 1
            events = [{'type': kind, 'task': 'point' if duration == 1 else 'range', 'axis': 'N',
                       'start_index': start, 'end_index': end,
                       'start_date': date(2025, 1, 1) + timedelta(days=start),
                       'end_date': date(2025, 1, 1) + timedelta(days=end),
                       'persistent': kind == 'trend',
                       'operation': 'native_extremum' if duration == 1 else 'add',
                       'target_offset_mm': 3, 'observed_end_mm': 3, 'sigma_mm': 1,
                       'source': 'fixture', 'native_parameters': {}}]
        return CaseTruth(case_id=cid, background_group='group',
                         measurement_noise_mm=[[0, 0, 0]] * 365,
                         anomaly_delta_mm=[[0, 0, 0]] * 365,
                         noise_seed=1, position_seed=2, tods_seed=3,
                         native_labels=[max(v) for v in labels], axis_labels=labels,
                         source_lock_sha256='fixture', events=events)
    truths = {'case_0001': native('case_0001'),
              'case_0002': native('case_0002', 'global_extremum'),
              'case_0003': native('case_0003', 'trend', duration=90)}
    trend = truths['case_0003'].events[0]
    start = trend.start_index
    prediction = RangeResult(case_id='case_0003', method='test', status='success',
        predictions={'N': [(start, start + 9), (start + 5, start + 14)], 'E': [(0, 9)], 'U': []})
    r = evaluate(truths, {'case_0003': prediction}, 'range')
    assert (r['daily']['tp'], r['daily']['fp'], r['daily']['fn']) == (15, 10, 75)
    assert r['normal']['failed_cases'] == 1 and r['failed_cases'] == 2
    assert evaluate(truths, {}, 'range')['daily']['fn'] == 90
    assert r['affiliation']['positive_axes'] == 1
    spike = truths['case_0002'].events[0]
    rows = {'case_0002': PointResult(case_id='case_0002', method='test', status='success',
            predictions={'N': [spike.start_index, spike.start_index], 'E': [], 'U': []}),
            'case_0003': PointResult(case_id='case_0003', method='test', status='success',
            predictions={'N': [start], 'E': [], 'U': []})}
    p = evaluate(truths, rows, 'point')
    assert p['daily']['tp'] == 1 and p['daily']['fp'] == 1
    assert evaluate({'case_0001': truths['case_0001']}, {}, 'point')['daily']['f1'] is None
    # Point duplicates do not inflate scores, and task-negative axes do not enter Affiliation.
    assert p['affiliation'] == {'positive_axes': 1, 'precision': 1., 'recall': 1., 'f1': 1.}
    shifted = rows['case_0002'].model_copy(deep=True)
    shifted.predictions.N = [spike.start_index + 1]
    actual = evaluate(truths, {'case_0002': shifted}, 'point')
    expected = pr_from_events([(101, 102)], [(100, 101)], Trange=(0, 365))
    assert actual['daily']['f1'] == 0
    assert actual['affiliation']['precision'] == pytest.approx(expected['precision'])
    assert actual['affiliation']['recall'] == pytest.approx(expected['recall'])
    assert 0 < actual['affiliation']['f1'] < 1
    for task in ('point', 'range'):
        assert evaluate(truths, {}, task)['affiliation'] == {
            'positive_axes': 1, 'precision': 0., 'recall': 0., 'f1': 0.}
        assert evaluate({'case_0001': truths['case_0001']}, {}, task)['affiliation'] == {
            'positive_axes': 0, 'precision': None, 'recall': None, 'f1': None}
    shifted.status = 'failed'
    assert evaluate(truths, {'case_0002': shifted}, 'point')['affiliation']['f1'] == 0
