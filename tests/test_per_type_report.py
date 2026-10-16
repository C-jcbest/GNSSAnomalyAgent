"""Keep anomaly types separate and distinguish absent outputs from detector errors."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from report_per_type import export, per_type_rows  # noqa: E402

from gnss_research.files import read_json, sha256, write_json  # noqa: E402
from gnss_research.metrics import counts_metrics  # noqa: E402


def sample_summary():
    counts = {'spike': (1, 0, 8), 'step': (1, 0, 3),
        'transient': (3, 0, 4), 'deformation': (4, 0, 2)}
    types = {kind: {'daily_micro': counts_metrics(*values)} for kind, values in counts.items()}
    return {'experiment_id': 'fixture', 'stations': 2, 'methods': {
        'full': {'original': {'types': types, 'types_daily_macro_f1': .999}, 'range': None},
        'image_pure': {'original': {'types': {'transient': types['transient']}}, 'range': None},
        'generic': {'original': {'types': None}, 'range': None}}}


def test_preserves_each_type_score_and_marks_unsupported_types_as_not_applicable():
    rows = per_type_rows(sample_summary())
    full = {row['kind']: row for row in rows if row['method'] == 'full'}
    assert [full[kind]['f1'] for kind in ('spike', 'step', 'transient', 'deformation')] == [.2, .4, .6, .8]
    pure = [row for row in rows if row['method'] == 'image_pure']
    assert next(row for row in pure if row['kind'] == 'transient')['f1'] == .6
    assert all(row['f1'] is None and row['evaluated'] is False for row in pure if row['kind'] != 'transient')
    assert all(row['f1'] is None for row in rows if row['method'] == 'generic')


def test_export_does_not_change_frozen_input_or_export_old_cross_type_average(tmp_path):
    source = tmp_path / 'summary.json'
    write_json(source, sample_summary())
    before = sha256(source)
    output = tmp_path / 'per-type'
    export(source, output)
    assert sha256(source) == before
    result = read_json(output / 'per-type.json')
    assert result['source_summary_sha256'] == before
    assert result['new_model_requests'] == 0
    assert len(result['rows']) == 12
    assert 'types_daily_macro_f1' not in (output / 'per-type.json').read_text(encoding='utf-8')
    assert '四类F1' not in (output / 'results.md').read_text(encoding='utf-8')


def test_primary_scores_follow_point_daily_and_range_affiliation_policy():
    summary = sample_summary()
    summary['methods']['full']['range'] = {
        kind: {'affiliation': {'precision': .9, 'recall': .7, 'station_macro_f1': .75},
            'negative_station_fp_days': 12}
        for kind in ('transient', 'deformation')}
    typed = {row['kind']: row for row in per_type_rows(summary) if row['method'] == 'full'}
    assert typed['spike']['primary_f1'] == .2
    assert typed['step']['primary_f1'] == .4
    for kind in ('transient', 'deformation'):
        assert typed[kind]['primary_metric'] == 'affiliation_f1'
        assert typed[kind]['primary_f1'] == .75
        assert typed[kind]['primary_precision'] == .9
        assert typed[kind]['primary_recall'] == .7
        assert typed[kind]['negative_station_fp_days'] == 12
    assert typed['transient']['f1'] == .6
