"""Paired multimetric evaluation, modality attribution and prespecified admission."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from exp29_data_baseline import DATASET, ROOT
from report_exp25 import aggregate_cost, aggregate_rows, bootstrap, request_receipts, score_record
from report_exp26 import write_table

from gnss_research.files import read_json, write_json

MAIN = 'vf_vl_image_complement'
CONTROLS = ['vector_nearest', 'numeric_all_pairs', 'vf_vl_text_complement', 'nf_vl',
            'vf_vl_image_pure', 'vf_vl_image_raw_complement', 'vf_vl_image_verified']


def compact(method, summary):
    transient = summary['range']['transient'] if summary['range'] is not None else None
    row = {'method': method, 'transient_aff_f1': None, 'transient_daily_f1': None,
        'four_type_daily_f1': summary['original'].get('types_daily_macro_f1'),
        'transient_tp': None, 'transient_fp': None, 'transient_fn': None,
        'transient_precision': None, 'transient_recall': None,
        'generic_daily_f1': summary['original']['generic_daily_micro']['f1'],
        'generic_aff_f1': summary['generic_affiliation']['station_macro_f1']}
    if transient is not None:
        row['transient_aff_f1'] = transient['affiliation']['station_macro_f1']
        row['transient_daily_f1'] = transient['daily_micro']['f1']
        for key in ('tp', 'fp', 'fn', 'precision', 'recall'):
            row['transient_' + key] = transient['daily_micro'][key]
    return row


def summarize(split, phase, output=None):
    visual = ROOT / f'{split}-visual-v1'
    phases = ['pilot'] if phase == 'pilot' else ['pilot', 'remaining']
    records = []
    for stage in phases:
        if not (visual / f'{stage}-materialized.json').exists():
            raise ValueError('materialize all planned stations before reporting')
        registration = read_json(visual / f'{stage}-registration.json')
        records.extend(registration['records'])
    baseline = Path(registration['baseline'])
    baseline_methods = read_json(baseline / 'registration.json')['methods']
    methods = sorted(set(baseline_methods) | {p.name for p in (visual / 'final').iterdir()
        if phase == 'pilot' or 'flash' not in p.name})
    rows = {method: [] for method in methods}
    dependencies = []
    candidate_stats = []
    for number, record in enumerate(records, 1):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        truth = read_json(DATASET / record['truth'])
        for method in methods:
            parent = baseline / 'predictions' if method in baseline_methods else visual / 'final'
            result = read_json(parent / method / station / 'result.json')
            dependencies.extend(result.get('evaluation_dependency_workdirs', []))
            rows[method].append({'id': station, 'scenario': truth['scenario'],
                'cohort_seed': record['cohort_seed'],
                'events': result.get('prediction'), **score_record(observation, truth, result)})
        tool = read_json(visual / 'predictions/vf_vl_image' / station / 'tool-evidence.json')
        candidate_stats.append({'station': station, 'visual_intervals': len(tool['receipts']),
            'snapped': len(tool['proposed']), 'verified': len(tool['verified'])})
        if number % 60 == 0:
            print('Scored', number, '/', len(records), flush=True)
    summary = {method: aggregate_rows(values) for method, values in rows.items()}
    comparison = [compact(method, values) for method, values in summary.items()]
    strata = {key: {method: {str(value): aggregate_rows([r for r in values if r[key] == value])
        for value in sorted({r[key] for r in values})} for method, values in rows.items()}
        for key in ('cohort_seed', 'scenario')}
    changes = {}
    for control in CONTROLS:
        pairs = []
        for a, b in zip(rows[MAIN], rows[control]):
            da, db = a['range']['transient'], b['range']['transient']
            if a['events'] != b['events']:
                fa, fb = da['affiliation']['f1'], db['affiliation']['f1']
                pairs.append({'station': a['id'], 'scenario': a['scenario'], 'cohort_seed': a['cohort_seed'],
                    'aff_delta': fa - fb if fa is not None and fb is not None else None,
                    'tp_delta': da['daily']['tp'] - db['daily']['tp'],
                    'fp_delta': da['daily']['fp'] - db['daily']['fp'],
                    'main_events': a['events'], 'control_events': b['events']})
        changes[control] = {'changes': pairs, 'bootstrap': bootstrap(rows[MAIN], rows[control]),
            'tp_improved_stations': sum(p['tp_delta'] > 0 for p in pairs),
            'fp_increased_stations': sum(p['fp_delta'] > 0 for p in pairs),
            'aff_only_improved_stations': sum(p['aff_delta'] is not None and p['aff_delta'] > 0
                                             and p['tp_delta'] <= 0 for p in pairs)}
    tab = {r['method']: r for r in comparison}
    admission = {'admitted': False, 'phase': phase, 'stations': len(records), 'criteria': {}}
    for control in ('vector_nearest', 'vf_vl_text_complement'):
        aff = tab[MAIN]['transient_aff_f1'] - tab[control]['transient_aff_f1']
        daily = tab[MAIN]['transient_daily_f1'] - tab[control]['transient_daily_f1']
        admission['criteria'][control] = {'aff_delta': aff, 'daily_delta': daily,
                                         'passed': aff >= .02 and daily >= -.01}
    admission['admitted'] = phase == 'full' and all(v['passed'] for v in admission['criteria'].values())
    output = output or Path(f'results/exp29-{split}-{phase}-v1')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', {'experiment_id': 'GNSS-VLM-EXP29',
        'stations': len(records), 'split': split, 'phase': phase, 'methods': summary})
    write_json(output / 'strata.json', strata)
    write_json(output / 'paired-changes.json', changes)
    write_json(output / 'admission.json', admission)
    write_json(output / 'candidates.json', candidate_stats)
    receipts = request_receipts(dependencies)
    write_json(output / 'costs.json', aggregate_cost(receipts))
    write_json(output / 'request-receipts.json', receipts)
    completed_workdirs = [str(path.parent) for path in (visual / 'predictions').rglob('request.json')
                          if (path.parent / 'result.json').exists()]
    all_receipts = request_receipts(completed_workdirs)
    write_json(output / 'all-experiment-costs.json', aggregate_cost(all_receipts))
    with gzip.open(output / 'per-station.json.gz', 'xt', encoding='utf-8') as stream:
        json.dump(rows, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    write_table(output, 'comparison.csv', comparison)
    ranges = []
    for method, report in summary.items():
        if report['range'] is None:
            continue
        for kind in ('transient', 'deformation', 'range_union'):
            values = report['range'][kind]
            ranges.append({'method': method, 'kind': kind,
                'aff_f1': values['affiliation']['station_macro_f1'],
                'aff_precision': values['affiliation']['precision'],
                'aff_recall': values['affiliation']['recall'],
                **{'daily_' + key: val for key, val in values['daily_micro'].items()},
                'truth_event_mean_coverage': values['truth_event_mean_coverage'],
                'negative_station_fp_days': values['negative_station_fp_days']})
    write_table(output, 'range-metrics.csv', ranges)
    lines = ['# EXP29：视觉先行定位与数值验证', '', f'{split}, {phase}, {len(records)} stations.', '',
        '| Method | Transient Aff-F1 | Daily F1 | Four-type F1 | FP days |',
        '|---|---:|---:|---:|---:|']
    for row in comparison:
        fields = [row[key] for key in ('transient_aff_f1', 'transient_daily_f1',
                                      'four_type_daily_f1', 'transient_fp')]
        lines.append('| ' + row['method'] + ' | ' + ' | '.join(
            '—' if value is None else f'{value:.4f}' for value in fields) + ' |')
    lines.extend(['', 'Admission: ' + str(admission['admitted'])])
    (output / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--phase', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    summarize(args.split, args.phase, args.out)
