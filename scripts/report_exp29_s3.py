"""Independent confirmation of the unchanged visual-range complement."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from exp29_data_baseline import DATASET, ROOT
from experiment_exp29_s3 import OUT
from report_exp25 import aggregate_cost, aggregate_rows, bootstrap, request_receipts, score_record
from report_exp26 import write_table
from report_exp29 import compact

from gnss_research.files import read_json, write_json

MAIN = 'vf_vl_image_raw_complement'


def summarize(phase):
    phases = ['pilot'] if phase == 'pilot' else ['pilot', 'remaining']
    records = []
    for stage in phases:
        if not (OUT / (stage + '-materialized.json')).exists():
            raise ValueError('complete frozen test phase first')
        records.extend(read_json(OUT / (stage + '-registration.json'))['records'])
    baseline = ROOT / 'test-baseline-v1'
    baseline_methods = read_json(baseline / 'registration.json')['methods']
    methods = sorted(set(baseline_methods) | {p.name for p in (OUT / 'final').iterdir()})
    rows = {mode: [] for mode in methods}
    for number, record in enumerate(records, 1):
        observation = read_json(DATASET / record['observation'])
        truth = read_json(DATASET / record['truth'])
        for mode in methods:
            folder = baseline / 'predictions' if mode in baseline_methods else OUT / 'final'
            result = read_json(folder / mode / record['station_id'] / 'result.json')
            rows[mode].append({'id': record['station_id'], 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': result.get('prediction'),
                **score_record(observation, truth, result)})
        if number % 60 == 0:
            print('S3 scored', number, '/', len(records), flush=True)
    summary = {mode: aggregate_rows(values) for mode, values in rows.items()}
    main = summary[MAIN]['range']['transient']
    paired = {}
    controls = ['vector_nearest', 'vf_vl_text_raw_complement', 'vf_vl_image_complement',
                'joint_vl_image', 'vf_vl_image_pure']
    for control in controls:
        changes = []
        for a, b in zip(rows[MAIN], rows[control]):
            x, y = a['range']['transient'], b['range']['transient']
            if a['events'] != b['events']:
                changes.append({'station': a['id'], 'cohort_seed': a['cohort_seed'], 'scenario': a['scenario'],
                    'main_events': a['events'], 'control_events': b['events'],
                    'tp_delta': x['daily']['tp'] - y['daily']['tp'],
                    'fp_delta': x['daily']['fp'] - y['daily']['fp'],
                    'main_transient': x, 'control_transient': y})
        paired[control] = {'changes': changes, 'bootstrap': bootstrap(rows[MAIN], rows[control]),
            'tp_improved_stations': sum(c['tp_delta'] > 0 for c in changes),
            'fp_increased_stations': sum(c['fp_delta'] > 0 for c in changes)}
    checks = {}
    for control in ('vector_nearest', 'vf_vl_text_raw_complement'):
        other = summary[control]['range']['transient']
        checks[control] = {'aff_delta': main['affiliation']['station_macro_f1'] - other['affiliation']['station_macro_f1'],
            'daily_delta': main['daily_micro']['f1'] - other['daily_micro']['f1']}
    added_tp = main['daily_micro']['tp'] - summary['vector_nearest']['range']['transient']['daily_micro']['tp']
    supported = phase == 'full' and added_tp > 0 and all(
        value['aff_delta'] >= .01 and value['daily_delta'] >= -.01 for value in checks.values())
    strata = {key: {mode: {str(value): aggregate_rows([r for r in values if r[key] == value])
        for value in sorted({r[key] for r in values})} for mode, values in rows.items()}
        for key in ('cohort_seed', 'scenario')}
    output = Path(f'results/exp29-s3-test-{phase}-v1')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', {'experiment_id': 'GNSS-VLM-EXP29-S3', 'split': 'test',
        'stations': len(records), 'phase': phase, 'methods': summary})
    write_json(output / 'paired-changes.json', paired)
    write_json(output / 'strata.json', strata)
    write_json(output / 'confirmation.json', {'supported': supported, 'checks': checks,
        'added_tp_days': added_tp, 'selection': 'Secondary chosen on development; test untouched before registration'})
    comparison = [compact(mode, report) for mode, report in summary.items()]
    write_table(output, 'comparison.csv', comparison)
    ranges = []
    for mode, report in summary.items():
        if report['range'] is None:
            continue
        for kind in ('transient', 'deformation', 'range_union'):
            metric = report['range'][kind]
            ranges.append({'method': mode, 'kind': kind,
                'aff_f1': metric['affiliation']['station_macro_f1'],
                'aff_precision': metric['affiliation']['precision'],
                'aff_recall': metric['affiliation']['recall'],
                **{'daily_' + key: value for key, value in metric['daily_micro'].items()},
                'coverage': metric['truth_event_mean_coverage'],
                'negative_station_fp_days': metric['negative_station_fp_days']})
    write_table(output, 'range-metrics.csv', ranges)
    with gzip.open(output / 'per-station.json.gz', 'xt', encoding='utf-8') as stream:
        json.dump(rows, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    workdirs = [str(OUT / 'predictions' / mode / r['station_id'])
                for mode in ('vf_vl_image', 'vf_vl_text') for r in records]
    receipts = request_receipts(workdirs)
    write_json(output / 'request-receipts.json', receipts)
    write_json(output / 'costs.json', aggregate_cost(receipts))
    lines = ['# EXP29-S3：视觉范围互补的独立确认', '', f'{len(records)} independent test stations.', '',
        '| Method | Transient Aff-F1 | Daily F1 | Four-type F1 | FP days |', '|---|---:|---:|---:|---:|']
    for row in comparison:
        keys = ('transient_aff_f1', 'transient_daily_f1', 'four_type_daily_f1', 'transient_fp')
        lines.append('| ' + row['method'] + ' | ' + ' | '.join('—' if row[k] is None else f'{row[k]:.4f}' for k in keys) + ' |')
    lines.append('\nConfirmation: ' + str(supported))
    (output / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('pilot', 'full'), default='pilot')
    summarize(parser.parse_args().phase)
