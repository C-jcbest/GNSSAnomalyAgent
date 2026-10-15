"""Full paired comparison of three-stage adjudication and all previous EXP29 controls."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from exp29_data_baseline import DATASET
from experiment_exp29_s2 import JOINT, MODES, OUT
from report_exp25 import aggregate_cost, aggregate_rows, bootstrap, request_receipts, score_record
from report_exp26 import write_table
from report_exp29 import compact

from gnss_research.files import read_json, write_json


def summarize(phase):
    phases = ['pilot'] if phase == 'pilot' else ['pilot', 'remaining']
    records = []
    for stage in phases:
        if not (OUT / (stage + '-materialized.json')).exists():
            raise ValueError('complete planned S2 phase')
        records.extend(read_json(OUT / (stage + '-registration.json'))['records'])
    original_report = Path('results/exp29-validation-pilot-v2' if phase == 'pilot'
                           else 'results/exp29-validation-full-v1')
    with gzip.open(original_report / 'per-station.json.gz', 'rt', encoding='utf-8') as stream:
        original = json.load(stream)
    rows = {}
    for mode, values in original.items():
        mapping = {r['id']: r for r in values}
        rows[mode] = [mapping[r['station_id']] for r in records]
    new_modes = {mode: OUT / 'final' for mode in MODES}
    new_modes.update({mode: JOINT for mode in ('joint_vl_image', 'joint_vl_text', 'joint_numeric_all_pairs')})
    rows.update({mode: [] for mode in new_modes})
    for record in records:
        observation = read_json(DATASET / record['observation'])
        truth = read_json(DATASET / record['truth'])
        for mode, folder in new_modes.items():
            result = read_json(folder / mode / record['station_id'] / 'result.json')
            rows[mode].append({'id': record['station_id'], 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': result['prediction'],
                **score_record(observation, truth, result)})
    summary = {mode: aggregate_rows(values) for mode, values in rows.items()}
    main = summary['vnv_image']['range']['transient']
    controls = ('vector_nearest', 'vnv_text_pipeline', 'vnv_text_final', 'vf_vl_image_raw_complement',
                'joint_vl_image', 'nf_vl', 'joint_numeric_all_pairs')
    paired = {}
    for control in controls:
        changes = []
        for a, b in zip(rows['vnv_image'], rows[control]):
            ta, tb = a['range']['transient'], b['range']['transient']
            if a['events'] != b['events']:
                changes.append({'station': a['id'], 'scenario': a['scenario'], 'cohort_seed': a['cohort_seed'],
                    'main_events': a['events'], 'control_events': b['events'],
                    'tp_delta': ta['daily']['tp'] - tb['daily']['tp'],
                    'fp_delta': ta['daily']['fp'] - tb['daily']['fp'],
                    'main_transient': ta, 'control_transient': tb})
        paired[control] = {'changes': changes, 'bootstrap': bootstrap(rows['vnv_image'], rows[control]),
            'tp_improved_stations': sum(c['tp_delta'] > 0 for c in changes),
            'fp_increased_stations': sum(c['fp_delta'] > 0 for c in changes)}
    checks = {}
    for control in ('vector_nearest', 'vnv_text_pipeline'):
        other = summary[control]['range']['transient']
        checks[control] = {'aff_delta': main['affiliation']['station_macro_f1'] - other['affiliation']['station_macro_f1'],
            'daily_delta': main['daily_micro']['f1'] - other['daily_micro']['f1']}
    seeds = {str(seed): {mode: aggregate_rows([r for r in values if r['cohort_seed'] == seed])
        for mode, values in rows.items()} for seed in sorted({r['cohort_seed'] for r in records})}
    positive_seeds = sum(s['vnv_image']['range']['transient']['affiliation']['station_macro_f1']
        > s['vector_nearest']['range']['transient']['affiliation']['station_macro_f1'] for s in seeds.values())
    improved = paired['vector_nearest']['tp_improved_stations']
    admitted = phase == 'full' and improved > 0 and positive_seeds >= 2 and all(
        value['aff_delta'] >= .02 and value['daily_delta'] >= -.01 for value in checks.values())
    output = Path(f'results/exp29-s2-validation-{phase}-v1')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', {'experiment_id': 'GNSS-VLM-EXP29-S2',
        'stations': len(records), 'phase': phase, 'methods': summary})
    write_json(output / 'paired-changes.json', paired)
    write_json(output / 'seeds.json', seeds)
    write_json(output / 'admission.json', {'admitted': admitted, 'checks': checks,
        'tp_improved_stations': improved, 'positive_seeds': positive_seeds})
    workdirs = [str(OUT / 'predictions' / mode / r['station_id']) for mode in MODES for r in records]
    receipts = request_receipts(workdirs)
    write_json(output / 'costs.json', aggregate_cost(receipts))
    write_json(output / 'request-receipts.json', receipts)
    comparison = [compact(mode, report) for mode, report in summary.items()]
    write_table(output, 'comparison.csv', comparison)
    with gzip.open(output / 'per-station.json.gz', 'xt', encoding='utf-8') as stream:
        json.dump(rows, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    lines = ['# EXP29-S2：视觉—数值—视觉终审', '', f'{len(records)} development stations.', '',
        '| Method | Transient Aff-F1 | Daily F1 | Four-type F1 | FP days |', '|---|---:|---:|---:|---:|']
    for row in comparison:
        keys = ('transient_aff_f1', 'transient_daily_f1', 'four_type_daily_f1', 'transient_fp')
        lines.append('| ' + row['method'] + ' | ' + ' | '.join('—' if row[k] is None else f'{row[k]:.4f}' for k in keys) + ' |')
    lines.append('\nAdmission: ' + str(admitted))
    (output / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('pilot', 'full'), default='pilot')
    summarize(parser.parse_args().phase)
