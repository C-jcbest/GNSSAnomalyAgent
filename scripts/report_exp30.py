"""Multimetric and process evaluation with explicit pilot and confirmation gates."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp28_data_baseline import METHODS
from exp30_data_baseline import DATASET, ROOT, records_for
from report_exp25 import aggregate_cost, aggregate_rows, bootstrap, request_receipts, score_record
from report_exp26 import write_table
from report_exp29 import compact

from gnss_research.files import read_json, write_json
from gnss_research.schema import Prediction, event_masks, observed_values

CONTROLS = ('vector_nearest', 'vrc_image', 'air_text_pipeline', 'air_text_final',
            'air_global', 'air_image_no_numeric')


def gates(table, paired, strata, phase):
    main = table['air_image']
    changes = {name: {key: main[key] - table[name][key]
        for key in ('transient_aff_f1', 'transient_daily_f1', 'four_type_daily_f1')}
        for name in CONTROLS}
    if phase == 'pilot':
        checks = {'actual_new_tp': any(paired[name]['tp_improved_stations'] > 0
            for name in ('vector_nearest', 'vrc_image')),
            'daily_loss_within_003': changes['vector_nearest']['transient_daily_f1'] >= -.03}
    else:
        checks = {name + '_aff_gain': changes[name]['transient_aff_f1'] >= .01
                  for name in ('vector_nearest', 'air_text_pipeline')}
        checks.update({name + '_daily_loss': changes[name]['transient_daily_f1'] >= -.01
                       for name in ('vector_nearest', 'air_text_pipeline')})
        checks.update(vrc_daily_gain=changes['vrc_image']['transient_daily_f1'] >= .005,
            vrc_aff_preserved=changes['vrc_image']['transient_aff_f1'] >= -.01,
            actual_new_tp=paired['vector_nearest']['tp_improved_stations'] > 0,
            two_seeds_positive=sum(strata['cohort_seed']['air_image'][seed]['range']['transient']['affiliation']['station_macro_f1']
                > strata['cohort_seed']['vector_nearest'][seed]['range']['transient']['affiliation']['station_macro_f1']
                for seed in strata['cohort_seed']['air_image']) >= 2)
    return {'admitted': all(checks.values()), 'checks': checks, 'deltas': changes,
        'scope': 'pilot mechanism expansion gate' if phase == 'pilot' else 'prespecified comprehensive gate'}


def report(split, phase):
    visual = ROOT / (split + '-visual-v1')
    phases = ['pilot'] if phase == 'pilot' else ['pilot', 'remaining']
    for stage in phases:
        if not (visual / (stage + '-materialized.json')).exists():
            raise ValueError('materialize all planned stations first')
    records = records_for(split, 'pilot' if phase == 'pilot' else 'full')
    methods = sorted(set(METHODS) | {p.name for p in (visual / 'final').iterdir()})
    rows = {method: [] for method in methods}
    process = {family: Counter() for family in ('image', 'text')}
    job_paths = []
    for stage in phases:
        for step in ('scan', 'local'):
            registration = read_json(visual / f'{stage}-{step}-registration.json')
            job_paths.extend(job['work'] for job in registration['jobs'])
    for index, record in enumerate(records, 1):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        truth = read_json(DATASET / record['truth'])
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        target = event_masks(Prediction.model_validate({'events': truth['events']}), observed)['transient']
        for family, totals in process.items():
            receipt = read_json(visual / 'predictions' / ('scan_' + family) / station / 'result.json')
            regions = receipt.get('decision', {}).get('regions', [])
            mask = np.zeros(len(observed), dtype=bool)
            for region in regions:
                mask[region['start']:region['end'] + 1] = True
            totals['stations'] += 1
            totals['positive_stations'] += int(target.any())
            totals['positive_without_region'] += int(target.any() and not regions)
            totals['positive_region_hits_truth'] += int(target.any() and (mask & target).any())
            totals['regions'] += len(regions)
            totals['failed_scans'] += int(receipt['status'] != 'success')
            totals['template_summary'] += int(receipt.get('decision', {}).get('summary', '') in ('short observations', 'observations'))
        for method in methods:
            parent = ROOT / (split + '-baseline-v1/predictions') if method in METHODS else visual / 'final'
            result = read_json(parent / method / station / 'result.json')
            rows[method].append({'id': station, 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': result.get('prediction'),
                **score_record(observation, truth, result)})
        if index % 24 == 0:
            print('Scored', index, '/', len(records), flush=True)
    summaries = {method: aggregate_rows(values) for method, values in rows.items()}
    comparison = [compact(method, summary) for method, summary in summaries.items()]
    table = {r['method']: r for r in comparison}
    strata = {key: {method: {str(value): aggregate_rows([r for r in values if r[key] == value])
        for value in sorted({r[key] for r in values})} for method, values in rows.items()}
        for key in ('cohort_seed', 'scenario')}
    paired = {}
    for control in CONTROLS:
        changes = []
        for main, base in zip(rows['air_image'], rows[control], strict=True):
            a, b = main['range']['transient'], base['range']['transient']
            aff_a, aff_b = a['affiliation']['f1'], b['affiliation']['f1']
            delta_aff = aff_a - aff_b if aff_a is not None and aff_b is not None else None
            delta_tp = a['daily']['tp'] - b['daily']['tp']
            delta_fp = a['daily']['fp'] - b['daily']['fp']
            if delta_tp or delta_fp or (delta_aff is not None and abs(delta_aff) > 1e-12):
                changes.append({'station': main['id'], 'scenario': main['scenario'],
                    'cohort_seed': main['cohort_seed'], 'tp_delta': delta_tp, 'fp_delta': delta_fp,
                    'aff_delta': delta_aff, 'main_events': main['events'], 'control_events': base['events']})
        paired[control] = {'changes': changes,
            'tp_improved_stations': sum(c['tp_delta'] > 0 for c in changes),
            'fp_increased_stations': sum(c['fp_delta'] > 0 for c in changes),
            'aff_only_improved_stations': sum(c['aff_delta'] is not None and c['aff_delta'] > 0 and c['tp_delta'] <= 0 for c in changes),
            'bootstrap': bootstrap(rows['air_image'], rows[control])}
    output = Path(f'results/exp30-{split}-{phase}-v1')
    output.mkdir(parents=True, exist_ok=False)
    receipts = request_receipts(job_paths)
    costs = aggregate_cost(receipts)
    admission = gates(table, paired, strata, phase)
    write_json(output / 'summary.json', {'experiment_id': 'GNSS-VLM-EXP30', 'stations': len(records),
        'split': split, 'phase': phase, 'methods': summaries, 'costs': costs})
    write_json(output / 'admission.json', admission)
    write_json(output / 'process.json', process)
    write_json(output / 'strata.json', strata)
    write_json(output / 'paired-changes.json', paired)
    write_json(output / 'request-receipts.json', receipts)
    with gzip.open(output / 'per-station.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(rows, stream, ensure_ascii=False, allow_nan=False)
    write_table(output, 'comparison.csv', comparison)
    ranges = []
    for method, summary in summaries.items():
        if summary['range'] is not None:
            for kind, metrics in summary['range'].items():
                if isinstance(metrics, dict) and 'affiliation' in metrics:
                    ranges.append({'method': method, 'kind': kind, **metrics['affiliation'],
                        **{'daily_' + key: value for key, value in metrics['daily_micro'].items()}})
    write_table(output, 'range-metrics.csv', ranges)
    lines = ['# EXP30：视觉粗筛与自适应局部重定界（AIR-GNSS）', '',
        f'{len(records)} stations, {split}, {phase}.', '',
        '| Method | Transient Aff-F1 | Daily F1 | Four-type F1 | FP days |',
        '|---|---:|---:|---:|---:|']
    for row in comparison:
        values = [row[key] for key in ('transient_aff_f1', 'transient_daily_f1', 'four_type_daily_f1', 'transient_fp')]
        lines.append('| ' + row['method'] + ' | ' + ' | '.join('—' if v is None else f'{v:.4f}' for v in values) + ' |')
    lines.extend(['', 'Expansion/adoption gate: ' + str(admission['admitted']), '',
        'Pure methods output transient only; four-type F1 is not a fair full-system comparison.'])
    (output / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    chosen = ['vector_nearest', 'vrc_image', 'vrc_text', *CONTROLS[2:], 'air_image', 'air_image_pure']
    figure, panels = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for panel, key, title in zip(panels, ('transient_aff_f1', 'transient_daily_f1', 'four_type_daily_f1'),
            ('Transient Affiliation F1', 'Transient daily F1', 'Four-type daily F1'), strict=True):
        panel.barh(chosen, [table[name][key] for name in chosen], color='#357a91')
        panel.set_title(title)
        panel.set_xlim(0, 1)
        panel.grid(axis='x', alpha=.2)
    panels[0].invert_yaxis()
    figure.tight_layout()
    figure.savefig(output / 'ablation.png', dpi=150)
    plt.close(figure)
    print(json.dumps({'admission': admission, 'costs': costs, 'process': process}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--phase', choices=('pilot', 'full'), default='pilot')
    args = parser.parse_args()
    report(args.split, args.phase)
