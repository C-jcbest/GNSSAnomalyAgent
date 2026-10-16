"""Attribute visual selection on fresh stations without conflating fusion gains."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import report_exp32 as shared
from exp28_data_baseline import METHODS
from experiment_exp32 import DATASET
from experiment_exp34 import MODES, ROOT, records_for
from report_exp25 import aggregate_cost, aggregate_rows, request_receipts, score_record
from report_exp26 import write_table
from report_exp29 import compact

from gnss_research.files import read_json, write_json
from gnss_research.schema import Prediction, event_masks, observed_values

PRIMARY = 'jcv_text_final_pure'
CONTROL = 'jcv_text_pipeline_pure'


def admission(table, strata, paired):
    def positive_seeds(main, control):
        groups = strata['cohort_seed']
        return sum(groups[main][seed]['range']['transient']['affiliation']['station_macro_f1']
            > groups[control][seed]['range']['transient']['affiliation']['station_macro_f1']
            for seed in groups[main])

    main, text = table[PRIMARY], table[CONTROL]
    mechanism = {'aff_gain_at_least_003': main['transient_aff_f1'] - text['transient_aff_f1'] >= .03,
        'daily_loss_within_002': main['transient_daily_f1'] - text['transient_daily_f1'] >= -.02,
        'actual_tp_gain': paired['visual_screen_vs_text_screen']['tp_delta'] > 0,
        'two_seeds_positive': positive_seeds(PRIMARY, CONTROL) >= 2}
    fused, base = table['jcv_text_final'], table['vector_nearest']
    system = {'aff_gain_at_least_001': fused['transient_aff_f1'] - base['transient_aff_f1'] >= .01,
        'actual_tp_gain': paired['fusion_vs_nearest']['tp_delta'] > 0,
        'daily_loss_within_002': fused['transient_daily_f1'] - base['transient_daily_f1'] >= -.02,
        'four_type_loss_within_001': fused['four_type_daily_f1'] - base['four_type_daily_f1'] >= -.01,
        'two_seeds_positive': positive_seeds('jcv_text_final', 'vector_nearest') >= 2}
    return {'mechanism_confirmed': all(mechanism.values()), 'mechanism_checks': mechanism,
        'stable_system_admitted': all(system.values()), 'system_checks': system,
        'scope': 'Fresh development confirmation; visual screening mechanism and stable-system utility are distinct.'}


def report(phase):
    visual = ROOT / 'validation-visual-v1'
    phases = ['pilot'] if phase == 'pilot' else ['pilot', 'remaining']
    jobs = []
    for part in phases:
        if not (visual / (part + '-materialized.json')).exists():
            raise ValueError('complete every registered station first')
        for stage in ('scan', 'joint'):
            jobs.extend(read_json(visual / f'{part}-{stage}-registration.json')['jobs'])
    baselines = [*METHODS, 'vector_nearest']
    methods = sorted([*baselines, *MODES, *(mode + '_pure' for mode in MODES)])
    rows = {method: [] for method in methods}
    process = {family: Counter() for family in ('image', 'text')}
    records = records_for(phase)
    for index, record in enumerate(records, 1):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        truth = read_json(DATASET / record['truth'])
        for method in methods:
            parent = ROOT / 'validation-baseline-v1/predictions' if method in baselines else visual / 'final'
            prediction = read_json(parent / method / station / 'result.json')
            rows[method].append({'id': station, 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': prediction.get('prediction'),
                **score_record(observation, truth, prediction)})
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        target = event_masks(Prediction.model_validate({'events': truth['events']}), observed)['transient']
        for family in process:
            scan = read_json(visual / 'predictions' / ('scan_' + family) / station / 'result.json')
            selected = np.zeros(len(observed), dtype=bool)
            for region in scan.get('decision', {}).get('regions', []):
                selected[region['start']:region['end'] + 1] = True
            process[family].update(stations=1, failed=int(scan['status'] != 'success'),
                positive_stations=int(target.any()), positive_region_hits_truth=int((selected & target).any()))
        if index % 12 == 0:
            print('Scored', index, '/', len(records), flush=True)
    summaries = {method: aggregate_rows(values) for method, values in rows.items()}
    table = {method: compact(method, summary) for method, summary in summaries.items()}
    for method, summary in summaries.items():
        ranges = summary['range']
        for key, kind in (('deformation_aff_f1', 'deformation'), ('range_union_aff_f1', 'range_union')):
            table[method][key] = ranges[kind]['affiliation']['station_macro_f1'] if ranges else None
        table[method]['range_types_aff_macro_f1'] = ranges['range_types_affiliation_macro_f1'] if ranges else None
    strata = {key: {method: {str(value): aggregate_rows([row for row in values if row[key] == value])
        for value in sorted({row[key] for row in values})} for method, values in rows.items()}
        for key in ('cohort_seed', 'scenario')}
    paired = {'visual_screen_vs_text_screen': shared.compare(rows, PRIMARY, CONTROL),
        'text_final_vs_image_final': shared.compare(rows, PRIMARY, 'jcv_image_pure'),
        'fusion_vs_nearest': shared.compare(rows, 'jcv_text_final', 'vector_nearest')}
    costs = aggregate_cost(request_receipts([job['work'] for job in jobs]))
    by_mode = {mode: aggregate_cost(request_receipts([job['work'] for job in jobs if job['mode'] == mode]))
        for mode in sorted({job['mode'] for job in jobs})}
    for mode in by_mode:
        images = 0
        for job in jobs:
            if job['mode'] != mode:
                continue
            for message in read_json(Path(job['work']) / 'request.json')['messages']:
                if isinstance(message['content'], list):
                    images += sum(block.get('type') == 'image_url' for block in message['content'])
        by_mode[mode]['requested_input_images'] = images
    output = Path(f'results/exp34-validation-{phase}-v1')
    output.mkdir(parents=True, exist_ok=False)
    decision = admission(table, strata, paired)
    write_json(output / 'summary.json', {'experiment_id': 'GNSS-VLM-EXP34', 'stations': len(records),
        'phase': phase, 'methods': summaries, 'costs': costs})
    for name, value in (('admission', decision), ('strata', strata), ('paired-changes', paired),
            ('process', process), ('costs-by-mode', by_mode)):
        write_json(output / (name + '.json'), value)
    with gzip.open(output / 'per-station.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(rows, stream, ensure_ascii=False, allow_nan=False)
    write_table(output, 'comparison.csv', list(table.values()))
    ranges = []
    for method, summary in summaries.items():
        for kind, metrics in (summary['range'] or {}).items():
            if isinstance(metrics, dict) and 'affiliation' in metrics:
                ranges.append({'method': method, 'kind': kind, **metrics['affiliation'],
                    **{'daily_' + key: value for key, value in metrics['daily_micro'].items()}})
    write_table(output, 'range-metrics.csv', ranges)
    keys = ('transient_aff_f1', 'transient_daily_f1', 'deformation_aff_f1', 'range_union_aff_f1',
        'four_type_daily_f1', 'transient_tp', 'transient_fp', 'transient_fn')
    lines = ['# EXP34：视觉选区—去叙述证据判定（VDE-GNSS）', '', f'{len(records)} fresh development stations, {phase}.', '',
        '| Method | Transient Aff | Daily F1 | Deformation Aff | Range union Aff | Four-type F1 | TP | FP | FN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method, row in table.items():
        lines.append('| ' + method + ' | ' + ' | '.join('—' if row[key] is None else f'{row[key]:.4f}' for key in keys) + ' |')
    lines.extend(['', 'Mechanism confirmed: ' + str(decision['mechanism_confirmed']),
        'Stable system admitted: ' + str(decision['stable_system_admitted']), '',
        'Pure outputs transient only and still receives numerical soft evidence.',
        'Affiliation averages truth-positive stations; read negative-station FP alongside it.'])
    (output / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    shared.PRIMARY = PRIMARY
    shared.CONTROLS = (CONTROL, 'jcv_image_pure')
    shared.draw_cases(output, rows, paired['visual_screen_vs_text_screen']['changes'])
    print(json.dumps({'admission': decision, 'costs': costs}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('pilot', 'full'), default='pilot')
    report(parser.parse_args().phase)
