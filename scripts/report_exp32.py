"""Separate model identification, full-system utility and all range categories."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp28_data_baseline import METHODS
from experiment_exp29 import plot_values
from experiment_exp31 import MODES
from experiment_exp32 import DATASET, ROOT, records_for
from report_exp25 import aggregate_cost, aggregate_rows, bootstrap, request_receipts, score_record
from report_exp26 import write_table
from report_exp29 import compact

from gnss_research.files import read_json, write_json
from gnss_research.schema import Prediction, event_masks, observed_values

PRIMARY = 'jcv_image_pure'
CONTROLS = ('jcv_text_final_pure', 'jcv_global_only_pure', 'jcv_local_only_pure',
    'jcv_image_no_numeric_pure', 'jcv_text_pipeline_pure')


def compare(rows, main, control):
    changes = []
    for primary, secondary in zip(rows[main], rows[control], strict=True):
        if primary['id'] != secondary['id']:
            raise ValueError('unpaired station order')
        a = primary['range']['transient']
        b = secondary['range']['transient']
        af, bf = a['affiliation']['f1'], b['affiliation']['f1']
        changes.append({'station': primary['id'], 'cohort_seed': primary['cohort_seed'],
            'scenario': primary['scenario'], 'tp_delta': a['daily']['tp'] - b['daily']['tp'],
            'fp_delta': a['daily']['fp'] - b['daily']['fp'],
            'aff_delta': af - bf if af is not None and bf is not None else None})
    return {'changes': changes, 'tp_delta': sum(row['tp_delta'] for row in changes),
        'fp_delta': sum(row['fp_delta'] for row in changes),
        'tp_improved_stations': sum(row['tp_delta'] > 0 for row in changes),
        'tp_harmed_stations': sum(row['tp_delta'] < 0 for row in changes),
        'bootstrap': bootstrap(rows[main], rows[control])}


def admission(table, strata, paired, phase):
    main, text = table[PRIMARY], table['jcv_text_final_pure']
    seeds = strata['cohort_seed']

    def positive_seeds(primary, control):
        return sum(seeds[primary][seed]['range']['transient']['affiliation']['station_macro_f1']
            > seeds[control][seed]['range']['transient']['affiliation']['station_macro_f1']
            for seed in seeds[primary])

    checks = {'aff_gain_at_least_003': main['transient_aff_f1'] - text['transient_aff_f1'] >= .03,
        'daily_loss_within_001': main['transient_daily_f1'] - text['transient_daily_f1'] >= -.01,
        'actual_tp_gain': paired['model_vs_text']['tp_delta'] > 0,
        'two_seeds_positive': positive_seeds(PRIMARY, 'jcv_text_final_pure') >= 2}
    if phase == 'full':
        for control in ('jcv_global_only_pure', 'jcv_local_only_pure'):
            checks[control + '_aff_gain_at_least_001'] = main['transient_aff_f1'] - table[control]['transient_aff_f1'] >= .01
    fused, base = table['jcv_image'], table['vector_nearest']
    system = {'aff_gain_at_least_001': fused['transient_aff_f1'] - base['transient_aff_f1'] >= .01,
        'actual_tp_gain': paired['fusion_vs_nearest']['tp_delta'] > 0,
        'daily_loss_within_001': fused['transient_daily_f1'] - base['transient_daily_f1'] >= -.01,
        'four_type_loss_within_0005': fused['four_type_daily_f1'] - base['four_type_daily_f1'] >= -.005,
        'two_seeds_positive': positive_seeds('jcv_image', 'vector_nearest') >= 2}
    return {'experiment_id': 'GNSS-VLM-EXP32', 'phase': phase,
        'mechanism_confirmed': all(checks.values()), 'mechanism_checks': checks,
        'stable_system_admitted': all(system.values()), 'system_checks': system,
        'scope': 'Independent development confirmation of an EXP31 exploratory hypothesis; not held-out test or a replacement of EXP31 admission.'}


def draw_cases(output, rows, changes):
    selected = {}
    for name, score in (
        ('added_tp', lambda change: change['tp_delta']),
        ('lost_tp', lambda change: -change['tp_delta']),
        ('added_fp', lambda change: change['fp_delta']),
        ('aff_only', lambda change: change['aff_delta']
            if change['aff_delta'] is not None and change['tp_delta'] <= 0 else 0),
    ):
        case = max(changes, key=score)
        if score(case) > 0:
            selected.setdefault(case['station'], {'selection': [], 'change': case})['selection'].append(name)
    folder = output / 'cases'
    folder.mkdir()
    for station, case in selected.items():
        observation = read_json(DATASET / 'observations' / (station + '.json'))
        truth = read_json(DATASET / 'truth' / (station + '.json'))
        values = plot_values(observation)
        first = observation['background_end'] + 1
        figure, panels = plt.subplots(4, 1, figsize=(16, 12), dpi=130, sharex=True)
        for axis, panel in enumerate(panels[:3]):
            panel.plot(np.arange(first, len(values)), values[first:, axis], '.-',
                linewidth=.7, markersize=2, color='#356d85')
            for event in truth['events']:
                if event['kind'] == 'transient':
                    panel.axvspan(event['start'], event['end'] + 1, color='#cf9c29', alpha=.2)
            panel.set_ylabel('NEU'[axis] + ' / station MAD')
            panel.grid(alpha=.2)
        predictions = {'truth': truth['events']}
        for method in (PRIMARY, *CONTROLS, 'jcv_image', 'vector_nearest'):
            predictions[method] = next(row['events']['events'] for row in rows[method] if row['id'] == station)
        for index, events in enumerate(predictions.values()):
            spans = [(event['start'], event['end'] - event['start'] + 1)
                for event in events if event['kind'] == 'transient']
            panels[3].broken_barh(spans, (index - .3, .6), facecolors='#cf9c29' if index == 0 else '#397e91')
        panels[3].set_yticks(range(len(predictions)), list(predictions))
        panels[3].set_ylim(-.7, len(predictions) - .3)
        panels[3].set_xlim(first, len(values) - 1)
        panels[3].set_xlabel('Original index; missing observations remain absent')
        panels[3].grid(axis='x', alpha=.2)
        figure.suptitle(station + ' / model vs matched final text: ' + ', '.join(case['selection']))
        figure.tight_layout()
        figure.savefig(folder / (station + '.png'))
        plt.close(figure)
        case.update(truth=truth, predictions=predictions)
    write_json(folder / 'cases.json', selected)


def report(phase):
    visual = ROOT / 'validation-visual-v1'
    phases = ['pilot'] if phase == 'pilot' else ['pilot', 'remaining']
    jobs = []
    for part in phases:
        if not (visual / (part + '-materialized.json')).exists():
            raise ValueError('complete every registered station first')
        for step in ('scan', 'joint'):
            jobs.extend(read_json(visual / f'{part}-{step}-registration.json')['jobs'])
    records = records_for(phase)
    baselines = [*METHODS, 'vector_nearest']
    methods = sorted([*baselines, *MODES, *(mode + '_pure' for mode in MODES)])
    rows = {method: [] for method in methods}
    process = {family: Counter() for family in ('image', 'text')}
    for index, record in enumerate(records, 1):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        truth = read_json(DATASET / record['truth'])
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        target = event_masks(Prediction.model_validate({'events': truth['events']}), observed)['transient']
        strong = read_json(ROOT / 'validation-baseline-v1/predictions/vector_nearest' / station / 'result.json')
        strong_score = score_record(observation, truth, strong)
        numerical_days = strong_score['range']['transient']['daily']
        if numerical_days['tp'] + numerical_days['fn'] == 0:
            numerical_status = 'truth_negative'
        elif numerical_days['fp'] == 0 and numerical_days['fn'] == 0:
            numerical_status = 'positive_numerically_perfect'
        else:
            numerical_status = 'positive_numerically_imperfect'
        for method in methods:
            parent = ROOT / 'validation-baseline-v1/predictions' if method in baselines else visual / 'final'
            prediction = strong if method == 'vector_nearest' else read_json(parent / method / station / 'result.json')
            score = strong_score if method == 'vector_nearest' else score_record(observation, truth, prediction)
            rows[method].append({'id': station, 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': prediction.get('prediction'),
                'numeric_transient_status': numerical_status, **score})
        for family in process:
            scan = read_json(visual / 'predictions' / ('scan_' + family) / station / 'result.json')
            regions = scan.get('decision', {}).get('regions', [])
            selected = np.zeros(len(observed), dtype=bool)
            for region in regions:
                selected[region['start']:region['end'] + 1] = True
            process[family].update(stations=1, failed=int(scan['status'] != 'success'), regions=len(regions),
                truth_positive_stations=int(target.any()), region_hits_truth=int((selected & target).any()))
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
        for key in ('cohort_seed', 'scenario', 'numeric_transient_status')}
    paired = {'model_vs_text': compare(rows, PRIMARY, 'jcv_text_final_pure'),
        'fusion_vs_nearest': compare(rows, 'jcv_image', 'vector_nearest')}
    for control in CONTROLS[1:]:
        paired[control] = compare(rows, PRIMARY, control)
    costs = aggregate_cost(request_receipts([job['work'] for job in jobs]))
    by_mode = {mode: aggregate_cost(request_receipts([job['work'] for job in jobs if job['mode'] == mode]))
        for mode in sorted({job['mode'] for job in jobs})}
    for mode in by_mode:
        images = 0
        for job in jobs:
            if job['mode'] != mode:
                continue
            request = read_json(Path(job['work']) / 'request.json')
            for message in request['messages']:
                if isinstance(message['content'], list):
                    images += sum(block.get('type') == 'image_url' for block in message['content'])
        by_mode[mode]['requested_input_images'] = images
    output = Path(f'results/exp32-validation-{phase}-v1')
    output.mkdir(parents=True, exist_ok=False)
    decision = admission(table, strata, paired, phase)
    write_json(output / 'summary.json', {'experiment_id': 'GNSS-VLM-EXP32', 'stations': len(records),
        'phase': phase, 'methods': summaries, 'costs': costs})
    for name, data in (('strata', strata), ('paired-changes', paired), ('admission', decision),
            ('process', process), ('costs-by-mode', by_mode)):
        write_json(output / (name + '.json'), data)
    with gzip.open(output / 'per-station.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(rows, stream, ensure_ascii=False, allow_nan=False)
    write_table(output, 'comparison.csv', list(table.values()))
    range_rows = []
    for method, summary in summaries.items():
        for kind, metrics in (summary['range'] or {}).items():
            if isinstance(metrics, dict) and 'affiliation' in metrics:
                range_rows.append({'method': method, 'kind': kind, **metrics['affiliation'],
                    **{'daily_' + key: value for key, value in metrics['daily_micro'].items()}})
    write_table(output, 'range-metrics.csv', range_rows)
    lines = ['# EXP32：联合上下文视觉识别确认（JVC-GNSS）', '', f'{len(records)} development stations, {phase}.', '',
        '| Method | Transient Aff | Daily F1 | Deformation Aff | Range union Aff | Four-type F1 | TP | FP | FN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys = ('transient_aff_f1', 'transient_daily_f1', 'deformation_aff_f1', 'range_union_aff_f1',
        'four_type_daily_f1', 'transient_tp', 'transient_fp', 'transient_fn')
    for method, row in table.items():
        lines.append('| ' + method + ' | ' + ' | '.join('—' if row[key] is None else f'{row[key]:.4f}' for key in keys) + ' |')
    lines.extend(['', 'Mechanism confirmed: ' + str(decision['mechanism_confirmed']),
        'Stable system admitted: ' + str(decision['stable_system_admitted']), '',
        'Pure outputs transient only; their four-type F1 is not a full-system ranking.',
        'Pure still receives numerical soft evidence except no_numeric.',
        'Affiliation averages truth-positive stations; negative-station false alarms are separate.'])
    (output / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    draw_cases(output, rows, paired['model_vs_text']['changes'])
    print(json.dumps({'admission': decision, 'costs': costs}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('pilot', 'full'), default='pilot')
    report(parser.parse_args().phase)
