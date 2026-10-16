"""Same-station range, daily and process evaluation for the frozen JCV experiment."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp28_data_baseline import METHODS
from exp31_data_baseline import DATASET, ROOT, records_for
from experiment_exp31 import MODES, merged_contexts
from report_exp25 import aggregate_cost, aggregate_rows, bootstrap, request_receipts, score_record
from report_exp26 import write_table
from report_exp29 import compact

from gnss_research.files import read_json, write_json
from gnss_research.schema import Prediction, event_masks, observed_values

MAIN = 'jcv_image'
CONTROLS = ('vector_nearest', 'vrc_image', 'air_image', 'air_global', 'jcv_global_only',
            'jcv_local_only', 'jcv_text_final', 'jcv_text_pipeline', 'jcv_image_no_numeric')


def admission(table, paired, strata, phase):
    changes = {name: {key: table[MAIN][key] - table[name][key]
        for key in ('transient_aff_f1', 'transient_daily_f1', 'four_type_daily_f1')}
        for name in CONTROLS}
    if phase == 'pilot':
        checks = {'actual_new_tp': any(paired[name]['tp_improved_stations'] > 0
                      for name in ('vector_nearest', 'vrc_image')),
                  'daily_loss_within_003': changes['vector_nearest']['transient_daily_f1'] >= -.03}
        return {'admitted': all(checks.values()), 'checks': checks, 'deltas': changes,
                'scope': 'Development pilot expansion gate, not an independent confirmation.'}
    checks = {name + '_aff_gain': changes[name]['transient_aff_f1'] >= .01
              for name in ('vector_nearest', 'jcv_text_pipeline')}
    checks['actual_new_tp'] = paired['vector_nearest']['tp_improved_stations'] > 0
    checks['two_seeds_positive'] = sum(
        strata['cohort_seed'][MAIN][seed]['range']['transient']['affiliation']['station_macro_f1']
        > strata['cohort_seed']['vector_nearest'][seed]['range']['transient']['affiliation']['station_macro_f1']
        for seed in strata['cohort_seed'][MAIN]) >= 2
    research = all(checks.values())
    stability = {'daily_loss_within_001': changes['vector_nearest']['transient_daily_f1'] >= -.01,
                 'four_type_loss_within_0005': changes['vector_nearest']['four_type_daily_f1'] >= -.005}
    return {'admitted': research, 'research_variant_admitted': research,
        'stable_system_admitted': research and all(stability.values()),
        'research_checks': checks, 'stability_checks': stability, 'deltas': changes,
        'scope': 'Preregistered range research and stable-system decisions are separate.'}


def interval_coordinates(prediction):
    return [(event['start'], event['end']) for event in prediction['events']
            if event['kind'] == 'transient']


def report(split, phase):
    visual = ROOT / (split + '-visual-v1')
    phases = ['pilot'] if phase == 'pilot' else ['pilot', 'remaining']
    jobs = []
    for stage in phases:
        if not (visual / (stage + '-joint-materialized.json')).exists():
            raise ValueError('finish every registered station before evaluation')
        for step in ('scan', 'local', 'joint'):
            registration = read_json(visual / f'{stage}-{step}-registration.json')
            jobs.extend(registration['jobs'])
    records = records_for(split, 'pilot' if phase == 'pilot' else 'full')
    methods = sorted(set(METHODS) | {path.name for path in (visual / 'final').iterdir()})
    rows = {method: [] for method in methods}
    process = {family: Counter() for family in ('image', 'text')}
    contributions = {method: Counter() for method in MODES}
    station_details = []
    for index, record in enumerate(records, 1):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        truth = read_json(DATASET / record['truth'])
        observed = np.isfinite(observed_values(observation)).all(axis=1)
        target = event_masks(Prediction.model_validate({'events': truth['events']}), observed)['transient']
        for method in methods:
            parent = ROOT / (split + '-baseline-v1/predictions') if method in METHODS else visual / 'final'
            prediction = read_json(parent / method / station / 'result.json')
            rows[method].append({'id': station, 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': prediction.get('prediction'),
                **score_record(observation, truth, prediction)})
        base = rows['vector_nearest'][-1]
        base_intervals = interval_coordinates(base['events'])
        for family, totals in process.items():
            scan = read_json(visual / 'predictions' / ('scan_' + family) / station / 'result.json')
            regions = scan.get('decision', {}).get('regions', [])
            contexts = merged_contexts(regions, observation)
            selected = np.zeros(len(observed), dtype=bool)
            for region in regions:
                selected[region['start']:region['end'] + 1] = True
            totals.update(stations=1, positive_stations=int(target.any()), regions=len(regions),
                merged_contexts=len(contexts), failed_scans=int(scan['status'] != 'success'),
                positive_region_hits_truth=int(target.any() and (selected & target).any()))
        for method in MODES:
            prediction = rows[method][-1]
            pure = rows[method + '_pure'][-1]
            work = visual / 'predictions' / method / station / 'result.json'
            receipt = read_json(work) if work.exists() else {}
            intervals = receipt.get('decision', {}).get('intervals', [])
            confirmed = [interval for interval in intervals if interval['status'] == 'confirmed']
            added = [interval for interval in interval_coordinates(prediction['events'])
                     if interval not in base_intervals]
            actual = prediction['range']['transient']['daily']
            numerical = base['range']['transient']['daily']
            tp_delta, fp_delta = actual['tp'] - numerical['tp'], actual['fp'] - numerical['fp']
            negative = numerical['tp'] + numerical['fn'] == 0
            contributions[method].update(stations=1, confirmed_proposals=len(confirmed),
                uncertain_proposals=len(intervals) - len(confirmed), added_final_intervals=len(added),
                recovered_tp_days=tp_delta, added_fp_days=fp_delta,
                tp_improved_stations=int(tp_delta > 0), fp_increased_stations=int(fp_delta > 0),
                negative_stations=int(negative), negative_added_fp_days=fp_delta if negative else 0,
                negative_stations_with_added_fp=int(negative and fp_delta > 0),
                pure_tp_days=pure['range']['transient']['daily']['tp'],
                pure_fp_days=pure['range']['transient']['daily']['fp'],
                confirmed_overlapping_numerical=sum(any(old_left <= interval['end']
                    and interval['start'] <= old_right for old_left, old_right in base_intervals)
                    for interval in confirmed))
            station_details.append({'method': method, 'station': station,
                'scenario': truth['scenario'], 'confirmed': confirmed,
                'added_final_intervals': added, 'tp_delta': tp_delta, 'fp_delta': fp_delta})
        if index % 12 == 0:
            print('Scored', index, '/', len(records), flush=True)
    summaries = {method: aggregate_rows(values) for method, values in rows.items()}
    comparison = [compact(method, summary) for method, summary in summaries.items()]
    table = {row['method']: row for row in comparison}
    strata = {key: {method: {str(value): aggregate_rows([row for row in values if row[key] == value])
        for value in sorted({row[key] for row in values})} for method, values in rows.items()}
        for key in ('cohort_seed', 'scenario')}
    paired = {}
    for control in CONTROLS:
        changes = []
        for main, other in zip(rows[MAIN], rows[control], strict=True):
            a, b = main['range']['transient'], other['range']['transient']
            aff_a, aff_b = a['affiliation']['f1'], b['affiliation']['f1']
            delta_aff = aff_a - aff_b if aff_a is not None and aff_b is not None else None
            tp_delta = a['daily']['tp'] - b['daily']['tp']
            fp_delta = a['daily']['fp'] - b['daily']['fp']
            if tp_delta or fp_delta or (delta_aff is not None and abs(delta_aff) > 1e-12):
                changes.append({'station': main['id'], 'scenario': main['scenario'],
                    'cohort_seed': main['cohort_seed'], 'tp_delta': tp_delta, 'fp_delta': fp_delta,
                    'aff_delta': delta_aff, 'main_events': main['events'], 'control_events': other['events']})
        paired[control] = {'changes': changes,
            'tp_improved_stations': sum(change['tp_delta'] > 0 for change in changes),
            'fp_increased_stations': sum(change['fp_delta'] > 0 for change in changes),
            'aff_only_improved_stations': sum(change['aff_delta'] is not None and change['aff_delta'] > 0
                and change['tp_delta'] <= 0 for change in changes),
            'bootstrap': bootstrap(rows[MAIN], rows[control])}
    receipts = request_receipts([job['work'] for job in jobs])
    costs = aggregate_cost(receipts)
    input_images = Counter()
    for job in jobs:
        request = read_json(Path(job['work']) / 'request.json')
        for message in request['messages']:
            if isinstance(message['content'], list):
                input_images[job['mode']] += sum(block.get('type') == 'image_url'
                    for block in message['content'])
    costs_by_mode = {}
    for mode in sorted({job['mode'] for job in jobs}):
        selected = {path: receipt for path, receipt in receipts.items()
                    if Path(path).relative_to(visual / 'predictions').parts[0] == mode}
        mode_cost = aggregate_cost(selected)
        mode_cost['requested_input_images'] = input_images[mode]
        mode_cost['receipted_image_tokens'] = sum((receipt['usage'] or {})
            .get('prompt_tokens_details', {}).get('image_tokens', 0) for receipt in selected.values())
        costs_by_mode[mode] = mode_cost
    output = Path(f'results/exp31-{split}-{phase}-v1')
    output.mkdir(parents=True, exist_ok=False)
    decision = admission(table, paired, strata, phase)
    write_json(output / 'summary.json', {'experiment_id': 'GNSS-VLM-EXP31', 'split': split,
        'phase': phase, 'stations': len(records), 'methods': summaries, 'costs': costs})
    write_json(output / 'admission.json', decision)
    write_json(output / 'strata.json', strata)
    write_json(output / 'process.json', process)
    write_json(output / 'contribution-audit.json', {'methods': contributions, 'details': station_details})
    write_json(output / 'paired-changes.json', paired)
    write_json(output / 'request-receipts.json', receipts)
    write_json(output / 'costs-by-mode.json', costs_by_mode)
    with gzip.open(output / 'per-station.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(rows, stream, ensure_ascii=False, allow_nan=False)
    write_table(output, 'comparison.csv', comparison)
    range_rows = []
    for method, summary in summaries.items():
        if summary['range'] is None:
            continue
        for kind, metrics in summary['range'].items():
            if isinstance(metrics, dict) and 'affiliation' in metrics:
                range_rows.append({'method': method, 'kind': kind, **metrics['affiliation'],
                    **{'daily_' + key: value for key, value in metrics['daily_micro'].items()}})
    write_table(output, 'range-metrics.csv', range_rows)
    lines = ['# EXP31：全局—局部联合视觉复核（JCV-GNSS）', '',
        f'{len(records)} stations, {split}, {phase}.', '',
        '| Method | Transient Aff-F1 | Daily F1 | Four-type F1 | TP | FP | FN |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in comparison:
        keys = ('transient_aff_f1', 'transient_daily_f1', 'four_type_daily_f1',
                'transient_tp', 'transient_fp', 'transient_fn')
        lines.append('| ' + row['method'] + ' | ' + ' | '.join(
            '—' if row[key] is None else f'{row[key]:.4f}' for key in keys) + ' |')
    lines.extend(['', 'Preregistered decision: ' + str(decision['admitted']), '',
        'Pure methods output transient only; their four-type F1 is not a full-system comparison.',
        'Affiliation averages truth-positive stations; read negative-station FP and daily scores alongside it.'])
    (output / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    chosen = [MAIN, *CONTROLS]
    figure, panels = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    for panel, key, title in zip(panels, ('transient_aff_f1', 'transient_daily_f1', 'four_type_daily_f1'),
            ('Transient Affiliation F1', 'Transient daily F1', 'Four-type daily F1'), strict=True):
        panel.barh(chosen, [table[method][key] for method in chosen], color='#357a91')
        panel.set_title(title)
        panel.set_xlim(0, 1)
        panel.grid(axis='x', alpha=.2)
    panels[0].invert_yaxis()
    figure.tight_layout()
    figure.savefig(output / 'ablation.png', dpi=150)
    plt.close(figure)
    print(json.dumps({'admission': decision, 'costs': costs, 'contributions': contributions}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--phase', choices=('pilot', 'full'), default='pilot')
    args = parser.parse_args()
    report(args.split, args.phase)
