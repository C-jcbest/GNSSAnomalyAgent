"""Per-type, paired and ablation evaluation for the registered EXP36 experiment."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from exp28_data_baseline import METHODS
from experiment_exp35 import aggregate_types
from experiment_exp36 import DATASET, ROOT, records_for
from report_exp25 import aggregate_cost, request_receipts, score_record
from report_per_type import export, per_type_rows

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import counts_metrics

PRIMARY = 'cev_image_verified'


def aggregate_method(rows):
    if rows[0]['original'].get('types') is not None:
        return aggregate_types(rows)
    generic = [row['original']['generic_daily'] for row in rows]
    return {'original': {'types': None, 'generic_daily_micro': counts_metrics(
        *(sum(row[key] for row in generic) for key in ('tp', 'fp', 'fn')))}, 'range': None}


def compare(rows, main, control):
    changes = []
    for current, previous in zip(rows[main], rows[control], strict=True):
        if current['id'] != previous['id']:
            raise ValueError('paired station order differs')
        a, b = (row['range']['transient'] for row in (current, previous))
        af, bf = (item['affiliation']['f1'] for item in (a, b))
        changes.append({'station': current['id'], 'scenario': current['scenario'],
            'cohort_seed': current['cohort_seed'],
            'aff_delta': af - bf if af is not None else None,
            'tp_delta': a['daily']['tp'] - b['daily']['tp'],
            'fp_delta': a['daily']['fp'] - b['daily']['fp']})
    delta = np.asarray([change['aff_delta'] for change in changes
        if change['aff_delta'] is not None])
    rng = np.random.default_rng(2026101736)
    draws = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(axis=1)
    return {'changes': changes, 'aff_delta': float(delta.mean()),
        'tp_delta': sum(change['tp_delta'] for change in changes),
        'fp_delta': sum(change['fp_delta'] for change in changes),
        'tp_improved_stations': sum(change['tp_delta'] > 0 for change in changes),
        'tp_harmed_stations': sum(change['tp_delta'] < 0 for change in changes),
        'positive_station_paired_bootstrap_ci95': np.quantile(draws, [.025, .975]).tolist(),
        'bootstrap_repeats': 2000, 'scope': 'descriptive positive-station bootstrap, not a significance gate'}


def admission(summary, strata):
    methods = summary['methods']

    def aff(method):
        return methods[method]['range']['transient']['affiliation']['station_macro_f1']

    def count(method, key):
        return methods[method]['original']['types']['transient']['daily_micro'][key]

    def seed_deltas(main, control):
        return [strata['cohort_seed'][main][seed]['range']['transient']['affiliation']['station_macro_f1']
            - strata['cohort_seed'][control][seed]['range']['transient']['affiliation']['station_macro_f1']
            for seed in strata['cohort_seed'][main]]

    image, text = 'cev_image_verified_pure', 'cev_text_verified_pure'
    mechanism = {'pure_aff_gain_at_least_002': aff(image) - aff(text) >= .02,
        'at_least_two_seed_directions_nonnegative': sum(value >= 0 for value in seed_deltas(image, text)) >= 2}
    alternative = {'aff_gain_vs_nearest_at_least_001': aff(PRIMARY) - aff('vector_nearest') >= .01,
        'aff_preserved_vs_old_with_20pct_fp_reduction': (
            aff(PRIMARY) - aff('gnv_image') >= -.01 and count('gnv_image', 'fp') > 0
            and count(PRIMARY, 'fp') <= .8 * count('gnv_image', 'fp'))}
    system = {'aff_gain_vs_nearest_at_least_002': aff(PRIMARY) - aff('vector_nearest') >= .02,
        'at_least_two_seeds_positive': sum(value > 0 for value in seed_deltas(PRIMARY, 'vector_nearest')) >= 2,
        'aff_gain_vs_text_system_at_least_001': aff(PRIMARY) - aff('cev_text_verified') >= .01,
        'actual_tp_gain': count(PRIMARY, 'tp') > count('vector_nearest', 'tp')}
    return {'expansion_admitted': all(mechanism.values()) and any(alternative.values()),
        'mechanism_checks': mechanism, 'system_expansion_alternatives': alternative,
        'full_development_system_admitted': all(system.values()), 'system_checks': system,
        'scope': 'registered development criteria; independent test not executed'}


def report(phase):
    parts = ('pilot', 'remaining') if phase == 'full' else (phase,)
    visual = ROOT / 'validation-visual-v1'
    jobs = []
    for part in parts:
        if not (visual / f'{part}-materialized.json').exists():
            raise ValueError('materialize all planned stations first')
        for stage in ('direct', 'scan', 'joint'):
            jobs.extend(read_json(visual / f'{part}-{stage}-registration.json')['jobs'])
    methods = [*METHODS, 'vector_nearest', *sorted(path.name for path in (visual / 'final').iterdir())]
    rows = {method: [] for method in methods}
    for record in records_for(phase):
        observation = read_json(DATASET / record['observation'])
        # Predictions are already frozen; target truth is used only for reporting.
        truth = read_json(DATASET / record['truth'])
        for method in methods:
            folder = ROOT / 'validation-baseline-v1/predictions' if method in (*METHODS, 'vector_nearest') else visual / 'final'
            result = read_json(folder / method / record['station_id'] / 'result.json')
            rows[method].append({'id': record['station_id'], 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': result['prediction'],
                **score_record(observation, truth, result)})
    summaries = {method: aggregate_method(values) for method, values in rows.items()}
    strata = {key: {method: {str(value): aggregate_method([row for row in values if row[key] == value])
        for value in sorted({row[key] for row in values})} for method, values in rows.items()}
        for key in ('cohort_seed', 'scenario')}
    paired = {control: compare(rows, PRIMARY, control) for control in (
        'vector_nearest', 'gnv_image', 'numeric_all_pairs', 'cev_text_verified',
        'cev_no_examples_verified', 'cev_image_raw')}
    paired['pure_image_vs_text'] = compare(rows, 'cev_image_verified_pure', 'cev_text_verified_pure')
    receipts = request_receipts([job['work'] for job in jobs])
    summary = {'experiment_id': 'GNSS-VLM-EXP36', 'phase': phase, 'stations': len(records_for(phase)),
        'methods': summaries, 'costs': aggregate_cost(receipts), 'scope': 'unused development stations; not test',
        'deformation_owner': 'frozen numerical vector_nearest, no new visual deformation output'}
    output = Path(f'results/exp36-{phase}-v1')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', summary)
    write_json(output / 'strata.json', strata)
    write_json(output / 'paired-changes.json', paired)
    write_json(output / 'admission.json', admission(summary, strata))
    write_json(output / 'request-receipts.json', receipts)
    write_json(output / 'provenance.json', {'script_sha256': sha256(Path(__file__)),
        'manifest_sha256': sha256(DATASET / 'manifest.json'),
        'protocol_sha256': sha256(Path('docs/experiment-exp36.md')),
        'registrations': {f'{part}-{stage}': sha256(visual / f'{part}-{stage}-registration.json')
            for part in parts for stage in ('direct', 'scan', 'joint')},
        'test_model_requests': 0, 'new_numerical_fit': False})
    with (output / 'per-station.json.gz').open('xb') as stream:
        stream.write(gzip.compress(json.dumps(rows, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    export(output / 'summary.json', output / 'per-type')
    print(json.dumps({'admission': admission(summary, strata), 'costs': summary['costs'],
        'transient': [row for row in per_type_rows(summary)
            if row['kind'] == 'transient' and (not row['method'].endswith('_pure') or row['method'] in
                ('cev_image_verified_pure', 'cev_text_verified_pure'))]}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('pilot', 'remaining', 'full'), default='pilot')
    report(parser.parse_args().phase)
