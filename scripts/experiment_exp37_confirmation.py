"""Confirm the fixed Flash image/text experiment on the last 54 unused stations."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import experiment_exp30 as air
import experiment_exp32 as numerical
import experiment_exp36 as parent
import experiment_exp37 as diagnostic
from exp28_data_baseline import METHODS
from experiment_exp26 import finite_rows
from experiment_exp29 import image_block, payload, plot_values, transient_event
from experiment_exp30 import LocalDecision
from experiment_exp30_s1 import complete_windows
from report_exp25 import aggregate_cost, request_receipts, score_record
from report_exp36 import aggregate_method, compare
from report_per_type import export

from gnss_research.files import read_json, sha256, source_hashes, write_json

ROOT = Path('artifacts/exp37/confirmation-v1')
MODES = {'owv_flash': 'image', 'owv_flash_text': 'text'}
DEPENDENCIES = [Path(__file__), diagnostic.PROTOCOL, Path('scripts/experiment_exp37.py'),
    *parent.DEPENDENCIES]


def records():
    return parent.records_for('remaining')


def guard_phase(_phase):
    if not read_json(Path('results/exp37-development-diagnostic-v1/admission.json'))['fresh_confirmation_admitted']:
        raise ValueError('development diagnostic failed; do not use new stations')


def register(output, phase, stage, selected, jobs):
    write_json(output / f'{phase}-{stage}-registration.json', {
        'experiment_id': 'GNSS-VLM-EXP37-CONFIRMATION', 'records': selected, 'jobs': jobs,
        'manifest_sha256': sha256(parent.DATASET / 'manifest.json'), 'source_hashes': source_hashes(),
        'frozen_files': {str(path): sha256(path) for path in DEPENDENCIES}})


def baseline():
    numerical.ROOT = ROOT
    numerical.records_for = lambda _phase: records()
    numerical.guard_phase = guard_phase
    numerical.register = register
    numerical.baseline('confirmation')


def prepare():
    if not (ROOT / 'validation-baseline-v1/confirmation-complete.json').exists():
        raise ValueError('complete same-station numerical baselines first')
    output = ROOT / 'validation-visual-v1'
    jobs = []
    for record in records():
        station = record['station_id']
        observation = read_json(parent.DATASET / record['observation'])
        values = plot_values(observation)
        image = output / 'inputs' / (station + '.png')
        image.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.render_sheet(observation, image)
        metadata = {'monitoring_start': observation['background_end'] + 1,
            'last_index': len(values) - 1, 'windows': complete_windows(observation),
            'window_end_convention': 'exclusive in metadata; plotted index labels inclusive'}
        for mode, view in MODES.items():
            content = [{'type': 'text', 'text': json.dumps(metadata)}]
            if view == 'image':
                content.append(image_block(image))
            else:
                for left, right in complete_windows(observation):
                    content.append({'type': 'text', 'text': json.dumps({'first_index': left,
                        'original_N_E_U': finite_rows(values[left:right])}, allow_nan=False)})
            work = output / 'predictions' / mode / station
            write_json(work / 'request.json', payload('qwen3.8-flash', diagnostic.PROMPT, LocalDecision, content))
            jobs.append({'work': str(work), 'mode': mode, 'station': station, 'kind': 'local',
                'request_sha256': sha256(work / 'request.json'), 'allowed_contexts': [[365, 729]]})
    register(output, 'confirmation', 'direct', records(), jobs)
    registration = read_json(output / 'confirmation-direct-registration.json')
    registration.update(scope='54 new development stations, independent of 18-station diagnosis; not test',
        diagnostic_summary_sha256=sha256(Path('results/exp37-development-diagnostic-v1/summary.json')),
        diagnostic_rows_sha256=sha256(Path('results/exp37-development-diagnostic-v1/per-station.json.gz')))
    write_json(ROOT / 'registration.json', registration)
    print('Prepared', len(jobs), 'confirmation requests', flush=True)


def infer():
    # The existing engine records failures without retries; both modes use this same model.
    parent.ROOT = ROOT
    air.MODEL = 'qwen3.8-flash'
    parent.infer('confirmation', 'direct')


def report():
    visual = ROOT / 'validation-visual-v1'
    if not (visual / 'confirmation-direct-complete.json').exists():
        raise ValueError('finish all confirmation requests first')
    registration = read_json(ROOT / 'registration.json')
    methods = [*METHODS, 'vector_nearest', *MODES, *[mode + '_pure' for mode in MODES]]
    rows = {method: [] for method in methods}
    for record in records():
        station = record['station_id']
        observation = read_json(parent.DATASET / record['observation'])
        base = read_json(ROOT / 'validation-baseline-v1/predictions/vector_nearest' / station / 'result.json')
        predictions = {method: read_json(ROOT / 'validation-baseline-v1/predictions' / method / station / 'result.json')
            for method in (*METHODS, 'vector_nearest')}
        for mode in MODES:
            receipt = read_json(visual / 'predictions' / mode / station / 'result.json')
            events = [transient_event(item['start'], item['end'], item['observation'])
                for item in receipt.get('decision', {}).get('intervals', []) if item['status'] == 'confirmed']
            predictions[mode] = parent.fuse(observation, base, events)
            predictions[mode + '_pure'] = parent.result_from_events(observation, events)
        truth = read_json(parent.DATASET / record['truth'])
        for method, result in predictions.items():
            if method in MODES or method.endswith('_pure'):
                write_json(visual / 'final' / method / station / 'result.json', result)
            rows[method].append({'id': station, 'cohort_seed': record['cohort_seed'],
                'scenario': truth['scenario'], 'events': result['prediction'], **score_record(observation, truth, result)})
    summaries = {method: aggregate_method(values) for method, values in rows.items()}
    strata = {key: {method: {str(value): aggregate_method([row for row in values if row[key] == value])
        for value in sorted({row[key] for row in values})} for method, values in rows.items()}
        for key in ('cohort_seed', 'scenario')}
    paired = {control: compare(rows, 'owv_flash', control)
        for control in ('vector_nearest', 'owv_flash_text')}
    paired['pure_image_vs_text'] = compare(rows, 'owv_flash_pure', 'owv_flash_text_pure')
    nearest, text = paired['vector_nearest'], paired['owv_flash_text']
    seeds = strata['cohort_seed']
    seed_deltas = {seed: seeds['owv_flash'][seed]['range']['transient']['affiliation']['station_macro_f1']
        - seeds['vector_nearest'][seed]['range']['transient']['affiliation']['station_macro_f1']
        for seed in seeds['owv_flash']}
    checks = {'aff_vs_nearest_at_least_002': nearest['aff_delta'] >= .02,
        'aff_vs_text_at_least_001': text['aff_delta'] >= .01,
        'at_least_two_seed_directions_positive': sum(delta > 0 for delta in seed_deltas.values()) >= 2,
        'actual_tp_gain': nearest['tp_delta'] > 0}
    for kind in ('spike', 'step', 'deformation'):
        metric = summaries['owv_flash']['range'][kind]['affiliation'] if kind == 'deformation' else summaries['owv_flash']['original']['types'][kind]
        reference = summaries['vector_nearest']['range'][kind]['affiliation'] if kind == 'deformation' else summaries['vector_nearest']['original']['types'][kind]
        checks[kind + '_preserved'] = metric == reference
    output = Path('results/exp37-fresh-confirmation-v1')
    output.mkdir(parents=True, exist_ok=False)
    receipts = request_receipts([job['work'] for job in registration['jobs']])
    summary = {'experiment_id': 'GNSS-VLM-EXP37-CONFIRMATION', 'stations': len(records()),
        'methods': summaries, 'costs': aggregate_cost(receipts), 'scope': registration['scope']}
    write_json(output / 'summary.json', summary)
    write_json(output / 'strata.json', strata)
    write_json(output / 'paired-changes.json', paired)
    write_json(output / 'confirmation.json', {'supported': all(checks.values()), 'checks': checks,
        'seed_aff_deltas_vs_nearest': seed_deltas, 'test_model_requests': 0})
    write_json(output / 'request-receipts.json', receipts)
    with (output / 'per-station.json.gz').open('xb') as stream:
        stream.write(gzip.compress(json.dumps(rows, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
    export(output / 'summary.json', output / 'per-type')
    print(json.dumps({'checks': checks, 'costs': summary['costs'],
        'paired': {key: {k: v for k, v in value.items() if k != 'changes'} for key, value in paired.items()}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('baseline', 'prepare', 'infer', 'report'))
    args = parser.parse_args()
    guard_phase('confirmation')
    {'baseline': baseline, 'prepare': prepare, 'infer': infer, 'report': report}[args.stage]()
