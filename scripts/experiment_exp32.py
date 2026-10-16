"""Confirm frozen joint-context visual readouts on untouched development stations."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import exp28_data_baseline as numerical
import experiment_exp30 as air
import experiment_exp31 as joint
from exp14_fingerprint import apply, free_graph
from exp15_vector import nearest, vectors
from exp30_data_baseline import DATASET
from experiment_exp26 import finite_rows
from experiment_exp29 import compose, image_block, plot_values, render_scan, transient_event
from experiment_exp30_s1 import complete_windows

from gnss_research.experiment import verify_dataset
from gnss_research.files import read_json, sha256, source_hashes, write_json

ROOT = Path('D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp32')
PROTOCOL = Path('docs/experiment-exp32.md')
DEPENDENCIES = [Path(__file__), PROTOCOL, Path('scripts/experiment_exp31.py'),
    Path('scripts/experiment_exp30.py'), Path('scripts/experiment_exp30_s1.py'),
    Path('scripts/experiment_exp29.py'), Path('scripts/experiment_exp26.py'),
    Path('scripts/exp28_data_baseline.py'), Path('scripts/exp14_fingerprint.py'),
    Path('scripts/exp15_vector.py')]


def records_for(phase):
    records = [record for record in read_json(DATASET / 'manifest.json')['records']
        if record['split'] == 'validation' and int(record['station_id'][1:]) % 120 >= 36]
    if phase == 'full':
        return records
    return [record for record in records
        if (int(record['station_id'][1:]) % 120 < 60) == (phase == 'pilot')]


def guard_phase(phase):
    if phase == 'remaining':
        decision = read_json(Path('results/exp32-validation-pilot-v1/admission.json'))
        if not decision['mechanism_confirmed']:
            raise ValueError('mechanism pilot failed; do not expand')


def register(output, phase, stage, records, jobs):
    write_json(output / f'{phase}-{stage}-registration.json', {
        'experiment_id': 'GNSS-VLM-EXP32', 'records': records, 'jobs': jobs,
        'source_hashes': source_hashes(), 'manifest_sha256': sha256(DATASET / 'manifest.json'),
        'frozen_files': {str(path): sha256(path) for path in DEPENDENCIES}})


def configure():
    air.ROOT = ROOT
    air.DATASET = DATASET
    air.register = register
    joint.ROOT = ROOT
    joint.DATASET = DATASET
    joint.records_for = lambda split, phase: records_for(phase)


def initialize_worker(model):
    numerical.DATASET = DATASET
    numerical.initialize_worker(model)


def predict_station(job):
    station = numerical.predict_station(job)
    record, output = job
    observation = read_json(DATASET / record['observation'])
    numeric = read_json(output / 'predictions/segment_numeric' / station / 'result.json')
    graph = free_graph(numeric)
    strong = apply(observation, numeric, graph, nearest(graph, vectors(observation, graph)))
    write_json(output / 'predictions/vector_nearest' / station / 'result.json', strong)
    return station


def baseline(phase):
    guard_phase(phase)
    verify_dataset(DATASET)
    training = read_json(numerical.TRAINING / 'training.json')
    if training['source_hashes'] != source_hashes():
        raise ValueError('frozen production sources changed')
    if training['model_sha256'] != sha256(numerical.TRAINING / 'models.joblib'):
        raise ValueError('frozen numerical fit changed')
    output = ROOT / 'validation-baseline-v1'
    records = records_for(phase)
    register(output, phase, 'baseline', records, [])
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    with ProcessPoolExecutor(max_workers=2, initializer=initialize_worker,
            initargs=(numerical.TRAINING / 'models.joblib',)) as executor:
        for index, station in enumerate(executor.map(predict_station,
                [(record, output) for record in records]), 1):
            if index % 12 == 0:
                print('Baseline', index, '/', len(records), station, flush=True)
    write_json(output / (phase + '-complete.json'), {'stations': len(records),
        'model_sha256': training['model_sha256'], 'new_fit': False})


def prepare_scan(phase):
    guard_phase(phase)
    if not (ROOT / 'validation-baseline-v1' / (phase + '-complete.json')).exists():
        raise ValueError('complete numerical phase first')
    output = ROOT / 'validation-visual-v1'
    jobs = []
    for record in records_for(phase):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        values = plot_values(observation)
        inputs = output / 'inputs' / station
        inputs.mkdir(parents=True, exist_ok=False)
        render_scan(observation, values, inputs / 'global.png')
        windows = complete_windows(observation)
        for index, (left, right) in enumerate(windows):
            air.render_window(values, left, right, inputs / f'window-{index}.png')
        for view in ('image', 'text'):
            info = {'monitoring_start': observation['background_end'] + 1,
                'last_index': len(values) - 1,
                'window_indices_inclusive': [[left, right - 1] for left, right in windows]}
            content = [{'type': 'text', 'text': json.dumps(info)}]
            for index, (left, right) in enumerate(windows):
                if view == 'image':
                    content.append(image_block(inputs / f'window-{index}.png'))
                else:
                    content.append({'type': 'text', 'text': json.dumps({'first_index': left,
                        'original_N_E_U': finite_rows(values[left:right])}, allow_nan=False)})
            jobs.append(air.store_request(output / 'predictions' / ('scan_' + view) / station,
                'scan_' + view, station, 'scan', content, air.SCAN_PROMPT, air.ScanDecision,
                allowed_context=[observation['background_end'] + 1, len(values) - 1]))
    register(output, phase, 'scan', records_for(phase), jobs)
    print('Prepared screening', len(jobs), flush=True)


def materialize(phase):
    output = ROOT / 'validation-visual-v1'
    if not (output / (phase + '-joint-complete.json')).exists():
        raise ValueError('complete joint inference first')
    air.check_registration(read_json(output / (phase + '-joint-registration.json')))
    for record in records_for(phase):
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        strong = read_json(ROOT / 'validation-baseline-v1/predictions/vector_nearest' / station / 'result.json')
        for mode, (family, _, _) in joint.MODES.items():
            scan_work = output / 'predictions' / ('scan_' + family) / station
            scan = read_json(scan_work / 'result.json')
            additions, dependencies = [], [str(scan_work)]
            if scan.get('decision', {}).get('regions'):
                work = output / 'predictions' / mode / station
                receipt = read_json(work / 'result.json')
                dependencies.append(str(work))
                additions = [transient_event(item['start'], item['end'], item['observation'])
                    for item in receipt.get('decision', {}).get('intervals', [])
                    if item['status'] == 'confirmed']
            for suffix, pure in (('', False), ('_pure', True)):
                result = compose(observation, strong, additions, pure=pure)
                result['evaluation_dependency_workdirs'] = dependencies
                write_json(output / 'final' / (mode + suffix) / station / 'result.json', result)
    write_json(output / (phase + '-materialized.json'), {'stations': len(records_for(phase))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('baseline', 'prepare_scan', 'infer_scan',
        'prepare_joint', 'infer_joint', 'materialize'))
    parser.add_argument('--phase', choices=('pilot', 'remaining'), default='pilot')
    args = parser.parse_args()
    configure()
    if args.stage == 'baseline':
        baseline(args.phase)
    elif args.stage == 'prepare_scan':
        prepare_scan(args.phase)
    elif args.stage == 'infer_scan':
        air.execute('validation', args.phase, 'scan')
    elif args.stage == 'prepare_joint':
        joint.prepare_joint('validation', args.phase)
    elif args.stage == 'infer_joint':
        if (ROOT / 'validation-visual-v1' / (args.phase + '-joint-started.json')).exists():
            raise ValueError('joint stage already started; no implicit retry')
        joint.execute_joint('validation', args.phase)
    else:
        materialize(args.phase)


if __name__ == '__main__':
    main()
