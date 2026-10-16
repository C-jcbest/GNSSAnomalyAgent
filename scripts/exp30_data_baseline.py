"""New balanced cohorts and staged predictions using the unchanged EXP28 numerical fit."""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import exp28_data_baseline as numerical
import numpy as np

from gnss_research.data import SCENARIOS, simulate
from gnss_research.experiment import verify_dataset
from gnss_research.files import read_json, sha256, source_hashes, write_json

ROOT = Path('D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp30')
DATASET = ROOT / 'data'
SEEDS = (2026101601, 2026101602, 2026101603)
PROTOCOL = Path('docs/experiment-exp30.md')


def records_for(split, phase):
    records = [r for r in read_json(DATASET / 'manifest.json')['records'] if r['split'] == split]
    if phase == 'full':
        return records
    return [r for r in records if (int(r['station_id'][1:]) % 120 < 24) == (phase == 'pilot')]


def generate():
    DATASET.mkdir(parents=True, exist_ok=False)
    case_seeds = set()
    old_datasets = (Path('data/exp28-expanded-1528-1530'), ROOT.parent / 'exp29/data')
    for dataset in old_datasets:
        for record in read_json(dataset / 'manifest.json')['records']:
            case_seeds.add(read_json(dataset / record['truth'])['seed'])
    records = []
    for group, seed in enumerate(SEEDS):
        streams = np.random.SeedSequence(seed).spawn(240)
        for split_index, split in enumerate(('validation', 'test')):
            for index in range(120):
                station = f'{"v" if split == "validation" else "e"}{group * 120 + index:04d}'
                case_seed = int(streams[split_index * 120 + index].generate_state(1)[0])
                if case_seed in case_seeds:
                    raise ValueError('mechanism seed overlaps a previous or current station')
                case_seeds.add(case_seed)
                observation, truth = simulate(case_seed, station, SCENARIOS[index % 6], 730, 365)
                observation_path = Path('observations') / (station + '.json')
                truth_path = Path('truth') / (station + '.json')
                write_json(DATASET / observation_path, observation)
                write_json(DATASET / truth_path, truth)
                records.append({'station_id': station, 'split': split, 'cohort_seed': seed,
                    'observation': str(observation_path), 'truth': str(truth_path),
                    'observation_sha256': sha256(DATASET / observation_path),
                    'truth_sha256': sha256(DATASET / truth_path)})
    write_json(DATASET / 'manifest.json', {'schema': 'gnss-research-dataset-v1',
        'seed': SEEDS[0], 'cohort_seeds': list(SEEDS), 'days': 730, 'background_days': 365,
        'split_counts': {'validation': 360, 'test': 360}, 'records': records,
        'origin': 'unchanged mechanism simulation, not field measurements',
        'sources': source_hashes(), 'script_sha256': sha256(Path(__file__))})
    verify_dataset(DATASET)
    print('Generated 720 independent stations', flush=True)


def initialize(model_path):
    numerical.DATASET = DATASET
    numerical.initialize_worker(model_path)


def predict(split, phase):
    if split == 'test' and not read_json(Path('results/exp30-validation-full-v1/admission.json'))['admitted']:
        raise ValueError('development gate failed; test remains sealed')
    training = read_json(numerical.TRAINING / 'training.json')
    if training['source_hashes'] != source_hashes() or training['model_sha256'] != sha256(numerical.TRAINING / 'models.joblib'):
        raise ValueError('frozen training/source changed')
    verify_dataset(DATASET)
    output = ROOT / (split + '-baseline-v1')
    registration = output / (phase + '-registration.json')
    if registration.exists():
        raise ValueError('phase already registered')
    if phase == 'remaining' and not (output / 'pilot-complete.json').exists():
        raise ValueError('complete pilot first')
    records = records_for(split, phase)
    write_json(registration, {'records': records, 'methods': numerical.METHODS,
        'model_sha256': training['model_sha256'], 'manifest_sha256': sha256(DATASET / 'manifest.json'),
        'source_hashes': source_hashes(), 'protocol_sha256': sha256(PROTOCOL)})
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    with ProcessPoolExecutor(max_workers=2, initializer=initialize,
            initargs=(numerical.TRAINING / 'models.joblib',)) as executor:
        jobs = [(r, output) for r in records]
        for index, station in enumerate(executor.map(numerical.predict_station, jobs), 1):
            if index % 12 == 0:
                print(split, phase, index, '/', len(records), station, flush=True)
    write_json(output / (phase + '-complete.json'), {'stations': len(records)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('generate', 'predict'))
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--phase', choices=('pilot', 'remaining'), default='pilot')
    args = parser.parse_args()
    if args.stage == 'generate':
        generate()
    else:
        predict(args.split, args.phase)
