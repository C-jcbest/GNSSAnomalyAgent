"""Independent EXP29 cohorts with the unchanged EXP28 fit and numerical paths."""
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

ROOT = Path('D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp29')
DATASET = ROOT / 'data'
SEEDS = (2026101532, 2026101533, 2026101534)
PROTOCOL = Path('docs/experiment-exp29.md')


def generate():
    DATASET.mkdir(parents=True, exist_ok=False)
    records = []
    old_seeds = {read_json(Path('data/exp28-expanded-1528-1530') / r['truth'])['seed']
                 for r in read_json(Path('data/exp28-expanded-1528-1530/manifest.json'))['records']}
    case_seeds = set(old_seeds)
    for group, seed in enumerate(SEEDS):
        streams = np.random.SeedSequence(seed).spawn(240)
        for split_index, split in enumerate(('validation', 'test')):
            for index in range(120):
                prefix = 'v' if split == 'validation' else 'e'
                station = f'{prefix}{group * 120 + index:04d}'
                case_seed = int(streams[split_index * 120 + index].generate_state(1)[0])
                if case_seed in case_seeds:
                    raise ValueError('seed overlap with previous or current cohort')
                case_seeds.add(case_seed)
                observation, truth = simulate(case_seed, station, SCENARIOS[index % 6], 730, 365)
                observed_path = Path('observations') / f'{station}.json'
                truth_path = Path('truth') / f'{station}.json'
                write_json(DATASET / observed_path, observation)
                write_json(DATASET / truth_path, truth)
                records.append({'station_id': station, 'split': split, 'cohort_seed': seed,
                    'observation': str(observed_path), 'truth': str(truth_path),
                    'observation_sha256': sha256(DATASET / observed_path),
                    'truth_sha256': sha256(DATASET / truth_path)})
    write_json(DATASET / 'manifest.json', {'schema': 'gnss-research-dataset-v1',
        'seed': SEEDS[0], 'days': 730, 'background_days': 365, 'records': records,
        'origin': 'mechanism-inspired simulation, not field measurements',
        'cohort_seeds': list(SEEDS), 'split_counts': {'validation': 360, 'test': 360},
        'sources': source_hashes(), 'script_sha256': sha256(Path(__file__))})
    verify_dataset(DATASET)
    print('Generated 720 stations; no overlap with EXP28 case seeds', flush=True)


def initialize(model_path):
    numerical.DATASET = DATASET
    numerical.initialize_worker(model_path)


def predict(split):
    training = numerical.TRAINING
    receipt = read_json(training / 'training.json')
    if receipt['source_hashes'] != source_hashes() or receipt['model_sha256'] != sha256(training / 'models.joblib'):
        raise ValueError('frozen fit changed')
    manifest = verify_dataset(DATASET)
    records = [r for r in manifest['records'] if r['split'] == split]
    output = ROOT / f'{split}-baseline-v1'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'registration.json', {'experiment_id': 'GNSS-VLM-EXP29',
        'dataset': str(DATASET), 'split': split, 'methods': numerical.METHODS,
        'manifest_sha256': sha256(DATASET / 'manifest.json'), 'records': records,
        'source_hashes': source_hashes(), 'shared_model_sha256': receipt['model_sha256'],
        'protocol_sha256': sha256(PROTOCOL), 'script_sha256': sha256(Path(__file__))})
    write_json(output / 'execution-started.json', {'retries': 0})
    write_json(output / 'training.json', receipt)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    with ProcessPoolExecutor(max_workers=2, initializer=initialize,
                             initargs=(training / 'models.joblib',)) as executor:
        for number, station in enumerate(executor.map(numerical.predict_station,
                [(r, output) for r in records]), 1):
            if number % 30 == 0:
                print(split, number, '/', len(records), station, flush=True)
    write_json(output / 'execution-complete.json', {'stations': len(records), 'requests': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('generate', 'predict'))
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    args = parser.parse_args()
    if args.stage == 'generate':
        generate()
    else:
        predict(args.split)
