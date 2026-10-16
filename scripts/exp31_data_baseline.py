"""Untouched EXP30 development stations with a separately registered JCV experiment."""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import exp28_data_baseline as numerical
from exp30_data_baseline import DATASET

from gnss_research.experiment import verify_dataset
from gnss_research.files import read_json, sha256, source_hashes, write_json

ROOT = Path('D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp31')
PROTOCOL = Path('docs/experiment-exp31.md')


def records_for(split, phase):
    records = [record for record in read_json(DATASET / 'manifest.json')['records']
               if record['split'] == split]
    if split == 'validation':
        records = [record for record in records if int(record['station_id'][1:]) % 120 >= 24]
    if phase == 'full':
        return records
    first = 24 if split == 'validation' else 0
    return [record for record in records
            if (first <= int(record['station_id'][1:]) % 120 < first + 12) == (phase == 'pilot')]


def initialize(model_path):
    numerical.DATASET = DATASET
    numerical.initialize_worker(model_path)


def predict(split, phase):
    if split == 'test':
        decision = read_json(Path('results/exp31-validation-full-v1/admission.json'))
        if not decision['research_variant_admitted']:
            raise ValueError('development research gate failed; test remains unused')
    if split == 'validation' and phase == 'remaining':
        decision = read_json(Path('results/exp31-validation-pilot-v1/admission.json'))
        if not decision['admitted']:
            raise ValueError('pilot gate failed; do not expand this method')
    verify_dataset(DATASET)
    training = read_json(numerical.TRAINING / 'training.json')
    if training['source_hashes'] != source_hashes():
        raise ValueError('frozen production sources changed')
    if training['model_sha256'] != sha256(numerical.TRAINING / 'models.joblib'):
        raise ValueError('frozen numerical fit changed')
    output = ROOT / (split + '-baseline-v1')
    if phase == 'remaining' and not (output / 'pilot-complete.json').exists():
        raise ValueError('complete pilot first')
    records = records_for(split, phase)
    write_json(output / (phase + '-registration.json'), {'experiment_id': 'GNSS-VLM-EXP31',
        'records': records, 'methods': numerical.METHODS, 'model_sha256': training['model_sha256'],
        'manifest_sha256': sha256(DATASET / 'manifest.json'), 'source_hashes': source_hashes(),
        'protocol_sha256': sha256(PROTOCOL)})
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    with ProcessPoolExecutor(max_workers=2, initializer=initialize,
            initargs=(numerical.TRAINING / 'models.joblib',)) as executor:
        for index, station in enumerate(executor.map(numerical.predict_station,
                [(record, output) for record in records]), 1):
            if index % 12 == 0:
                print(split, phase, index, '/', len(records), station, flush=True)
    write_json(output / (phase + '-complete.json'), {'stations': len(records)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--phase', choices=('pilot', 'remaining'), default='pilot')
    args = parser.parse_args()
    predict(args.split, args.phase)
