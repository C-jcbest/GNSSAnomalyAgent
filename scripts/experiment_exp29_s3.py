"""Confirm the unchanged exploratory visual-range complement on untouched stations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from exp14_fingerprint import apply, free_graph
from exp15_vector import nearest, vectors
from exp29_data_baseline import DATASET, ROOT
from exp29_joint_boundaries import joint_ranges
from experiment_exp26 import finite_rows
from experiment_exp29 import (
    PROMPT,
    RangeDecision,
    compose,
    image_block,
    infer,
    payload,
    phase_records,
    plot_values,
    refine_ranges,
    render_scan,
    transient_event,
    verify_frozen,
)

from gnss_research.files import read_json, sha256, source_hashes, write_json

OUT = ROOT / 'test-visual-v1'
PROTOCOL = Path('docs/experiment-exp29-s3.md')


def prepare(phase):
    baseline = ROOT / 'test-baseline-v1'
    if not (baseline / 'execution-complete.json').exists():
        raise ValueError('complete frozen test numerical baseline')
    if phase == 'pilot':
        OUT.mkdir(parents=True, exist_ok=False)
    else:
        if not (OUT / 'pilot-completed.json').exists():
            raise ValueError('complete test pilot first')
        verify_frozen(read_json(OUT / 'pilot-registration.json'))
    target = OUT / (phase + '-registration.json')
    if target.exists():
        raise ValueError('already registered')
    records = phase_records('test', phase)
    jobs = []
    for record in records:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        inputs = OUT / 'inputs' / station
        inputs.mkdir(parents=True)
        values = plot_values(observation)
        render_scan(observation, values, inputs / 'scan.png')
        first = observation['background_end'] + 1
        metadata = {'monitoring_start': first, 'last_index': len(values) - 1,
                    'historical_reference': [first - 45, first - 1]}
        for family in ('image', 'text'):
            mode = 'vf_vl_' + family
            work = OUT / 'predictions' / mode / station
            work.mkdir(parents=True)
            content = [{'type': 'text', 'text': json.dumps(metadata)}]
            if family == 'image':
                content.append(image_block(inputs / 'scan.png'))
            else:
                content.append({'type': 'text', 'text': json.dumps({'first_index': first - 45,
                    'original_N_E_U': finite_rows(values[first - 45:])}, allow_nan=False)})
            write_json(work / 'request.json', payload('qwen3-vl-plus', PROMPT, RangeDecision, content))
            jobs.append({'station': station, 'mode': mode, 'model': 'qwen3-vl-plus',
                         'kind': 'range', 'request_sha256': sha256(work / 'request.json')})
    dependencies = [Path(__file__), PROTOCOL, Path('scripts/experiment_exp29.py'),
                    Path('scripts/exp29_joint_boundaries.py'), Path('docs/experiment-exp29.md')]
    write_json(target, {'experiment_id': 'GNSS-VLM-EXP29-S3', 'split': 'test', 'phase': phase,
        'records': records, 'jobs': jobs, 'dataset': str(DATASET), 'baseline': str(baseline),
        'manifest_sha256': sha256(DATASET / 'manifest.json'), 'source_hashes': source_hashes(),
        'frozen_files': {str(p): sha256(p) for p in dependencies},
        'selection': 'Development-selected, unchanged, preregistered secondary confirmation'})
    print('Prepared S3', phase, len(records), 'stations', len(jobs), 'requests', flush=True)


def materialize(phase):
    registration = read_json(OUT / (phase + '-registration.json'))
    verify_frozen(registration)
    if not (OUT / (phase + '-completed.json')).exists():
        raise ValueError('complete test inference first')
    baseline = ROOT / 'test-baseline-v1'
    for record in registration['records']:
        station = record['station_id']
        observation = read_json(DATASET / record['observation'])
        numeric = read_json(baseline / 'predictions/segment_numeric' / station / 'result.json')
        graph = free_graph(numeric)
        strong = apply(observation, numeric, graph, nearest(graph, vectors(observation, graph)))
        write_json(OUT / 'final/vector_nearest' / station / 'result.json', strong)
        for family in ('image', 'text'):
            mode = 'vf_vl_' + family
            work = OUT / 'predictions' / mode / station
            receipt = read_json(work / 'result.json')
            decision = receipt.get('range_decision', {'intervals': [], 'evidence': 'No validated proposals'})
            raw = [transient_event(v['start'], v['end'], decision['evidence']) for v in decision['intervals']]
            _, hard, hard_receipts = refine_ranges(observation, numeric, decision)
            joint, joint_receipts = joint_ranges(observation, numeric, decision)
            write_json(work / 'tool-evidence.json', {'hard': hard_receipts, 'joint': joint_receipts})
            variants = {mode + '_raw_complement': compose(observation, strong, raw),
                mode + '_pure': compose(observation, strong, raw, pure=True),
                mode + '_complement': compose(observation, strong, hard),
                'joint_vl_' + family: compose(observation, strong, joint)}
            for name, result in variants.items():
                result['visual_dependency_status'] = receipt['status']
                result['evaluation_dependency_workdirs'] = [str(work)]
                if name.endswith('_pure') and receipt['status'] != 'success':
                    result['status'] = 'failed'
                write_json(OUT / 'final' / name / station / 'result.json', result)
    write_json(OUT / (phase + '-materialized.json'), {'stations': len(registration['records'])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'infer', 'materialize'))
    parser.add_argument('--phase', choices=('pilot', 'remaining'), default='pilot')
    args = parser.parse_args()
    if args.stage == 'infer':
        infer('test', args.phase)
    else:
        {'prepare': prepare, 'materialize': materialize}[args.stage](args.phase)
