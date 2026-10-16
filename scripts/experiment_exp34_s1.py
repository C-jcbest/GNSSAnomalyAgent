"""Separately registered confirmation of the secondary image-final readout."""
from __future__ import annotations

import argparse
from pathlib import Path

import experiment_exp30 as air
import experiment_exp31 as joint
import experiment_exp32 as workflow
import experiment_exp34 as parent

from gnss_research.files import read_json, sha256, source_hashes, write_json

ROOT = parent.ROOT / 'image-confirmation-s1'
PROTOCOL = Path('docs/experiment-exp34-s1.md')


def records_for(phase):
    records = [record for record in read_json(workflow.DATASET / 'manifest.json')['records']
        if record['split'] == 'validation' and int(record['station_id'][1:]) % 120 >= 84]
    if phase == 'full':
        return records
    return [record for record in records
        if (int(record['station_id'][1:]) % 120 < 96) == (phase == 'pilot')]


def guard_phase(phase):
    if phase == 'remaining':
        if not read_json(Path('results/exp34-s1-validation-pilot-v1/admission.json'))['mechanism_confirmed']:
            raise ValueError('S1 image mechanism gate failed; do not expand')


def register(output, phase, stage, records, jobs):
    dependencies = [*parent.DEPENDENCIES, Path(__file__), PROTOCOL]
    write_json(output / f'{phase}-{stage}-registration.json', {
        'experiment_id': 'GNSS-VLM-EXP34-S1', 'records': records, 'jobs': jobs,
        'source_hashes': source_hashes(), 'manifest_sha256': sha256(workflow.DATASET / 'manifest.json'),
        'frozen_files': {str(path): sha256(path) for path in dependencies}})


def configure():
    parent.configure()
    workflow.ROOT = air.ROOT = joint.ROOT = ROOT
    workflow.records_for = records_for
    workflow.guard_phase = guard_phase
    workflow.register = air.register = register
    joint.records_for = lambda split, phase: records_for(phase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('baseline', 'prepare_scan', 'infer_scan',
        'prepare_joint', 'infer_joint', 'materialize'))
    parser.add_argument('--phase', choices=('pilot', 'remaining'), default='pilot')
    args = parser.parse_args()
    configure()
    guard_phase(args.phase)
    if args.stage == 'baseline':
        workflow.baseline(args.phase)
    elif args.stage == 'prepare_scan':
        workflow.prepare_scan(args.phase)
    elif args.stage == 'infer_scan':
        air.execute('validation', args.phase, 'scan')
    elif args.stage == 'prepare_joint':
        joint.prepare_joint('validation', args.phase)
    elif args.stage == 'infer_joint':
        if (ROOT / 'validation-visual-v1' / (args.phase + '-joint-started.json')).exists():
            raise ValueError('S1 joint stage already started; no implicit retry')
        joint.execute_joint('validation', args.phase)
    else:
        workflow.materialize(args.phase)
