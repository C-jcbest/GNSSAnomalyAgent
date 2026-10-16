"""Fresh-station confirmation of visual screening followed by deanchored evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

import experiment_exp30 as air
import experiment_exp31 as joint
import experiment_exp32 as workflow
from experiment_exp29 import payload
from experiment_exp33 import remove_narratives

from gnss_research.files import read_json, sha256, source_hashes, write_json

ROOT = workflow.ROOT.parent / 'exp34'
PROTOCOL = Path('docs/experiment-exp34.md')
MODES = {mode: joint.MODES[mode] for mode in ('jcv_image', 'jcv_text_final', 'jcv_text_pipeline')}
DEPENDENCIES = [Path(__file__), PROTOCOL, Path('scripts/experiment_exp33.py'),
    *[path for path in workflow.DEPENDENCIES if path.name != 'experiment-exp32.md']]


def records_for(phase):
    records = [record for record in read_json(workflow.DATASET / 'manifest.json')['records']
        if record['split'] == 'validation' and int(record['station_id'][1:]) % 120 >= 60]
    if phase == 'full':
        return records
    return [record for record in records
        if (int(record['station_id'][1:]) % 120 < 84) == (phase == 'pilot')]


def guard_phase(phase):
    if phase == 'remaining':
        if not read_json(Path('results/exp34-validation-pilot-v1/admission.json'))['mechanism_confirmed']:
            raise ValueError('pilot mechanism gate failed; do not expand')


def register(output, phase, stage, records, jobs):
    write_json(output / f'{phase}-{stage}-registration.json', {
        'experiment_id': 'GNSS-VLM-EXP34', 'records': records, 'jobs': jobs,
        'source_hashes': source_hashes(), 'manifest_sha256': sha256(workflow.DATASET / 'manifest.json'),
        'frozen_files': {str(path): sha256(path) for path in DEPENDENCIES}})


def store_deanchored_request(work, mode, station, kind, content, prompt, schema, **metadata):
    request = remove_narratives(payload(air.MODEL, prompt, schema, content))
    write_json(work / 'request.json', request)
    return {'work': str(work), 'mode': mode, 'station': station, 'kind': kind,
        'request_sha256': sha256(work / 'request.json'), **metadata}


def configure():
    workflow.ROOT = ROOT
    workflow.PROTOCOL = PROTOCOL
    workflow.records_for = records_for
    workflow.guard_phase = guard_phase
    workflow.register = register
    workflow.configure()
    joint.MODES = MODES
    joint.store_request = store_deanchored_request


def main():
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
            raise ValueError('joint stage already started; no implicit retry')
        joint.execute_joint('validation', args.phase)
    else:
        workflow.materialize(args.phase)


if __name__ == '__main__':
    main()
