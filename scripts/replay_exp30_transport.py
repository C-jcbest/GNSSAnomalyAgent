"""Explicit exact-request technical replication after a recorded connection outage."""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import exp30_data_baseline as data
import experiment_exp30 as experiment
import report_exp30 as reporting
from exp30_data_baseline import ROOT

from gnss_research.files import read_json, sha256, write_json

TECHNICAL = ROOT / 'technical-v2'


def prepare():
    original = ROOT / 'validation-visual-v1'
    out = TECHNICAL / 'validation-visual-v1'
    registration = read_json(original / 'pilot-scan-registration.json')
    experiment.check_registration(registration)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(original / 'inputs', out / 'inputs')
    shutil.copytree(ROOT / 'validation-baseline-v1', TECHNICAL / 'validation-baseline-v1')
    for job in registration['jobs']:
        source = Path(job['work'])
        target = out / source.relative_to(original)
        target.mkdir(parents=True)
        shutil.copyfile(source / 'request.json', target / 'request.json')
        if sha256(target / 'request.json') != job['request_sha256']:
            raise ValueError('technical replication changed request bytes')
        job['original_work'] = str(source)
        job['work'] = str(target)
    registration.update(technical_attempt=2, original_registration_sha256=sha256(original / 'pilot-scan-registration.json'))
    registration['frozen_files'][str(Path(__file__))] = sha256(Path(__file__))
    registration['frozen_files']['docs/exp30-transport-replication.md'] = sha256(Path('docs/exp30-transport-replication.md'))
    write_json(out / 'pilot-scan-registration.json', registration)
    print('Registered 288 exact-request technical replications', flush=True)


def smoke_then_execute():
    output = TECHNICAL / 'validation-visual-v1'
    registration = read_json(output / 'pilot-scan-registration.json')
    experiment.check_registration(registration)
    marker = output / 'pilot-scan-started.json'
    if marker.exists():
        raise ValueError('technical replication already started')
    write_json(marker, {'registered_requests': len(registration['jobs']), 'retries': 0, 'smoke_jobs': 4})

    def execute(job):
        work = Path(job['work'])
        if sha256(work / 'request.json') != job['request_sha256']:
            raise ValueError('request changed')
        receipt, content = experiment.send_request(read_json(work / 'request.json'), work, experiment.settings(experiment.MODEL))
        if receipt['status'] == 'success':
            try:
                schema = experiment.ScanDecision if job['kind'] == 'scan' else experiment.RangeDecision
                decision = schema.model_validate_json(content).model_dump()
                left, right = job['allowed_context']
                if any(not left <= v['start'] <= v['end'] <= right
                       for v in decision.get('regions', decision.get('intervals', []))):
                    raise ValueError('decision outside original monitoring range')
                receipt['decision'] = decision
            except (ValueError, TypeError, KeyError) as error:
                receipt.update(status='failed', error_type=type(error).__name__)
        write_json(work / 'result.json', receipt)
        return job['mode'], job['station'], receipt

    for job in registration['jobs'][:4]:
        mode, station, receipt = execute(job)
        print('technical smoke', mode, station, receipt['status'], flush=True)
        if receipt.get('http_status') != 200:
            write_json(output / 'pilot-scan-aborted.json', {'reason': 'smoke transport failed'})
            return
    with ThreadPoolExecutor(max_workers=3) as executor:
        for index, (mode, station, receipt) in enumerate(executor.map(execute, registration['jobs'][4:]), 5):
            if index % 12 == 0 or receipt['status'] != 'success':
                print(index, '/', len(registration['jobs']), mode, station, receipt['status'], flush=True)
    write_json(output / 'pilot-scan-complete.json', {'requests': len(registration['jobs']), 'technical_attempt': 2})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'prepare_scan', 'infer_scan', 'prepare_local', 'infer_local', 'materialize', 'baseline', 'report'))
    parser.add_argument('--phase', choices=('pilot', 'remaining', 'full'), default='pilot')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    args = parser.parse_args()
    experiment.ROOT = TECHNICAL
    experiment.DEPENDENCIES.extend([Path(__file__), Path('docs/exp30-transport-replication.md')])
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'infer_scan' and args.phase == 'pilot' and args.split == 'validation':
        smoke_then_execute()
    elif args.stage.startswith('infer_'):
        experiment.execute(args.split, args.phase, args.stage.removeprefix('infer_'))
    elif args.stage == 'baseline':
        data.ROOT = TECHNICAL
        data.predict(args.split, args.phase)
    elif args.stage == 'report':
        reporting.ROOT = TECHNICAL
        reporting.report(args.split, args.phase)
    else:
        {'prepare_scan': experiment.prepare_scan, 'prepare_local': experiment.prepare_local,
            'materialize': experiment.materialize}[args.stage](args.split, args.phase)
