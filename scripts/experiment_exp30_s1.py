"""Explicit three-window implementation correction; unchanged VRC requests are reused."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import exp30_data_baseline as data
import experiment_exp30 as experiment
import report_exp30 as reporting
from exp30_data_baseline import ROOT

from gnss_research.files import read_json, sha256, write_json

S1 = ROOT / 'implementation-s1'
PROTOCOL = Path('docs/experiment-exp30-s1.md')


def complete_windows(observation):
    first, length = observation['background_end'] + 1, len(observation['dates'])
    result = []
    for left in range(first, length, 120):
        right = min(left + 180, length)
        result.append((left, right))
        if right == length:
            break
    return result


def prepare_scan(split, phase):
    if split == 'validation' and phase == 'pilot':
        S1.mkdir(parents=True, exist_ok=False)
        shutil.copytree(ROOT / 'validation-baseline-v1', S1 / 'validation-baseline-v1')
    experiment.prepare_scan(split, phase)
    output = S1 / (split + '-visual-v1')
    path = output / f'{phase}-scan-registration.json'
    registration = read_json(path)
    # Only the pre-existing, byte-identical pilot VRC controls can be reused.
    if split == 'validation' and phase == 'pilot':
        for job in registration['jobs']:
            if not job['mode'].startswith('vrc_'):
                continue
            old_work = ROOT / 'technical-v2/validation-visual-v1/predictions' / job['mode'] / job['station']
            if sha256(old_work / 'request.json') != job['request_sha256']:
                raise ValueError('cannot reuse a changed VRC request')
            receipt = read_json(old_work / 'result.json')
            receipt.update(request_count=0, reused_original_request_count=1,
                reused_original_work=str(old_work), reused_receipt_sha256=sha256(old_work / 'result.json'))
            work = Path(job['work'])
            write_json(work / 'result.json', receipt)
            if (old_work / 'response.json').exists():
                shutil.copyfile(old_work / 'response.json', work / 'response.json')
            job['reused_from'] = str(old_work)
    fresh = {**registration, 'jobs': [job for job in registration['jobs'] if 'reused_from' not in job],
        'reused_controls': [job for job in registration['jobs'] if 'reused_from' in job]}
    write_json(output / f'{phase}-fresh-scan-registration.json', fresh)
    print('Fresh scan requests', len(fresh['jobs']), 'reused controls', len(fresh['reused_controls']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare_scan', 'infer_scan', 'prepare_local', 'infer_local', 'materialize', 'report', 'baseline'))
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--phase', choices=('pilot', 'remaining', 'full'), default='pilot')
    args = parser.parse_args()
    experiment.ROOT = S1
    experiment.windows = complete_windows
    experiment.DEPENDENCIES.extend([Path(__file__), PROTOCOL])
    if args.stage == 'prepare_scan':
        prepare_scan(args.split, args.phase)
    elif args.stage == 'infer_scan':
        experiment.execute(args.split, args.phase, 'fresh-scan')
        write_json(S1 / (args.split + '-visual-v1') / (args.phase + '-scan-complete.json'),
            {'fresh_registration': args.phase + '-fresh-scan-registration.json', 'includes_explicit_reused_controls': True})
    elif args.stage == 'infer_local':
        experiment.execute(args.split, args.phase, 'local')
    elif args.stage == 'baseline':
        data.ROOT = S1
        data.predict(args.split, args.phase)
    elif args.stage == 'report':
        reporting.ROOT = S1
        reporting.report(args.split, args.phase)
    else:
        {'prepare_local': experiment.prepare_local, 'materialize': experiment.materialize}[args.stage](args.split, args.phase)
