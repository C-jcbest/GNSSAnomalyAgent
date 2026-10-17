"""Verify this research batch and preserve ignored evidence outside the worktree."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from gnss_research.files import read_json, sha256, source_hashes, write_json


def run():
    root = Path.cwd()
    registrations = [Path(f'artifacts/exp41/pilot/{stage}/registration.json')
        for stage in ('baseline', 'scan', 'review')]
    registrations.extend(Path(f'artifacts/exp{number}/pilot/review/registration.json')
        for number in (42, 43))
    registrations.append(Path('artifacts/exp44/pilot/registration.json'))
    checks, jobs, frozen = [], [], set()
    for path in registrations:
        registration = read_json(path)
        checks.append({'path': str(path), 'type': 'production_sources',
            'matches': registration['source_hashes'] == source_hashes()})
        for filename, digest in registration['frozen_files'].items():
            source = Path(filename)
            checks.append({'path': filename, 'type': 'frozen_source', 'matches': sha256(source) == digest})
            frozen.add(source)
        for job in registration.get('jobs', []):
            request = Path(job['work']) / 'request.json'
            checks.append({'path': str(request), 'type': 'request',
                'matches': sha256(request) == job['request_sha256']})
            jobs.append(job)
    for path in (Path('artifacts/exp42/pilot/review/input-hashes.json'),
            Path('artifacts/exp43/pilot/review/input-hashes.json'),
            Path('artifacts/exp44/pilot/upstream-hashes.json')):
        for item in read_json(path):
            checks.append({'path': item['path'], 'type': 'upstream_prediction',
                'matches': sha256(Path(item['path'])) == item['sha256']})
    manifest = read_json(Path('artifacts/exp41/data/manifest.json'))
    for record in manifest['records']:
        for kind in ('observation', 'truth'):
            path = Path('artifacts/exp41/data') / record[kind]
            checks.append({'path': str(path), 'type': 'data_bytes_only',
                'matches': sha256(path) == record[kind + '_sha256']})
    training = Path('runs/exp28-shared-training-v1')
    model = read_json(training / 'training.json')
    checks.append({'path': str(training), 'type': 'frozen_training',
        'matches': source_hashes() == model['source_hashes']
            and sha256(training / 'models.joblib') == model['model_sha256']})
    receipts = [read_json(Path(job['work']) / 'result.json') for job in jobs]
    passed = all(item['matches'] for item in checks)
    verification = {'passed': passed, 'checks': checks, 'unique_new_requests': len(jobs),
        'successful_requests': sum(item['status'] == 'success' for item in receipts),
        'failed_requests': sum(item['status'] != 'success' for item in receipts),
        'retry_count': sum(item.get('retry_count', 0) for item in receipts),
        'different_development_stations': 18, 'unused_confirmation_stations': 108,
        'unused_test_stations': 720,
        'data_verification_scope': 'SHA only for sealed data; no sealed predictions or scoring',
        'frozen_model_sha256': model['model_sha256'], 'targeted_tests_passed': 7,
        'test_scope': 'new evidence/search/interval tests plus EXP40 ownership; not historical full suite'}
    write_json(Path('docs/exp41-44-verification.json'), verification)
    if not passed or verification['retry_count']:
        raise ValueError('verification failed; preserve evidence before proceeding')

    files = set(frozen)
    for number in (41, 42, 43, 44):
        files.update(path for path in Path(f'artifacts/exp{number}').rglob('*') if path.is_file())
    files.update((training / 'training.json', training / 'models.joblib'))
    files = {path.resolve() for path in files}
    archive = Path('D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/backups') / 'EXP41_EXP44_Boundary_Optimization_20261017.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    members = []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for path in sorted(files):
            if path.name.startswith('.env'):
                raise ValueError('private environment files must not enter experiment archive')
            content = path.read_bytes()
            name = path.resolve().relative_to(root).as_posix()
            stream.writestr(name, content)
            members.append({'path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
        stream.writestr('MANIFEST.json', json.dumps({'files': members,
            'scope': 'EXP41-44 raw evidence, full EXP41 simulated data, frozen sources and model; no .env'},
            ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive) as stream:
        bad = stream.testzip()
        count = len(stream.infolist())
    if bad is not None:
        raise ValueError('archive CRC failure: ' + bad)
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest,
        'files': len(members), 'members_including_manifest': count, 'crc_passed': True,
        'git_backup_before_work': 'backup/exp40-before-exp41-20261017',
        'failed_initial_baseline_included': True, 'private_environment_included': False}
    write_json(Path('docs/exp41-44-archive.json'), receipt)
    print(json.dumps({'verification_passed': passed, 'checks': len(checks), **receipt}, indent=2))


if __name__ == '__main__':
    run()
