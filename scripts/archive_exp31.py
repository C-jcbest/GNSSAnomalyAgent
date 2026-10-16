"""Archive raw requests once and keep compact, reviewable response evidence in Git."""
from __future__ import annotations

import gzip
import json
import shutil
import zipfile
from pathlib import Path

from exp31_data_baseline import DATASET, ROOT
from experiment_exp30 import check_registration
from report_exp25 import aggregate_cost, request_receipts

from gnss_research.files import read_json, sha256, write_json


def archive_path(path):
    if path.is_relative_to(ROOT):
        return Path('exp31') / path.relative_to(ROOT)
    if path.is_relative_to(DATASET):
        return Path('data/exp30') / path.relative_to(DATASET)
    return path


def main():
    evidence = Path('results/exp31-evidence-v1')
    evidence.mkdir(parents=True, exist_ok=False)
    costs = {}
    for split in ('validation', 'test'):
        run = ROOT / (split + '-visual-v1')
        if not run.exists():
            continue
        target = evidence / split
        target.mkdir()
        jobs = []
        for registration_path in run.glob('*-registration.json'):
            registration = read_json(registration_path)
            check_registration(registration)
            shutil.copyfile(registration_path, target / registration_path.name)
            jobs.extend(registration['jobs'])
        for path in run.glob('*preparation*.json'):
            shutil.copyfile(path, target / path.name)
        with gzip.open(target / 'responses.jsonl.gz', 'xt', encoding='utf-8') as stream:
            for job in jobs:
                work = Path(job['work'])
                if sha256(work / 'request.json') != job['request_sha256']:
                    raise ValueError('registered request changed')
                result = read_json(work / 'result.json')
                response = read_json(work / 'response.json') if (work / 'response.json').exists() else None
                row = {'work': str(work.relative_to(run)), 'request_sha256': job['request_sha256'],
                    'result_sha256': sha256(work / 'result.json'), 'result': result,
                    'response_sha256': sha256(work / 'response.json') if response is not None else None,
                    'response': response}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        costs[split] = aggregate_cost(request_receipts([job['work'] for job in jobs]))
    write_json(evidence / 'costs.json', costs)
    folders = [ROOT, DATASET, *[Path(path) for path in ('src', 'scripts', 'tests', 'configs', 'docs',
        'papers/exp31-joint-context', 'vendor/affiliation-metrics-py', 'runs/exp28-shared-training-v1')],
        *Path('results').glob('exp31-*')]
    files = {path for folder in folders for path in folder.rglob('*') if path.is_file()
        and not any(part in ('__pycache__', '.git', '.pytest_cache', '.ruff_cache') for part in path.parts)
        and not path.name.startswith('.env')}
    files.update(Path(path) for path in ('README.md', 'AGENTS.md', 'pyproject.toml', 'uv.lock', '.gitignore', '.gitattributes'))
    files = sorted(files)
    write_json(evidence / 'archive-files.json', {str(archive_path(path)): {
        'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in files})
    files.append(evidence / 'archive-files.json')
    archive = ROOT.parent / 'backups/EXP31_JointContextVisual_20261016.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
        for index, path in enumerate(files, 1):
            stream.write(path, archive_path(path))
            if index % 1000 == 0:
                print('Archived', index, '/', len(files), flush=True)
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError('backup CRC failed')
    receipt = {'experiment_id': 'GNSS-VLM-EXP31', 'archive': str(archive), 'sha256': sha256(archive),
        'bytes': archive.stat().st_size, 'files': len(files), 'crc_verified': True, 'costs': costs,
        'prior_backup': {key: read_json(Path('docs/exp30-backup.json'))[key] for key in ('archive', 'sha256')}}
    write_json(Path('docs/exp31-backup.json'), receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
