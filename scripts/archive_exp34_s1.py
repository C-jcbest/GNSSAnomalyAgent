"""Preserve full EXP34-S1 inputs once, and commit compact response evidence."""
from __future__ import annotations

import gzip
import json
import shutil
import zipfile
from pathlib import Path

from experiment_exp30 import check_registration
from experiment_exp32 import DATASET
from experiment_exp34_s1 import ROOT, configure
from report_exp25 import aggregate_cost, request_receipts

from gnss_research.files import read_json, sha256, write_json


def archive_path(path):
    if path.is_relative_to(ROOT):
        return Path('exp34-s1') / path.relative_to(ROOT)
    if path.is_relative_to(DATASET):
        return Path('data/exp30') / path.relative_to(DATASET)
    return path


def main():
    configure()
    evidence = Path('results/exp34-s1-evidence-v1')
    evidence.mkdir(parents=True, exist_ok=False)
    visual = ROOT / 'validation-visual-v1'
    jobs = []
    for registration_path in visual.glob('*-registration.json'):
        registration = read_json(registration_path)
        check_registration(registration)
        shutil.copyfile(registration_path, evidence / registration_path.name)
        jobs.extend(registration['jobs'])
    with gzip.open(evidence / 'responses.jsonl.gz', 'xt', encoding='utf-8') as stream:
        for job in jobs:
            work = Path(job['work'])
            if sha256(work / 'request.json') != job['request_sha256']:
                raise ValueError('registered request changed')
            result = read_json(work / 'result.json')
            response = read_json(work / 'response.json') if (work / 'response.json').exists() else None
            row = {'work': str(work.relative_to(visual)), 'request_sha256': job['request_sha256'],
                'result_sha256': sha256(work / 'result.json'), 'result': result,
                'response_sha256': sha256(work / 'response.json') if response is not None else None,
                'response': response}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    costs = aggregate_cost(request_receipts([job['work'] for job in jobs]))
    write_json(evidence / 'costs.json', costs)
    folders = [ROOT, DATASET, *[Path(path) for path in ('src', 'scripts', 'tests', 'configs', 'docs',
        'papers/exp32-context-confirmation', 'vendor/affiliation-metrics-py', 'runs/exp28-shared-training-v1')],
        *Path('results').glob('exp34-s1-*')]
    files = {path for folder in folders for path in folder.rglob('*') if path.is_file()
        and not any(part in ('__pycache__', '.git', '.pytest_cache', '.ruff_cache') for part in path.parts)
        and not path.name.startswith('.env')}
    files.update(Path(path) for path in ('README.md', 'AGENTS.md', 'pyproject.toml', 'uv.lock', '.gitignore', '.gitattributes'))
    files = sorted(files)
    write_json(evidence / 'archive-files.json', {str(archive_path(path)): {
        'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in files})
    files.append(evidence / 'archive-files.json')
    archive = ROOT.parents[1] / 'backups/EXP34_S1_DeanchoredImageConfirmation_20261016.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
        for index, path in enumerate(files, 1):
            stream.write(path, archive_path(path))
            if index % 1000 == 0:
                print('Archived', index, '/', len(files), flush=True)
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError('backup CRC failed')
    receipt = {'experiment_id': 'GNSS-VLM-EXP34-S1', 'archive': str(archive), 'sha256': sha256(archive),
        'bytes': archive.stat().st_size, 'files': len(files), 'crc_verified': True, 'costs': costs,
        'prior_backup': {key: read_json(Path('docs/exp34-backup.json'))[key] for key in ('archive', 'sha256')}}
    write_json(Path('docs/exp34-s1-backup.json'), receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
