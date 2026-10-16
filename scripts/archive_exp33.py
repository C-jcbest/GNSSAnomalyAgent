"""Save the diagnostic and its exact reused parent evidence without new API calls."""
from __future__ import annotations

import gzip
import json
import shutil
import zipfile
from pathlib import Path

from experiment_exp33 import DATASET, PARENT, ROOT
from report_exp25 import aggregate_cost, request_receipts

from gnss_research.files import read_json, sha256, write_json


def archive_path(path):
    for folder, prefix in ((ROOT, 'exp33'), (PARENT, 'exp32'), (DATASET, 'data/exp30')):
        if path.is_relative_to(folder):
            return Path(prefix) / path.relative_to(folder)
    return path


def main():
    evidence = Path('results/exp33-evidence-v1')
    evidence.mkdir(parents=True, exist_ok=False)
    jobs = []
    for folder in (ROOT, ROOT / 'extension-s1'):
        registration = read_json(folder / 'registration.json')
        jobs.extend(registration['jobs'])
        for source, digest in registration['frozen_files'].items():
            if sha256(Path(source)) != digest:
                raise ValueError('registered dependency changed')
        target = evidence / ('main' if folder == ROOT else 's1')
        target.mkdir()
        for path in folder.glob('*registration*.json'):
            shutil.copyfile(path, target / path.name)
        for path in folder.glob('*correction*.json'):
            shutil.copyfile(path, target / path.name)
    with gzip.open(evidence / 'responses.jsonl.gz', 'xt', encoding='utf-8') as stream:
        for job in jobs:
            work = Path(job['work'])
            if sha256(work / 'request.json') != job['request_sha256']:
                raise ValueError('request changed')
            row = {'work': str(work.relative_to(ROOT)), 'request_sha256': job['request_sha256'],
                'result': read_json(work / 'result.json'),
                'response': read_json(work / 'response.json') if (work / 'response.json').exists() else None}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    costs = aggregate_cost(request_receipts([job['work'] for job in jobs]))
    write_json(evidence / 'costs.json', costs)
    folders = [ROOT, *[Path(path) for path in ('src', 'scripts', 'tests', 'docs', 'configs',
        'vendor/affiliation-metrics-py', 'runs/exp28-shared-training-v1', 'papers/exp32-context-confirmation')],
        *Path('results').glob('exp33-*')]
    files = {path for folder in folders for path in folder.rglob('*') if path.is_file()
        and not any(part in ('__pycache__', '.git', '.pytest_cache', '.ruff_cache') for part in path.parts)
        and not path.name.startswith('.env')}
    records = read_json(ROOT / 'registration.json')['records']
    for record in records:
        files.update(DATASET / record[key] for key in ('observation', 'truth'))
        for mode in ('scan_image', 'scan_text', 'jcv_image', 'jcv_text_final', 'jcv_text_pipeline'):
            files.update(path for path in (PARENT / 'validation-visual-v1/predictions' / mode / record['station_id']).rglob('*') if path.is_file())
        files.update(path for path in (PARENT / 'validation-baseline-v1/predictions/vector_nearest' / record['station_id']).rglob('*') if path.is_file())
    files.add(DATASET / 'manifest.json')
    files.add(Path('results/exp32-validation-pilot-v1/per-station.json.gz'))
    files.update(Path(path) for path in ('README.md', 'AGENTS.md', 'pyproject.toml', 'uv.lock', '.gitignore', '.gitattributes'))
    files = sorted(files)
    write_json(evidence / 'archive-files.json', {str(archive_path(path)): {'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in files})
    files.append(evidence / 'archive-files.json')
    archive = ROOT.parent / 'backups/EXP33_DeanchoredVisual_20261016.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
        for path in files:
            stream.write(path, archive_path(path))
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError('backup CRC failed')
    receipt = {'experiment_id': 'GNSS-VLM-EXP33', 'archive': str(archive), 'sha256': sha256(archive),
        'bytes': archive.stat().st_size, 'files': len(files), 'crc_verified': True, 'costs': costs,
        'prior_backup': {key: read_json(Path('docs/exp32-backup.json'))[key] for key in ('archive', 'sha256')}}
    write_json(Path('docs/exp33-backup.json'), receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
