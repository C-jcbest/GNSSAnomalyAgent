"""Preserve EXP29 inputs, every attempted request and offline tool evidence."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from exp29_data_baseline import ROOT
from experiment_exp29 import verify_frozen
from experiment_exp29_s2 import check as check_s2
from report_exp25 import aggregate_cost, request_receipts

from gnss_research.files import read_json, sha256, write_json


def archive_path(path):
    if path.is_absolute() and path.is_relative_to(ROOT):
        return Path('exp29') / path.relative_to(ROOT)
    return path


def main():
    evidence = Path('results/exp29-evidence-v1')
    evidence.mkdir(parents=True, exist_ok=False)
    costs = {}
    for split in ('validation', 'test'):
        visual = ROOT / f'{split}-visual-v1'
        if not visual.exists():
            continue
        for registration_path in visual.glob('*-registration.json'):
            registration = read_json(registration_path)
            verify_frozen(registration)
            shutil.copyfile(registration_path, evidence / (split + '-' + registration_path.name))
            for job in registration['jobs']:
                work = visual / 'predictions' / job['mode'] / job['station']
                if sha256(work / 'request.json') != job['request_sha256']:
                    raise ValueError('frozen request changed')
                target = evidence / 'responses' / split / job['mode'] / job['station']
                target.mkdir(parents=True)
                for name in ('result.json', 'response.json', 'tool-evidence.json'):
                    if (work / name).exists():
                        shutil.copyfile(work / name, target / name)
        costs[split] = aggregate_cost(request_receipts([str(visual / 'predictions')]))
    joint = ROOT / 'validation-joint-v1'
    if joint.exists():
        joint_registration = read_json(joint / 'registration.json')
        for path, digest in joint_registration['frozen_files'].items():
            if sha256(Path(path)) != digest:
                raise ValueError('joint source changed')
        for path, digest in joint_registration['reused_response_sha256'].items():
            if sha256(Path(path)) != digest:
                raise ValueError('reused first-stage response changed')
        shutil.copyfile(joint / 'registration.json', evidence / 'joint-registration.json')
        for mode in ('joint_vl_image', 'joint_vl_text', 'joint_numeric_all_pairs'):
            for work in (joint / mode).iterdir():
                target = evidence / 'joint-tools' / mode / work.name
                target.mkdir(parents=True)
                for name in ('result.json', 'tool-evidence.json'):
                    shutil.copyfile(work / name, target / name)
    adjudication = ROOT / 'validation-adjudication-v1'
    if adjudication.exists():
        for registration_path in adjudication.glob('*-registration.json'):
            registration = read_json(registration_path)
            check_s2(registration)
            shutil.copyfile(registration_path, evidence / ('s2-' + registration_path.name))
            for job in registration['jobs']:
                work = adjudication / 'predictions' / job['mode'] / job['station']
                if sha256(work / 'request.json') != job['request_sha256']:
                    raise ValueError('adjudication request changed')
                target = evidence / 'responses/s2' / job['mode'] / job['station']
                target.mkdir(parents=True)
                for name in ('result.json', 'response.json', 'candidates.json'):
                    if (work / name).exists():
                        shutil.copyfile(work / name, target / name)
        costs['validation_s2'] = aggregate_cost(request_receipts([str(adjudication / 'predictions')]))
    write_json(evidence / 'costs.json', costs)
    folders = [Path(p) for p in ('src', 'scripts', 'tests', 'configs', 'docs',
        'vendor/affiliation-metrics-py', 'papers/exp29-visual-first',
        'runs/exp28-shared-training-v1')]
    folders.extend(p for p in Path('results').glob('exp29-*') if p.is_dir())
    folders.append(ROOT)
    files = {p for folder in folders for p in folder.rglob('*') if p.is_file()
        and not any(part in ('__pycache__', '.git', '.pytest_cache', '.ruff_cache') for part in p.parts)
        and not p.name.startswith('.env')}
    files.update(Path(p) for p in ('README.md', 'AGENTS.md', 'pyproject.toml', 'uv.lock',
                                  '.gitignore', '.gitattributes'))
    files = sorted(files)
    write_json(evidence / 'archive-files.json', {str(archive_path(p)): {
        'original_path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files})
    files.append(evidence / 'archive-files.json')
    archive = ROOT.parent / 'backups/EXP29_VisualFirst_720Stations_20261015.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
        for index, path in enumerate(files, 1):
            stream.write(path, archive_path(path))
            if index % 2000 == 0:
                print('Archived', index, '/', len(files), flush=True)
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError('archive CRC failed')
    receipt = {'experiment_id': 'GNSS-VLM-EXP29', 'archive': str(archive),
        'sha256': sha256(archive), 'bytes': archive.stat().st_size, 'files': len(files),
        'costs': costs, 'contains_credentials': False,
        'prior_backup': {key: read_json(Path('docs/exp28-backup.json'))[key]
                         for key in ('archive', 'sha256')}}
    write_json(Path('docs/exp29-backup.json'), receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
