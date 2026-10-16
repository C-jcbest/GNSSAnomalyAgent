"""Archive independent EXP30 attempts and all inputs, including failed transport receipts."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from exp30_data_baseline import ROOT
from experiment_exp30 import check_registration
from report_exp25 import aggregate_cost, request_receipts

from gnss_research.files import read_json, sha256, write_json


def archive_path(path):
    return Path('exp30') / path.relative_to(ROOT) if path.is_absolute() else path


def main():
    evidence = Path('results/exp30-evidence-v1')
    evidence.mkdir(parents=True, exist_ok=False)
    costs = {}
    for run in (ROOT / 'validation-visual-v1', ROOT / 'technical-v2/validation-visual-v1',
                ROOT / 'implementation-s1/validation-visual-v1', ROOT / 'implementation-s1/test-visual-v1'):
        if not run.exists():
            continue
        attempt = 'original-failure' if run.parent == ROOT else (run.parent.name + '-' + run.name)
        target = evidence / attempt
        target.mkdir(parents=True)
        for registration_path in run.glob('*-registration.json'):
            registration = read_json(registration_path)
            check_registration(registration)
            shutil.copyfile(registration_path, target / registration_path.name)
            for job in registration['jobs']:
                source = Path(job['work'])
                if sha256(source / 'request.json') != job['request_sha256']:
                    raise ValueError('registered request changed')
                copy = target / 'responses' / source.relative_to(run / 'predictions')
                copy.mkdir(parents=True, exist_ok=True)
                for name in ('result.json', 'response.json'):
                    if (source / name).exists():
                        shutil.copyfile(source / name, copy / name)
        completed = [str(p.parent) for p in (run / 'predictions').rglob('request.json')
            if (p.parent / 'result.json').exists()]
        costs[attempt] = aggregate_cost(request_receipts(completed))
        costs[attempt]['unreceipted_prepared_jobs'] = sum(not (p.parent / 'result.json').exists()
            for p in (run / 'predictions').rglob('request.json'))
        aborted = run / 'pilot-local-aborted.json'
        if aborted.exists():
            costs[attempt]['possible_unreceipted_inflight_upper_bound'] = read_json(aborted)['possible_unreceipted_inflight_requests_upper_bound']
    for path in ROOT.glob('health*.json'):
        shutil.copyfile(path, evidence / path.name)
    write_json(evidence / 'costs.json', costs)
    folders = [ROOT, *[Path(p) for p in ('src', 'scripts', 'tests', 'configs', 'docs',
        'papers/exp30-adaptive-localization', 'vendor/affiliation-metrics-py',
        'runs/exp28-shared-training-v1')], *Path('results').glob('exp30-*')]
    files = {p for folder in folders for p in folder.rglob('*') if p.is_file()
        and not any(part in ('__pycache__', '.git', '.pytest_cache', '.ruff_cache') for part in p.parts)
        and not p.name.startswith('.env')}
    files.update(Path(p) for p in ('README.md', 'AGENTS.md', 'pyproject.toml', 'uv.lock', '.gitignore', '.gitattributes'))
    files = sorted(files)
    write_json(evidence / 'archive-files.json', {str(archive_path(p)): {
        'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files})
    files.append(evidence / 'archive-files.json')
    archive = ROOT.parent / 'backups/EXP30_AdaptiveVisual_20261016.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
        for index, path in enumerate(files, 1):
            stream.write(path, archive_path(path))
            if index % 2000 == 0:
                print('Archived', index, '/', len(files), flush=True)
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError('backup CRC failed')
    receipt = {'experiment_id': 'GNSS-VLM-EXP30', 'archive': str(archive),
        'sha256': sha256(archive), 'bytes': archive.stat().st_size, 'files': len(files),
        'costs': costs, 'health_checks': 'GET /models; separately recorded, not detection requests',
        'prior_backup': {key: read_json(Path('docs/exp29-backup.json'))[key] for key in ('archive', 'sha256')}}
    write_json(Path('docs/exp30-backup.json'), receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
