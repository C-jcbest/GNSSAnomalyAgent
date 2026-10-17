"""Preserve experimental evidence outside the worktree, excluding local credentials."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from experiment_exp36 import DATASET, TRAIN_DATASET

from gnss_research.files import read_json, sha256, write_json


def archive(target):
    files = {}
    for folder in ('src', 'scripts', 'tests', 'configs', 'vendor/affiliation-metrics-py',
                   'papers/exp36-contrastive-evidence', 'papers/exp37-domain-search',
                   'papers/evidence', 'artifacts/exp36', 'artifacts/exp37'):
        for path in Path(folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and '.git' not in path.parts:
                files[path.as_posix()] = path
    for pattern in ('docs/*.md', 'results/exp36-*', 'results/exp37-*'):
        for path in Path('.').glob(pattern):
            paths = path.rglob('*') if path.is_dir() else [path]
            for item in paths:
                if item.is_file():
                    files[item.as_posix()] = item
    for name in ('AGENTS.md', 'README.md', 'pyproject.toml', 'uv.lock', 'docs/evaluation-policy.md'):
        files[name] = Path(name)
    for name in ('models.joblib', 'training.json'):
        path = Path('runs/exp28-shared-training-v1') / name
        files[path.as_posix()] = path
    # Keep complete manifests with all their files, so dataset integrity checks remain usable.
    for dataset, prefix in (
        (DATASET, 'datasets/exp30'),
        (TRAIN_DATASET, 'datasets/exp28'),
    ):
        files[prefix + '/manifest.json'] = dataset / 'manifest.json'
        for record in read_json(dataset / 'manifest.json')['records']:
            for key in ('observation', 'truth'):
                path = dataset / record[key]
                files[prefix + '/' + Path(record[key]).as_posix()] = path
    manifest = {name: {'sha256': sha256(path), 'bytes': path.stat().st_size}
        for name, path in sorted(files.items())}
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for name, path in sorted(files.items()):
            package.write(path, name)
        package.writestr('RESTORE.md',
            'EXP36/37 evidence archive; no credentials included.\n\n'
            'Extract into a separate directory; do not overwrite live frozen runs.\n'
            'Rebuild the environment from pyproject.toml/uv.lock. Provide your own .env.\n'
            'datasets/exp30 and datasets/exp28 contain every manifest record.\n'
            'Historical scripts use absolute D: dataset paths; restore data there or use a\n'
            'separate launcher to assign module DATASET paths before executing. Do not\n'
            'rewrite frozen registrations or scripts to pretend original hashes match.\n'
            'Existing requests are evidence, not instructions to replay them.\n')
    with zipfile.ZipFile(target) as package:
        if package.testzip() is not None:
            raise ValueError('archive CRC failed')
    receipt = {'archive': str(target.resolve()), 'sha256': sha256(target),
        'bytes': target.stat().st_size, 'files': len(files), 'crc_verified': True,
        'credentials_included': False, 'manifest': manifest}
    write_json(Path('docs/exp36-37-backup.json'), receipt)
    print({key: value for key, value in receipt.items() if key != 'manifest'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    archive(parser.parse_args().out)
