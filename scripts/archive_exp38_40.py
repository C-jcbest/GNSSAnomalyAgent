"""Save new experimental evidence outside the worktree, without local credentials."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from gnss_research.files import read_json, sha256, write_json


def archive(target):
    root = Path.cwd()
    files = {}
    folders = ['src', 'scripts', 'tests', 'configs', 'papers/exp38-reference-residual',
        'artifacts/exp38', 'artifacts/exp39', 'artifacts/exp40']
    for pattern in ('results/exp38-*', 'results/exp39-*', 'results/exp40-*'):
        folders.extend(str(p) for p in Path('.').glob(pattern) if p.is_dir())
    for folder in folders:
        for path in Path(folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and '.git' not in path.parts:
                files[path.as_posix()] = path
    for pattern in ('docs/*.md', 'docs/exp38*.json', 'docs/exp39*.json', 'docs/exp40*.json'):
        for path in Path('.').glob(pattern):
            files[path.as_posix()] = path
    for name in ('AGENTS.md', 'README.md', '.gitattributes', '.gitignore', 'pyproject.toml', 'uv.lock'):
        files[name] = Path(name)
    if any(Path(name).name.startswith('.env') for name in files):
        raise ValueError('credentials must not enter evidence archive')
    manifest = {name: {'sha256': sha256(path), 'bytes': path.stat().st_size}
        for name, path in sorted(files.items())}
    previous = read_json(Path('docs/exp38-backup.json'))
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for name, path in sorted(files.items()):
            package.write(path, name)
        package.writestr('MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        package.writestr('RESTORE.md',
            'EXP38-40 supplementary evidence; no credentials.\n\n'
            'Restore into a separate directory, never overwrite registered live runs.\n'
            'For original EXP28 models/data and EXP30 source data, use the verified\n'
            'EXP36/37 archive named in docs/exp38-backup.json. The new full EXP38\n'
            'manifest and 792 observation/truth pairs are under artifacts/exp38/data.\n'
            'Keep original source hashes, requests, failures and absolute-path provenance.\n'
            'Use a separate launcher if restored paths differ; do not edit frozen receipts.\n'
            'Saved requests are evidence, not instructions to replay billable calls.\n')
    with zipfile.ZipFile(target) as package:
        if package.testzip() is not None:
            raise ValueError('archive CRC verification failed')
    receipt = {'archive': str(target.resolve()), 'sha256': sha256(target),
        'bytes': target.stat().st_size, 'source_files': len(files), 'archive_members': len(files) + 2,
        'crc_verified': True, 'credentials_included': False, 'source_root': str(root),
        'full_manifest_inside_archive': 'MANIFEST.json',
        'shared_model_archive': previous['existing_full_artifact_archive'],
        'shared_model_archive_sha256': previous['archive_sha256']}
    write_json(Path('docs/exp38-40-archive.json'), receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    archive(parser.parse_args().out)
