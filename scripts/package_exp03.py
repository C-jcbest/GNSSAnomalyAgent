"""Bundle EXP03 evidence and frozen runs; deliberately exclude credentials and Git state."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from gnss_research.files import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "artifacts/exp03-package/exp03-supplement.zip"
    destination.parent.mkdir(parents=True, exist_ok=False)
    paths = [ROOT / name for name in ("README.md", "AGENTS.md", "pyproject.toml", "uv.lock",
                                     "configs", "docs", "src", "scripts", "tests",
                                     "papers/evidence", "vendor/upstream")]
    for directory in ("data", "runs", "results"):
        paths.extend((ROOT / directory).glob("exp03*"))
    files = []
    for path in paths:
        for item in ([path] if path.is_file() else path.rglob("*")):
            if not item.is_file():
                continue
            relative = item.relative_to(ROOT)
            if any(part in (".git", "__pycache__", ".pytest_cache", ".ruff_cache")
                   for part in relative.parts):
                continue
            if item.name.startswith(".env") or item.name.lower() in (
                    "credentials.yml", "credentials.yaml") or item.suffix in (".pyc", ".pyo"):
                continue
            files.append(item)
    inventory = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                  "sha256": sha256(p)} for p in sorted(set(files))]
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=5) as archive:
        for entry in inventory:
            archive.write(ROOT / entry["path"], entry["path"])
        archive.writestr("exp03-file-inventory.json", json.dumps(inventory, ensure_ascii=False,
                                                             indent=2))
    with ZipFile(destination) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt archived file: {bad}")
    receipt = {"path": str(destination.resolve()), "bytes": destination.stat().st_size,
               "sha256": sha256(destination), "source_files": len(inventory), "crc_check": "passed",
               "exclusions": ["environment credentials", "upstream credentials.yml", "Git metadata",
                              "Python caches"], "inventory": "exp03-file-inventory.json"}
    write_json(ROOT / "results/exp03-two-seeds/supplement-manifest.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
