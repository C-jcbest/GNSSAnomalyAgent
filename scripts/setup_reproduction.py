"""Materialize hash-locked upstream examples without changing the application environment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK = ROOT / "artifacts/reproduction-2026-10-07"


def materialize_sources(work: Path) -> None:
    lock = json.loads((ROOT / "configs/reproduction-sources.json").read_text(encoding="utf-8"))
    destination_root = (work / "upstream").resolve()
    receipts = []
    for entry in lock["files"]:
        destination = (destination_root / entry["path"]).resolve()
        if not destination.is_relative_to(destination_root):
            raise ValueError("Source destination escapes upstream directory")
        cache = ROOT / entry["cache"]
        if destination.exists():
            content = destination.read_bytes()
        elif cache.exists():
            content = cache.read_bytes()
        else:
            with urlopen(entry["url"], timeout=60) as response:
                content = response.read()
        digest = hashlib.sha256(content).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Upstream hash mismatch: {entry['path']}")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        receipts.append({"path": entry["path"], "url": entry["url"], "sha256": digest})
    work.mkdir(parents=True, exist_ok=True)
    (work / "source-receipts.json").write_text(
        json.dumps(receipts, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(receipts)} unmodified upstream files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    materialize_sources(parser.parse_args().work)
