"""Save inspected MedTVL source blobs at the exact author revision."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from gnss_sim.artifacts import sha, write_json

REVISION = "4f05ba8ff55a8387b95dcee38b0e20a6c99aa1d7"
FILES = ("README.md", "Models/Medtvl/model.py", "Config/model_config.json")


def export(repository, out):
    actual = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != REVISION:
        raise ValueError("Author HEAD differs from the inspected source")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for relative in FILES:
        blob = subprocess.check_output(
            ["git", "-C", str(repository), "show", REVISION + ":" + relative]
        )
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        rows.append(dict(
            repository="start2020/MedTVL", commit=REVISION, path=relative,
            url=f"https://github.com/start2020/MedTVL/blob/{REVISION}/{relative}",
            sha256=sha(target), read_scope="Complete static file", executed=False,
        ))
    write_json(out / "source-ledger.json", rows)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {
        str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()
    })
    print(dict(files=len(rows), revision=REVISION, author_model_executed=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.repository.resolve(), args.out.resolve())
