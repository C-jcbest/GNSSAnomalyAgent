"""Save the two inspected VTBench source files at the verified author commit."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from gnss_sim.artifacts import sha, write_json

REVISION = "7e20537261af86ca60b21c963a91ebea385316a0"
FILES = (
    "vtbench/models/multimodal/fusion.py",
    "vtbench/models/multimodal/one_chart_numerical.py",
)


def export(repository, out):
    actual = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != REVISION:
        raise ValueError("Author source revision differs from the inspected commit")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for relative in FILES:
        blob = subprocess.check_output(
            ["git", "-C", str(repository), "show", REVISION + ":" + relative]
        )
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        rows.append(
            dict(
                repository="via-cs/vtbench",
                commit=REVISION,
                path=relative,
                sha256=sha(target),
                url=f"https://github.com/via-cs/vtbench/blob/{REVISION}/{relative}",
                read_scope="complete file; static source inspection",
                executed=False,
            )
        )
    write_json(out / "source-ledger.json", rows)
    (out / "export-source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "freeze.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(dict(files=len(rows), commit=REVISION, author_model_executed=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.repository.resolve(), args.out.resolve())
