"""Export targeted author source blobs at exact Git revisions, without running them."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from gnss_sim.artifacts import sha, write_json


def export(base, out):
    out.mkdir(parents=True, exist_ok=False)
    selections = {
        "TShape": (
            base / "morphology-author-code-v2/TShape",
            "CSTCloudOps/TShape",
            ["readme.md", "layyer.py"],
        ),
        "VAN-AD": (
            base / "morphology-author-code-v1/VAN-AD",
            "PenyChen/VAN-AD",
            [
                "README.md",
                "layers/Learnable_TimeSeries_To_Image.py",
                "vanad/vanad_model.py",
                "vanad/rec_model.py",
            ],
        ),
    }
    rows = []
    for name, (repository, remote, paths) in selections.items():
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        for relative in paths:
            blob = subprocess.check_output(
                ["git", "-C", str(repository), "show", revision + ":" + relative]
            )
            target = out / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            rows.append(
                dict(
                    repository=remote,
                    commit=revision,
                    path=relative,
                    sha256=sha(target),
                    url=f"https://github.com/{remote}/blob/{revision}/{relative}",
                    executed=False,
                )
            )
    write_json(out / "source-ledger.json", rows)
    (out / "export-source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "freeze.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(rows, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    export(args.base.resolve(), args.out.resolve())
