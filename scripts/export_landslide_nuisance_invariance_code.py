"""Save selected immutable author loss/training files without executing them."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json

FILES = ("README.md", "utils/losses.py", "scripts/timeseries_MCC5/train.py", "scripts/timeseries_MCC5/configs.py")


def export(index, out):
    verify_manifest(index, "freeze.json")
    repository = read_json(index / "retrieval.json")
    if "commit" not in repository or not set(FILES).issubset(repository["paths"]):
        raise ValueError("Exact selected source paths need a successful immutable index")
    out.mkdir(parents=True, exist_ok=False)

    def fetch(relative):
        url = f"https://raw.githubusercontent.com/{repository['repository']}/{repository['commit']}/{relative}"
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.get(url)
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
            return dict(path=relative, url=url, status=response.status_code, sha256=sha(target))

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch, FILES))
    write_json(out / "retrieval.json", dict(repository=repository["repository"], commit=repository["commit"],
                                          index_manifest_sha256=sha(index / "freeze.json"), rows=rows, author_model_run=False))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    print(rows, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    export(arguments.index.resolve(), arguments.out.resolve())
