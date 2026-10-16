"""Retain official primary documents for numerical guidance and bounded agents."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_landslide_morphology_sources import fetch

from gnss_sim.artifacts import sha, write_json

URLS = {
    "medtvl-abs": "https://arxiv.org/abs/2608.28605",
    "medtvl-html": "https://arxiv.org/html/2608.28605",
    "vesta-abs": "https://arxiv.org/abs/2606.00384",
    "vesta-html": "https://arxiv.org/html/2606.00384",
}


def retrieve(out):
    out.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda pair: fetch(out, pair), URLS.items()))
    write_json(out / "fetch-ledger.json", rows)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(rows, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    retrieve(parser.parse_args().out.resolve())
