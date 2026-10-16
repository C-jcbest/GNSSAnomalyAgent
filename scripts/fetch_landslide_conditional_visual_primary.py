"""Fetch primary complementary-embedding and time-series fusion sources."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_landslide_morphology_sources import fetch

from gnss_sim.artifacts import sha, write_json

URLS = {
    "rescomemb-abs": "https://arxiv.org/abs/2609.37225",
    "rescomemb-html": "https://arxiv.org/html/2609.37225",
    "vtbench-abs": "https://arxiv.org/abs/2604.27259",
    "vtbench-html": "https://arxiv.org/html/2604.27259",
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
