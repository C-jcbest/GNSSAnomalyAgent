"""Archive nearest-neighbor methods and an explicitly separate query strategy."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_landslide_morphology_sources import fetch

from gnss_sim.artifacts import sha, write_json

URLS = {
    "vlm4ts-abs": "https://arxiv.org/abs/2506.06836",
    "vlm4ts-html": "https://arxiv.org/html/2506.06836",
    "refining-abs": "https://arxiv.org/abs/2503.21833",
    "refining-html": "https://arxiv.org/html/2503.21833",
    "dqs-abs": "https://arxiv.org/abs/2509.05663",
    "dqs-html": "https://arxiv.org/html/2509.05663",
}


def retrieve(out):
    out.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda pair: fetch(out, pair), URLS.items()))
    write_json(out / "fetch-ledger.json", rows)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir()})
    print(rows, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    retrieve(parser.parse_args().out.resolve())
