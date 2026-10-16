"""Archive primary descriptions of visual alignment and physics-supervised fusion."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_landslide_morphology_sources import fetch

from gnss_sim.artifacts import sha, write_json

URLS = {
    "vista-abs": "https://arxiv.org/abs/2609.31448",
    "vista-html": "https://arxiv.org/html/2609.31448",
    "utopya-abs": "https://arxiv.org/abs/2605.18188",
    "utopya-html": "https://arxiv.org/html/2605.18188",
}


def retrieve(out):
    out.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda pair: fetch(out, pair), URLS.items()))
    write_json(out / "fetch-ledger.json", rows)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {path.name: sha(path) for path in out.iterdir()})
    print(rows, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    retrieve(parser.parse_args().out.resolve())
