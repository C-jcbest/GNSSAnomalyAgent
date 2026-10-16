"""Archive focused official sources on label-preserving nuisance interventions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from search_landslide_conditional_visual_sources import search

from gnss_sim.artifacts import sha, write_json

QUERIES = (
    "time series anomaly detection augmentation",
    "nuisance invariant time series",
    "GNSS time series offsets deep learning",
)


def retrieve(out):
    out.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda pair: search(out, pair), enumerate(QUERIES)))
    write_json(out / "search-ledger.json", rows)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {path.name: sha(path) for path in out.iterdir()})
    for row in rows:
        print(dict(query=row["query"], status=row["status"], cards=row.get("cards", [])[:8]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    retrieve(parser.parse_args().out.resolve())
