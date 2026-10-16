"""Retain targeted literature discovery and fixed VLM4TS plotting source."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from gnss_sim.artifacts import sha, write_json

QUERIES = [
    "vision language time series anomaly detection normalization scale",
    "time series visualization logarithmic scale anomaly detection",
    "GNSS time series deformation offsets normalization",
    "vision time series adaptive visualization scaling",
]


def retrieve(out, index, query):
    url = "https://api.openalex.org/works"
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        try:
            response = client.get(
                url,
                params={
                    "search": query,
                    "per-page": 8,
                    "filter": "from_publication_date:2024-01-01,to_publication_date:2026-10-12",
                    "sort": "relevance_score:desc",
                },
            )
            (out / f"search-{index}.json").write_bytes(response.content)
            response.raise_for_status()
            records = []
            for work in response.json()["results"]:
                positions = {
                    position: word
                    for word, locations in (work.get("abstract_inverted_index") or {}).items()
                    for position in locations
                }
                records.append(
                    {
                        "title": work["title"],
                        "year": work["publication_year"],
                        "doi": work.get("doi"),
                        "location": work["primary_location"],
                        "abstract": " ".join(positions[position] for position in sorted(positions)),
                    }
                )
            return {
                "query": query,
                "url": str(response.url),
                "status": response.status_code,
                "records": records,
            }
        except (httpx.HTTPError, ValueError, KeyError) as error:
            return {"query": query, "url": url, "error": str(error)}


def search(out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "search-protocol.json",
        {
            "queries": QUERIES,
            "date_limit": "2026-10-12",
            "scope": "Targeted discovery, not a systematic review or proof of absence",
            "candidate": "Noise-scaled target-anchored asinh plotting; must check closest preprocessing sources, not claim generic nonlinear scaling is new",
        },
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda pair: retrieve(out, *pair), enumerate(QUERIES)))
    write_json(out / "search-results.json", responses)
    for item in responses:
        print(
            item["query"],
            item.get("error") or [(row["title"], row["year"]) for row in item["records"]],
            flush=True,
        )
    commit = "8ab8c16414eb2c1a861dfc3e76f458180035a879"
    url = f"https://raw.githubusercontent.com/ZLHe0/VLM4TS/{commit}/src/preprocessing/preprocess.py"
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        try:
            response = client.get(url)
            (out / "vlm4ts-preprocess.py").write_bytes(response.content)
            response.raise_for_status()
            write_json(
                out / "primary-source.json",
                {
                    "url": url,
                    "commit": commit,
                    "status": response.status_code,
                    "sha256": sha(out / "vlm4ts-preprocess.py"),
                },
            )
        except httpx.HTTPError as error:
            write_json(out / "primary-source.json", {"url": url, "error": str(error)})
    write_json(
        out / "freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    search(parser.parse_args().out.resolve())
