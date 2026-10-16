"""Retain targeted search results and source bytes for anchored visual learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
from run_landslide_comparison import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    queries = ["multimodal residual correction gradient boosted trees",
               "time series anomaly detection numerical visual residual",
               "GNSS anomaly multimodal foundation model"]
    records, retrieval = [], []
    with httpx.Client(timeout=35, follow_redirects=True) as client:
        for index, query in enumerate(queries):
            try:
                response = client.get("https://api.openalex.org/works", params={
                    "search": query, "per-page": 6, "filter": "from_publication_date:2024-01-01",
                    "sort": "relevance_score:desc"})
                (args.out / f"search-{index}.json").write_bytes(response.content)
                response.raise_for_status()
                for work in response.json()["results"]:
                    positions = {position: word for word, locations in (work.get("abstract_inverted_index") or {}).items()
                                 for position in locations}
                    records.append({"query": query, "title": work["title"], "year": work["publication_year"],
                                    "doi": work.get("doi"), "location": work["primary_location"],
                                    "abstract": " ".join(positions[position] for position in sorted(positions))})
            except (httpx.HTTPError, KeyError, ValueError) as error:
                retrieval.append({"query": query, "error": str(error)})
        for name, url in {
            "stacking-metadata": "https://api.openalex.org/works/https://doi.org/10.1016/S0893-6080(05)80023-1",
            "deepgbm-metadata": "https://api.openalex.org/works/https://doi.org/10.1145/3292500.3330858",
            "deepgbm-github": "https://api.github.com/repos/motefly/DeepGBM/commits/master",
        }.items():
            try:
                response = client.get(url)
                (args.out / f"{name}.json").write_bytes(response.content)
                response.raise_for_status()
                retrieval.append({"name": name, "url": url, "status": response.status_code})
            except httpx.HTTPError as error:
                retrieval.append({"name": name, "url": url, "error": str(error)})
    save_json(args.out / "records.json", records)
    save_json(args.out / "retrieval.json", retrieval)
    print([(row["title"], row["year"], row["doi"]) for row in records], flush=True)
    print(retrieval, flush=True)
    save_json(args.out / "search-manifest.json", {file.name: sha(file) for file in args.out.iterdir() if file.is_file()})


if __name__ == "__main__":
    main()
