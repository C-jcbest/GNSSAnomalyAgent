"""Save a bounded literature search for coordinate-sensitive chart understanding."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
from run_landslide_comparison import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    queries = [
        "chart understanding vision language model robustness axis",
        "time series anomaly detection vision language model augmentation",
        "vision language model chart transformation consistency",
    ]
    results = []
    with httpx.Client(timeout=45, follow_redirects=True) as client:
        for index, query in enumerate(queries):
            url = "https://api.openalex.org/works"
            try:
                response = client.get(url, params={"search": query, "per-page": 6})
                (args.out / f"search-{index}.json").write_bytes(response.content)
                response.raise_for_status()
                for work in response.json()["results"]:
                    abstract_index = work.get("abstract_inverted_index") or {}
                    positions = {position: word for word, locations in abstract_index.items()
                                 for position in locations}
                    abstract = " ".join(positions[position] for position in sorted(positions))
                    results.append({"query": query, "title": work["title"],
                                    "year": work["publication_year"], "doi": work.get("doi"),
                                    "url": work["primary_location"].get("landing_page_url"),
                                    "abstract": abstract})
            except (httpx.HTTPError, ValueError, KeyError) as error:
                results.append({"query": query, "error_type": type(error).__name__})
    save_json(args.out / "records.json", results)
    save_json(args.out / "manifest.json", {
        "date": "2026-10-10", "scope": "Three targeted searches, duplicates retained; not a systematic review",
        "files": {file.name: sha(file) for file in args.out.iterdir() if file.is_file()},
    })
    print(results, flush=True)


if __name__ == "__main__":
    main()
