"""Save official searches for conditional and residual multimodal representations."""

from __future__ import annotations

import argparse
import html
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from fetch_landslide_morphology_sources import TextExtractor

from gnss_sim.artifacts import sha, write_json

QUERIES = ("multimodal residual", "multimodal orthogonal", "time series visual numerical fusion")


def search(out, pair):
    index, query = pair
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(
            "https://arxiv.org/search/",
            params=dict(
                query=query,
                searchtype="all",
                abstracts="show",
                order="-announced_date_first",
                size=50,
            ),
        )
        target = out / f"search-{index}.html"
        target.write_bytes(response.content)
        row = dict(
            query=query, url=str(response.url), status=response.status_code, sha256=sha(target)
        )
        if response.status_code != 200:
            return row
        parser = TextExtractor()
        parser.feed(response.text)
        (out / f"search-{index}.txt").write_text("\n".join(parser.parts), encoding="utf-8")
        cards = []
        for card in re.findall(r'<li class="arxiv-result">([\s\S]*?)</li>', response.text):
            identifier = re.search(r'href="https://arxiv.org/abs/([^"?]+)', card)
            title = re.search(r'<p class="title is-5 mathjax">([\s\S]*?)</p>', card)
            if identifier and title:
                cards.append(
                    dict(
                        arxiv=identifier.group(1),
                        title=html.unescape(re.sub("<[^>]+>", "", title.group(1))).strip(),
                    )
                )
        row["cards"] = cards
        return row


def retrieve(out):
    out.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda pair: search(out, pair), enumerate(QUERIES)))
    write_json(out / "search-ledger.json", rows)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(rows, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    retrieve(parser.parse_args().out.resolve())
