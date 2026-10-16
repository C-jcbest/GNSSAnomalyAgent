"""Save public primary texts for the scaling candidate and new GNSS nearest work."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

import httpx

from gnss_sim.artifacts import sha, write_json

URLS = {
    "gnss-fm-abs": "https://arxiv.org/abs/2606.07725",
    "gnss-fm-html": "https://arxiv.org/html/2606.07725",
    "itf-tad-abs": "https://arxiv.org/abs/2408.14756",
    "itf-tad-html": "https://arxiv.org/html/2408.14756",
    "van-ad-readme": "https://github.com/PenyChen/VAN-AD",
    "vlm4ts-preprocess-github": "https://github.com/ZLHe0/VLM4TS/blob/8ab8c16414eb2c1a861dfc3e76f458180035a879/src/preprocessing/preprocess.py",
}


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def fetch(out, pair):
    name, url = pair
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        try:
            response = client.get(url)
            target = out / (name + ".html")
            target.write_bytes(response.content)
            response.raise_for_status()
            parser = TextExtractor()
            parser.feed(response.text)
            (out / (name + ".txt")).write_text("\n".join(parser.parts), encoding="utf-8")
            return {
                "name": name,
                "url": url,
                "status": response.status_code,
                "sha256": sha(target),
                "text_chars": sum(map(len, parser.parts)),
            }
        except httpx.HTTPError as error:
            return {"name": name, "url": url, "error": str(error)}


def retrieve(out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    with ThreadPoolExecutor(max_workers=4) as pool:
        ledger = list(pool.map(lambda pair: fetch(out, pair), URLS.items()))
    write_json(out / "fetch-ledger.json", ledger)
    write_json(
        out / "freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(ledger, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    retrieve(parser.parse_args().out.resolve())
