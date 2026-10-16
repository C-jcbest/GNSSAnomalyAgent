"""Save bounded public arXiv topic discovery without bibliographic resolver APIs."""

from __future__ import annotations

import argparse
import html
import re
import urllib.parse
import urllib.request
from pathlib import Path

from gnss_sim.artifacts import sha, write_json


def search(out):
    out.mkdir(parents=True, exist_ok=False)
    receipts = []
    for index, query in enumerate(
        ("time series counterfactual normalization", "time series visualization anomaly")
    ):
        url = "https://arxiv.org/search/?" + urllib.parse.urlencode(
            dict(
                query=query,
                searchtype="all",
                abstracts="show",
                order="-announced_date_first",
                size=25,
            )
        )
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 GNSS-research"})
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                raw = response.read()
            source = out / f"query-{index}.html"
            source.write_bytes(raw)
            content = raw.decode("utf-8")
            rows = []
            # The raw page is retained. Discovery cards do not establish that
            # an article's full text or implementation has been read.
            for result in re.findall(r'<li class="arxiv-result">(.*?)</li>', content, re.S):
                title = re.search(r'<p class="title[^\"]*">(.*?)</p>', result, re.S)
                link = re.search(r'href="(https://arxiv.org/abs/[^\"]+)"', result)
                if title is None or link is None:
                    continue
                rows.append(
                    dict(
                        title=" ".join(
                            html.unescape(re.sub(r"<[^>]+>", " ", title.group(1))).split()
                        ),
                        url=link.group(1),
                    )
                )
            write_json(out / f"query-{index}-records.json", rows)
            receipts.append(
                dict(
                    query=query,
                    url=url,
                    status="retrieved",
                    source_sha256=sha(source),
                    discovered=len(rows),
                    papers_read=0,
                )
            )
        except Exception as error:
            receipts.append(
                dict(query=query, url=url, status="failed", error=str(error), papers_read=0)
            )
    write_json(out / "receipts.json", receipts)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(receipts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    search(parser.parse_args().out.resolve())
