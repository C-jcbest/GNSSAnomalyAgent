"""Fetch two discovered primary articles and preserve retrieval failures."""

from __future__ import annotations

import argparse
import html
import re
import urllib.request
from pathlib import Path

from gnss_sim.artifacts import sha, write_json


def retrieve(out):
    out.mkdir(parents=True, exist_ok=False)
    receipts = []
    for identifier in ("2610.06180", "2606.08935"):
        for surface in ("html", "abs"):
            url = f"https://arxiv.org/{surface}/{identifier}"
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 GNSS-research"}
            )
            try:
                with urllib.request.urlopen(request, timeout=25) as response:
                    raw = response.read()
                source = out / f"{identifier}-{surface}.html"
                source.write_bytes(raw)
                title_match = re.search(r"<title>(.*?)</title>", raw.decode("utf-8"), re.S)
                title = "" if title_match is None else html.unescape(title_match.group(1))
                expected = "anlu" if identifier == "2610.06180" else "amplitude"
                if expected not in title.lower():
                    receipts.append(
                        dict(
                            url=url,
                            status="retrieved_content_mismatch",
                            source_sha256=sha(source),
                            document_title=title,
                            full_text=False,
                            read_scope="none; requested article identity not verified",
                        )
                    )
                    continue
                content = re.sub(
                    r"<(script|style)\b[^>]*>.*?</\1>", "", raw.decode("utf-8"), flags=re.S
                )
                text = html.unescape(re.sub(r"<[^>]+>", "\n", content))
                (out / f"{identifier}-{surface}.txt").write_text(text, encoding="utf-8")
                receipts.append(
                    dict(
                        url=url,
                        status="retrieved",
                        source_sha256=sha(source),
                        full_text=surface == "html",
                        read_scope="pending actual reading",
                    )
                )
                break
            except Exception as error:
                receipts.append(dict(url=url, status="failed", error=str(error), read_scope="none"))
    write_json(out / "receipts.json", receipts)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(receipts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    retrieve(parser.parse_args().out.resolve())
