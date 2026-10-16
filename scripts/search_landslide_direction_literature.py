"""Save a small official-source search, including failures and actual reading scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def search(out):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    queries = [
        ("arxiv", "https://export.arxiv.org/api/query", {
            "search_query": 'all:"time series" AND all:anomaly AND (all:vision OR all:multimodal)',
            "start": 0, "max_results": 8, "sortBy": "submittedDate", "sortOrder": "descending"}),
        ("crossref", "https://api.crossref.org/works", {
            "query.bibliographic": "GNSS landslide displacement vector direction monitoring", "rows": 8}),
    ]
    ledger = []
    for provider, base, params in queries:
        url = base + "?" + urllib.parse.urlencode(params)
        entry = {"provider": provider, "url": url, "params": params,
                 "retrieved_utc": datetime.now(timezone.utc).isoformat()}
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GNSS-research-source-check/1.0"})
            with urllib.request.urlopen(request, timeout=35) as response:
                content = response.read()
                entry["http_status"] = response.status
            (out / f"{provider}.response").write_bytes(content)
            entry["sha256"] = hashlib.sha256(content).hexdigest()
            if provider == "arxiv":
                root = ET.fromstring(content)
                namespace = {"a": "http://www.w3.org/2005/Atom"}
                records = []
                for record in root.findall("a:entry", namespace):
                    records.append({key: record.findtext("a:" + key, namespaces=namespace).strip()
                                    for key in ("title", "id", "published", "summary")})
            else:
                records = [{key: record.get(key) for key in ("title", "DOI", "published", "abstract")}
                           for record in json.loads(content)["message"]["items"]]
            entry["records"] = records
            print(provider, [(row["title"], row.get("id", row.get("DOI"))) for row in records], flush=True)
        except Exception as error:
            entry["error"] = repr(error)
            print(provider, type(error).__name__, flush=True)
        ledger.append(entry)
    (out / "search-ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    search(parser.parse_args().out)
