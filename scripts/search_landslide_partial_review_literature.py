"""Archive a bounded official-source search for partial visual review and abstention."""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

QUERIES = {
    "recent_visual_agents": 'all:"time series" AND all:anomaly AND (all:agent OR all:vision OR all:multimodal)',
    "partial_visual_decisions": 'all:abstention AND (all:multimodal OR all:vision OR all:"time series")',
}


def retrieve(name, query, out):
    params = {"search_query": query, "start": 0, "max_results": 8,
              "sortBy": "submittedDate", "sortOrder": "descending"}
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    entry = {"name": name, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
             "reading_scope": "API metadata and abstracts only; no claim to have read the full papers"}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "GNSS-method-research/1.0"})
        with urllib.request.urlopen(request, timeout=35) as response:
            content = response.read()
            entry["http_status"] = response.status
        (out / (name + ".xml")).write_bytes(content)
        entry["sha256"] = hashlib.sha256(content).hexdigest()
        namespace = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for record in ET.fromstring(content).findall("a:entry", namespace):
            value = {key: record.findtext("a:" + key, default="", namespaces=namespace).strip()
                     for key in ("title", "id", "published", "summary")}
            value["authors"] = [author.findtext("a:name", namespaces=namespace)
                                for author in record.findall("a:author", namespace)]
            records.append(value)
        entry["records"] = records
    except Exception as error:
        entry["error"] = repr(error)
    return entry


def search(out):
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(retrieve, name, query, out) for name, query in QUERIES.items()]
        entries = [job.result() for job in jobs]
    (out / "search-ledger.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    for entry in entries:
        print(entry["name"], entry.get("error", [(row["title"], row["id"]) for row in entry.get("records", [])]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    search(args.out.resolve())
