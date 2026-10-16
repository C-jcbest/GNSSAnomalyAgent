"""Archive one official recent search about numerical/visual time-series evidence."""

import argparse
import json
from pathlib import Path

from search_landslide_partial_review_literature import retrieve

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    out.mkdir(exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    query = 'all:"time series" AND (all:visual OR all:multimodal) AND (all:numerical OR all:textual)'
    result = retrieve("signed_visual_facts", query, out)
    (out / "search-ledger.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.get("error", [(row["title"], row["id"]) for row in result.get("records", [])]))
