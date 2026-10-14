"""Select a development pilot by observed candidate ambiguity, never by truth."""
from __future__ import annotations

import shutil
from pathlib import Path

from gnss_research.files import read_json, write_json
from gnss_research.segmented_review import review_candidates


def main():
    dataset = Path("data/exp06-development")
    manifest = read_json(dataset / "manifest.json")
    ranked = []
    for record in manifest["records"]:
        if record["split"] != "validation":
            continue
        numerical = read_json(Path("results/exp06-segmentation-diagnosis-v2")
                              / (record["station_id"] + ".json"))
        count = sum(proposal["status"] == "uncertain"
                    for proposal in review_candidates(numerical))
        if count:
            ranked.append((-count, record["station_id"]))
    pilot_ids = [station for _, station in sorted(ranked)[:6]]
    for name, pilot in (("pilot", True), ("remaining", False)):
        out = dataset.with_name("exp06-focus-" + name)
        out.mkdir(parents=True, exist_ok=False)
        records = [record for record in manifest["records"] if record["split"] == "train"
                   or (record["split"] == "validation"
                       and (record["station_id"] in pilot_ids) == pilot)]
        for record in records:
            for kind in ("observation", "truth"):
                target = out / record[kind]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dataset / record[kind], target)
        write_json(out / "manifest.json", {**manifest, "records": records,
                   "split_counts": {"train": 96, "validation": len(records) - 96, "test": 0},
                   "selection": "observational alternative pair count; no truth used",
                   "pilot_ids": pilot_ids})
    print({"pilot_ids": pilot_ids, "pilot_requests": len(pilot_ids),
           "remaining_requests": 24 - len(pilot_ids)})


if __name__ == "__main__":
    main()
