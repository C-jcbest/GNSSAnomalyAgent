"""Generate fresh EXP04 datasets and disjoint pilot/remaining validation subsets."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from gnss_research.data import generate
from gnss_research.files import read_json, sha256, write_json


def validation_slice(source: Path, target: Path, start: int, end: int):
    manifest = read_json(source / "manifest.json")
    validation = [r for r in manifest["records"] if r["split"] == "validation"]
    selected = {r["station_id"] for r in validation[start:end]}
    records = [r for r in manifest["records"]
               if r["split"] != "validation" or r["station_id"] in selected]
    target.mkdir(parents=True, exist_ok=False)
    for record in records:
        for kind in ("observation", "truth"):
            destination = target / record[kind]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / record[kind], destination)
    manifest.update(records=records, split_counts={
        split: sum(r["split"] == split for r in records) for split in ("train", "validation", "test")},
        subset={"source_manifest_sha256": sha256(source / "manifest.json"),
                "rule": f"validation manifest positions [{start}:{end}], no label selection"})
    write_json(target / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    generate(args.out, args.seed, 24, 730, 96)
    if args.development:
        validation_slice(args.out, args.out.with_name(args.out.name + "-small"), 0, 6)
        validation_slice(args.out, args.out.with_name(args.out.name + "-remaining"), 6, 24)
    print({"dataset": str(args.out), "training": 96, "validation": 24, "test": 24})


if __name__ == "__main__":
    main()
