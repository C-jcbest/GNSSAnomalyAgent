"""Nested training-size datasets with byte-identical validation/test records."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from gnss_research.data import generate
from gnss_research.files import read_json, sha256, write_json


def subset(source: Path, target: Path, train_count: int, validation_count: int):
    manifest = read_json(source / "manifest.json")
    counts = {"train": 0, "validation": 0, "test": 0}
    records = []
    limits = {"train": train_count, "validation": validation_count, "test": 24}
    target.mkdir(parents=True, exist_ok=False)
    for record in manifest["records"]:
        split = record["split"]
        if counts[split] >= limits[split]:
            continue
        counts[split] += 1
        records.append(record)
        for kind in ("observation", "truth"):
            destination = target / record[kind]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / record[kind], destination)
    manifest.update(records=records, split_counts=counts,
                    subset={"source_manifest_sha256": sha256(source / "manifest.json"),
                            "rule": "first training/validation stations in manifest; no label selection"})
    write_json(target / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args.out, args.seed, 24, 730, 96)
    subset(args.out, args.out.with_name(args.out.name + "-train24"), 24, 24)
    subset(args.out, args.out.with_name(args.out.name + "-small"), 96, 6)
    print({"dataset": str(args.out), "training": [24, 96], "validation": 24, "test": 24,
           "small_validation": 6})


if __name__ == "__main__":
    main()
