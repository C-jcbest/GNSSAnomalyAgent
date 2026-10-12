"""Verify completed artifacts without network requests or response replacement."""
from __future__ import annotations

import argparse
from pathlib import Path

from gnss_research.experiment import verify_dataset
from gnss_research.files import read_json, sha256
from gnss_research.upstream import ROOT


def audit(run: Path, repeat: Path | None, upstream_repeat: Path | None = None,
          dataset_override: Path | None = None) -> dict:
    config = read_json(run / "registration.json")
    dataset = dataset_override or Path(config["dataset"])
    verify_dataset(dataset)
    if not (run / "execution-complete.json").exists():
        raise ValueError("incomplete run")
    if sha256(dataset / "manifest.json") != config["manifest_sha256"]:
        raise ValueError("manifest differs from registration")
    for name, digest in config["source_hashes"].items():
        if sha256(run / "source" / name) != digest:
            raise ValueError(f"frozen source changed: {name}")
    if sha256(run / "source/protocol.md") != config["protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    requests = 0
    for method in config["methods"]:
        for station in config["station_ids"]:
            work = run / "predictions" / method / station
            result = read_json(work / "result.json")
            requests += result.get("request_count", 0)
            for receipt in result.get("subrequests", [result]):
                if receipt.get("retry_count", 0) != 0:
                    raise ValueError("unexpected automatic retry")
            if (work / "input-receipt.json").exists():
                receipt = read_json(work / "input-receipt.json")
                if sha256(work / "input.png") != receipt["plot_sha256"]:
                    raise ValueError("input plot changed")
                if sha256(work / "request.json") != receipt["request_sha256"]:
                    raise ValueError("visual request changed")
    if requests != config["expected_requests"]:
        raise ValueError("actual request count differs from registration")
    rf_stations = 0
    identical_author_requests = 0
    if repeat is not None:
        second = read_json(repeat / "registration.json")
        for key in ("station_ids", "manifest_sha256", "seed", "model", "source_hashes"):
            if config[key] != second[key]:
                raise ValueError(f"repeat differs: {key}")
        audit(repeat, None, dataset_override=dataset)
        for station in config["station_ids"]:
            first = read_json(run / "predictions/random_forest" / station / "result.json")
            replayed = read_json(repeat / "predictions/random_forest" / station / "result.json")
            for key in ("prediction", "generic_mask", "generic_score", "type_scores"):
                if first[key] != replayed[key]:
                    raise ValueError(f"RF replay differs: {station} {key}")
            rf_stations += 1
    if upstream_repeat is not None:
        second = read_json(upstream_repeat / "registration.json")
        for key in ("station_ids", "manifest_sha256", "seed", "model"):
            if config[key] != second[key]:
                raise ValueError(f"author repeat differs: {key}")
        audit(upstream_repeat, None, dataset_override=dataset)
        for station in config["station_ids"]:
            for axis in "NEU":
                for task in ("point", "range"):
                    relative = Path("predictions/visualtime_gnss") / station / f"{axis}-{task}/request.json"
                    if sha256(run / relative) != sha256(upstream_repeat / relative):
                        raise ValueError(f"author repeat request changed: {station} {axis}-{task}")
                    identical_author_requests += 1
    evidence = ROOT / "papers/evidence"
    sources = read_json(evidence / "manifest.json")["sources"]
    verified = 0
    for source in sources:
        if source.get("file") and source.get("sha256"):
            if sha256(evidence / source["file"]) != source["sha256"]:
                raise ValueError(f"paper evidence changed: {source['id']}")
            verified += 1
    return {"run": str(run.resolve()), "actual_requests": requests,
            "frozen_sources_and_inputs": "verified", "RF_identical_stations": rf_stations,
            "identical_author_repeat_requests": identical_author_requests,
            "evidence_hashes_verified": verified}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--upstream-repeat", type=Path)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    import json

    print(json.dumps(audit(args.run, args.repeat, args.upstream_repeat, args.dataset), indent=2))
