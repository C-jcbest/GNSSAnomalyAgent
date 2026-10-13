"""Verify completed EXP04 inputs, frozen code and same-response derivations."""
from __future__ import annotations

import argparse
from pathlib import Path

from gnss_research.experiment import verify_dataset
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.fusion import DERIVED_METHODS


def audit(runs, out):
    receipts = []
    for run in runs:
        config = read_json(run / "registration.json")
        assert (run / "execution-complete.json").exists(), run
        assert config["source_hashes"] == source_hashes(), run
        for name, digest in config["source_hashes"].items():
            assert sha256(run / "source" / name) == digest, (run, name)
        assert sha256(run / "source/method-protocol.md") == config["method_protocol_sha256"]
        assert sha256(Path(config["experiment"]["protocol_path"])) == config["method_protocol_sha256"]
        assert read_json(run / "source/experiment.json") == config["experiment"]
        dataset = Path(config["dataset"])
        verify_dataset(dataset)
        assert sha256(dataset / "manifest.json") == config["manifest_sha256"]
        methods = {}
        requests = 0
        for method in config["methods"]:
            failed = 0
            for station in config["station_ids"]:
                work = run / "predictions" / method / station
                result = read_json(work / "result.json")
                requests += result["request_count"]
                failed += result["status"] != "success"
                if method in DERIVED_METHODS:
                    parent = run / "predictions" / DERIVED_METHODS[method] / station / "result.json"
                    assert result["source_result_sha256"] == sha256(parent)
                if result["request_count"]:
                    assert result["retry_count"] == 0
                    assert result["requested_model"] == config["model"]
            methods[method] = {"stations": len(config["station_ids"]), "failures": failed}
        assert requests == config["expected_requests"], (run, requests)
        receipts.append({"run": run.name, "registration_sha256": sha256(run / "registration.json"),
                         "request_attempts": requests, "source_and_dataset_verified": True,
                         "methods": methods})
    write_json(out, {"runs": receipts, "request_attempts": sum(
        receipt["request_attempts"] for receipt in receipts),
        "scope": "completed runs, all failures retained, no retries; no paid replay"})
    print({"audit": str(out), "request_attempts": sum(r["request_attempts"] for r in receipts)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.runs, args.out)
