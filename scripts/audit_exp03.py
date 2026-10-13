"""Collect request costs and the provenance needed to interpret EXP03 results."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from gnss_research.files import read_json, sha256, source_hashes, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    recoveries = {Path(read_json(path)["original_run"]).name: path.parent.name
                  for path in (ROOT / "runs").glob("exp03*/recovery.json")}
    for run in sorted((ROOT / "runs").glob("exp03*")):
        config = read_json(run / "registration.json")
        requests = 0
        uncertain_attempts = 0
        tokens = Counter()
        statuses = Counter()
        errors = Counter()
        for path in (run / "predictions").glob("*/*/result.json"):
            result = read_json(path)
            requests += result.get("request_count", 0)
            uncertain_attempts += result.get("unresolved_attempt_upper_bound", 0)
            if not result.get("request_count", 0):
                continue
            statuses[result["status"]] += 1
            for part in result.get("subrequests", [result]):
                if part.get("error_type"):
                    errors[part["error_type"]] += 1
                for key, value in (part.get("usage") or {}).items():
                    if isinstance(value, int):
                        tokens[key] += value
        rows.append({"run": run.name, "requests": requests,
                     "unresolved_attempt_upper_bound": uncertain_attempts,
                     "counted_in_total": run.name not in recoveries,
                     "recovered_in": recoveries.get(run.name), "planned": config["expected_requests"],
                     "paid_result_statuses": dict(statuses), "error_types": dict(errors),
                     "reported_usage": dict(tokens), "complete": (run / "execution-complete.json").exists(),
                     "source_matches_current": config["source_hashes"] == source_hashes(),
                     "method_protocol_matches_current": config["method_protocol_sha256"] == sha256(
                         ROOT / config["experiment"]["protocol_path"]),
                     "protocol_version": config["experiment"]["protocol_version"]})
    evidence = read_json(ROOT / "papers/evidence/manifest.json")
    mismatches = [entry["id"] for entry in evidence["sources"]
                  if "file" in entry and sha256(ROOT / "papers/evidence" / entry["file"])
                  != entry["sha256"]]
    write_json(ROOT / "results/exp03-two-seeds/audit.json", {
        "runs": rows, "requests_lower_bound": sum(row["requests"] for row in rows
                                                   if row["counted_in_total"]),
        "unresolved_attempt_upper_bound": sum(row["unresolved_attempt_upper_bound"] for row in rows
                                               if row["counted_in_total"]),
        "evidence_file_count": sum("file" in entry for entry in evidence["sources"]),
        "evidence_hash_mismatches": mismatches,
        "note": "v1 development used earlier source/protocol; formal v2 must match current"})
    print({"requests_lower_bound": sum(row["requests"] for row in rows if row["counted_in_total"]),
           "runs": rows,
           "evidence_mismatches": mismatches})


if __name__ == "__main__":
    main()
