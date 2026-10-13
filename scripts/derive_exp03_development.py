"""Develop EXP03 v2 on saved v1 validation responses, without new inference."""
from pathlib import Path

import numpy as np

from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.fusion import ranked_boundary_review
from gnss_research.metrics import aggregate, evaluate_record
from gnss_research.schema import observed_values

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT / "runs/exp03-small-v1"
    out = ROOT / "results/exp03-small-v2"
    out.mkdir(exist_ok=False)
    config = read_json(run / "registration.json")
    dataset = Path(config["dataset"])
    reports = {}
    for method, parent in (("ranked_boundary_fusion", "guided_fusion"),
                           ("ranked_no_local", "guided_no_local"),
                           ("ranked_no_scores", "guided_no_scores")):
        rows = []
        for record in read_json(dataset / "manifest.json")["records"]:
            if record["station_id"] not in config["station_ids"]:
                continue
            station = record["station_id"]
            work = run / "predictions" / parent / station
            observation = read_json(dataset / record["observation"])
            truth = read_json(dataset / record["truth"])
            result = ranked_boundary_review(observation, read_json(work / "result.json"),
                                            read_json(work / "candidates.json"))
            result["source_result_sha256"] = sha256(work / "result.json")
            write_json(out / method / station / "result.json", result)
            observed = np.isfinite(observed_values(observation)).all(axis=1)
            rows.append(evaluate_record(truth, result, observed, observation["background_end"]))
        reports[method] = aggregate(rows)
    write_json(out / "summary.json", {"scope": "used six validation stations, offline development",
        "request_count": 0, "source_hashes": source_hashes(), "methods": reports,
        "parent_registration_sha256": sha256(run / "registration.json")})
    print({m: {k: v["daily_micro"]["f1"] for k, v in values["types"].items()}
           for m, values in reports.items()})


if __name__ == "__main__":
    main()
