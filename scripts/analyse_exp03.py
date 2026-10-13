"""Point-error uncertainty and actual training-event support, after frozen evaluation."""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from report_exp03 import difference

from gnss_research.files import read_json, sha256, write_json
from gnss_research.schema import observed_values

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = [ROOT / "runs" / name for name in ("exp03-main-v2-recovered",
                                              "exp03-seed2026101208-v2")]
    groups = [read_json(run / "evaluation/per-station.json") for run in runs]
    contrasts = {kind: {baseline: difference(groups, "ranked_boundary_fusion", baseline, kind)
                       for baseline in ("rf_split_temporal", "boundary_fusion", "guided_fusion")}
                 for kind in ("spike", "step")}
    training = []
    for run in runs + [ROOT / "runs" / name for name in (
            "exp03-train24-seed2026101207-v2", "exp03-train24-seed2026101208-v2")]:
        config = read_json(run / "registration.json")
        dataset = Path(config["dataset"])
        mechanism = Counter()
        observable = Counter()
        for record in read_json(dataset / "manifest.json")["records"]:
            if record["split"] != "train":
                continue
            truth = read_json(dataset / record["truth"])
            observed = np.isfinite(observed_values(read_json(
                dataset / record["observation"]))).all(axis=1)
            for event in truth["events"]:
                mechanism[event["kind"]] += 1
                observable[event["kind"]] += int(observed[event["start"]:event["end"] + 1].any())
        training.append({"run": run.name,
                         "stations": len(read_json(run / "training.json")["training_stations"]),
                         "mechanism_events": dict(mechanism), "observable_events": dict(observable)})
    out = ROOT / "results/exp03-two-seeds"
    write_json(out / "point-analysis.json", {"contrasts": contrasts, "training": training,
        "scope": "after frozen testing; not validation-based tuning",
        "registration_hashes": [sha256(run / "registration.json") for run in runs]})
    for path in (Path(__file__), Path(__file__).with_name("report_exp03.py")):
        shutil.copyfile(path, out / "analysis-source" / ("point-analysis-" + path.name))
    print(contrasts)


if __name__ == "__main__":
    main()
