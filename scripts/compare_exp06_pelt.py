"""Unmodified author-library examples versus GNSS adaptation on an edge-only task.

The source library emits change points, not four GNSS types or daily activity masks.
Its edge scores therefore stay separate from the primary four-type daily F1 table.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import ruptures as rpt

from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import counts_metrics, match_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reports = []
    for run in args.runs:
        config = read_json(run / "registration.json")
        dataset = Path(config["dataset"])
        for record in read_json(dataset / "manifest.json")["records"]:
            if record["station_id"] not in config["station_ids"]:
                continue
            obs = read_json(dataset / record["observation"])
            evidence = numerical_evidence(obs)
            eligible = evidence.observed & (np.arange(len(evidence.values)) >= evidence.background_end - 89)
            time = np.flatnonzero(eligible)
            signal = evidence.residual[eligible] / evidence.noise
            observed = evidence.observed.copy()
            observed[:evidence.background_end + 1] = False
            truth = read_json(dataset / record["truth"])
            target = set()
            for event in truth["events"]:
                if event["kind"] == "spike":
                    continue
                target.add(event["start"])
                if (event["kind"] in ("transient", "deformation")
                        and event["end"] + 1 < len(evidence.values)):
                    target.add(event["end"] + 1)
            target_spans = [(index, index) for index in sorted(target)
                            if observed[index] and observed[index - 1]]
            predictions = {}
            for name, algorithm in (("author_api_defaults_l2_pen10", rpt.Pelt()),
                                    ("author_readme_rbf_pen10", rpt.Pelt(model="rbf"))):
                ends = algorithm.fit(signal).predict(pen=10)
                predictions[name] = [int(time[end]) for end in ends[:-1]
                                     if time[end] > evidence.background_end]
            numeric = read_json(run / "predictions/segment_numeric" / record["station_id"] / "result.json")
            edges = set()
            for event in numeric["prediction"]["events"]:
                if event["kind"] == "spike":
                    continue
                edges.add(event["start"])
                if event["kind"] in ("transient", "deformation") and event["end"] + 1 < len(observed):
                    edges.add(event["end"] + 1)
            predictions["gnss_displacement_structure_adapter"] = sorted(edges)
            for name, dates in predictions.items():
                spans = [(index, index) for index in dates]
                reports.append({"method": name, "dataset_seed": read_json(dataset / "manifest.json")["seed"],
                                "station_id": record["station_id"], "truth_edges": sorted(target),
                                "predicted_edges": dates,
                                "strict": match_events(target_spans, spans, observed, .5),
                                "tolerance3": match_events(target_spans, spans, observed, .5, 3)})
    methods = {}
    for method in sorted({row["method"] for row in reports}):
        rows = [row for row in reports if row["method"] == method]
        methods[method] = {endpoint: counts_metrics(*(sum(row[endpoint][key] for row in rows)
                                                    for key in ("tp", "fp", "fn")))
                           for endpoint in ("strict", "tolerance3")}
    write_json(args.out / "summary.json", {
        "experiment_id": "GNSS-VLM-EXP06", "methods": methods,
        "scope": "change-point edges only, excluding spikes; not the primary typed daily anomaly task",
        "library_version": rpt.__version__, "pelt_source_sha256": sha256(Path(inspect.getfile(rpt.Pelt))),
        "original_parameters": {"Pelt defaults": "model=l2, min_size=2, jump=5",
                                "README example": "model=rbf, pen=10, other constructor defaults",
                                "penalty_note": "pen=10 is the README example value, not an automatic GNSS default"},
        "shared_input_adapter": "normal-prefix residual MAD standardization; missing observations excluded with original index mapping; normal tail=90 days",
        "adapted_parameters": numeric["parameters"], "new_model_requests": 0})
    write_json(args.out / "per-station.json", reports)
    print({name: {key: value["f1"] for key, value in metrics.items()}
           for name, metrics in methods.items()})


if __name__ == "__main__":
    main()
