"""Prepare fresh EXP05 development data and diagnose step localization, without vision."""
from __future__ import annotations

import argparse
from pathlib import Path

from prepare_exp04 import validation_slice

from gnss_research.baselines import fit_baselines, predict_numerical
from gnss_research.data import generate
from gnss_research.features import numerical_evidence
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.guided import inspection_windows
from gnss_research.localization import locate_step


def diagnose(dataset: Path, out: Path, seed: int) -> dict:
    manifest = read_json(dataset / "manifest.json")
    models, training = fit_baselines(dataset, manifest["records"], seed)
    stations = []
    events = []
    for record in manifest["records"]:
        if record["split"] != "validation":
            continue
        observation = read_json(dataset / record["observation"])
        evidence = numerical_evidence(observation)
        numerical = predict_numerical(models["rf_split_temporal"], "rf_split_temporal",
                                      observation, evidence)
        windows = [window for window in inspection_windows(observation, numerical)
                   if window["inspection_kind"] == "step"]
        fits = [locate_step(evidence, window) for window in windows]
        stations.append({"station_id": record["station_id"], "windows": windows, "fits": fits})
        # Labels are read only AFTER candidate construction, solely for this diagnostic.
        truth = read_json(dataset / record["truth"])
        for event in truth["events"]:
            day = event["start"]
            if event["kind"] != "step" or not evidence.observed[day]:
                continue
            centers = [window["peak_index"] for window in windows]
            dates = [fit["index"] for fit in fits if fit["index"] is not None]
            events.append({"station_id": record["station_id"], "truth_day": day,
                           "window_covers": any(w["start"] <= day <= w["end"] for w in windows),
                           "rf_exact": day in centers,
                           "rf_within3": any(abs(day - center) <= 3 for center in centers),
                           "fit_exact": day in dates,
                           "fit_within3": any(abs(day - index) <= 3 for index in dates)})
    counts = {key: sum(event[key] for event in events) for key in
              ("window_covers", "rf_exact", "rf_within3", "fit_exact", "fit_within3")}
    result = {"experiment_id": "GNSS-VLM-EXP05", "split": "validation",
              "seed": seed, "manifest_sha256": sha256(dataset / "manifest.json"),
              "source_hashes": source_hashes(), "observable_truth_steps": len(events),
              "counts": counts, "stations": stations, "events": events,
              "interpretation": "Candidate localization only; no visual confirmation, "
              "no final detection result, no test data inspected"}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "step-localization-diagnosis.json", result)
    write_json(out / "training.json", training)
    return {"observable_truth_steps": len(events), "counts": counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/exp05-development"))
    parser.add_argument("--out", type=Path, default=Path("results/exp05-numeric-diagnosis"))
    args = parser.parse_args()
    seed = 2026101401
    generate(args.dataset, seed, 24, 730, 96)
    validation_slice(args.dataset, args.dataset.with_name(args.dataset.name + "-small"), 0, 6)
    validation_slice(args.dataset, args.dataset.with_name(args.dataset.name + "-remaining"), 6, 24)
    print(diagnose(args.dataset, args.out, seed))


if __name__ == "__main__":
    main()
