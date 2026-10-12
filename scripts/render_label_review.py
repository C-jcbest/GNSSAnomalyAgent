"""Render fixed first-per-scenario cases from saved predictions, never model inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from gnss_research.files import read_json, sha256, write_json
from gnss_research.plots import export_labels, render_observation
from gnss_research.schema import Prediction


def render(run: Path, out: Path) -> None:
    config = read_json(run / "registration.json")
    dataset = Path(config["dataset"])
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    # Fixed balanced order from the registered test set, not selected by score.
    for station in config["station_ids"][:6]:
        observation = read_json(dataset / "observations" / f"{station}.json")
        truth = read_json(dataset / "truth" / f"{station}.json")
        for method in ("mechanism_truth", "random_forest", "visual_only", "fusion"):
            if method == "mechanism_truth":
                prediction = Prediction.model_validate({"events": truth["events"]})
                # Display the current activity endpoint, not unpredicted internal phases.
                for event in prediction.events:
                    event.phases = []
                status = "simulation mechanism, not expert annotation"
            else:
                result = read_json(run / "predictions" / method / station / "result.json")
                status = result["status"]
                prediction = Prediction.model_validate(result["prediction"]) \
                    if status == "success" else None
            directory = out / station / method
            directory.mkdir(parents=True)
            plot = directory / ("activity.png" if prediction is not None else "failed_no_prediction.png")
            render_observation(observation, plot, prediction=prediction)
            if prediction is not None:
                export_labels(observation, prediction, directory / "labels.csv")
            cases.append({"station": station, "scenario": truth["scenario"], "method": method,
                          "status": status, "plot": str(plot.relative_to(out)),
                          "plot_sha256": sha256(plot)})
    write_json(out / "manifest.json", {"selection": "first registered case of each scenario",
               "source_run": str(run.resolve()), "model_requests": 0, "cases": cases,
               "note": "display-only correction; original inference artifacts remain unchanged"})
    print(out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.run, args.out)
