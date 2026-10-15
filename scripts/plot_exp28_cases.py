"""Display post-test positive and negative cases without changing predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnss_research.files import read_json, write_json
from gnss_research.schema import observed_values

CASES = {
    "e0095": "Image recovers the full transient across an independent step",
    "e0311": "Image recovers a transient missed by the matched text control",
    "e0119": "Failure: image selects the interval after the true transient",
    "e0035": "Metric caution: nearby image interval has zero true-positive days",
}


def plot_cases(visual: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    registration = read_json(visual / "remaining-registration.json")
    dataset = Path(registration["dataset"])
    baseline = Path(registration["baseline"])
    rows = {}
    for station, title in CASES.items():
        observation = read_json(dataset / "observations" / f"{station}.json")
        truth = read_json(dataset / "truth" / f"{station}.json")
        sources = {
            "Truth": truth["events"],
            "CEV-VL image": read_json(visual / "final/ce_vl_image" / station / "result.json")["prediction"]["events"],
            "Matched VL text": read_json(visual / "final/ce_vl_text" / station / "result.json")["prediction"]["events"],
            "Vector nearest": read_json(visual / "final/vector_nearest" / station / "result.json")["prediction"]["events"],
            "Numeric base": read_json(baseline / "predictions/segment_numeric" / station / "result.json")["prediction"]["events"],
        }
        spans = {
            name: [(event["start"], event["end"]) for event in events if event["kind"] == "transient"]
            for name, events in sources.items()
        }
        endpoints = [day for ranges in spans.values() for pair in ranges for day in pair]
        left, right = max(365, min(endpoints) - 25), min(730, max(endpoints) + 26)
        values = observed_values(observation)
        figure, panels = plt.subplots(4, 1, figsize=(12, 8.5), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1, 1, 1.25]})
        for axis, panel in enumerate(panels[:3]):
            panel.plot(np.arange(left, right), values[left:right, axis], ".-",
                       color="#276a85", linewidth=.85, markersize=3)
            for start, end in spans["Truth"]:
                panel.axvspan(start - .5, end + .5, color="#8ca98c", alpha=.18)
            panel.set_ylabel(f"{'NEU'[axis]} (mm)")
            panel.grid(alpha=.18)
        colors = ("#788b72", "#256d91", "#d29935", "#9361a4", "#777777")
        for index, ((name, ranges), color) in enumerate(zip(spans.items(), colors)):
            for start, end in ranges:
                panels[3].broken_barh([(start - .5, end - start + 1)], (index - .3, .6), facecolors=color)
        panels[3].set_yticks(range(len(spans)), list(spans))
        panels[3].invert_yaxis()
        panels[3].set_xlabel("Original daily index; shaded observations = true transient; gaps unfilled")
        panels[3].grid(axis="x", alpha=.18)
        panels[3].set_xlim(left, right - 1)
        figure.suptitle(f"{station}: {title}", fontsize=12)
        figure.tight_layout()
        figure.savefig(output / f"{station}.png", dpi=150)
        plt.close(figure)
        rows[station] = {"title": title, "transient_intervals": spans,
                         "truth_events": truth["events"], "window": [left, right],
                         "selection": "Post-test illustrative cases, including failures; no method tuning"}
    write_json(output / "cases.json", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot_cases(args.visual, args.out)
