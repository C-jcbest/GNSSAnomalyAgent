"""Show the first test example of each morphology, without selecting by performance."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_research.baselines import contiguous_runs
from gnss_research.files import read_json, write_json
from gnss_research.schema import EVENT_TYPES, Prediction, event_masks, observed_values


def plot_cases(run: Path, out: Path):
    config = read_json(run / "registration.json")
    dataset = Path(config["dataset"])
    records = read_json(dataset / "manifest.json")["records"]
    selected = {}
    for record in records:
        if record["station_id"] not in config["station_ids"]:
            continue
        truth = read_json(dataset / record["truth"])
        kind = truth["scenario"]
        if kind in EVENT_TYPES and kind not in selected:
            selected[kind] = (record, truth)
    methods = [("truth", "Mechanism label"), ("rf_split_temporal", "Temporal RF"),
               ("visual_only", "Visual only"), ("ranked_boundary_fusion", "EXP03 fusion"),
               ("morphology_fusion", "EXP04 fusion")]
    fig, panels = plt.subplots(2, 4, figsize=(15, 5.7), height_ratios=[2.2, 1],
                              constrained_layout=True)
    receipts = []
    for column, kind in enumerate(EVENT_TYPES):
        record, truth = selected[kind]
        observation = read_json(dataset / record["observation"])
        values = observed_values(observation)
        observed = np.isfinite(values).all(axis=1)
        target = next(event for event in truth["events"] if event["kind"] == kind)
        left = max(observation["background_end"] + 1, target["start"] - 45)
        right = min(len(values) - 1, target["end"] + 45)
        raw, labels = panels[:, column]
        for axis, name, color in zip(range(3), ("N", "E", "U"), ("#267294", "#bc7833", "#8764a6")):
            raw.plot(np.arange(len(values)), values[:, axis], label=name, color=color, linewidth=.9)
        raw.set_title(f"{kind.capitalize()} | {record['station_id']}", fontsize=10)
        raw.set_ylabel("Displacement (mm)")
        raw.legend(fontsize=7)
        statuses = {}
        for index, (method, _) in enumerate(methods):
            if method == "truth":
                prediction = Prediction.model_validate({"events": truth["events"]})
                statuses[method] = "synthetic mechanism label"
            else:
                result = read_json(run / "predictions" / method / record["station_id"] / "result.json")
                prediction = Prediction.model_validate(result.get("prediction") or {"events": []})
                statuses[method] = result["status"]
            mask = event_masks(prediction, observed)[kind]
            position = len(methods) - 1 - index
            color = "#19846c" if method == "morphology_fusion" else "#60798c"
            if kind in ("spike", "step"):
                days = np.flatnonzero(mask)
                labels.scatter(days, np.full(len(days), position), marker="|", s=130, color=color)
            else:
                spans = [(start - .5, end - start + 1) for start, end in contiguous_runs(mask)]
                labels.broken_barh(spans, (position - .28, .56), facecolors=color)
        labels.set_yticks(range(len(methods)), [name for _, name in reversed(methods)], fontsize=7)
        labels.set_ylim(-.6, len(methods) - .4)
        labels.set_xlabel("Original daily index")
        for panel in (raw, labels):
            panel.set_xlim(left, right)
            panel.grid(alpha=.16)
            panel.spines[["top", "right"]].set_visible(False)
        receipts.append({"station_id": record["station_id"], "scenario": kind,
                         "display_indices": [left, right], "result_statuses": statuses})
    fig.suptitle("GNSS-VLM-EXP04 | First test example per morphology (not selected by score)", fontsize=12)
    out.mkdir(parents=True, exist_ok=False)
    fig.savefig(out / "cases.png", dpi=200)
    fig.savefig(out / "cases.svg")
    plt.close(fig)
    write_json(out / "selection.json", {"run": str(run.resolve()), "cases": receipts,
        "selection": "first test station per pure morphology in manifest; no performance selection",
        "display": "post-inference truth-centered display only; never used as model input; missing masked"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    plot_cases(args.run, args.out)
