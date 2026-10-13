"""Scientific comparison figure from saved EXP03 evaluation, not inference input."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_research.files import read_json


def plot(out: Path):
    summary = read_json(out / "summary.json")
    methods = ("rf_split_temporal", "anchored_fusion", "guided_fusion", "ranked_boundary_fusion")
    labels = ("Numerical", "Locked fusion", "Free visual", "Ranked + revised")
    colors = ("#8093a2", "#af755d", "#b29952", "#207c79")
    kinds = ("spike", "step", "transient", "deformation")
    figure, panels = plt.subplots(1, 2, figsize=(12.5, 4.5), dpi=160)
    x = np.arange(len(kinds))
    width = .18
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        metrics = summary["combined"][method]["types"]
        values = [metrics[kind]["daily_micro"]["f1"] for kind in kinds]
        panels[0].bar(x + (index - 1.5) * width, values, width, label=label, color=color)
    panels[0].set_xticks(x, [k.title() for k in kinds])
    panels[0].set_ylim(0, 1)
    panels[0].set_ylabel("Daily micro-F1 (no point adjustment)")
    panels[0].set_title("48 simulated holdout stations, 96 training stations per seed")
    panels[0].legend(fontsize=8, loc="upper left")
    errors = read_json(out / "activity-errors.json")
    keys = ("missed_activity_days", "false_activity_days_before_onset",
            "false_activity_days_after_arrest")
    x = np.arange(3)
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = [sum(row[key] for row in errors if row["method"] == method) for key in keys]
        bars = panels[1].bar(x + (index - 1.5) * width, values, width, label=label, color=color)
        panels[1].bar_label(bars, fontsize=7, padding=2)
    panels[1].set_xticks(x, ("Missed active", "Early false", "After-arrest false"))
    panels[1].set_ylabel("Observed days in activity-bearing stations")
    panels[1].set_title("Activity-boundary errors (missing days excluded)")
    for panel in panels:
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="y", alpha=.2)
        panel.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(out / "comparison.png")
    figure.savefig(out / "comparison.svg")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    plot(args.report)
