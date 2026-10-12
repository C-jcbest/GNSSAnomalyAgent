"""Truth-free plots and a single mask for displayed and exported phase labels."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_research.baselines import contiguous_runs
from gnss_research.schema import Prediction, event_masks, observed_values, phase_labels


def render_observation(observation: dict, path: Path, axes: tuple[int, ...] = (0, 1, 2),
                       show_background: bool = True, prediction: Prediction | None = None) -> None:
    values = observed_values(observation)
    observed = np.isfinite(values).all(axis=1)
    labels = phase_labels(prediction, observed) if prediction is not None else [None] * len(values)
    masks = event_masks(prediction, observed) if prediction is not None else {}
    figure, panels = plt.subplots(len(axes), 1, figsize=(12, 2.5 * len(axes)), sharex=True,
                                  squeeze=False, dpi=120)
    colors = {"acceleration": "#f2b980", "steady": "#8cbec5", "deceleration": "#b9afd8"}
    event_colors = {"spike": "#b74f39", "step": "#755197", "transient": "#d49a32",
                    "deformation": "#3b9a88"}
    for panel, axis_index in zip(panels[:, 0], axes):
        panel.plot(np.arange(len(values)), values[:, axis_index], color="#23495b", linewidth=0.8)
        panel.set_ylabel(f"{('N', 'E', 'U')[axis_index]} (mm)")
        panel.grid(alpha=0.2)
        if show_background:
            panel.axvspan(0, observation["background_end"], color="#bfc9c6", alpha=0.25,
                          label="historical reference")
        if prediction is not None:
            for kind, mask in masks.items():
                color = event_colors[kind]
                if kind in ("spike", "step"):
                    days = np.flatnonzero(mask)
                    if len(days):
                        panel.scatter(days, values[days, axis_index], color=color,
                                      marker="x", s=24, zorder=3, label=kind)
                else:
                    for index, (start, end) in enumerate(contiguous_runs(mask)):
                        panel.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.20,
                                      label=kind if index == 0 else None)
            for day, label in enumerate(labels):
                if label is not None:
                    panel.axvspan(day - 0.5, day + 0.5, color=colors[label], alpha=0.35)
            if any(mask.any() for mask in masks.values()) or show_background:
                panel.legend(loc="best", fontsize=8)
    panels[-1, 0].set_xlim(0, len(values) - 1)
    panels[-1, 0].set_xlabel("Original daily index (gaps retained)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def export_labels(observation: dict, prediction: Prediction, path: Path) -> None:
    from gnss_research.schema import EVENT_TYPES, event_masks

    observed = np.isfinite(observed_values(observation)).all(axis=1)
    masks = event_masks(prediction, observed)
    phases = phase_labels(prediction, observed)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "date", "observed", *EVENT_TYPES, "phase"])
        for day, date in enumerate(observation["dates"]):
            labels = [int(masks[kind][day]) if observed[day] else "" for kind in EVENT_TYPES]
            writer.writerow([day, date, int(observed[day]), *labels, phases[day] or ""])
