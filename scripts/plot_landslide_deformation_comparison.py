"""Export daily and event F1 side by side for the exposed diverse validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(run, out):
    out.mkdir(parents=True, exist_ok=False)
    rows = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    table = {row["method"]: row for row in rows}
    methods = {
        "visual_numeric_plain": "Visual + numeric, plain",
        "visual_numeric_augmentation": "Same augmentation, no pairing",
        "visual_numeric_pair": "PP-VN pairing",
        "visual_numeric_shuffled_pairs": "Shuffled pairing",
        "numeric_pair": "Numeric pairing",
        "numeric_augmentation": "Numeric augmentation",
        "xgb_raw84_augmentation": "XGBoost raw84 + augmentation",
        "xgb_chart384_augmentation": "XGBoost chart384 + augmentation",
        "dq_xgb_raw84": "XGBoost raw84, original data",
    }
    positions = np.arange(len(methods))
    colors = ["#b95b45" if name == "visual_numeric_pair" else "#52869a" for name in methods]
    figure, panels = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    for panel, metric, title in zip(
        panels, ("f1", "event_f1_iou03"), ("Daily activity F1", "Event F1 (IoU 0.3)"), strict=True
    ):
        values = [table[name][metric] for name in methods]
        panel.barh(positions, values, color=colors)
        panel.set_xlim(0, 1)
        panel.set_title(title)
        panel.grid(axis="x", alpha=0.2)
        panel.set_axisbelow(True)
        for position, value in zip(positions, values, strict=True):
            panel.text(value + 0.012, position, f"{value:.4f}", va="center", fontsize=9)
    panels[0].set_yticks(positions, list(methods.values()))
    panels[0].invert_yaxis()
    figure.suptitle("Diverse GNSS validation: 24 records used for model selection")
    figure.text(
        0.01,
        0.01,
        "Reserved evaluation not opened; bars are frozen selected configurations, not independent confirmation.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(out / "daily-event-comparison.png", dpi=180)
    figure.savefig(out / "daily-event-comparison.pdf")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.run.resolve(), args.out.resolve())
