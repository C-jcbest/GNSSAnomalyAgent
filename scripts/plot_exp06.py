"""Publication figures from saved EXP06 counts; no model calls or metric tuning."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gnss_research.files import read_json
from gnss_research.schema import EVENT_TYPES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = read_json(args.report / "summary.json")
    methods = report["methods"]
    selected = {
        "rf_balanced": ("Balanced RF", "#8c99a6"),
        "rf_split_temporal": ("Temporal numerical", "#527c9f"),
        "morphology_fusion": ("MF-VLM (old full views)", "#b9865a"),
        "localized_fusion": ("Localized fusion (old full views)", "#ab839e"),
        "segment_numeric": ("Displacement structure, no VLM", "#68a89c"),
        "pair_focus_repair": ("SDR-VLM (frozen primary)", "#146b64"),
        "pair_raw_repair": ("SDR-VLM, no slow subtraction", "#95b94c"),
    }
    selected = {name: description for name, description in selected.items() if name in methods}
    figure, panels = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    width = .82 / len(selected)
    for position, (name, (label, color)) in enumerate(selected.items()):
        row = methods[name]
        offset = (position - (len(selected) - 1) / 2) * width
        panels[0].bar(np.arange(3) + offset, [row["types_daily_macro_f1"],
                      row["types_event_macro_f1"]["iou_0.5"], row["generic_daily_micro"]["f1"]],
                      width, color=color, label=label)
        panels[1].bar(np.arange(4) + offset,
                      [row["types"][kind]["daily_micro"]["f1"] for kind in EVENT_TYPES],
                      width, color=color)
    panels[0].set_xticks(np.arange(3), ["Type daily macro\n(primary)",
                                      "Type event macro\nIoU >= 0.5", "Generic daily micro"])
    panels[1].set_xticks(np.arange(4), ["Spike", "Step (strict)", "Transient", "Activity"])
    for panel in panels:
        panel.set_ylim(0, 1.04)
        panel.set_ylabel("F1, no point adjustment")
        panel.grid(axis="y", alpha=.2)
        panel.set_axisbelow(True)
        panel.spines[["top", "right"]].set_visible(False)
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=8)
    panels[0].set_title("Primary and secondary endpoints")
    panels[1].set_title("Detection by GNSS morphology")
    scope = "development validation" if report["splits"] == ["validation"] else "fresh frozen test"
    figure.suptitle(f"GNSS-VLM-EXP06 | {report['unique_stations']} {scope} stations")
    for suffix in ("png", "svg"):
        figure.savefig(args.report / ("comparison." + suffix), dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
