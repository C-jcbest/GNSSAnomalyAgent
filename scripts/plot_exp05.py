"""Plot the saved EXP05 comparison; do not infer new scores from figures."""
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
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = read_json(args.report / "summary.json")
    methods = report["methods"]
    selected = {
        "rf_balanced": ("Balanced RF", "#8696a6"),
        "rf_split_temporal": ("Temporal numerical", "#3a71a1"),
        "visual_only": ("Visual only", "#d1a04e"),
        "morphology_fusion": ("EXP04 MF-VLM, 8 images", "#a66e98"),
        "morphology_no_local": ("Single image, original dates", "#73a59a"),
        "localized_global": ("SG-LocFusion, single image", "#187d68"),
    }
    selected = {method: value for method, value in selected.items() if method in methods}
    figure, panels = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    width = .8 / len(selected)
    for index, (method, (label, color)) in enumerate(selected.items()):
        values = methods[method]
        offset = (index - (len(selected) - 1) / 2) * width
        panels[0].bar(np.arange(3) + offset, [values["types_daily_macro_f1"],
                      values["types_event_macro_f1"]["iou_0.5"],
                      values["generic_daily_micro"]["f1"]], width, label=label, color=color)
        panels[1].bar(np.arange(4) + offset,
                      [values["types"][kind]["daily_micro"]["f1"] for kind in EVENT_TYPES],
                      width, color=color)
    panels[0].set_xticks(np.arange(3), ["Type daily macro\n(primary)",
                                      "Type event macro\nIoU >= 0.5", "Generic daily micro"])
    panels[1].set_xticks(np.arange(4), [kind.capitalize() for kind in EVENT_TYPES])
    for panel in panels:
        panel.set_ylim(0, 1.03)
        panel.set_ylabel("F1, no point adjustment")
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="y", alpha=.18)
        panel.set_axisbelow(True)
    panels[0].legend(fontsize=7, loc="upper left")
    panels[0].set_title("Prespecified primary and secondary endpoints")
    panels[1].set_title("Daily detection by morphology")
    figure.suptitle(f"GNSS-VLM-EXP05 | {report['unique_stations']} simulated "
                   + "/".join(report["splits"]) + " stations")
    for suffix in ("png", "svg"):
        figure.savefig(args.report / f"comparison.{suffix}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
