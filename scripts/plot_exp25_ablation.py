"""Display both positive and negative image ablations on their actual metrics."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnss_research.files import read_json


def main():
    target = Path("results/exp25-test-1516-full-v1")
    methods = read_json(target / "summary.json")["methods"]
    names = ["segment_numeric", "numeric_gr", "sdr_gr_flash", "sdr_gr_pipeline_noimage",
             "sdr_flash", "sdr_gr_recovery_noimage", "sdr_gr_numeric_recovery",
             "sdr_cr_flash", "sdr_cr_vl", "sdr_raw", "sdr_vl", "sdr_vl_noimage"]
    endpoints = {
        "Transient Affiliation F1": [methods[n]["range"]["transient"]["affiliation"]["station_macro_f1"] for n in names],
        "Transient daily F1": [methods[n]["range"]["transient"]["daily_micro"]["f1"] for n in names],
        "Four-type daily F1": [methods[n]["original"]["types_daily_macro_f1"] for n in names],
        "Transient false-positive days": [methods[n]["range"]["transient"]["false_positive_days"] for n in names],
    }
    figure, panels = plt.subplots(2, 2, figsize=(16, 10), dpi=170)
    x = np.arange(len(names))
    for panel, (title, values) in zip(panels.flat, endpoints.items(), strict=True):
        panel.bar(x, values, color=["#2b927c" if name == "sdr_gr_flash" else "#5785ac" for name in names])
        panel.set_title(title)
        panel.set_ylim(0, max(values) * 1.16 if "days" in title else 1.12)
        panel.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
        panel.grid(axis="y", alpha=.2)
        for index, value in enumerate(values):
            label = str(value) if "days" in title else f"{value:.3f}"
            panel.text(index, value + max(values) * .015, label, ha="center", fontsize=8)
    figure.suptitle("EXP25 / held-out 48 stations / fixed methods and evaluation")
    figure.tight_layout()
    figure.savefig(target / "ablation-comparison.png")
    figure.savefig(target / "ablation-comparison.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
