"""Collect range evidence and plot the paired image ablation on identical stations."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnss_research.files import read_json, sha256, write_json


def main():
    output = Path("results/exp20-22-assessment")
    output.mkdir(parents=True, exist_ok=False)
    reports = {
        "original_48": Path("results/exp20-exp06-range-v3/summary.json"),
        "boundary_24": Path("results/exp20-boundaries-range-v2/summary.json"),
        "development_1509": Path("results/exp20-exp17-range-v2/summary.json"),
        "development_1512": Path("results/exp20-exp18-range-v3/summary.json"),
        "direct_range_pilot": Path("results/exp21-development-1515-pilot-v1/summary.json"),
        "paired_image_48": Path("results/exp22-image-ablation-full-v1/summary.json")}
    ablation = read_json(reports["paired_image_48"])
    names = ["segment_numeric", "historical_sdr", "flash_image", "flash_noimage", "vl_image", "vl_noimage"]
    labels = ["Numeric", "SDR saved", "Flash image", "Flash no image", "VL image", "VL no image"]
    colors = ["#777777", "#436d9c", "#1e8176", "#79b2a8", "#b57939", "#d7b68b"]
    figure, panels = plt.subplots(2, 2, figsize=(13, 8), dpi=180, sharey=True)
    for column, kind in enumerate(("transient", "deformation")):
        for row, endpoint in enumerate(("affiliation", "event_iou_0.5")):
            panel = panels[row, column]
            values = []
            for method in names:
                metrics = ablation["methods"][method][kind]
                values.append(metrics["affiliation"]["station_macro_f1"] if endpoint == "affiliation"
                              else metrics["events_micro"]["iou_0.5"]["f1"])
            panel.bar(np.arange(len(names)), values, color=colors, width=.7)
            for index, value in enumerate(values):
                panel.text(index, value + .015, f"{value:.3f}", ha="center", fontsize=9)
            panel.set_xticks(np.arange(len(names)), labels, rotation=25, ha="right")
            panel.set_ylim(0, 1.08)
            panel.set_title(f"{kind.capitalize()}: {'Affiliation F1 (station macro)' if row == 0 else 'Event F1 (IoU >= 0.5)'}")
            panel.set_ylabel("F1")
            panel.grid(axis="y", alpha=.2)
            panel.set_axisbelow(True)
    figure.suptitle("EXP20/22: same 48 stations; fresh image/no-image ablation; post-hoc analysis")
    figure.tight_layout()
    figure.savefig(output / "range-image-comparison.png")
    figure.savefig(output / "range-image-comparison.svg")
    plt.close(figure)
    requests = []
    for root in (Path("runs/exp21-development-1515-visual-v1"), Path("runs/exp22-image-ablation-v1")):
        for path in root.glob("predictions/**/result.json"):
            result = read_json(path)
            if not result.get("request_count"):
                continue
            response = path.with_name("response.json")
            requests.append({"result": str(path), "result_sha256": sha256(path),
                "response_sha256": sha256(response) if response.exists() else None,
                "status": result["status"], "http_status": result.get("http_status"),
                "requested_model": result.get("requested_model"), "served_model": result.get("served_model"),
                "retry_count": result.get("retry_count"), "usage": result.get("usage")})
    equality = {}
    prediction_root = Path("results/exp22-image-ablation-full-v1/predictions")
    for model in ("flash", "vl"):
        differences = []
        compared = 0
        for path in sorted((prediction_root / (model + "_image")).glob("*/result.json")):
            image = read_json(path)
            noimage = read_json(prediction_root / (model + "_noimage") / path.parent.name / "result.json")
            image_events = sorted((e["kind"], e["start"], e["end"], e["status"]) for e in image["prediction"]["events"])
            noimage_events = sorted((e["kind"], e["start"], e["end"], e["status"]) for e in noimage["prediction"]["events"])
            if image_events != noimage_events:
                differences.append(path.parent.name)
            compared += 1
        equality[model] = {"compared_stations": compared, "different_event_signatures": differences}
    write_json(output / "evidence.json", {"reports": {key: {"path": str(path), "sha256": sha256(path)}
        for key, path in reports.items()}, "request_count": len(requests),
        "failed_count": sum(row["status"] != "success" for row in requests),
        "http_success_count": sum(row["http_status"] == 200 for row in requests),
        "retry_count": sum(row["retry_count"] or 0 for row in requests),
        "paired_final_predictions": equality, "requests": requests})
    print({"requests": len(requests), "failures": sum(row["status"] != "success" for row in requests),
           "paired_prediction_equality": equality}, flush=True)


if __name__ == "__main__":
    main()
