"""Summarize actual range scores and request receipts without additional inference."""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnss_research.files import read_json, sha256, write_json

TARGET = Path("results/exp23-24-assessment")
STUDIES = {
    "historical": (Path("runs/exp23-historical-recovery-v2"),
                   Path("results/exp23-historical-full-v1"), Path("results/exp24-historical-gap-v1")),
    "development_1516": (Path("runs/exp23-development-1516-recovery-v1"),
                         Path("results/exp23-development-1516-full-v1"), Path("results/exp24-development-1516-gap-v1")),
}


def main():
    TARGET.mkdir(parents=True, exist_ok=False)
    evidence = {"experiment_ids": ["GNSS-VLM-EXP23", "GNSS-VLM-EXP24"], "studies": {}, "requests": []}
    selected = []
    for name, (run, unguarded, guarded) in STUDIES.items():
        first, second = read_json(unguarded / "summary.json"), read_json(guarded / "summary.json")
        comparisons = {"Numeric": first["methods"]["segment_numeric"],
                       "SDR-Flash-Aff": first["methods"]["base_flash"],
                       "SDR-CR (VL)": first["methods"]["vl_image"],
                       "SDR-GR (Flash)": second["methods"]["gap_flash_image"]}
        evidence["studies"][name] = {"stations": first["stations"], "methods": comparisons,
            "unrestricted_summary": str(unguarded / "summary.json"),
            "gap_summary": str(guarded / "summary.json"), "historical": first["historical"]}
        selected.append(comparisons)
        snapshots = TARGET / "sources" / name
        snapshots.mkdir(parents=True)
        for path in (run / "registration.json", run / "experiment-source.py"):
            shutil.copyfile(path, snapshots / path.name)
        for request in sorted(run.rglob("request.json")):
            result_path = request.parent / "result.json"
            if not result_path.exists():
                raise ValueError(f"request has no saved result: {request}")
            result = read_json(result_path)
            relative = request.parent.relative_to(run)
            output = TARGET / "responses" / name / relative
            output.mkdir(parents=True)
            for filename in ("result.json", "response.json"):
                source = request.parent / filename
                if source.exists():
                    shutil.copyfile(source, output / filename)
            payload = read_json(request)
            images = sum(item.get("type") == "image_url" for message in payload["messages"]
                         if isinstance(message["content"], list) for item in message["content"])
            evidence["requests"].append({"study": name, "path": str(request),
                "request_sha256": sha256(request), "result_sha256": sha256(result_path),
                "image_count": images, "status": result["status"], "http_status": result.get("http_status"),
                "retry_count": result.get("retry_count", 0), "requested_model": result.get("requested_model"),
                "served_model": result.get("served_model"), "usage": result.get("usage")})
        for source in sorted((run / "inputs").glob("*/local.png")):
            dest = TARGET / "images" / name / source.parent.name / source.name
            dest.parent.mkdir(parents=True)
            shutil.copyfile(source, dest)
    requests = evidence["requests"]
    evidence["request_summary"] = {"total": len(requests),
        "failed": sum(r["status"] != "success" for r in requests),
        "http_200": sum(r["http_status"] == 200 for r in requests),
        "retry_count": sum(r["retry_count"] for r in requests)}
    write_json(TARGET / "evidence.json", evidence)
    figure, panels = plt.subplots(2, 3, figsize=(14, 7.4), dpi=170)
    colors = ["#747b87", "#3a82b5", "#df9855", "#389782"]
    names = list(selected[0])
    for row, (study, comparisons) in enumerate(zip(STUDIES, selected, strict=True)):
        for column, (label, key) in enumerate((("Affiliation F1", "aff"), ("Daily F1", "daily"), ("False positive days", "fp"))):
            values = []
            for metric in comparisons.values():
                metric = metric["transient"]
                values.append(metric["affiliation"]["station_macro_f1"] if key == "aff" else
                              metric["daily_micro"]["f1"] if key == "daily" else metric["false_positive_days"])
            panel = panels[row, column]
            panel.bar(np.arange(len(names)), values, color=colors, width=.65)
            panel.set_xticks(np.arange(len(names)), names, rotation=18, ha="right", fontsize=8)
            panel.set_title(f"{'Historical 48 (post-hoc)' if study == 'historical' else 'New validation 48 (development)'}\n{label}")
            panel.set_ylim(0, 1.12 if key != "fp" else max(135, max(values) * 1.2))
            for position, value in enumerate(values):
                panel.text(position, value + (.018 if key != "fp" else 2),
                           f"{value:.3f}" if key != "fp" else str(value), ha="center", fontsize=9)
            panel.grid(axis="y", alpha=.2)
    figure.tight_layout()
    figure.savefig(TARGET / "affiliation-and-errors.png")
    figure.savefig(TARGET / "affiliation-and-errors.svg")
    plt.close(figure)
    print(evidence["request_summary"])


if __name__ == "__main__":
    main()
