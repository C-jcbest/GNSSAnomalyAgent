"""Offline multi-metric comparison and paired image attribution for EXP26."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from report_exp25 import aggregate_rows, bootstrap, request_receipts, score_record

from gnss_research.files import read_json, write_json


def write_table(output, name, records):
    with (output / name).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize(visual: Path, output: Path, phase: str):
    phases = ("pilot",) if phase == "pilot" else ("pilot", "remaining")
    records = []
    for name in phases:
        if not (visual / (name + "-completed.json")).exists():
            raise ValueError("planned phase is incomplete")
        registration = read_json(visual / (name + "-registration.json"))
        records.extend(registration["records"])
    dataset, baseline = Path(registration["dataset"]), Path(registration["baseline"])
    baseline_registration = read_json(baseline / "registration.json")
    baseline_methods = baseline_registration["methods"]
    methods = sorted(set(baseline_methods) | {path.name for path in (visual / "final").iterdir()})
    rows = {method: [] for method in methods}
    predictions = {method: {} for method in methods}
    dependencies = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        truth = read_json(dataset / record["truth"])
        for method in methods:
            parent = baseline / "predictions" if method in baseline_methods else visual / "final"
            result = read_json(parent / method / station / "result.json")
            predictions[method][station] = result
            dependencies.extend(result.get("evaluation_dependency_workdirs", []))
            rows[method].append({"id": station, "scenario": truth["scenario"],
                                 **score_record(observation, truth, result)})
    split = baseline_registration["split"]
    summary = {"experiment_id": "GNSS-VLM-EXP26", "phase": phase, "stations": len(records),
               "dataset": str(dataset), "split": split, "methods": {
                   method: aggregate_rows(reports) for method, reports in rows.items()}}
    comparison, ranges = [], []
    for method, report in summary["methods"].items():
        original = report["original"]
        row = {"method": method, "four_type_daily_f1": original.get("types_daily_macro_f1"),
               "generic_daily_f1": original["generic_daily_micro"]["f1"],
               "generic_affiliation_f1": report["generic_affiliation"]["station_macro_f1"]}
        for kind in ("spike", "step", "transient", "deformation"):
            row[kind + "_daily_f1"] = original["types"][kind]["daily_micro"]["f1"] if "types" in original else None
        if report["range"] is not None:
            for kind in ("transient", "deformation", "range_union"):
                score = report["range"][kind]
                ranges.append({"method": method, "kind": kind,
                               "aff_precision": score["affiliation"]["precision"],
                               "aff_recall": score["affiliation"]["recall"],
                               "aff_f1": score["affiliation"]["station_macro_f1"],
                               **{"daily_" + key: value for key, value in score["daily_micro"].items()},
                               "coverage": score["truth_event_mean_coverage"],
                               "negative_station_fp_days": score["negative_station_fp_days"],
                               "event_f1_05": score["events_micro"]["iou_0.5"]["f1"]})
            row["transient_aff_f1"] = report["range"]["transient"]["affiliation"]["station_macro_f1"]
            row["transient_fp_days"] = report["range"]["transient"]["false_positive_days"]
        else:
            row.update(transient_aff_f1=None, transient_fp_days=None)
        comparison.append(row)
    paired = {}
    for family in ("vl", "flash"):
        main = f"cf_{family}_image"
        controls = [f"cf_{family}_{variant}" for variant in ("noimage", "raw", "text")]
        controls += ["segment_numeric", "vector_nearest", "sdr_flash", "sdr_raw"]
        paired[main] = {}
        for control in controls:
            if control not in rows:
                continue
            changes = []
            for a, b in zip(rows[main], rows[control]):
                station = a["id"]
                first = predictions[main][station]["prediction"]["events"]
                second = predictions[control][station]["prediction"]["events"]
                signature_a = sorted((e["kind"], e["start"], e["end"]) for e in first)
                signature_b = sorted((e["kind"], e["start"], e["end"]) for e in second)
                if signature_a != signature_b:
                    changes.append({"station": station, "scenario": a["scenario"],
                                    "image_events": signature_a, "control_events": signature_b,
                                    "image_transient": a["range"]["transient"],
                                    "control_transient": b["range"]["transient"]})
            paired[main][control] = {"changed_stations": len(changes), "changes": changes,
                                     "bootstrap": bootstrap(rows[main], rows[control])}
    receipts = request_receipts(dependencies)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_json(output / "per-station.json", rows)
    write_json(output / "paired-changes.json", paired)
    write_json(output / "request-receipts.json", receipts)
    write_table(output, "comparison.csv", comparison)
    write_table(output, "range-metrics.csv", ranges)
    lines = ["# EXP26：反事实位移连续性视觉复核", "", f"{len(records)}站，{split}数据，阶段：{phase}。",
             "", "| 方法 | 暂态Aff-F1 | 暂态日F1 | 四类日F1 | 暂态FP日 |", "|---|---:|---:|---:|---:|"]
    for row in comparison:
        values = [row[key] for key in ("transient_aff_f1", "transient_daily_f1", "four_type_daily_f1", "transient_fp_days")]
        lines.append("| " + row["method"] + " | " + " | ".join("—" if value is None else f"{value:.4f}" for value in values) + " |")
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    chosen = ["segment_numeric", "vector_nearest", "sdr_flash", "sdr_raw", "cf_vl_image", "cf_vl_noimage",
              "cf_vl_raw", "cf_vl_text", "cf_flash_image", "cf_flash_noimage", "cf_flash_raw", "cf_flash_text"]
    table = {row["method"]: row for row in comparison}
    chosen = [method for method in chosen if method in table]
    figure, panels = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    for panel, key, title in zip(panels, ("transient_aff_f1", "transient_daily_f1", "four_type_daily_f1"),
                                  ("Transient Affiliation F1", "Transient daily F1", "Four-type daily F1")):
        panel.barh(chosen, [table[name][key] for name in chosen], color="#397e91")
        panel.set_xlim(0, 1)
        panel.set_title(title)
        panel.grid(axis="x", alpha=.2)
    panels[0].invert_yaxis()
    figure.tight_layout()
    figure.savefig(output / "ablation.png", dpi=170)
    plt.close(figure)
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "full"), default="pilot")
    args = parser.parse_args()
    summarize(args.visual, args.out, args.phase)
