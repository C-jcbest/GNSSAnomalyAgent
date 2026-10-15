"""Offline evaluation with unchanged Affiliation and daily metric implementations."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from report_exp25 import aggregate_rows, bootstrap, request_receipts, score_record
from report_exp26 import write_table

from gnss_research.files import read_json, write_json


def summarize(visual: Path, output: Path, phase: str):
    phases = ("pilot",) if phase == "pilot" else ("pilot", "remaining")
    records = []
    for name in phases:
        if not (visual / f"{name}-completed.json").exists():
            raise ValueError("planned phase incomplete")
        registration = read_json(visual / f"{name}-registration.json")
        records.extend(registration["records"])
    dataset, baseline = Path(registration["dataset"]), Path(registration["baseline"])
    baseline_methods = read_json(baseline / "registration.json")["methods"]
    methods = sorted(set(baseline_methods) | {path.name for path in (visual / "final").iterdir()})
    rows = {method: [] for method in methods}
    predictions = {method: {} for method in methods}
    dependencies = []
    for record in records:
        station = record["station_id"]
        observation, truth = read_json(dataset / record["observation"]), read_json(dataset / record["truth"])
        for method in methods:
            parent = baseline / "predictions" if method in baseline_methods else visual / "final"
            result = read_json(parent / method / station / "result.json")
            predictions[method][station] = result
            dependencies.extend(result.get("evaluation_dependency_workdirs", []))
            rows[method].append({"id": station, "scenario": truth["scenario"],
                                 **score_record(observation, truth, result)})
    summary = {"experiment_id": "GNSS-VLM-EXP27", "stations": len(records), "phase": phase,
               "split": registration["split"], "seed": registration["seed"],
               "dataset": str(dataset), "methods": {
                   method: aggregate_rows(reports) for method, reports in rows.items()}}
    comparison, ranges = [], []
    for method, report in summary["methods"].items():
        original = report["original"]
        row = {"method": method, "four_type_daily_f1": original.get("types_daily_macro_f1"),
               "generic_daily_f1": original["generic_daily_micro"]["f1"],
               "generic_affiliation_f1": report["generic_affiliation"]["station_macro_f1"]}
        for kind in ("spike", "step", "transient", "deformation"):
            row[kind + "_daily_f1"] = original["types"][kind]["daily_micro"]["f1"] if "types" in original else None
        row.update(transient_aff_f1=None, transient_fp_days=None)
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
            transient = report["range"]["transient"]
            row["transient_aff_f1"] = transient["affiliation"]["station_macro_f1"]
            row["transient_fp_days"] = transient["false_positive_days"]
        comparison.append(row)
    paired = {}
    for main, controls in {
        "edge_flash_image": ["cf_flash_image", "edge_flash_text", "edge_flash_raw", "edge_flash_noimage", "vector_nearest"],
        "trend_flash_image": ["cf_flash_image", "edge_flash_image", "trend_flash_text", "trend_flash_raw", "trend_flash_noimage", "trend_nearest", "vector_nearest"],
        "trend_vl_image": ["trend_vl_noimage", "trend_flash_image", "trend_nearest"],
    }.items():
        paired[main] = {}
        for control in controls:
            changes = []
            for first, second in zip(rows[main], rows[control]):
                station = first["id"]
                events_a = predictions[main][station]["prediction"]["events"]
                events_b = predictions[control][station]["prediction"]["events"]
                signature_a = sorted((e["kind"], e["start"], e["end"]) for e in events_a)
                signature_b = sorted((e["kind"], e["start"], e["end"]) for e in events_b)
                if signature_a != signature_b:
                    changes.append({"station": station, "scenario": first["scenario"],
                        "image_events": signature_a, "control_events": signature_b,
                        "image_transient": first["range"]["transient"],
                        "control_transient": second["range"]["transient"]})
            paired[main][control] = {"changed_stations": len(changes), "changes": changes,
                                     "bootstrap": bootstrap(rows[main], rows[control])}
    scenarios = {}
    for method, reports in rows.items():
        scenarios[method] = {}
        for scenario in sorted({row["scenario"] for row in reports}):
            scenarios[method][scenario] = aggregate_rows([row for row in reports if row["scenario"] == scenario])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_json(output / "per-station.json", rows)
    write_json(output / "paired-changes.json", paired)
    write_json(output / "scenario-strata.json", scenarios)
    write_json(output / "request-receipts.json", request_receipts(dependencies))
    write_table(output, "comparison.csv", comparison)
    write_table(output, "range-metrics.csv", ranges)
    lines = ["# EXP27：局部趋势分离与双边缘放大", "",
             f"{len(records)}站，{registration['split']}，seed={registration['seed']}，{phase}。", "",
             "| 方法 | 暂态Aff-F1 | 暂态日F1 | 四类日F1 | 暂态FP日 |", "|---|---:|---:|---:|---:|"]
    for row in comparison:
        values = [row[key] for key in ("transient_aff_f1", "transient_daily_f1", "four_type_daily_f1", "transient_fp_days")]
        lines.append("| " + row["method"] + " | " + " | ".join("—" if v is None else f"{v:.4f}" for v in values) + " |")
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    selected = [row for row in comparison if row["transient_aff_f1"] is not None]
    figure, panels = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    for panel, key, title in zip(panels, ("transient_aff_f1", "transient_daily_f1", "four_type_daily_f1"),
                                  ("Transient Affiliation F1", "Transient daily F1", "Four-type daily F1")):
        panel.barh([row["method"] for row in selected], [row[key] for row in selected], color="#397e91")
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
