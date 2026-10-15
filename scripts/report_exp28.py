"""Large-cohort all-station, intervention-subset, seed and scenario evaluation."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
from exp14_fingerprint import free_graph
from report_exp25 import aggregate_cost, aggregate_rows, bootstrap, request_receipts, score_record
from report_exp26 import write_table

from gnss_research.files import read_json, write_json


def summarize(visual: Path, output: Path, phase: str):
    records = []
    phases = ("pilot",) if phase == "pilot" else ("pilot", "remaining")
    for name in phases:
        if not (visual / f"{name}-completed.json").exists():
            raise ValueError("planned inference incomplete")
        registration = read_json(visual / f"{name}-registration.json")
        records.extend(registration["records"])
    dataset, baseline = Path(registration["dataset"]), Path(registration["baseline"])
    baseline_methods = read_json(baseline / "registration.json")["methods"]
    methods = sorted(set(baseline_methods) | {p.name for p in (visual / "final").iterdir()})
    rows = {method: [] for method in methods}
    dependencies, candidates, thin_rows = [], {}, []
    for number, record in enumerate(records, 1):
        station = record["station_id"]
        observation, truth = read_json(dataset / record["observation"]), read_json(dataset / record["truth"])
        graph = free_graph(read_json(baseline / "predictions/segment_numeric" / station / "result.json"))
        candidates[station] = {"pairs": len(graph["pairs"]), "jumps": len(graph["jumps"]),
                               "cohort_seed": record["cohort_seed"], "scenario": truth["scenario"]}
        for method in methods:
            parent = baseline / "predictions" if method in baseline_methods else visual / "final"
            result = read_json(parent / method / station / "result.json")
            dependencies.extend(result.get("evaluation_dependency_workdirs", []))
            row = {"id": station, "cohort_seed": record["cohort_seed"], "scenario": truth["scenario"],
                   "triggered": bool(graph["pairs"]), "candidate_pairs": len(graph["pairs"]),
                   "event_signature": sorted((event["kind"], event["start"], event["end"])
                       for event in (result.get("prediction") or {}).get("events", [])),
                   "pair_decision": result.get("pair_decision"), **score_record(observation, truth, result)}
            rows[method].append(row)
            thin = {"method": method, "station": station, "cohort_seed": row["cohort_seed"],
                    "scenario": row["scenario"], "triggered": row["triggered"],
                    "transient_aff_f1": None, "transient_tp": None, "transient_fp": None, "transient_fn": None}
            if row["range"] is not None:
                metric = row["range"]["transient"]
                thin.update(transient_aff_f1=metric["affiliation"]["f1"],
                            **{"transient_" + key: metric["daily"][key] for key in ("tp", "fp", "fn")})
            thin_rows.append(thin)
        if number % 60 == 0:
            print("scored", number, "/", len(records), flush=True)
    summary = {"experiment_id": "GNSS-VLM-EXP28", "phase": phase, "split": registration["split"],
               "stations": len(records), "cohort_seeds": registration["cohort_seeds"],
               "triggered_stations": sum(bool(v["pairs"]) for v in candidates.values()),
               "candidate_pairs": sum(v["pairs"] for v in candidates.values()),
               "methods": {method: aggregate_rows(reports) for method, reports in rows.items()}}
    strata = {"scenario": {}, "seed": {}, "intervention_subset": {}}
    for method, reports in rows.items():
        for label, key in (("scenario", "scenario"), ("seed", "cohort_seed")):
            strata[label][method] = {str(value): {"stations": sum(row[key] == value for row in reports),
                **aggregate_rows([row for row in reports if row[key] == value])}
                for value in sorted({row[key] for row in reports})}
        strata["intervention_subset"][method] = {}
        for name, subset in (("triggered", [row for row in reports if row["triggered"]]),
                             ("multiple_pairs", [row for row in reports if row["candidate_pairs"] > 1])):
            if subset:
                strata["intervention_subset"][method][name] = {"stations": len(subset), **aggregate_rows(subset)}
    paired = {}
    main = "ce_flash_image"
    for control in ("cf_flash_image", "ce_flash_cfonly", "ce_flash_raw", "ce_flash_text",
                    "ce_flash_noimage", "vector_nearest", "sdr_flash", "edge_flash_image", "trend_flash_image"):
        changes = []
        for first, second in zip(rows[main], rows[control]):
            if first["event_signature"] != second["event_signature"]:
                changes.append({"station": first["id"], "cohort_seed": first["cohort_seed"],
                    "scenario": first["scenario"], "main_decision": first["pair_decision"],
                    "control_decision": second["pair_decision"],
                    "main_events": first["event_signature"], "control_events": second["event_signature"],
                    "main_transient": first["range"]["transient"],
                    "control_transient": second["range"]["transient"]})
        paired[control] = {"changed_event_stations": len(changes), "changes": changes,
                           "bootstrap": bootstrap(rows[main], rows[control])}
    comparison, ranges = [], []
    for method, report in summary["methods"].items():
        original = report["original"]
        row = {"method": method, "four_type_daily_f1": original.get("types_daily_macro_f1"),
               "generic_daily_f1": original["generic_daily_micro"]["f1"],
               "generic_affiliation_f1": report["generic_affiliation"]["station_macro_f1"],
               "transient_aff_f1": None, "transient_daily_f1": None, "transient_fp_days": None}
        if report["range"] is not None:
            for kind in ("transient", "deformation", "range_union"):
                metric = report["range"][kind]
                ranges.append({"method": method, "kind": kind,
                    "aff_precision": metric["affiliation"]["precision"], "aff_recall": metric["affiliation"]["recall"],
                    "aff_f1": metric["affiliation"]["station_macro_f1"],
                    **{"daily_" + key: value for key, value in metric["daily_micro"].items()},
                    "coverage": metric["truth_event_mean_coverage"],
                    "negative_station_fp_days": metric["negative_station_fp_days"],
                    "event_f1_05": metric["events_micro"]["iou_0.5"]["f1"]})
            transient = report["range"]["transient"]
            row.update(transient_aff_f1=transient["affiliation"]["station_macro_f1"],
                       transient_daily_f1=transient["daily_micro"]["f1"],
                       transient_fp_days=transient["false_positive_days"])
        for kind in ("spike", "step", "deformation"):
            row[kind + "_daily_f1"] = original["types"][kind]["daily_micro"]["f1"] if "types" in original else None
        comparison.append(row)
    output.mkdir(parents=True, exist_ok=False)
    receipts = request_receipts(dependencies)
    write_json(output / "summary.json", summary)
    write_json(output / "strata.json", strata)
    write_json(output / "candidates.json", candidates)
    write_json(output / "paired-changes.json", paired)
    write_json(output / "request-receipts.json", receipts)
    write_json(output / "costs.json", aggregate_cost(receipts))
    # Full station metrics stay available without committing a giant uncompressed JSON file.
    with gzip.open(output / "per-station.json.gz", "xt", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    write_table(output, "per-station.csv", thin_rows)
    write_table(output, "comparison.csv", comparison)
    write_table(output, "range-metrics.csv", ranges)
    lines = ["# EXP28：竞争解释反事实视觉复核", "",
             f"{len(records)}站，{registration['split']}，{phase}，{summary['triggered_stations']}站触发自由配对复核。", "",
             "| 方法 | 暂态Aff-F1 | 暂态日F1 | 四类F1 | 暂态FP日 |", "|---|---:|---:|---:|---:|"]
    for row in comparison:
        values = [row[key] for key in ("transient_aff_f1", "transient_daily_f1", "four_type_daily_f1", "transient_fp_days")]
        lines.append("| " + row["method"] + " | " + " | ".join("—" if v is None else f"{v:.4f}" for v in values) + " |")
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    selected = [row for row in comparison if row["transient_aff_f1"] is not None]
    figure, panels = plt.subplots(1, 3, figsize=(16, 7.5), sharey=True)
    for panel, key, title in zip(panels, ("transient_aff_f1", "transient_daily_f1", "four_type_daily_f1"),
                                  ("Transient Affiliation F1", "Transient daily F1", "Four-type daily F1")):
        panel.barh([row["method"] for row in selected], [row[key] for row in selected], color="#397e91")
        panel.set_xlim(0, 1)
        panel.set_title(title)
        panel.grid(axis="x", alpha=.2)
    panels[0].invert_yaxis()
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "full"), default="full")
    args = parser.parse_args()
    summarize(args.visual, args.out, args.phase)
