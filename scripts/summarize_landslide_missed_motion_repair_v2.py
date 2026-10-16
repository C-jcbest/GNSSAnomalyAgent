"""Verify cached repair probes and export complete activity, event and stage evidence."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_background_benchmark import recompute_numerical
from summarize_landslide_motion_competition import verify_metrics

from gnss_sim.landslide_evaluation import aggregate_scores, score_counts


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_bindings(run):
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError(f"Frozen repair output changed: {name}")
    for binding in ("input-bindings.json", "reference-binding.json"):
        for name, expected in read_json(run / binding).items():
            if sha(Path(name)) != expected:
                raise ValueError(f"Bound input changed: {name}")
    executing_sources = {name: expected for name, expected in read_json(run / "input-bindings.json").items()
                         if Path(name).parent == ROOT / "scripts"}
    if len(executing_sources) != 1 or sha(run / "source.py") != next(iter(executing_sources.values())):
        raise ValueError("Executed repair source differs from its bound snapshot")


def summarize_run(run, source, labels, records, out):
    check_bindings(run)
    predictions = read_json(run / "predictions.json")
    results = read_json(run / "results.json")
    changes = read_json(run / "changes.json")
    weak = read_json(run / "weak-results.json")
    boundaries = read_json(run / "boundaries.json")
    stages = verify_metrics(predictions, results, labels, records)
    permissions = read_json(run / "permissions.json")
    original = read_json(source / "predictions.json")
    metrics, per_record, groups = [], [], []
    known_days = sum(int(row["activity_label"]) >= 0 for rows in labels.values() for row in rows)
    positive_days = sum(int(row["activity_label"]) == 1 for rows in labels.values() for row in rows)
    weak_days = sum(int(row["weak_signal"]) == 1 for rows in labels.values() for row in rows)
    mapping = read_json(ROOT / read_json(source / "protocol.json")["source_bundle"] / "background-mapping.json")
    # Family/background metadata is diagnostic, never an input to the repair.
    truth = {cid: read_json(ROOT / read_json(source / "protocol.json")["source_bundle"]
                           / "mechanism-reference" / cid / "truth.json") for cid in labels}
    for method, values in predictions.items():
        aggregate = results[method]["aggregate"]
        if aggregate["known_days"] != known_days or aggregate["daily"]["tp"] + aggregate["daily"]["fn"] != positive_days:
            raise ValueError("A repair changed the scoring population")
        detected_weak = 0
        matched = []
        event05 = [0, 0, 0]
        for cid, value in values.items():
            activity = np.asarray(value["activity"])
            actual = np.asarray([int(row["activity_label"]) for row in labels[cid]])
            weak_mask = np.asarray([int(row["weak_signal"]) == 1 for row in labels[cid]])
            detected_weak += int((weak_mask & (activity == 1)).sum())
            details = boundaries[method][cid]
            matched.extend(details["0.3"]["matched_pairs"])
            for index, field in enumerate(("matched_pairs", "unmatched_prediction_events", "missed_events")):
                value05 = details["0.5"][field]
                event05[index] += len(value05) if isinstance(value05, list) else value05
            if method in changes:
                rep = int(method[-1])
                before = np.asarray(original[f"direct_visual_r{rep}"][cid]["activity"])
                measured = {
                    "gained_tp": int(((actual == 1) & (before != 1) & (activity == 1)).sum()),
                    "lost_tp": int(((actual == 1) & (before == 1) & (activity != 1)).sum()),
                    "added_fp": int(((actual == 0) & (before != 1) & (activity == 1)).sum()),
                    "removed_fp": int(((actual == 0) & (before == 1) & (activity != 1)).sum()),
                }
                if measured != changes[method]["per_case"][cid]:
                    raise ValueError("Saved repair changes differ from daily predictions")
                if "__" in method and "bidirectional" not in method:
                    if np.any((before == 1) & (activity != 1)):
                        raise ValueError("One-sided repair removed an initial active date")
                    arm = method.split("__")[0]
                    permission = permissions[f"{arm}_r{rep}-{cid}"]
                    field = "visual_permitted_days" if "visual_additions_without_raw" in method else "raw_permitted_days"
                    allowed = np.zeros(len(activity), dtype=bool)
                    allowed[permission[field]] = True
                    if np.any((activity != before) & ~allowed):
                        raise ValueError("A repair changed a date without permission")
                    if "numerical_additions_without_visual" not in method:
                        local = np.asarray(original[f"{arm}_r{rep}"][cid]["activity"])
                        if np.any((before != 1) & (activity == 1) & (local != 1)):
                            raise ValueError("Numerical evidence fabricated a visual confirmation")
            score = results[method]["per_case"][cid]
            per_record.append({"run": run.name, "method": method, "case_id": cid,
                               "family": truth[cid]["family"], "background": truth[cid]["background_group"],
                               **score["daily"], "event_f1": score["events"]["f1"]})
        if detected_weak != weak[method]["detected"] or weak[method]["days"] != weak_days:
            raise ValueError("Weak recall uses a different annotation population")
        if len(matched) != aggregate["events"]["tp"]:
            raise ValueError("Independent event matching differs")
        metrics.append({
            "run": run.name, "method": method, **aggregate["daily"],
            "weak_recall": weak[method]["recall"], "weak_days": weak_days,
            "event_f1_iou03": aggregate["events"]["f1"],
            "event_f1_iou05": score_counts(*event05)["f1"],
            "predicted_events": aggregate["events"]["tp"] + aggregate["events"]["fp"],
            "matched_events": len(matched),
            "onset_mae_days": float(np.mean([abs(pair["onset_error_days"]) for pair in matched])) if matched else None,
            "offset_mae_days": float(np.mean([abs(pair["offset_error_days"]) for pair in matched])) if matched else None,
            "coverage": aggregate["coverage"], "stage_macro_f1": aggregate["stage_macro_f1"],
            "internal_stage_macro_f1": stages[method]["macro_f1"],
        })
        for field in ("family", "background_group"):
            for group in sorted({row[field] for row in truth.values()}, key=str):
                ids = [cid for cid in labels if truth[cid][field] == group]
                score = aggregate_scores([results[method]["per_case"][cid] for cid in ids])
                denominator = sum(int(row["weak_signal"]) == 1 for cid in ids for row in labels[cid])
                detected = sum(int(row["weak_signal"]) == 1 and values[cid]["activity"][day] == 1
                               for cid in ids for day, row in enumerate(labels[cid]))
                groups.append({"run": run.name, "method": method, "field": field, "group": group,
                               "records": len(ids), **score["daily"],
                               "weak_days": denominator, "weak_recall": detected / denominator if denominator else None})
    write_rows(out / f"{run.name}-metrics.csv", metrics)
    write_rows(out / f"{run.name}-per-record.csv", per_record)
    write_rows(out / f"{run.name}-groups.csv", groups)
    save_json(out / f"{run.name}-internal-stages.json", stages)
    # Retain the mapping hash instead of duplicating source calendars or claiming
    # that the 24 records are 24 independent observed backgrounds.
    save_json(out / f"{run.name}-verification.json", {
        "status": "passed", "configurations": len(predictions), "known_days": known_days,
        "activity_days": positive_days, "weak_days": weak_days,
        "group_mapping_entries": len(mapping), "new_api_requests": 0,
    })
    return predictions, metrics


def export_timeline(records, labels, predictions, out):
    names = [f"{arm}_r{rep}" for rep in (0, 1) for arm in ("direct_visual", "adaptive_no_feedback")]
    names += [f"adaptive_no_feedback__raw_visual_additions_r{rep}" for rep in (0, 1)]
    names += ["raw_multiscale_rule", "xgboost_84"]
    rows = []
    for cid, (case, _) in records.items():
        for day, reference in enumerate(labels[cid]):
            rows.append({"case_id": cid, "day_index": day, "date": str(case.dates[day]),
                         "activity_label": reference["activity_label"], "feature": reference["feature"],
                         "weak_signal": reference["weak_signal"],
                         **{f"{run}__{name}": values[name][cid]["activity"][day]
                            for run, values in predictions.items() for name in names}})
    write_rows(out / "complete-primary-timelines.csv", rows)
    return len(rows)


def render_records(records, labels, predictions, out):
    original, calibrated, persistent = predictions.values()
    cmap = ListedColormap(["#c5c9cc", "#fff4dc", "#377dab"])
    for cid, (case, _) in records.items():
        matrix = [[int(row["activity_label"]) for row in labels[cid]]]
        titles = ["Known injection"]
        for rep in (0, 1):
            for title, values, method in (
                ("Direct", original, f"direct_visual_r{rep}"),
                ("Whole local", original, f"adaptive_no_feedback_r{rep}"),
                ("Exact addition", original, f"adaptive_no_feedback__raw_visual_additions_r{rep}"),
                ("Calibrated addition", calibrated, f"adaptive_no_feedback__raw_visual_additions_r{rep}"),
                ("14-day addition", persistent, f"adaptive_no_feedback__raw_visual_additions_r{rep}"),
            ):
                titles.append(f"{title} / {rep + 1}")
                matrix.append(values[method][cid]["activity"])
        for method in ("raw_multiscale_rule", "xgboost_84"):
            titles.append(method)
            matrix.append(original[method][cid]["activity"])
        values = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(13, 11),
                                 gridspec_kw={"height_ratios": [1, 1, 1, 3]})
        for component, name in enumerate(("N", "E", "U")):
            axes[component].plot(values[:, component], linewidth=.8, color="#24887b")
            axes[component].set_ylabel(f"{name} (mm)")
            axes[component].grid(alpha=.2)
        axes[3].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1, cmap=cmap,
                       extent=[-.5, len(values) - .5, len(matrix) - .5, -.5])
        axes[3].set_yticks(range(len(matrix)), titles, fontsize=8)
        axes[3].set_xlim(0, len(values) - 1)
        axes[3].set_xlabel("Original day | blue activity; cream quiet; gray unknown/missing")
        fig.suptitle(f"{cid} | exposed development, injection recovery only\n"
                     "Exact-date additions can fragment events; background motion has no expert reference")
        fig.tight_layout()
        fig.savefig(out / f"{cid}.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.out / "source.py")
    source = Path(read_json(args.runs[0] / "protocol.json")["source"])
    config = read_json(source / "protocol.json")
    records = {cid: load_case_features(source, cid) for cid in config["case_ids"]}
    labels = load_labels(source / "analysis-v1/reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    numerical = recompute_numerical(ROOT / config["source_bundle"], records)
    frozen = read_json(source / "numeric-predictions.json")
    for name, values in numerical.items():
        if values != frozen[name]:
            raise ValueError(f"Common refitted numerical baseline changed: {name}")
    predictions, metrics = {}, []
    for run in args.runs:
        if Path(read_json(run / "protocol.json")["source"]) != source:
            raise ValueError("Repair runs use different source populations")
        predictions[run.name], rows = summarize_run(run, source, labels, records, args.out)
        metrics.extend(rows)
    write_rows(args.out / "all-metrics.csv", metrics)
    timeline_rows = export_timeline(records, labels, predictions, args.out)
    render_records(records, labels, predictions, args.out)
    save_json(args.out / "state.json", {
        "status": "complete", "verified_configurations": len(metrics),
        "numerical_predictions_recomputed": len(numerical) * len(records),
        "timeline_rows": timeline_rows, "full_record_figures": len(records), "new_api_requests": 0,
        "independent_reference": False,
    })
    print(read_json(args.out / "state.json"), flush=True)


if __name__ == "__main__":
    main()
