"""Verify scale-gated cached reviews and the same-information learning control."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from finalize_landslide_adaptive_evidence import verify_request_receipts
from fit_landslide_scale_controls import scale_information
from matplotlib.colors import ListedColormap
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_background_benchmark import recompute_numerical
from summarize_landslide_missed_motion_repair import write_rows
from summarize_landslide_motion_competition import verify_metrics
from xgboost import XGBClassifier

from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    score_counts,
)
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_scale_review import compressed_review_targets, historical_scale_gains


def analyze(run, controls, out):
    source = Path(read_json(run / "protocol.json")["source"])
    config = read_json(source / "protocol.json")
    data = ROOT / config["source_bundle"]
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError("Frozen scale-review prediction changed")
    for binding in (run / "input-bindings.json", run / "reference-binding.json", controls / "pre-fit-freeze.json"):
        for name, expected in read_json(binding).items():
            if sha(Path(name)) != expected:
                raise ValueError("Bound scale-review source or data changed")
    for name, expected in read_json(controls / "fitting-freeze.json").items():
        if sha(controls / name) != expected:
            raise ValueError("Frozen scale-information model changed")
    if sha(controls / "predictions.json") != read_json(controls / "manifest.json")["prediction_sha256"]:
        raise ValueError("Frozen scale control predictions changed")
    records = {cid: load_case_features(source, cid) for cid in config["case_ids"]}
    labels = load_labels(source / "analysis-v1/reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    original = read_json(source / "predictions.json")
    predictions = {**read_json(run / "predictions.json"), **read_json(controls / "predictions.json")}
    numerical = recompute_numerical(data, records)
    for name, values in numerical.items():
        if values != predictions[name]:
            raise ValueError("Original refitted numerical control differs")
    model, head = XGBClassifier(), XGBClassifier()
    model.load_model(controls / "scale-xgboost.json")
    head.load_model(source / "stage-xgboost.json")
    threshold = read_json(controls / "frozen-selection.json")["threshold"]
    selections = read_json(run / "selections.json")
    permissions = read_json(run / "permissions.json")
    authority_checks = 0
    for cid, (case, feature) in records.items():
        matrix, support = scale_information(case, feature)
        gate = activity_from_mask(model.predict_proba(matrix)[:, 1] >= threshold, support)
        recomputed = learned_stages(head.predict(feature.matrix).astype(int) + 1, gate, feature.supported).to_dict()
        if recomputed != predictions["xgboost_scale_information"][cid]:
            raise ValueError("Scale-augmented numerical predictions differ")
        gains = historical_scale_gains(case, feature.noise, regular_windows(len(case.dates), 180, 90))
        for rep in (0, 1):
            selected = selections[f"{cid}-r{rep}"]
            if gains != selected["gains"] or compressed_review_targets(selected["original"], gains) != selected["main"]:
                raise ValueError("Scale admission does not reproduce from observed inputs")
            initial = np.asarray(original[f"direct_visual_r{rep}"][cid]["activity"])
            for method, values in predictions.items():
                permission_key = f"{method}-{cid}"
                if not method.endswith(f"_r{rep}") or permission_key not in permissions:
                    continue
                allowed = np.zeros(len(initial), dtype=bool)
                allowed[permissions[permission_key]] = True
                if np.any(allowed & ~feature.observed):
                    raise ValueError("A scale review acquired authority over missing data")
                prior = "adaptive_no_feedback"
                if method.startswith("fixed"):
                    prior = "fixed_image_review"
                elif method.startswith("scale_gate_numeric"):
                    prior = "adaptive_numeric_feedback"
                local = np.asarray(original[f"{prior}_r{rep}"][cid]["activity"])
                expected = np.where(allowed, local, initial)
                if not np.array_equal(expected, values[cid]["activity"]):
                    raise ValueError("Final activity differs from the admitted cached local judgment")
                authority_checks += 1
    receipts = verify_request_receipts(source)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    results = {}
    for method, cases in predictions.items():
        per_case = {cid: evaluate_case(labels[cid], ActivityPrediction(
            np.asarray(value["activity"]), np.asarray(value["stage"]), status=value["status"],
            gated=value["gated"], stage_status=value["stage_status"])) for cid, value in cases.items()}
        results[method] = {"aggregate": aggregate_scores(list(per_case.values())), "per_case": per_case}
    for method, score in read_json(run / "results.json").items():
        if results[method] != score:
            raise ValueError("Saved scale-review scores differ from fresh evaluation")
    stages = verify_metrics(predictions, results, labels, records)
    summary, per_record, groups, timelines = [], [], [], []
    costs = read_json(run / "costs.json")
    truth = {cid: read_json(data / "mechanism-reference" / cid / "truth.json") for cid in labels}
    for method, values in predictions.items():
        aggregate = results[method]["aggregate"]
        matched, event05 = [], [0, 0, 0]
        weak_days, weak_detected = 0, 0
        for cid, value in values.items():
            activity = np.asarray(value["activity"])
            for day, reference in enumerate(labels[cid]):
                weak_day = int(reference["weak_signal"]) == 1
                weak_days += weak_day
                weak_detected += weak_day and activity[day] == 1
                timelines.append({"method": method, "case_id": cid, "day_index": day,
                                  "date": reference["date"], "actual_activity": reference["activity_label"],
                                  "actual_stage": reference["feature"], "weak_signal": reference["weak_signal"],
                                  "predicted_activity": value["activity"][day], "predicted_stage": value["stage"][day]})
            boundary03 = boundary_errors(labels[cid], activity, .3)
            boundary05 = boundary_errors(labels[cid], activity, .5)
            matched.extend(boundary03["matched_pairs"])
            event05[0] += len(boundary05["matched_pairs"])
            event05[1] += boundary05["unmatched_prediction_events"]
            event05[2] += boundary05["missed_events"]
            per_record.append({"method": method, "case_id": cid, "family": truth[cid]["family"],
                               "background": truth[cid]["background_group"], **results[method]["per_case"][cid]["daily"]})
        if len(matched) != aggregate["events"]["tp"]:
            raise ValueError("Independent event matches differ")
        usage = costs.get(method, {"requests": 0, "reported_tokens": 0})
        summary.append({"method": method, **aggregate["daily"], "weak_days": weak_days,
                        "weak_recall": int(weak_detected) / weak_days, "coverage": aggregate["coverage"],
                        "event_f1_iou03": aggregate["events"]["f1"], "event_f1_iou05": score_counts(*event05)["f1"],
                        "onset_mae_days": float(np.mean([abs(row["onset_error_days"]) for row in matched])) if matched else None,
                        "offset_mae_days": float(np.mean([abs(row["offset_error_days"]) for row in matched])) if matched else None,
                        "internal_stage_macro_f1": stages[method]["macro_f1"], "stage_macro_f1": aggregate["stage_macro_f1"],
                        "requests": usage["requests"], "reported_tokens": usage["reported_tokens"]})
        for field in ("family", "background_group"):
            for group in sorted({row[field] for row in truth.values()}, key=str):
                ids = [cid for cid in labels if truth[cid][field] == group]
                score = aggregate_scores([results[method]["per_case"][cid] for cid in ids])
                groups.append({"method": method, "field": field, "group": group, "records": len(ids), **score["daily"]})
    write_rows(out / "summary.csv", summary)
    write_rows(out / "per-record.csv", per_record)
    write_rows(out / "groups.csv", groups)
    write_rows(out / "daily-predictions.csv", timelines)
    save_json(out / "combined-predictions.json", predictions)
    save_json(out / "combined-results.json", results)
    save_json(out / "internal-stages.json", stages)
    methods = [f"{arm}_r{rep}" for rep in (0, 1) for arm in
               ("direct_visual", "old_adaptive_image", "scale_gate_image", "matched_budget_image", "fixed_scale_gate_image")]
    methods += ["raw_multiscale_rule", "xgboost_84", "xgboost_scale_information"]
    cmap = ListedColormap(["#c5c9cc", "#fff4dc", "#377dab"])
    for cid, (case, _) in records.items():
        matrix = [[int(row["activity_label"]) for row in labels[cid]]]
        matrix.extend(predictions[name][cid]["activity"] for name in methods)
        raw = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(13, 11), gridspec_kw={"height_ratios": [1, 1, 1, 3]})
        for component, name in enumerate(("N", "E", "U")):
            axes[component].plot(raw[:, component], linewidth=.8, color="#24887b")
            axes[component].set_ylabel(f"{name} (mm)")
            axes[component].grid(alpha=.2)
        axes[3].imshow(matrix, aspect="auto", interpolation="nearest", vmin=-1, vmax=1, cmap=cmap,
                       extent=[-.5, len(raw) - .5, len(matrix) - .5, -.5])
        axes[3].set_yticks(range(len(matrix)), ["Known injection"] + methods, fontsize=7)
        axes[3].set_xlim(0, len(raw) - 1)
        axes[3].set_xlabel("Original day | blue activity; cream quiet; gray unknown/missing")
        fig.suptitle(f"{cid} | previously exposed development, not field truth\n"
                     "Compression gate changes observation admission; same cached local answers")
        fig.tight_layout()
        fig.savefig(out / f"{cid}.png", dpi=130)
        plt.close(fig)
    save_json(out / "state.json", {
        "status": "complete", "configurations": len(predictions), "authority_checks": authority_checks,
        "numerical_predictions_recomputed": (len(numerical) + 1) * len(records),
        "cached_payload_receipts_verified": len(receipts), "group_rows": len(groups),
        "daily_prediction_rows": len(timelines), "full_record_figures": len(records), "new_api_requests": 0,
    })
    print(read_json(out / "state.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.controls, args.out)


if __name__ == "__main__":
    main()
