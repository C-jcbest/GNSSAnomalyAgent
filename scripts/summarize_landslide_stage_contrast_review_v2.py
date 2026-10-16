"""Independently verify stage repair, export every configuration and aligned figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from finalize_landslide_adaptive_evidence import verify_metrics, verify_request_receipts
from run_landslide_comparison import STAGES, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case, runs
from gnss_sim.landslide_stage_raw_consistency_v2 import StageContrastPolicy, stage_conflict_mask
from gnss_sim.landslide_visual import parse_stages


def render_comparison(case, labels, predictions, path):
    values = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    fig, axes = plt.subplots(3 + len(predictions) + 1, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2] + [1] * (len(predictions) + 1)})
    for axis, name in enumerate(("N", "E", "U")):
        axes[axis].plot(np.arange(len(values)), values[:, axis], linewidth=0.7)
        axes[axis].set_ylabel(name + " mm")
        axes[axis].grid(alpha=0.2)
    reference = np.array([STAGES.get(row["feature"], -1) if int(row["activity_label"]) != 0 else 0
                          for row in labels])
    bands = [("Known injected stage", reference)] + [(name, np.asarray(value["stage"]))
                                                      for name, value in predictions.items()]
    colors = {-1: "#dedede", 1: "#dd7259", 2: "#a986c7", 3: "#70a7c0"}
    for axis, (name, stages) in zip(axes[3:], bands, strict=True):
        for code, color in colors.items():
            for left, right in runs(stages == code):
                axis.axvspan(left - 0.5, right - 0.5, color=color)
        axis.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
        axis.set_yticks([])
    axes[-1].set_xlim(-0.5, len(values) - 0.5)
    axes[-1].set_xlabel("Original day | A orange; S purple; D blue; unknown/missing grey; none white")
    fig.suptitle(case.case_id + " | Controlled development, not independent field accuracy")
    fig.subplots_adjust(left=0.22, right=0.98, top=0.94, bottom=0.08, hspace=0.15)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--additional", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.out
    out.mkdir(parents=True, exist_ok=False)
    config = read_json(run / "protocol.json")
    source = Path(config["source"])
    if read_json(run / "state.json")["status"] != "complete":
        raise ValueError("Run must finish first")
    initial, local = read_json(run / "initial-predictions.json"), read_json(run / "local-predictions.json")
    predictions, results = read_json(run / "predictions.json"), read_json(run / "results.json")
    receipts = verify_request_receipts(run)
    plan = read_json(run / "request-plan.json")
    failures = read_json(run / "failures.json")
    replayed = 0
    for receipt in receipts:
        request_id = receipt["request_id"]
        item = plan[request_id]
        cid, rep = item["case_id"], item["rep"]
        _, features = load_case_features(source, cid)
        gate = np.asarray(initial[f"direct_visual_native_stage_r{rep}"][cid]["activity"]).copy()
        left, right = item["target"]
        gate[:left], gate[right:] = 0, 0
        gate[~features.observed] = -1
        try:
            prediction = parse_stages(json.loads(receipt["output"]), gate, features.observed)
            prediction.stage[(gate == 1) & ~features.supported] = -1
        except ValueError:
            if request_id not in failures:
                raise
            prediction = ActivityPrediction(gate, np.full(len(gate), -1), stage_status="failed")
        if prediction.to_dict() != local[request_id]:
            raise ValueError("Local answer replay differs")
        replayed += 1
    # Two stronger raw-persistence controls were trained/calibrated on the same split.
    additional = read_json(args.additional / "predictions.json")
    cases = {cid: load_case_features(source, cid)[0] for cid in initial["direct_visual_native_stage_r0"]}
    labels = load_labels(source / "reference-subset.csv", cases)
    for method, values in additional.items():
        subset = {cid: values[cid] for cid in cases}
        predictions[method] = subset
        scores = {cid: evaluate_case(labels[cid], ActivityPrediction(np.asarray(value["activity"]),
                                                                    np.asarray(value["stage"]),
                                                                    stage_status=value["stage_status"]))
                  for cid, value in subset.items()}
        results[method] = {"aggregate": aggregate_scores(list(scores.values())), "per_case": scores}
    internal = verify_metrics(predictions, results, source / "reference-subset.csv", source)
    for method, row in internal.items():
        if "internal" in results[method]:
            if not np.isclose(row["macro_f1"], results[method]["internal"]["macro_f1"]):
                raise ValueError("Independent internal stage metric differs")
        results[method]["independent_internal"] = row
    selections, contrasts = read_json(run / "selections.json"), read_json(run / "contrasts.json")
    permissions = 0
    for method, values in predictions.items():
        if not method.endswith(("_r0", "_r1")):
            continue
        rep = int(method[-1])
        for cid, value in values.items():
            original = initial[f"direct_visual_native_stage_r{rep}"][cid]
            if value["activity"] != original["activity"]:
                raise ValueError("Stage repair changed activity")
            changed = np.asarray(value["stage"]) != np.asarray(original["stage"])
            selection = selections[f"{cid}-r{rep}"]
            mask = np.zeros(len(changed), dtype=bool)
            mask[selection["conflict_days"]] = True
            if "full_target" in method:
                mask[:] = False
                if selection["adaptive"] is not None:
                    left, right = selection["adaptive"]
                    mask[left:right] = np.asarray(original["activity"])[left:right] == 1
            elif method.startswith("fixed_image") or method.startswith("adaptive_"):
                target = selection["fixed" if method.startswith("fixed_image") else "adaptive"]
                target_mask = np.zeros(len(mask), dtype=bool)
                if target is not None:
                    target_mask[target[0]:target[1]] = True
                mask &= target_mask
            if np.any(changed & ~mask):
                raise ValueError("Stage repair escaped registered permissions")
            if "_veto_r" in method and "_no_veto_" not in method:
                contradictions = stage_conflict_mask(np.asarray(value["stage"]),
                                                     {42: np.asarray(contrasts[f"{cid}-r{rep}"], dtype=float)},
                                                     StageContrastPolicy(separations=(42,), minimum_conflict_days=1))
                if np.any(contradictions & mask):
                    raise ValueError("A contradictory replacement survived the veto")
            permissions += 1
    save_json(out / "combined-predictions.json", predictions)
    save_json(out / "combined-results.json", results)
    fields = ["method", "case_id", "day_index", "date", "reference_activity", "reference_stage",
              "activity", "stage", "original_raw_conflict", "observed"]
    rows_written = 0
    with (out / "daily-predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, values in predictions.items():
            for cid, value in values.items():
                rep = int(method[-1]) if method.endswith(("_r0", "_r1")) else None
                conflicts = set(selections[f"{cid}-r{rep}"]["conflict_days"]) if rep is not None else set()
                for day, row in enumerate(labels[cid]):
                    writer.writerow(dict(zip(fields, [method, cid, day, row["date"], row["activity_label"],
                                                      row["feature"], value["activity"][day], value["stage"][day],
                                                      int(day in conflicts),
                                                      int(cases[cid].displacement_mm[day] is not None)], strict=True)))
                    rows_written += 1
    (out / "figures").mkdir()
    for cid, case in cases.items():
        rows = {}
        for rep in range(2):
            for arm, title in (("initial", "Initial"), ("raw_contrast_abstention", "Raw veto"),
                               ("adaptive_numeric_feedback_conflict_scope_veto", "Local + numerical")):
                rows[f"{title} / {rep + 1}"] = predictions[f"{arm}_r{rep}"][cid]
        render_comparison(case, labels[cid], rows, out / "figures" / f"{cid}.png")
    # Confirm old and prepared sources, local and full predictions have not changed.
    checked = 0
    for manifest in ("freeze.json", "dependency-freeze.json"):
        for name, expected in read_json(run / manifest).items():
            path = Path(name) if manifest.startswith("dependency") else run / name
            if sha(path) != expected:
                raise ValueError("Frozen source changed during analysis")
            checked += 1
    if sha(run / "predictions.json") != read_json(run / "prediction-freeze.json")["sha256"]:
        raise ValueError("Full prediction freeze changed")
    save_json(out / "verification.json", {
        "actual_request_replays": replayed, "successful_provider_responses": len(receipts),
        "local_stage_contract_failures": len(failures), "independent_metric_configs": len(results),
        "activity_and_permission_checks": permissions, "daily_csv_rows": rows_written,
        "full_record_figures": len(cases), "frozen_files_checked": checked,
        "additional_control_binding": {"path": str(args.additional / "predictions.json"),
                                       "sha256": sha(args.additional / "predictions.json")},
        "source_sha256": sha(Path(__file__)), "independent_field_accuracy": "not verified",
    })
    print(read_json(out / "verification.json"), flush=True)


if __name__ == "__main__":
    main()
