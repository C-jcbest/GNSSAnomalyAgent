"""Test interval-scale accumulation on frozen predictions, without new VLM calls.

This is a development falsification probe. Previously viewed confirmation and
unlabelled field outputs cannot confirm a method chosen from this replay.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_evaluation import (
    ActivityPrediction,
    aggregate_scores,
    evaluate_case,
    event_masks,
)

POLICIES = ("contradiction_only", "require_support")
PROTOCOL = {
    "schema": "interval-consistency-development-probe-v1",
    "scope": "Cached development replay, including already-viewed old confirmation. "
             "Field data have no accuracy reference. No new VLM inference or model fitting.",
    "hypothesis": "Whole predicted intervals expose reversal or absent accumulated "
                  "motion that day-centred evidence and local images can miss.",
    "intervals": "Common event_masks: bridge at most seven unknown days, never negatives. "
                 "Only predicted positive days may be changed to unknown.",
    "blocks": "Disjoint first, middle and last 14 calendar days of each event; "
              "each requires eight complete raw N/E/U observations. Intervals shorter "
              "than 42 days or with unsupported blocks are unchanged.",
    "noise": "Existing per-axis whole-record raw first-difference MAD noise proxy. "
             "Whiten both changes before comparing direction; no significance claim.",
    "minimum_change_over_noise": 2.0,
    "consistent_direction": "Whitened change cosine strictly greater than zero",
    "policies": {
        "contradiction_only": "Both changes exceed noise proxy and cosine <= 0: abstain",
        "require_support": "Both changes exceed noise proxy and cosine > 0: retain; "
                           "otherwise abstain if the three blocks are observable",
    },
    "fairness": "Apply exactly the same deterministic check to all four visual arms "
                "and all six gated numerical baselines on mechanism/confirmation/field "
                "batches, retaining original stage head. Pilot uses only visual arms: "
                "its historical numeric serialization lacks stage_status and is not adapted.",
    "denominator": "Unknown predictions retain positive-label FN exposure and known-day "
                   "coverage denominator. Report TP lost, FP removed and weak recall.",
    "selection": "No threshold sweep. A candidate failing a stable benefit/risk check "
                 "is not promoted. Any selected revision requires a new frozen confirmation.",
}


def interval_evidence(values, noise, start, stop):
    """Inspect one whole event; insufficient observations are not negative evidence."""
    if stop - start < 42:
        return {"status": "too_short", "block_ranges": []}
    starts = (start, start + (stop - start - 14) // 2, stop - 14)
    medians = []
    blocks = []
    for left in starts:
        sample = values[left:left + 14]
        complete = sample[np.isfinite(sample).all(axis=1)]
        blocks.append({"start": left, "stop": left + 14, "observed": len(complete)})
        if len(complete) >= 8:
            medians.append(np.median(complete, axis=0))
    if len(medians) != 3:
        return {"status": "insufficient_blocks", "block_ranges": blocks}
    changes = np.diff(np.asarray(medians), axis=0) / noise
    lengths = np.linalg.norm(changes, axis=1)
    cosine = None
    if lengths[0] > 0 and lengths[1] > 0:
        cosine = float(np.dot(changes[0], changes[1]) / np.prod(lengths))
    supported = bool(np.all(lengths >= PROTOCOL["minimum_change_over_noise"]))
    reversed_motion = supported and cosine is not None and cosine <= 0
    consistent = supported and cosine is not None and cosine > 0
    return {
        "status": "observable", "block_ranges": blocks,
        "raw_block_medians_mm": np.asarray(medians).tolist(),
        "change_over_noise": lengths.tolist(), "whitened_cosine": cosine,
        "reversal": reversed_motion, "consistent_accumulation": consistent,
    }


def check_prediction(case, noise, prediction, policy):
    """Return a new prediction, preserving negatives, existing unknowns and failures."""
    if policy not in POLICIES:
        raise ValueError("Unknown interval consistency policy")
    values = np.asarray([
        row if row is not None else [np.nan] * 3 for row in case.displacement_mm
    ], dtype=float)
    noise = np.asarray(noise, dtype=float)
    if noise.shape != (3,) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Noise must contain three finite positive axis proxies")
    prediction.validate(len(values))
    observed = np.isfinite(values).all(axis=1)
    if np.any(prediction.activity[~observed] != -1):
        raise ValueError("Missing observations must remain unknown")
    activity = prediction.activity.copy()
    stage = prediction.stage.copy()
    details = []
    for mask in event_masks(prediction.activity):
        indices = np.flatnonzero(mask)
        start, stop = int(indices[0]), int(indices[-1] + 1)
        evidence = interval_evidence(values, noise, start, stop)
        abstain = False
        if evidence["status"] == "observable":
            if policy == "contradiction_only":
                abstain = evidence["reversal"]
            else:
                abstain = not evidence["consistent_accumulation"]
        affected = mask & (prediction.activity == 1)
        if abstain:
            activity[affected] = -1
            stage[affected] = -1
        details.append({"start": start, "stop": stop, "abstain": abstain,
                        "changed_positive_days": int(affected.sum()) if abstain else 0,
                        "evidence": evidence})
    result = ActivityPrediction(activity, stage, status=prediction.status,
                                gated=prediction.gated, stage_status=prediction.stage_status)
    result.validate(len(values))
    return result, details


def cached_predictions(run, extra_path=None, include_numeric=True):
    for name, expected in read_json(run / "freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError(f"Frozen input/source changed: {run / name}")
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError(f"Frozen prediction changed: {run / name}")
    predictions = read_json(run / "predictions.json")
    if include_numeric:
        for name, cases in read_json(run / "numeric-predictions.json").items():
            if name != "ungated_rule":
                predictions[name] = cases
    if extra_path is not None:
        for name, cases in read_json(extra_path).items():
            if name in predictions:
                raise ValueError("Duplicate baseline")
            predictions[name] = cases
    return predictions


def run_probe(out):
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", PROTOCOL)
    shutil.copyfile(Path(__file__), out / "source.py")
    save_json(out / "state.json", {"status": "running", "new_vlm_requests": 0})
    started = time.monotonic()
    root = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
    runs = (
        ("pilot-v3", None),
        ("mechanism-v1", root / "raw-baselines-v1/predictions.json"),
        ("confirmation-v1", root / "confirmation-data-v1/additional-baselines/predictions.json"),
        ("field-transfer-v1", None),
    )
    summary, changed_rows, results, bindings = [], [], {}, {}
    for name, extra in runs:
        if time.monotonic() - started > 600:
            raise RuntimeError("Development replay exceeded ten-minute wall-time budget")
        run = root / name
        destination = out / name
        destination.mkdir()
        methods = cached_predictions(run, extra, include_numeric=name != "pilot-v3")
        case_ids = read_json(run / "protocol.json")["case_ids"]
        cases, features = {}, {}
        for cid in case_ids:
            cases[cid], features[cid] = load_case_features(run, cid)
        # Complete every candidate output before opening reference labels.
        predictions, diagnostics = {}, {}
        for method, records in methods.items():
            for policy in ("original", *POLICIES):
                arm = f"{method}__{policy}"
                predictions[arm], diagnostics[arm] = {}, {}
                for cid in case_ids:
                    value = records[cid]
                    original = ActivityPrediction(
                        np.array(value["activity"]), np.array(value["stage"]),
                        status=value["status"], gated=value["gated"],
                        stage_status=value["stage_status"],
                    )
                    original.validate(len(cases[cid].dates))
                    if policy == "original":
                        prediction, detail = original, []
                    else:
                        prediction, detail = check_prediction(
                            cases[cid], features[cid].noise, original, policy)
                    predictions[arm][cid] = prediction.to_dict()
                    diagnostics[arm][cid] = detail
        save_json(destination / "predictions.json", predictions)
        save_json(destination / "diagnostics.json", diagnostics)
        save_json(destination / "prediction-freeze.json", {
            file: sha(destination / file) for file in ("predictions.json", "diagnostics.json")
        })
        labelled = name != "field-transfer-v1"
        labels = load_labels(run / "reference-subset.csv", cases) if labelled else None
        results[name] = {}
        for arm, records in predictions.items():
            method, policy = arm.split("__")
            row = {"run": name, "method": method, "policy": policy, "labelled": labelled,
                   "activity_days": 0, "unknown_observed_days": 0,
                   "changed_tp_to_unknown": 0, "changed_fp_to_unknown": 0,
                   "weak_days": 0, "weak_detected_days": 0}
            per_case, boundaries = {}, []
            for cid, value in records.items():
                activity, stage = np.array(value["activity"]), np.array(value["stage"])
                original = np.array(methods[method][cid]["activity"])
                changed = (original == 1) & (activity == -1)
                observed = features[cid].observed
                row["activity_days"] += int(np.count_nonzero(activity == 1))
                row["unknown_observed_days"] += int(np.count_nonzero((activity == -1) & observed))
                detail = {"run": name, "method": method, "policy": policy, "case_id": cid,
                          "changed_positive_days": int(changed.sum())}
                if labelled:
                    actual = np.array([int(item["activity_label"]) for item in labels[cid]])
                    weak = np.array([
                        int(item.get("weak_signal", 0)) == 1
                        or item["feature"] == "slow_displacement" for item in labels[cid]
                    ])
                    lost_tp = int(np.count_nonzero(changed & (actual == 1)))
                    removed_fp = int(np.count_nonzero(changed & (actual == 0)))
                    row["changed_tp_to_unknown"] += lost_tp
                    row["changed_fp_to_unknown"] += removed_fp
                    row["weak_days"] += int(weak.sum())
                    row["weak_detected_days"] += int(np.count_nonzero(weak & (activity == 1)))
                    detail.update(lost_tp=lost_tp, removed_fp=removed_fp)
                    prediction = ActivityPrediction(activity, stage)
                    per_case[cid] = evaluate_case(labels[cid], prediction)
                    boundaries.extend(boundary_errors(labels[cid], activity, .3)["matched_pairs"])
                changed_rows.append(detail)
            if labelled:
                score = aggregate_scores(list(per_case.values()))
                results[name][arm] = {"aggregate": score, "per_case": per_case,
                                      "boundaries": boundaries}
                row.update(score["daily"])
                row.update(event_f1=score["events"]["f1"], coverage=score["coverage"],
                           stage_macro_f1=score["stage_macro_f1"],
                           weak_recall=row["weak_detected_days"] / row["weak_days"]
                           if row["weak_days"] else None,
                           onset_mae_matched_days=float(np.mean([
                               abs(pair["onset_error_days"]) for pair in boundaries
                           ])) if boundaries else None,
                           offset_mae_matched_days=float(np.mean([
                               abs(pair["offset_error_days"]) for pair in boundaries
                           ])) if boundaries else None)
            summary.append(row)
        bound = {file: sha(run / file) for file in (
            "freeze.json", "prediction-freeze.json", "protocol.json",
            "predictions.json", "numeric-predictions.json",
        )}
        if labelled:
            bound["reference-subset.csv"] = sha(run / "reference-subset.csv")
        if extra is not None:
            bound[str(extra)] = sha(extra)
        bindings[name] = bound
        print(f"Cached replay complete: {name}, {len(case_ids)} records, "
              f"{len(methods)} methods, no new VLM requests", flush=True)
    save_json(out / "results.json", results)
    save_json(out / "source-bindings.json", bindings)
    for file, rows in (("summary.csv", summary), ("per-record-changes.csv", changed_rows)):
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (out / file).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    save_json(out / "manifest.json", {
        "files": {str(path.relative_to(out)): sha(path)
                  for path in out.rglob("*") if path.is_file() and path.name != "state.json"}
    })
    save_json(out / "state.json", {"status": "complete", "new_vlm_requests": 0,
                                   "elapsed_seconds": time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_probe(args.out)


if __name__ == "__main__":
    main()
