"""Describe frozen field outputs and provenance without manufacturing accuracy labels."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from prepare_landslide_field_probe import (
    STATIONS,
    daily_coordinates,
    horizontal_reference_rate,
    load_ngl_records,
    relative_displacement,
)
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import save_json, sha
from run_landslide_window_optimization import read_json

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_evaluation import ActivityPrediction, runs

VISUAL_METHODS = ("direct_visual", "fixed_image_review", "adaptive_no_feedback", "adaptive_numeric_feedback")
DISPLAY = {
    "direct_visual": "direct", "fixed_image_review": "fixed",
    "adaptive_no_feedback": "adaptive image", "adaptive_numeric_feedback": "adaptive numeric",
    "robust_rule": "robust", "pelt_rule": "PELT", "xgboost": "XGBoost",
    "robust_xgboost_stage": "robust+XGB stage", "raw_multiscale_rule": "multiscale rule",
    "xgboost_raw_persistence": "long XGBoost",
}


def verify_inputs(run):
    verify_freeze(run)
    protocol = read_json(run / "protocol.json")
    source = Path(protocol["source_bundle"])
    for path, expected in read_json(run / "training-bindings.json").items():
        if sha(Path(path)) != expected:
            raise ValueError("Transferred model/selection binding changed")
    mappings = read_json(run / "field-mapping.json")
    records = {}
    for station in STATIONS:
        path = source / f"{station}.tenv3"
        expected = next(value["source_sha256"] for value in mappings.values() if value["station"] == station)
        if sha(path) != expected:
            raise ValueError("Official field data changed")
        records[station] = load_ngl_records(path)
    cases = {cid: LandslideInput.model_validate_json(
        (run / "inputs" / cid / "input.json").read_text(encoding="utf-8")) for cid in protocol["case_ids"]}
    calendar = next(iter(cases.values())).dates
    rate = horizontal_reference_rate(daily_coordinates(records["P227"], calendar))
    np.testing.assert_allclose(rate * 1000, read_json(run / "reference-correction.json")["rate_NEU_mm_day"])
    for cid, case in cases.items():
        if case.dates != calendar:
            raise ValueError("Field representations have different calendars")
        mapping = mappings[cid]
        coordinates = daily_coordinates(records[mapping["station"]], calendar)
        expected, origin = relative_displacement(
            coordinates, rate if mapping["representation"] == "P227_horizontal_rate_removed" else None)
        actual = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        reconstructed = np.array([row if row is not None else [np.nan] * 3 for row in expected])
        np.testing.assert_allclose(actual, reconstructed, equal_nan=True, atol=1e-6, rtol=0)
        np.testing.assert_allclose(origin, mapping["relative_origin_m"])
        if sha(run / "inputs" / cid / "input.json") != mapping["input_sha256"]:
            raise ValueError("Field input hash differs")
    if (run / "reference-labels.csv").exists() or (run / "reference-subset.csv").exists():
        raise ValueError("This diagnostic protocol has no reference labels")
    return cases, mappings


def verify_receipts(run):
    responses = {}
    for started_path in sorted((run / "requests").glob("*.started.json")):
        started = read_json(started_path)
        response = read_json(started_path.with_name(started_path.name.replace(".started.json", ".response.json")))
        content = [{"type": "text", "text": started["prompt"]}]
        for name, expected in started["image_sha256"].items():
            path = run / "images" / name
            if sha(path) != expected:
                raise ValueError("Field actual input image changed")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
        payload = {
            "model": started["model"], "messages": [{"role": "user", "content": content}],
            "temperature": .1, "top_p": .3, "max_tokens": 2048,
            "enable_thinking": False, "response_format": {"type": "json_object"},
        }
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if actual != started["request_sha256"] or actual != response["request_sha256"]:
            raise ValueError("Field request/response payload binding differs")
        responses[started["request_id"]] = response
    return responses


def comparison_days(left, right, observed):
    return {
        "observed_days": int(observed.sum()),
        "all_state_disagreement_days": int(np.count_nonzero((left != right) & observed)),
        "positive_vs_negative_days": int(np.count_nonzero(((left == 1) & (right == 0)
                                                          | (left == 0) & (right == 1)) & observed)),
        "decided_vs_unknown_days": int(np.count_nonzero(((left == -1) != (right == -1)) & observed)),
        "both_positive_days": int(np.count_nonzero((left == 1) & (right == 1) & observed)),
    }


def plot_case(case, predictions, path, mapping):
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    methods = list(predictions)
    figure, panels = plt.subplots(3 + len(methods), 1, figsize=(13, 5 + .36 * len(methods)),
                                 sharex=True, gridspec_kw={"height_ratios": [3] * 3 + [1] * len(methods)})
    for axis in range(3):
        panels[axis].plot(values[:, axis], linewidth=.7, color=("#147d72", "#cc6d4b", "#a38a32")[axis])
        panels[axis].set_ylabel(("N", "E", "U")[axis] + " mm")
        panels[axis].grid(alpha=.2)
    colors = ListedColormap(["#ead8a2", "#e3e5e8", "#278b82"])
    for index, method in enumerate(methods, 3):
        panels[index].imshow(np.array(predictions[method]["activity"])[None, :], aspect="auto",
                             extent=(-.5, len(values) - .5, 0, 1), cmap=colors, vmin=-1, vmax=1)
        base = method.rsplit("_r", 1)[0] if method.rsplit("_r", 1)[-1].isdigit() else method
        suffix = method[len(base):]
        panels[index].set_ylabel(DISPLAY[base] + suffix, rotation=0, ha="right", va="center", fontsize=8)
        panels[index].set_yticks([])
    panels[-1].set_xlabel("Day index; teal=predicted activity, gray=negative, yellow=unknown; NO reference labels")
    panels[-1].set_xlim(0, len(values) - 1)
    figure.suptitle(f"{mapping['station']} | {mapping['representation']} | unlabelled field diagnostic")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def analyze(run, out):
    if read_json(run / "state.json")["status"] != "predictions_complete":
        raise ValueError("Analyze a complete frozen inference batch")
    out.mkdir(parents=True, exist_ok=False)
    cases, mapping = verify_inputs(run)
    for name, expected in read_json(run / "prediction-freeze.json").items():
        if sha(run / name) != expected:
            raise ValueError("Frozen field inference changed")
    receipts = verify_receipts(run)
    predictions = read_json(run / "predictions.json")
    predictions.update(read_json(run / "numeric-predictions.json"))
    summary, repeat_differences, representation_differences = [], [], []
    for method, records in predictions.items():
        for cid, value in records.items():
            activity, stage = np.array(value["activity"]), np.array(value["stage"])
            observed = np.array([row is not None for row in cases[cid].displacement_mm])
            prediction = ActivityPrediction(activity, stage, status=value["status"], stage_status=value["stage_status"])
            prediction.validate(len(observed))
            if np.any(activity[~observed] != -1) or np.any(stage[~observed] != -1):
                raise ValueError("Field missing-day decision invented")
            summary.append({
                "method": method, "case_id": cid, "station": mapping[cid]["station"],
                "representation": mapping[cid]["representation"], "observed_days": int(observed.sum()),
                "predicted_activity_days": int(np.count_nonzero(activity == 1)),
                "unknown_observed_days": int(np.count_nonzero((activity == -1) & observed)),
                "predicted_A_days": int(np.count_nonzero(stage == 1)),
                "predicted_S_days": int(np.count_nonzero(stage == 2)),
                "predicted_D_days": int(np.count_nonzero(stage == 3)),
                "unknown_stage_in_predicted_activity_days": int(np.count_nonzero((stage == -1) & (activity == 1))),
                "inclusive_predicted_activity_intervals": [[start, stop - 1] for start, stop in runs(activity == 1)],
            })
    for method in VISUAL_METHODS:
        for cid, case in cases.items():
            observed = np.array([row is not None for row in case.displacement_mm])
            left, right = predictions[method + "_r0"][cid], predictions[method + "_r1"][cid]
            difference = comparison_days(np.array(left["activity"]), np.array(right["activity"]), observed)
            repeat_differences.append({"method": method, "case_id": cid, **difference})
    for method, records in predictions.items():
        for index, station in enumerate(STATIONS):
            raw_id, corrected_id = f"case_{101 + 2 * index:04d}", f"case_{102 + 2 * index:04d}"
            observed = np.array([row is not None for row in cases[raw_id].displacement_mm])
            difference = comparison_days(np.array(records[raw_id]["activity"]),
                                         np.array(records[corrected_id]["activity"]), observed)
            representation_differences.append({"method": method, "station": station, **difference})
    costs = {}
    selections = read_json(run / "selections.json")
    for repetition in range(2):
        for method in VISUAL_METHODS:
            request_ids = []
            for cid in cases:
                request_ids.append(f"{cid}-direct-r{repetition}")
                if method == "direct_visual":
                    continue
                choice = selections[f"{cid}-r{repetition}"]
                indices = choice["fixed_indices"] if method == "fixed_image_review" else [row["index"] for row in choice["selected"]]
                arm = "numeric" if method == "adaptive_numeric_feedback" else "image"
                request_ids.extend(f"{cid}-window-{index:02d}-{arm}-r{repetition}" for index in indices)
            rows = [receipts[request_id] for request_id in request_ids]
            costs[f"{method}_r{repetition}"] = {
                "requests": len(rows), "tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in rows),
                "sum_response_seconds": sum(row["seconds"] for row in rows),
                "shared_requests": True,
            }
    for cid, case in cases.items():
        plot_case(case, {name: value[cid] for name, value in predictions.items()}, out / f"{cid}-activity.png", mapping[cid])
    for name, rows in (("summary", summary), ("repeat-differences", repeat_differences),
                       ("representation-differences", representation_differences)):
        with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_json(out / "diagnostics.json", {
        "scope": "Unlabelled, paired representation diagnostic; no reference accuracy or sample independence claim",
        "summary": summary, "repeat_differences": repeat_differences,
        "representation_differences": representation_differences, "costs": costs,
        "unique_actual_requests": len(receipts),
        "unique_total_tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts.values()),
        "failures": read_json(run / "failures.json"),
    })
    save_json(out / "verification.json", {
        "status": "verified", "reference_labels": "absent_not_scorable",
        "input_calendar_and_reconstruction": len(cases), "method_case_mask_gate_groups": len(summary),
        "actual_payload_and_image_bindings": len(receipts), "frozen_files": len(read_json(run / "freeze.json")),
        "source_sha256": sha(Path(__file__)),
        "prediction_sha256": sha(run / "predictions.json"),
    })
    print("Field diagnostic verified:", len(receipts), "actual requests; no accuracy computed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.out)


if __name__ == "__main__":
    main()
