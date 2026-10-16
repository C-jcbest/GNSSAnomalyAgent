"""Independently verify stage repair, export every configuration and aligned figures."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from finalize_landslide_adaptive_evidence import verify_metrics
from run_landslide_comparison import STAGES, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case, runs
from gnss_sim.landslide_stage_raw_consistency_v2 import StageContrastPolicy, stage_conflict_mask
from gnss_sim.landslide_visual import parse_stages


def verify_all_receipts(run):
    """Bind every attempted request, including connection failures, to its input."""
    receipts = []
    for path in sorted((run / "requests").glob("*.response.json")):
        receipt = read_json(path)
        started = read_json(path.with_name(path.name.replace(".response.json", ".started.json")))
        content = [{"type": "text", "text": started["prompt"]}]
        for name, expected in started["image_sha256"].items():
            image = run / "images" / name
            if sha(image) != expected:
                raise ValueError("Attempted request image changed")
            encoded = base64.b64encode(image.read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + encoded}})
        payload = {"model": started["model"], "messages": [{"role": "user", "content": content}],
                   "temperature": 0.1, "top_p": 0.3, "max_tokens": 2048,
                   "enable_thinking": False, "response_format": {"type": "json_object"}}
        binding = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if binding != started["request_sha256"] or binding != receipt["request_sha256"]:
            raise ValueError("Attempted request binding differs")
        if receipt["status"] not in ("failed", "returned_json"):
            raise ValueError("Unknown receipt status")
        receipts.append(receipt)
    return receipts


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
    receipts = verify_all_receipts(run)
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
        if receipt["status"] == "failed":
            if request_id not in failures:
                raise ValueError("Failed receipt missing from failure ledger")
            prediction = ActivityPrediction(gate, np.full(len(gate), -1), stage_status="failed")
        else:
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
    # Post-hoc completion of ordinary fixed review: the registered fixed-scope
    # arm shared exact conflict permissions and therefore cannot measure the
    # value/risk of replacing an entire fixed target. These controls use the
    # already frozen answers, never become a selectable primary, and add no API.
    selections, contrasts = read_json(run / "selections.json"), read_json(run / "contrasts.json")
    fixed_changes = {}
    for rep in range(2):
        for feedback in (False, True):
            method = f"fixed_{'numeric' if feedback else 'image'}_full_target_veto_r{rep}"
            values = {}
            for cid, original in initial[f"direct_visual_native_stage_r{rep}"].items():
                key = f"{cid}-r{rep}"
                target = selections[key]["fixed"]
                stage = np.asarray(original["stage"]).copy()
                activity = np.asarray(original["activity"])
                stage_status = original["stage_status"]
                if target is not None:
                    left, right = target
                    request_id = f"{key}-target-{left}-{right}-{'numeric' if feedback else 'image'}"
                    replacement = np.asarray(local[request_id]["stage"]).copy()
                    replacement[stage_conflict_mask(
                        replacement, {42: np.asarray(contrasts[key], dtype=float)},
                        StageContrastPolicy(separations=(42,), minimum_conflict_days=1))] = -1
                    permission = np.zeros(len(stage), dtype=bool)
                    permission[left:right] = activity[left:right] == 1
                    stage[permission] = replacement[permission]
                    stage_status = "success"
                prediction = ActivityPrediction(activity.copy(), stage, status=original["status"],
                                                stage_status=stage_status)
                prediction.validate(len(stage))
                values[cid] = prediction.to_dict()
            predictions[method] = values
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
    for method, values in predictions.items():
        if method in results:
            continue
        scores = {cid: evaluate_case(labels[cid], ActivityPrediction(np.asarray(value["activity"]),
                                                                    np.asarray(value["stage"]),
                                                                    stage_status=value["stage_status"]))
                  for cid, value in values.items()}
        results[method] = {"aggregate": aggregate_scores(list(scores.values())), "per_case": scores,
                           "scope": "Post-hoc ordinary fixed-target control; not a registered primary"}
        rep = int(method[-1])
        changes = {}
        for cid, value in values.items():
            before = np.asarray(initial[f"direct_visual_native_stage_r{rep}"][cid]["stage"])
            after = np.asarray(value["stage"])
            expected = np.asarray([STAGES.get(row["feature"], -1) for row in labels[cid]])
            active = np.asarray([int(row["activity_label"]) for row in labels[cid]])
            evaluable = (active == 1) & (expected > 0)
            changes[cid] = {
                "wrong_to_correct": int((evaluable & (before != expected) & (after == expected)).sum()),
                "correct_to_wrong_or_unknown": int((evaluable & (before == expected) & (after != expected)).sum()),
                "wrong_to_unknown": int((evaluable & (before > 0) & (before != expected) & (after == -1)).sum()),
            }
        fixed_changes[method] = {"per_case": changes,
                                "total": {key: sum(row[key] for row in changes.values())
                                          for key in next(iter(changes.values()))}}
    internal = verify_metrics(predictions, results, source / "reference-subset.csv", source)
    for method, row in internal.items():
        if "internal" in results[method]:
            if not np.isclose(row["macro_f1"], results[method]["internal"]["macro_f1"]):
                raise ValueError("Independent internal stage metric differs")
        results[method]["independent_internal"] = row
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
                selector = "fixed" if method.startswith("fixed_") else "adaptive"
                if selection[selector] is not None:
                    left, right = selection[selector]
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
    save_json(out / "ordinary-fixed-review-changes.json", fixed_changes)
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
                               ("adaptive_numeric_feedback_conflict_scope_veto", "Local + numerical"),
                               ("fixed_numeric_full_target_veto", "Fixed full (post hoc)")):
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
        "actual_request_replays": replayed,
        "successful_provider_responses": sum(row["status"] == "returned_json" for row in receipts),
        "failed_receipts": sum(row["status"] == "failed" for row in receipts),
        "local_stage_contract_failures": len(failures), "independent_metric_configs": len(results),
        "activity_and_permission_checks": permissions, "daily_csv_rows": rows_written,
        "full_record_figures": len(cases), "frozen_files_checked": checked,
        "additional_control_binding": {"path": str(args.additional / "predictions.json"),
                                       "sha256": sha(args.additional / "predictions.json")},
        "source_sha256": sha(Path(__file__)), "independent_field_accuracy": "not verified",
        "ordinary_fixed_controls": "Post-hoc full-target controls; predictions share frozen answers, no API",
    })
    print(read_json(out / "verification.json"), flush=True)


if __name__ == "__main__":
    main()
