"""Extend the frozen numerical pilot with visual localization and matched ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from run_landslide_comparison import (
    DEFAULT_EVIDENCE,
    ROOT,
    ActivityPrediction,
    LandslideDiagnostics,
    LandslideInput,
    load_labels,
    observation_features,
    rule_stages,
    runs,
    save_json,
    sha,
    write_report,
)

from gnss_sim.landslide_diagnostics import render_diagnostics
from gnss_sim.landslide_visual import (
    ACTIVITY_TASK,
    EVOLUTION_TASK,
    VisualRequests,
    parse_activity,
    parse_stages,
)


def render_raw(case, path):
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    fig, panels = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    for axis, panel in enumerate(panels):
        panel.plot(values[:, axis], linewidth=.65, color=("#147d72", "#cc6d4b", "#a38a32")[axis])
        low, high = min(0, np.nanmin(values[:, axis])), max(0, np.nanmax(values[:, axis]))
        center, span = (low + high) / 2, max(60, high - low)
        panel.set_ylim(center - .75 * span, center + .75 * span)
        panel.set_ylabel(("N", "E", "U")[axis] + " displacement (mm)")
        panel.grid(alpha=.2)
    panels[-1].set_xlim(0, len(values) - 1)
    panels[-1].set_xticks(np.unique(np.r_[np.arange(0, len(values), 100), len(values) - 1]))
    panels[-1].set_xlabel("Zero-based day index, original daily calendar; gaps are missing observations")
    fig.suptitle(case.case_id + " | Raw observed displacement only")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def inclusive_spans(activity):
    return [[start, stop - 1] for start, stop in runs(activity == 1)]


def infer_stages(transport, request_id, activity, observed, prompt, images, failures, gated=True):
    try:
        payload = transport.ask(request_id, prompt, images)
        return parse_stages(payload, activity, observed, gated=gated)
    except (ValueError, RuntimeError) as error:
        failures[request_id] = str(error)
        # A failed second layer does not erase a successful first-layer localization.
        stage = np.full(len(activity), -1, dtype=int)
        return ActivityPrediction(activity.copy(), stage, gated=gated, stage_status="failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--localization-from", type=Path,
                        help="Reuse a completed visual run's first layer; never resample its localization")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = json.loads((args.numeric / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Numerical calibration must finish first")
    choices = json.loads((args.numeric / "frozen-selection.json").read_text(encoding="utf-8"))
    change = choices["rule_stage"]["relative_change_60d"]
    case_ids = manifest["split"]["development_evaluation"]
    cached_localization = None
    if args.localization_from is not None:
        cache_manifest = json.loads((args.localization_from / "visual-manifest.json").read_text(encoding="utf-8"))
        if (cache_manifest["status"] != "complete" or cache_manifest["case_ids"] != case_ids
                or cache_manifest["numeric_manifest_sha256"] != sha(args.numeric / "manifest.json")
                or cache_manifest["model"] != args.model):
            raise ValueError("Localization cache has a different model, dataset or split")
        cache_predictions = json.loads((args.localization_from / "predictions.json").read_text(encoding="utf-8"))
        cached_localization = cache_predictions["visual_rule"]
    paths = {cid: Path(manifest["data"]) / "cases" / cid / "input.json" for cid in case_ids}
    if any(sha(path) != manifest["input_sha256"][cid] for cid, path in paths.items()):
        raise ValueError("Input changed since numeric calibration")
    if sha(Path(manifest["labels"])) != manifest["labels_sha256"]:
        raise ValueError("Labels changed since numeric calibration")
    all_cases = {path.parent.name: LandslideInput.model_validate_json(path.read_text(encoding="utf-8"))
                 for path in (Path(manifest["data"]) / "cases").glob("*/input.json")}
    previous = json.loads((args.numeric / "predictions.json").read_text(encoding="utf-8"))
    methods = {name: {cid: ActivityPrediction(np.array(row["activity"]), np.array(row["stage"]),
                                             row["status"], row["gated"]) for cid, row in predictions.items()}
               for name, predictions in previous.items()}
    for name in ("visual_rule", "visual_model", "numeric_model", "visual_model_raw", "visual_model_ungated", "single_visual"):
        methods[name] = {}
    budget = (5 if cached_localization is not None else 6) * len(case_ids)
    transport = VisualRequests(args.out / "requests", ROOT / ".env", args.model, budget=budget)
    run = {"status": "running", "numeric_manifest_sha256": sha(args.numeric / "manifest.json"),
           "numeric_selection_sha256": sha(args.numeric / "frozen-selection.json"), "model": args.model,
           "case_ids": case_ids, "budget": budget, "prompt_frozen_before_calls": True,
           "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                             [Path(__file__), ROOT / "src/gnss_sim/landslide_visual.py"]},
           "activity_task": ACTIVITY_TASK, "evolution_task": EVOLUTION_TASK,
           "raw_axis_policy": "per-axis min span 60mm, padded to 1.5 times observed span; fixed for every arm"}
    if args.localization_from is not None:
        run["localization_source"] = str(args.localization_from.resolve())
        run["localization_predictions_sha256"] = sha(args.localization_from / "predictions.json")
    save_json(args.out / "visual-manifest.json", run)
    failures = {}
    for cid in case_ids:
        case = all_cases[cid]
        motions = {window: LandslideDiagnostics.model_validate_json(
            (DEFAULT_EVIDENCE / cid / f"motion-{window}.json").read_text(encoding="utf-8")) for window in (31, 61, 91)}
        canonical = hashlib.sha256(case.model_dump_json().encode()).hexdigest()
        if any(motion.input_sha256 != canonical or motion.case_id != cid or motion.window_days != window
               for window, motion in motions.items()):
            raise ValueError("Visual auxiliary evidence does not match the observation input")
        feature = observation_features(case, motions)
        raw = args.out / f"{cid}-raw.png"
        auxiliary = args.out / f"{cid}-auxiliary.png"
        render_raw(case, raw)
        auxiliary.write_bytes(render_diagnostics(case, motions[61]))
        task = ACTIVITY_TASK.format(last=len(case.dates) - 1)
        activity_prompt = task + '\nReturn exactly {"activity":[[start,end]],"uncertain":[[start,end]]}. No stage labels or other keys.'
        try:
            if cached_localization is None:
                payload = transport.ask(cid + "-locate", activity_prompt, [raw])
                activity = parse_activity(payload, feature.observed)
            else:
                cached = cached_localization[cid]
                prediction = ActivityPrediction(np.array(cached["activity"]), np.array(cached["stage"]),
                                                 cached["status"], cached["gated"], cached["stage_status"])
                prediction.validate(len(case.dates))
                if prediction.status != "success":
                    raise ValueError("Cached localization was unsuccessful")
                activity = prediction.activity.copy()
        except (ValueError, RuntimeError) as error:
            failures[cid + "-locate"] = str(error)
            activity = np.full(len(case.dates), -1, dtype=int)
        locator_failed = cid + "-locate" in failures
        rule_prediction = rule_stages(feature, activity, change)
        if locator_failed:
            rule_prediction = ActivityPrediction(activity.copy(), np.full(len(activity), -1), status="failed")
        methods["visual_rule"][cid] = rule_prediction
        for arm in ("visual_model", "numeric_model", "visual_model_raw", "visual_model_ungated"):
            gate = methods["robust_rule"][cid].activity if arm == "numeric_model" else activity
            if locator_failed and arm != "numeric_model":
                methods[arm][cid] = ActivityPrediction(gate.copy(), np.full(len(gate), -1), status="failed")
                continue
            gated = arm != "visual_model_ungated"
            instruction = "Only label stages inside the supplied confirmed activity intervals. Do not extend, revise or invent activity."
            if not gated:
                instruction = "Activity intervals are supplied as context, but for this ablation label evolution anywhere in the record; do not restrict stages to activity."
            prompt = task + EVOLUTION_TASK + instruction
            prompt += "\nConfirmed inclusive activity intervals: " + json.dumps(inclusive_spans(gate))
            prompt += '\nReturn exactly {"stages":[[start,end,"A"],[start,end,"S"],[start,end,"D"]]} with actual intervals; empty lists are valid. No other keys.'
            images = [raw] if arm == "visual_model_raw" else [raw, auxiliary]
            methods[arm][cid] = infer_stages(transport, cid + "-" + arm, gate, feature.observed,
                                            prompt, images, failures, gated=gated)
        prompt = task + EVOLUTION_TASK + '\nIn ONE response return exactly {"activity":[[start,end]],"uncertain":[[start,end]],"stages":[[start,end,"A"]]}. Use A/S/D as appropriate, stage intervals must lie inside confirmed activity; no other keys.'
        try:
            payload = transport.ask(cid + "-single", prompt, [raw, auxiliary])
            if not isinstance(payload, dict) or set(payload) != {"activity", "uncertain", "stages"}:
                raise ValueError("Single-stage output requires exactly activity, uncertain, stages")
            single_activity = parse_activity({key: payload[key] for key in ("activity", "uncertain")}, feature.observed)
            methods["single_visual"][cid] = parse_stages({"stages": payload["stages"]}, single_activity, feature.observed)
        except (ValueError, RuntimeError) as error:
            failures[cid + "-single"] = str(error)
            methods["single_visual"][cid] = ActivityPrediction(np.full(len(activity), -1), np.full(len(activity), -1), status="failed")
        save_json(args.out / "inference-checkpoint.json", {name: {key: value.to_dict() for key, value in predictions.items()}
                                                           for name, predictions in methods.items()})
        save_json(args.out / "failures.json", failures)
        print(f"{cid}: inference complete; cumulative failed calls/contracts={len(failures)}", flush=True)
    # Evaluation labels are loaded only after every model output has been saved.
    labels = load_labels(Path(manifest["labels"]), all_cases)
    table = write_report(args.out, all_cases, labels, methods, case_ids)
    responses = [json.loads(path.read_text(encoding="utf-8")) for path in (args.out / "requests").glob("*.response.json")]
    run.update(status="complete", seconds=time.monotonic() - started, calls=len(responses),
               failed_calls_or_contracts=len(failures),
               total_tokens=sum(row.get("usage", {}).get("total_tokens", 0) for row in responses))
    save_json(args.out / "visual-manifest.json", run)
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()
