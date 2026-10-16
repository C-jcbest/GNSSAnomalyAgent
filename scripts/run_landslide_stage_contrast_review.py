"""Bounded development stage repair with raw-contrast selection and exact permissions."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from probe_landslide_stage_raw_consistency_v2 import serial_contrasts
from run_landslide_comparison import ROOT, STAGES, load_labels, save_json, sha
from run_landslide_window_optimization import read_json

from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case, runs
from gnss_sim.landslide_stage_raw_consistency_v2 import (
    StageContrastPolicy,
    abstain_on_stage_conflicts,
    raw_stage_contrasts,
    stage_conflict_mask,
)
from gnss_sim.landslide_visual import EVOLUTION_TASK, VisualRequests, parse_stages

PLOT = {"single_plot_inches": [10.24, 6.3], "single_plot_dpi": 150}


def activity_interiors(activity, observed):
    return [(left, right) for left, right in runs((activity == 1) | ~observed)
            if np.any(activity[left:right] == 1)]


def local_auxiliary(source, cid, left, right, path):
    motion = read_json(source / "inputs" / cid / "motion-61.json")
    fig, panels = plt.subplots(2, 1, figsize=(10.24, 4.8), sharex=True)
    for panel, field, unit in zip(panels, ("speed_mm_day", "tangential_acceleration_mm_day2"),
                                  ("Speed mm/day", "Tangential mm/day²"), strict=True):
        values = np.asarray(motion[field], dtype=float)
        panel.plot(np.arange(left, right), values[left:right], linewidth=1)
        panel.set_ylabel(unit)
        panel.grid(alpha=0.2)
        panel.set_xlim(left, right - 1)
    panels[1].axhline(0, color="grey", linewidth=0.6)
    panels[1].set_xlabel("Original day; 61-day centered estimates are auxiliary only")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def numeric_table(contrasts, left, right):
    rows = []
    for start in range(left, right, 14):
        stop = min(start + 14, right)
        values = contrasts[start:stop]
        valid = np.isfinite(values).all(axis=1)
        sample = values[valid]
        rows.append({"days_inclusive": [start, stop - 1], "available_days": len(sample),
                     "before_rate_mm_day": float(np.median(sample[:, 0])) if len(sample) else None,
                     "after_rate_mm_day": float(np.median(sample[:, 1])) if len(sample) else None,
                     "noise_proxy_mm_day": float(np.median(sample[:, 2])) if len(sample) else None})
    return rows


def stage_prompt(cid, activity, stage, target, contrasts, feedback):
    left, right = target
    intervals = [[a, b - 1] for a, b in runs((activity == 1) & (np.arange(len(activity)) >= left)
                                           & (np.arange(len(activity)) < right))]
    old_stages = sorted([[a, b - 1, code] for code in (1, 2, 3)
                         for a, b in runs((stage == code) & (np.arange(len(stage)) >= left)
                                          & (np.arange(len(stage)) < right))])
    names = {1: "A", 2: "S", 3: "D"}
    old_stages = [[a, b, names[code]] for a, b, code in old_stages]
    prompt = (f"Retrospectively review GNSS stages for {cid}, target days {left}..{right - 1}.\n"
              + EVOLUTION_TASK
              + "Images are the full raw record, local raw displacement, and local auxiliary "
              "61-day centered speed/tangential estimates, in that order. Raw displacement "
              "must establish continuing motion; directions N/E/U are not A/S/D labels. "
              "Activity is already frozen: do not revise or extend it. Examine the rate "
              "throughout each proposed stage and retain unknown near ambiguous transitions.\n"
              + "Confirmed inclusive activity intervals within target: " + json.dumps(intervals)
              + "\nInitial stage claims (may be wrong): " + json.dumps(old_stages)
              + '\nReturn exactly {"stages":[[start,end,"A"],[start,end,"S"],'
                '[start,end,"D"]]} using actual sorted nonoverlapping inclusive intervals. '
                "Only label supplied activity within the target. Empty stages are valid; no other keys.")
    if feedback:
        prompt += ("\nSame-source raw rate contrast: three disjoint 14-day median N/E/U blocks, "
                   "centers 42 days apart. Before/after rates are successive displacement norm/42. "
                   "Only envelopes inside confirmed observed activity are available. Noise proxy "
                   "uses daily-difference MAD, not confidence, independent evidence or true labels. "
                   "Empty evidence does not refute movement. Do not label from a single derivative "
                   "peak or one summary block.\n" + json.dumps(numeric_table(contrasts, left, right)))
    return prompt


def prepare(native, gate_probe, out):
    from run_landslide_window_optimization import load_case_features, render_window

    if not read_json(gate_probe / "decision.json")["allows_visual_review"]:
        raise ValueError("Registered transfer criterion did not permit a visual probe")
    native_config = read_json(native / "protocol.json")
    source = Path(native_config["activity_run"])
    if sha(native / "predictions.json") != read_json(native / "prediction-freeze.json")["sha256"]:
        raise ValueError("Native predictions changed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    save_json(out / "protocol.json", {
        "scope": "Second already seen 12-record controlled development; no independent validation claim",
        "native": str(native.resolve()), "source": str(source), "model": native_config["model"],
        "gate_probe": str(gate_probe.resolve()), "gate_decision_sha256": sha(gate_probe / "decision.json"),
        "policy": {"separation": 42, "relative_change": 0.25, "noise_multiplier": 2.0,
                   "minimum_conflict_days": 14, "maximum_reviews_per_record": 1},
        "primary": "adaptive_numeric_feedback_conflict_scope_veto",
        "ablation": ["fixed_image_conflict_scope_veto", "adaptive_image_conflict_scope_veto",
                     "adaptive_numeric_feedback_conflict_scope_no_veto",
                     "adaptive_numeric_feedback_full_target_veto", "raw_contrast_abstention"],
        "fixed_selection": "Longest confirmed observed activity; ties earliest, at most one",
        "adaptive_selection": "Most conflicting days in one activity, minimum14; ties earliest",
        "revision_scope": "Exact raw-conflict days only, no permission to alter activity or other stages",
        "veto": "Any replacement still contradicting the same raw42 evidence remains unknown",
        "requests_maximum": 96, "repetitions": 2, "retries": 0,
        "hard_timeout_seconds": 3600,
        "continue_condition": "Both repeats: internal and end-to-end macro-F1 increase vs initial "
                              "and vs numeric abstention; more wrong->correct than correct->wrong/unknown; "
                              "wrong->correct in at least two records; no activity change. "
                              "Report every ablation and strong control, never replace primary by best arm.",
    })
    predictions = read_json(native / "predictions.json")
    selections, request_plan, contrasts_saved = {}, {}, {}
    policy = StageContrastPolicy(separations=(42,))
    for rep in range(2):
        method = f"direct_visual_native_stage_r{rep}"
        for cid, value in predictions[method].items():
            case, features = load_case_features(source, cid)
            activity, stage = np.asarray(value["activity"]), np.asarray(value["stage"])
            contrasts = raw_stage_contrasts(case, activity, features.noise, 42)
            mask = stage_conflict_mask(stage, {42: contrasts}, policy)
            interiors = activity_interiors(activity, features.observed)
            fixed = max(interiors, key=lambda pair: (int((activity[pair[0]:pair[1]] == 1).sum()),
                                                   -pair[0])) if interiors else None
            ranked = sorted(interiors, key=lambda pair: (-int(mask[pair[0]:pair[1]].sum()), pair[0]))
            adaptive = ranked[0] if ranked and mask[ranked[0][0]:ranked[0][1]].sum() >= 14 else None
            key = f"{cid}-r{rep}"
            selections[key] = {"fixed": fixed, "adaptive": adaptive,
                               "conflict_days": np.flatnonzero(mask).tolist(),
                               "interiors": interiors}
            contrasts_saved[key] = serial_contrasts(contrasts)
            raw = out / "images" / f"{cid}-global-raw.png"
            if not raw.exists():
                shutil.copyfile(source / "images" / f"{cid}-raw.png", raw)
            targets = sorted({tuple(target) for target in (fixed, adaptive) if target is not None})
            for left, right in targets:
                stem = f"{key}-target-{left}-{right}"
                local = out / "images" / f"{stem}-local-raw.png"
                auxiliary = out / "images" / f"{stem}-local-auxiliary.png"
                display_left, display_right = max(0, left - 28), min(len(activity), right + 28)
                render_window(case, features.noise, display_left, display_right, local, PLOT)
                local_auxiliary(source, cid, display_left, display_right, auxiliary)
                for feedback in (False, True):
                    request_id = stem + ("-numeric" if feedback else "-image")
                    request_plan[request_id] = {
                        "case_id": cid, "rep": rep, "target": [left, right], "feedback": feedback,
                        "prompt": stage_prompt(cid, activity, stage,
                                               (left, right), contrasts, feedback),
                        "images": [str(path.relative_to(out)) for path in (raw, local, auxiliary)],
                    }
    save_json(out / "initial-predictions.json", predictions)
    save_json(out / "selections.json", selections)
    save_json(out / "contrasts.json", contrasts_saved)
    save_json(out / "request-plan.json", request_plan)
    dependencies = [Path(__file__), ROOT / "src/gnss_sim/landslide_stage_raw_consistency_v2.py",
                    ROOT / "src/gnss_sim/landslide_visual.py", ROOT / "src/gnss_sim/landslide_evaluation.py",
                    native / "predictions.json"]
    for cid in native_config["case_ids"]:
        dependencies.extend((source / "inputs" / cid).glob("*.json"))
    save_json(out / "dependency-freeze.json", {str(path.resolve()): sha(path) for path in dependencies})
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path)
                                   for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared", "unique_planned_requests": len(request_plan)})
    print("Prepared unique requests:", len(request_plan), flush=True)


def run(out):
    from run_landslide_window_optimization import load_case_features

    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Frozen prepared evidence changed")
    for name, expected in read_json(out / "dependency-freeze.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Executing dependency changed")
    config = read_json(out / "protocol.json")
    source = Path(config["source"])
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Do not rerun inference")
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"],
                               config["requests_maximum"])
    initial = read_json(out / "initial-predictions.json")
    plan = read_json(out / "request-plan.json")
    outputs, failures = {}, {}
    save_json(out / "state.json", {"status": "running"})
    started = time.monotonic()
    # Counterbalance image/numeric order, retaining all failures without retries.
    groups = {}
    for request_id, item in plan.items():
        groups.setdefault(request_id.rsplit("-", 1)[0], []).append(request_id)
    for ordinal, request_ids in enumerate(groups.values()):
        order = request_ids if ordinal % 2 == 0 else request_ids[::-1]
        for request_id in order:
            if time.monotonic() - started > config["hard_timeout_seconds"]:
                raise RuntimeError("Registered wall-time budget exceeded")
            item = plan[request_id]
            cid, rep = item["case_id"], item["rep"]
            _, features = load_case_features(source, cid)
            value = initial[f"direct_visual_native_stage_r{rep}"][cid]
            gate = np.asarray(value["activity"]).copy()
            left, right = item["target"]
            gate[:left], gate[right:] = 0, 0
            gate[~features.observed] = -1
            try:
                payload = transport.ask(request_id, item["prompt"],
                                        [out / name for name in item["images"]])
                prediction = parse_stages(payload, gate, features.observed)
                prediction.stage[(gate == 1) & ~features.supported] = -1
            except (ValueError, RuntimeError) as error:
                failures[request_id] = str(error)
                prediction = ActivityPrediction(gate, np.full(len(gate), -1), stage_status="failed")
            outputs[request_id] = prediction.to_dict()
            save_json(out / "local-predictions.json", outputs)
            save_json(out / "failures.json", failures)
            print("Completed", len(outputs), "/", len(plan), request_id, flush=True)
    save_json(out / "local-prediction-freeze.json", {"sha256": sha(out / "local-predictions.json")})
    save_json(out / "state.json", {"status": "local_predictions_complete",
                                   "seconds": time.monotonic() - started})


def score(out):
    from sklearn.metrics import precision_recall_fscore_support

    config = read_json(out / "protocol.json")
    source = Path(config["source"])
    if read_json(out / "state.json")["status"] != "local_predictions_complete":
        raise ValueError("Freeze every response before compiling and scoring")
    if sha(out / "local-predictions.json") != read_json(out / "local-prediction-freeze.json")["sha256"]:
        raise ValueError("Local predictions changed")
    initial = read_json(out / "initial-predictions.json")
    local = read_json(out / "local-predictions.json")
    selections, contrasts = read_json(out / "selections.json"), read_json(out / "contrasts.json")
    predictions = {}
    for rep in range(2):
        original_method = f"direct_visual_native_stage_r{rep}"
        for cid, value in initial[original_method].items():
            key = f"{cid}-r{rep}"
            selection = selections[key]
            activity, stage = np.asarray(value["activity"]), np.asarray(value["stage"])
            evidence = {42: np.asarray(contrasts[key], dtype=float)}
            mask = np.zeros(len(stage), dtype=bool)
            mask[selection["conflict_days"]] = True
            original = ActivityPrediction(activity, stage, status=value["status"],
                                          stage_status=value["stage_status"])
            predictions.setdefault(f"initial_r{rep}", {})[cid] = original.to_dict()
            predictions.setdefault(f"raw_contrast_abstention_r{rep}", {})[cid] = (
                abstain_on_stage_conflicts(original, mask).to_dict())
            arms = (
                ("fixed_image_conflict_scope_veto", "fixed", False, False, True),
                ("adaptive_image_conflict_scope_veto", "adaptive", False, False, True),
                ("adaptive_numeric_feedback_conflict_scope_veto", "adaptive", True, False, True),
                ("adaptive_numeric_feedback_conflict_scope_no_veto", "adaptive", True, False, False),
                ("adaptive_numeric_feedback_full_target_veto", "adaptive", True, True, True),
            )
            for arm, selector, feedback, full, veto in arms:
                target = selection[selector]
                updated = stage.copy()
                stage_status = original.stage_status
                if target is not None:
                    left, right = target
                    stem = f"{key}-target-{left}-{right}"
                    request_id = stem + ("-numeric" if feedback else "-image")
                    replacement = np.asarray(local[request_id]["stage"])
                    permission = np.zeros(len(stage), dtype=bool)
                    permission[left:right] = activity[left:right] == 1
                    if not full:
                        permission &= mask
                    if veto:
                        replacement = replacement.copy()
                        replacement[stage_conflict_mask(replacement, evidence,
                                                       StageContrastPolicy(separations=(42,),
                                                                           minimum_conflict_days=1))] = -1
                    updated[permission] = replacement[permission]
                    if permission.any():
                        # Final mixed output may retain initial valid stages outside
                        # a failed target; the local failure is kept separately.
                        stage_status = "success"
                prediction = ActivityPrediction(activity.copy(), updated, status=original.status,
                                                stage_status=stage_status)
                prediction.validate(len(stage))
                predictions.setdefault(f"{arm}_r{rep}", {})[cid] = prediction.to_dict()
    # Strong calibrated controls from the same train/validation population.
    for method, cases in read_json(source / "numeric-predictions.json").items():
        if all(cases[cid]["gated"] for cid in initial["direct_visual_native_stage_r0"]):
            predictions[method] = {cid: cases[cid] for cid in initial["direct_visual_native_stage_r0"]}
    save_json(out / "predictions.json", predictions)
    save_json(out / "prediction-freeze.json", {"sha256": sha(out / "predictions.json")})
    from run_landslide_window_optimization import load_case_features

    cases = {cid: load_case_features(source, cid)[0] for cid in initial["direct_visual_native_stage_r0"]}
    labels = load_labels(source / "reference-subset.csv", cases)
    results, changes = {}, {}
    for method, values in predictions.items():
        per_case, truth, predicted, per_changes = {}, [], [], {}
        for cid, value in values.items():
            prediction = ActivityPrediction(np.asarray(value["activity"]), np.asarray(value["stage"]),
                                            status=value["status"], stage_status=value["stage_status"])
            per_case[cid] = evaluate_case(labels[cid], prediction)
            before = (initial[f"direct_visual_native_stage_r{method.rsplit('_r', 1)[1]}"][cid]["stage"]
                      if method.endswith(("_r0", "_r1")) else None)
            counts = {"wrong_to_correct": 0, "correct_to_wrong_or_unknown": 0,
                      "wrong_to_unknown": 0, "changed_unscored": 0, "changed_nonactive": 0}
            for day, row in enumerate(labels[cid]):
                expected = STAGES.get(row["feature"], -1)
                active = int(row["activity_label"])
                after = value["stage"][day]
                if active == 1 and expected > 0:
                    truth.append(expected)
                    predicted.append(after)
                if before is None or before[day] == after:
                    continue
                if active < 0 or (active == 1 and expected < 0):
                    counts["changed_unscored"] += 1
                elif active == 0:
                    counts["changed_nonactive"] += 1
                elif before[day] != expected and after == expected:
                    counts["wrong_to_correct"] += 1
                elif before[day] == expected:
                    counts["correct_to_wrong_or_unknown"] += 1
                elif after == -1:
                    counts["wrong_to_unknown"] += 1
            per_changes[cid] = counts
        p, r, f, n = precision_recall_fscore_support(truth, predicted, labels=[1, 2, 3], zero_division=0)
        results[method] = {"aggregate": aggregate_scores(list(per_case.values())), "per_case": per_case,
                           "internal": {"precision_ASD": p.tolist(), "recall_ASD": r.tolist(),
                                        "f1_ASD": f.tolist(), "support_ASD": n.tolist(),
                                        "macro_f1": float(f.mean()),
                                        "coverage": float(np.mean(np.asarray(predicted) > 0))}}
        changes[method] = {"per_case": per_changes,
                           "total": {key: sum(row[key] for row in per_changes.values())
                                     for key in next(iter(per_changes.values()))}}
    save_json(out / "results.json", results)
    save_json(out / "changes.json", changes)
    decisions = []
    for rep in range(2):
        method = f"adaptive_numeric_feedback_conflict_scope_veto_r{rep}"
        row = changes[method]
        delta_initial = results[method]["internal"]["macro_f1"] - results[f"initial_r{rep}"]["internal"]["macro_f1"]
        delta_abstain = results[method]["internal"]["macro_f1"] - results[f"raw_contrast_abstention_r{rep}"]["internal"]["macro_f1"]
        e2e_initial = results[method]["aggregate"]["stage_macro_f1"] - results[f"initial_r{rep}"]["aggregate"]["stage_macro_f1"]
        e2e_abstain = results[method]["aggregate"]["stage_macro_f1"] - results[f"raw_contrast_abstention_r{rep}"]["aggregate"]["stage_macro_f1"]
        records = sum(value["wrong_to_correct"] > 0 for value in row["per_case"].values())
        passed = (min(delta_initial, delta_abstain, e2e_initial, e2e_abstain) > 0
                  and row["total"]["wrong_to_correct"] > row["total"]["correct_to_wrong_or_unknown"]
                  and records >= 2)
        decisions.append({"method": method, "passed": passed, "internal_delta_initial": delta_initial,
                          "internal_delta_abstention": delta_abstain, "e2e_delta_initial": e2e_initial,
                          "e2e_delta_abstention": e2e_abstain, "corrected_records": records,
                          **row["total"]})
    receipts = [read_json(path) for path in (out / "requests").glob("*.response.json")]
    save_json(out / "costs.json", {"requests": len(receipts),
                                   "tokens": sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts),
                                   "seconds": sum(row["seconds"] for row in receipts)})
    save_json(out / "decision.json", {"allows_new_confirmation": all(row["passed"] for row in decisions),
                                      "repeats": decisions})
    print(decisions, flush=True)
    state = read_json(out / "state.json")
    state["status"] = "complete"
    save_json(out / "state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--native", type=Path)
    parser.add_argument("--gate-probe", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.native, args.gate_probe, args.out)
    elif args.action == "run":
        run(args.out)
    else:
        score(args.out)


if __name__ == "__main__":
    main()
