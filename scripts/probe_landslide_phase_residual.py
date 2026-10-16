"""Freeze and run a small activity-contained residual-view phase correction probe."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from probe_landslide_endpoint_fit import BASE, ROOT, csv_rows, export
from probe_landslide_visual_invariance import TRAINING
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import save_json
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha
from gnss_sim.landslide_phase_residual import phase_evidence_window, phase_residual_evidence
from gnss_sim.landslide_visual import EVOLUTION_TASK, VisualRequests, parse_stages

SOURCE_REPORT = BASE / "temporal-readout-calibration-report-v1"
ARMS = ("raw_local", "residual_view", "residual_feedback", "numeric_only")
STAGE_CODES = {"acceleration": 1, "steady_motion": 2, "deceleration": 3}
TASK = """Review phases only in the queried part of an observed GNSS activity.
The evidence envelope lies inside an activity interval already selected by a frozen
detector. Check raw continuing displacement before interpreting a phase. Evidence
of noise, a step or uncertain motion does not justify a definite phase. Do not
expand activity, infer latent mechanisms, or force a label on uncertain dates.
Optional motion-coordinate evidence uses endpoint block medians and daily-difference
noise proxies to project the SAME observed NEU onto its local net-motion direction.
It then subtracts a fitted straight line. This does not add independent evidence.
A curve above/below the line is not itself A/D: assess slope CHANGE near the queried
dates. Approximately constant nonzero slope is S even if its magnitude is large.
A/D require increasing/decreasing rate, not merely a residual sign. Direction is
chosen so that endpoint net motion is positive; changing direction can still limit
this representation. Noise proxy units are descriptive, not significance tests.
Return exactly a JSON object {"stages":[[start,end,"A|S|D"]]} with inclusive,
sorted, nonoverlapping original day endpoints ONLY inside the requested query.
Omitted dates are unknown; an empty stages list is valid. No additional keys.
""" + EVOLUTION_TASK


def input_values(case):
    return np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])


def reference_rows(source):
    labels = {}
    for row in csv_rows(source / "reference-labels.csv"):
        labels.setdefault(row["case_id"], []).append(row)
    return labels


def render_views(case, evidence, query, raw_path, residual_path):
    values = input_values(case)
    start, stop = evidence["start"], evidence["stop"]
    calendar = np.arange(start, stop)
    ticks = [start, query[0], query[1] - 1, stop - 1]
    for day in range(start, stop, 14):
        if all(abs(day - tick) >= 7 for tick in ticks):
            ticks.append(day)
    ticks = sorted(set(ticks))
    colors = ("#147d72", "#cc6d4b", "#a38a32")
    fig, panels = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    for axis, panel in enumerate(panels):
        panel.plot(calendar, values[start:stop, axis], linewidth=.7, color=colors[axis])
        span = max(10, 6 * evidence["noise_proxy_mm"][axis], np.nanmax(values[start:stop, axis]) - np.nanmin(values[start:stop, axis]))
        center = (np.nanmax(values[start:stop, axis]) + np.nanmin(values[start:stop, axis])) / 2
        panel.set_ylim(center - .65 * span, center + .65 * span)
        panel.set_ylabel(("N", "E", "U")[axis] + " (mm)")
        panel.axvspan(query[0], query[1] - 1, alpha=.12, color="#426da8")
        panel.grid(alpha=.2)
    panels[-1].set_xlim(start, stop - 1)
    panels[-1].set_xticks(ticks)
    panels[-1].set_xlabel("Original daily calendar; shaded query only; missing observations remain gaps")
    fig.suptitle(f"{case.case_id} | Raw NEU | query [{query[0]}, {query[1]})")
    fig.tight_layout()
    fig.savefig(raw_path, dpi=120)
    plt.close(fig)
    fig, panels = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    panels[0].plot(calendar, evidence["raw_progression"], linewidth=.65, color="#708897", label="Observed projected displacement")
    panels[0].plot(evidence["block_centers"], evidence["progression"], "o", color="#147d72", label="14-day median block projections")
    panels[0].plot(evidence["block_centers"], evidence["linear_reference"], "--", color="#cc6d4b", label="Fitted constant-rate reference")
    panels[0].set_ylabel("Noise-proxy units")
    panels[0].legend(fontsize=8)
    panels[1].plot(calendar, evidence["raw_residual"], linewidth=.65, color="#708897")
    panels[1].plot(evidence["block_centers"], evidence["residual"], "o", color="#147d72")
    panels[1].axhline(0, color="#cc6d4b", linestyle="--")
    residual_span = max(3, 1.2 * np.nanmax(np.abs([x if x is not None else np.nan for x in evidence["raw_residual"]])))
    panels[1].set_ylim(-residual_span, residual_span)
    panels[1].set_ylabel("Observed minus line\n(noise-proxy units)")
    for panel in panels:
        panel.axvspan(query[0], query[1] - 1, alpha=.12, color="#426da8")
        panel.grid(alpha=.2)
    panels[-1].set_xlim(start, stop - 1)
    panels[-1].set_xticks(ticks)
    panels[-1].set_xlabel("Original dates; same observations, no filled gaps or latent trend")
    fig.suptitle(f"{case.case_id} | Motion-coordinate evidence | query [{query[0]}, {query[1]})")
    fig.tight_layout()
    fig.savefig(residual_path, dpi=120)
    plt.close(fig)


def numeric_table(evidence):
    names = ("start", "stop", "block_centers", "complete_days", "median_neu_mm", "noise_proxy_mm",
             "scaled_direction", "progression", "linear_reference", "residual",
             "adjacent_block_rates_proxy_per_day", "linear_rate_proxy_per_day")
    return {name: evidence[name] for name in names}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    candidates, pairs, labels, predictions = [], {}, {}, {}
    for target, source in TRAINING.items():
        labels[target] = reference_rows(source)
        predictions[target] = read_json(SOURCE_REPORT / target / "predictions.json")["xgboost_raw84"]
        for cid, baseline in predictions[target].items():
            case, features = load_case_features(source, cid)
            pairs[target, cid] = case, features
            truth = np.array([STAGE_CODES.get(row["feature"], -1) if int(row["activity_label"]) == 1 else -1
                              for row in labels[target][cid]])
            activity, stage = np.array(baseline["activity"]), np.array(baseline["stage"])
            for left in range(0, len(activity) - 13, 14):
                right = left + 14
                window = phase_evidence_window(activity, features.observed, left, right)
                if window is None:
                    continue
                known = (truth[left:right] > 0) & features.supported[left:right]
                if known.sum() < 8:
                    continue
                evidence = phase_residual_evidence(input_values(case), features.noise, activity, *window)
                if evidence is None:
                    continue
                candidates.append({"target": target, "case_id": cid, "left": left, "right": right,
                                   "window": list(window), "known_days": int(known.sum()),
                                   "baseline_errors": int(((stage[left:right] != truth[left:right]) & known).sum()),
                                   "reference_counts": np.bincount(truth[left:right][known], minlength=4).tolist()})
    candidates.sort(key=lambda row: (-row["baseline_errors"], row["target"], row["case_id"], row["left"]))
    chosen, error_records = [], set()
    for row in candidates:
        identity = row["target"], row["case_id"]
        if row["baseline_errors"] > 0 and identity not in error_records:
            chosen.append({**row, "selection_role": "largest eligible error per distinct record"})
            error_records.add(identity)
        if len(chosen) == 5:
            break
    for code in (1, 2, 3):
        controls = [row for row in candidates if row["baseline_errors"] == 0
                    and row["reference_counts"][code] == row["known_days"]
                    and not any(row["target"] == selected["target"] and row["case_id"] == selected["case_id"]
                                and row["left"] < selected["right"] and selected["left"] < row["right"] for selected in chosen)]
        controls.sort(key=lambda row: (-row["known_days"], row["target"], row["case_id"], row["left"]))
        if controls:
            chosen.append({**controls[0], "selection_role": "correct baseline control for " + str(code)})
    if len(error_records) != 5 or len(chosen) != 8:
        raise ValueError("The declared five-error/three-control pilot lacks eligible records")
    packets, references, baseline_targets = {}, {}, {}
    for index, selected in enumerate(chosen):
        target, cid = selected["target"], selected["case_id"]
        case, features = pairs[target, cid]
        baseline = predictions[target][cid]
        key = f"probe_{index + 1:02d}"
        activity = np.array(baseline["activity"])
        evidence = phase_residual_evidence(input_values(case), features.noise, activity, *selected["window"])
        export(out / (key + "-evidence.json"), evidence)
        export(out / (key + "-input.json"), case.model_dump(mode="json"))
        render_views(case, evidence, (selected["left"], selected["right"]), out / (key + "-raw.png"), out / (key + "-residual.png"))
        prompt = TASK + f"\nOriginal query inclusive {selected['left']}..{selected['right'] - 1}; evidence envelope inclusive {evidence['start']}..{evidence['stop'] - 1}.\n"
        common = {"target": target, "case_id": cid, "query": [selected["left"], selected["right"]],
                  "window": selected["window"], "observed": features.observed.tolist(),
                  "stage_supported": features.supported.tolist(), "activity": activity.tolist()}
        packets[key] = {}
        for arm in ARMS:
            images = []
            if arm != "numeric_only":
                images.append(key + "-raw.png")
            if arm in ("residual_view", "residual_feedback"):
                images.append(key + "-residual.png")
            text = prompt
            if arm in ("residual_feedback", "numeric_only"):
                text += "\nMeasured block evidence JSON (no phase labels):\n" + json.dumps(numeric_table(evidence), separators=(",", ":"))
            packets[key][arm] = {**common, "prompt": text, "images": images}
        references[key] = labels[target][cid]
        baseline_targets[key] = baseline
        selected["probe_id"] = key
    export(out / "candidate-windows.json", candidates)
    export(out / "diagnostic-selection.json", {"selected": chosen, "post_score_development_selection": True,
                                               "independent_confirmation": False, "reference_used_for_packets": False})
    export(out / "packets.json", packets)
    export(out / "reference-labels.json", references)
    export(out / "baseline-predictions.json", baseline_targets)
    schedule = []
    rng = np.random.default_rng(20261014)
    for repetition in range(2):
        for key in packets:
            for arm in rng.permutation(ARMS).tolist():
                schedule.append({"probe_id": key, "arm": arm, "repetition": repetition,
                                 "request_id": f"{key}-{arm}-r{repetition}"})
    export(out / "schedule.json", schedule)
    export(out / "protocol.json", {
        "candidate": "Activity-contained motion-coordinate residual phase review",
        "workflow": "academic-research-suite/experiment-agent; run/validate inline",
        "model": "qwen3.8-flash", "arms": list(ARMS), "budget": 64, "repetitions": 2,
        "task": "Same fourteen original query dates, activity/stage support and reference for all arms",
        "scope": "Eight post-score selected development targets, not full-record or field effectiveness",
        "window_rule": "Longest of 98/70/42 original days contained in predicted XGB84 activity, shifted inside if needed; no known negative bridging; at most7 missing dates, >=8 complete NEU per14day block",
        "mechanism": "Endpoint noise-scaled motion direction; original14day block medians; subtract shared least-squares straight line; preserve raw observations and missing dates",
        "baseline": "Frozen fairness-refitted raw84 XGB activity and its original target-specific stage head",
        "stage_update": "Only queried observed active supported dates; no activity expansion; outside exact baseline; failed review preserves baseline and is reported separately",
        "continuation_gate": "Both repetitions: main residual_feedback macroF1 > raw_local, residual_view and numeric_only; positive net exact phase correction vs strong baseline in both process and author groups; gains on >=2 distinct error records and no losses on registered correct control dates. Otherwise stop this fixed pilot; no seed/window/prompt rescue.",
        "no_retries": True, "no_new_records": True, "no_reference_revision": True,
        "limitations": "Direction and subtraction are deterministic same-observation transforms; no novelty claim for generic detrending/projection. All records already exposed. Pilot references used for diagnostic selection only, never sent to VLM.",
    })
    sources = [Path(__file__), ROOT / "src/gnss_sim/landslide_phase_residual.py",
               ROOT / "src/gnss_sim/landslide_visual.py", ROOT / "src/gnss_sim/landslide_evaluation.py",
               ROOT / "tests/test_landslide_phase_residual.py"]
    source_hashes = {}
    for path in sources:
        destination = out / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        source_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
    export(out / "source-bindings.json", source_hashes)
    export(out / "state.json", {"status": "prepared", "new_api_calls": 0})
    export(out / "pre-inference-freeze.json", {str(path.relative_to(out)).replace("\\", "/"): sha(path)
                                              for path in out.rglob("*") if path.is_file() and path.name != "state.json"})
    print("Prepared eight targets, four input arms, two repetitions; no model requests", flush=True)


def infer(out):
    verify_bindings(out, "pre-inference-freeze.json")
    for name, expected in read_json(out / "source-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen probe source changed")
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Inference starts once from a prepared pilot")
    protocol = read_json(out / "protocol.json")
    packets = read_json(out / "packets.json")
    transport = VisualRequests(out / "requests", ROOT / ".env", protocol["model"], protocol["budget"])
    save_json(out / "state.json", {"status": "running", "budget": protocol["budget"]})
    outputs, failures = {}, {}
    started = time.perf_counter()
    for item in read_json(out / "schedule.json"):
        key, arm, repetition = item["probe_id"], item["arm"], item["repetition"]
        packet = packets[key][arm]
        baseline = read_json(out / "baseline-predictions.json")[key]
        stages = np.array(baseline["stage"], dtype=int)
        try:
            payload = transport.ask(item["request_id"], packet["prompt"], [out / name for name in packet["images"]])
            for interval in payload.get("stages", []):
                if len(interval) != 3 or not packet["query"][0] <= interval[0] <= interval[1] < packet["query"][1]:
                    raise ValueError("Phase response escaped the fixed query")
            prediction = parse_stages(payload, np.array(packet["activity"]), np.array(packet["observed"]))
            permitted = np.zeros(len(stages), dtype=bool)
            permitted[slice(*packet["query"])] = True
            permitted &= np.array(packet["observed"]) & np.array(packet["stage_supported"]) & (np.array(packet["activity"]) == 1)
            stages[permitted] = prediction.stage[permitted]
            status = "parsed"
        except (ValueError, RuntimeError, TypeError) as error:
            failures[item["request_id"]] = type(error).__name__ + ": " + str(error)
            status = "failed_baseline_preserved"
        outputs[item["request_id"]] = {**baseline, "stage": stages.tolist(), "review_status": status,
                                        "probe_id": key, "arm": arm, "repetition": repetition}
        save_json(out / "outputs.json", outputs)
        save_json(out / "failures.json", failures)
        print(f"{len(outputs)}/{protocol['budget']} {item['request_id']} {status}", flush=True)
    save_json(out / "state.json", {"status": "inference_complete", "attempts": len(outputs),
                                   "failures": len(failures), "wall_seconds": time.perf_counter() - started})
    export(out / "inference-freeze.json", {str(path.relative_to(out)).replace("\\", "/"): sha(path)
                                          for path in out.rglob("*") if path.is_file() and path.name not in ("inference-freeze.json", "state.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "infer"))
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.mode == "prepare":
        prepare(arguments.out.resolve())
    else:
        infer(arguments.out.resolve())
