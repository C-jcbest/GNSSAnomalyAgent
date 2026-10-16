"""Freeze and test balanced motion exemplars on eight label-free nominated queries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from probe_landslide_endpoint_fit import BASE, ROOT, export
from probe_landslide_feature_discovery import PROPOSALS
from probe_landslide_visual_invariance import TRAINING, calibration_labels
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import save_json
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_contrastive_exemplars import (
    contrastive_neighbors,
    fixed_class_examples,
    motion_shape,
)
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_visual_invariance import invariance_inputs

SOURCE = TRAINING["primary_process"]
ANCHOR = BASE / "feature-discovery-calibration-v1"
ARMS = ("retrieved_contrast", "target_only", "fixed_contrast", "retrieved_no_numeric")
TASK = """Review sustained GNSS displacement ONLY on the14 shaded queried dates.
Weak cumulative motion counts as active; an elevated stationary plateau, isolated
spike, single step or numerical derivative response alone does not. First check
continuing raw displacement near the query. Do not infer physical failure, latent
parameters or unobserved data. Return unknown when evidence is inconclusive.

Panels are original NEU observations, with a constant reference removed and each
axis divided by that record's daily-difference noise proxy. No detrending, amplitude
matching, gap filling or compressed time. Noise units are descriptive, not calibrated
confidence. All panels share relative dates-91..90 and center band-7..6, but belong
to DIFFERENT training records unless marked repeated target. Training labels apply
only to observed dates inside their shaded band, not the entire182-day window.
ACTIVE and INACTIVE examples are class evidence, not simultaneous observations or
exact counterfactuals. Similarity does not transfer a label automatically. Compare
continuing change in TARGET; no reference example supplies its true boundary.
Optional block tables describe the same observations, not model scores or labels.

Return exactly JSON {"labels":[14 integers]} in original consecutive query order,
each1=confirmed continuing displacement,0=confirmed stationary/no sustained motion,
-1=unknown. Do not output phases, probabilities, example labels or extra keys.
"""


def raw_values(case):
    return np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])


def serial_vector(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def block_packet(case, features, day, role, label):
    normalized, available, counts, _ = discovery_block_matrix(raw_values(case), features.noise)
    return {"role": role, "center_training_label": label,
            "normalized_neu_blocks": [serial_vector(row) for row in normalized[day]],
            "available": available[day].tolist(), "complete_days": counts[day].tolist(),
            "relative_block_centers": (-84.5 + 14 * np.arange(13)).tolist(),
            "noise_proxy_mm": features.noise.tolist()}


def render_packet(columns, path):
    """Keep target pixels identical across all arms, without query-dependent scaling."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True)
    for column, (case, features, day, role, label) in enumerate(columns):
        values = raw_values(case)
        _, available, _, medians = discovery_block_matrix(values, features.noise)
        first = np.flatnonzero(available[day])[0]
        local = (values[day - 91:day + 91] - medians[day, first]) / features.noise
        blocks = (medians[day] - medians[day, first]) / features.noise
        for axis in range(3):
            panel = axes[axis, column]
            panel.plot(np.arange(-91, 91), local[:, axis], linewidth=.65, color="#197c78")
            panel.plot(-84.5 + 14 * np.arange(13), blocks[:, axis], "o", markersize=3, color="#535d65")
            panel.axvspan(-7, 6, color="#5289ba", alpha=.15)
            center = (np.nanmax(local[:, axis]) + np.nanmin(local[:, axis])) / 2
            span = max(6., np.nanmax(local[:, axis]) - np.nanmin(local[:, axis]))
            panel.set_ylim(center - .65 * span, center + .65 * span)
            panel.set_ylabel(("N", "E", "U")[axis] + " / noise proxy")
            panel.set_xticks([-84, -42, 0, 42, 84])
            panel.grid(alpha=.2)
            if axis == 0:
                title = role
                if label is not None:
                    title += " | band " + ("ACTIVE" if label == 1 else "INACTIVE")
                panel.set_title(title, fontsize=10)
            if axis == 2:
                panel.set_xlabel("Original days relative to query center")
    axes[0, 1].set_title("TARGET | shaded band labels UNKNOWN", fontsize=10)
    axes[-1, 1].set_xlim(-91, 90)
    fig.suptitle("Center-labelled training contrasts | no query labels | same target in all arms")
    fig.tight_layout()
    fig.savefig(path, dpi=115)
    plt.close(fig)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    verify_bindings(ANCHOR, "result-freeze.json")
    verify_bindings(PROPOSALS, "proposal-freeze.json")
    split = read_json(SOURCE / "protocol.json")["split"]
    ids = split["train"] + split["validation"]
    pairs = {cid: load_case_features(SOURCE, cid) for cid in ids}
    labels = calibration_labels(SOURCE, pairs)
    blocks, supports = {}, {}
    for cid, pair in pairs.items():
        blocks[cid] = discovery_block_matrix(raw_values(pair[0]), pair[1].noise)
        supports[cid] = invariance_inputs(*pair)[2]
        export(out / "inputs" / (cid + ".json"), pair[0].model_dump(mode="json"))
    bank = []
    for cid in split["train"]:
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        for day in range(98, len(actual) - 98, 14):
            query = slice(day - 7, day + 7)
            eligible = supports[cid][query]
            codes = actual[query][eligible]
            observed_codes = actual[query][pairs[cid][1].observed[query]]
            if (eligible.sum() < 8 or len(set(codes.tolist())) != 1 or codes[0] < 0
                    or len(set(observed_codes.tolist())) != 1 or observed_codes[0] < 0
                    or blocks[cid][1][day].sum() < 10):
                continue
            descriptor = motion_shape(blocks[cid][0][day])
            bank.append({"example_id": cid + "-d" + str(day), "case_id": cid, "day": day,
                         "center_label": int(codes[0]), "descriptor": serial_vector(descriptor),
                         "eligible_center_days": int(eligible.sum())})
    fixed = fixed_class_examples(bank)
    frozen = read_json(ANCHOR / "primary_process/validation-predictions.json")
    probability = read_json(ANCHOR / "primary_process/validation-probabilities.json")
    head = XGBClassifier()
    head.load_model(SOURCE / "stage-xgboost.json")
    stage_codes = {cid: (head.predict(pairs[cid][1].matrix).astype(int) + 1).tolist()
                   for cid in split["validation"]}
    candidates = []
    for cid in split["validation"]:
        before = np.array(frozen["base18"][cid]["activity"])
        after = np.array(frozen["base_expert"][cid]["activity"])
        p_before = np.array(probability["base18"][cid])
        p_after = np.array(probability["base_expert"][cid])
        for day in range(98, len(before) - 98, 14):
            query = slice(day - 7, day + 7)
            if supports[cid][query].sum() < 10 or blocks[cid][1][day].sum() < 10:
                continue
            neighbors = contrastive_neighbors(motion_shape(blocks[cid][0][day]), bank)
            if neighbors is None:
                continue
            candidates.append({"case_id": cid, "day": day,
                               "changed_days": int((before[query] != after[query]).sum()),
                               "probability_difference": float(np.abs(p_before[query] - p_after[query]).mean()),
                               "expert_activity_days": int((after[query] == 1).sum()),
                               "expert_mean_probability": float(p_after[query].mean()),
                               "neighbors": neighbors})
    selected, used = [], set()
    # Select from predictions only. Validation reference never determines which
    # disagreements or correct controls receive a model call.
    disagreements = sorted([row for row in candidates if row["changed_days"] >= 7],
                           key=lambda row: (-row["probability_difference"], row["case_id"], row["day"]))
    for row in disagreements:
        if row["case_id"] not in used:
            selected.append({**row, "role": "detector_disagreement"})
            used.add(row["case_id"])
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError("Four distinct prediction-only disagreements unavailable")
    for active in (False, True):
        controls = [row for row in candidates if row["changed_days"] == 0
                    and ((row["expert_mean_probability"] >= .8) if active
                         else (row["expert_mean_probability"] <= .2))]
        controls.sort(key=lambda row: (row["probability_difference"], row["case_id"], row["day"]))
        added = 0
        for row in controls:
            if row["case_id"] in used:
                continue
            selected.append({**row, "role": "predicted_active_control" if active else "predicted_inactive_control"})
            used.add(row["case_id"])
            added += 1
            if added == 2:
                break
        if added != 2:
            raise ValueError("Two independent-record agreement controls unavailable")
    packets = {}
    for index, selected_row in enumerate(selected):
        key = f"q{index + 1}"
        cid, day = selected_row["case_id"], selected_row["day"]
        case, features = pairs[cid]
        original = frozen["base_expert"][cid]
        export(out / (key + "-baseline.json"), original)
        export(out / (key + "-reference.json"), labels[cid])
        for arm in ARMS:
            if arm == "target_only":
                columns = [(case, features, day, "REPEATED TARGET", None),
                           (case, features, day, "TARGET", None),
                           (case, features, day, "REPEATED TARGET", None)]
            else:
                references = fixed if arm == "fixed_contrast" else selected_row["neighbors"]
                quiet, active = (bank[references[name]["index"]] for name in ("inactive", "active"))
                columns = [(*pairs[quiet["case_id"]], quiet["day"], "TRAIN EXAMPLE", 0),
                           (case, features, day, "TARGET", None),
                           (*pairs[active["case_id"]], active["day"], "TRAIN EXAMPLE", 1)]
            chart = key + "-" + arm + ".png"
            render_packet(columns, out / chart)
            prompt = TASK + "\nOriginal query dates:" + json.dumps(list(range(day - 7, day + 7)))
            if arm != "retrieved_no_numeric":
                tables = [block_packet(*column) for column in columns]
                prompt += "\nSame-observation block measurements:" + json.dumps(tables, separators=(",", ":"))
            packets[key, arm] = {"prompt": prompt, "images": [chart], "case_id": cid, "day": day,
                                 "query": [day - 7, day + 7], "support": supports[cid].tolist(),
                                 "observed": features.observed.tolist(), "stage_support": features.supported.tolist(),
                                 "stage_codes": stage_codes[cid]}
    export(out / "training-bank.json", bank)
    export(out / "fixed-examples.json", fixed)
    export(out / "nomination-candidates.json", candidates)
    export(out / "selected-queries.json", selected)
    export(out / "packets.json", {key + ":" + arm: value for (key, arm), value in packets.items()})
    rng = np.random.default_rng(20261016)
    schedule = []
    for repetition in range(2):
        for key in rng.permutation([f"q{i + 1}" for i in range(8)]).tolist():
            for arm in rng.permutation(ARMS).tolist():
                schedule.append({"query_id": key, "arm": arm, "repetition": repetition,
                                 "request_id": f"{key}-{arm}-r{repetition}"})
    export(out / "schedule.json", schedule)
    export(out / "protocol.json", {
        "scope": "Eight prediction-only nominated14day development queries; not full-record efficacy or independent confirmation",
        "model": "qwen3.8-flash", "budget": 64, "repetitions": 2, "arms": list(ARMS),
        "nomination": "Four distinct >=7day base18/expert disagreement queries ranked by absolute probability difference; "
                      "two predicted active andtwoinactive agreement controls from four further records. No query label used.",
        "retrieval": "Own24training records only; homogeneous known/supported centerband,>=8days;>=10available14day blocks. "
                     "12log adjacent norms+11cosines;mean squared distance on>=12shared coordinates;center increments5/6 required. "
                     "One nearest per class, deterministic example-id tie break. No thresholds or distance-grid search.",
        "fixed_control": "One training-only class medoid per class; target pixels/time/noise normalization identical.",
        "output": "14 original-day activity codes;only original expert-known supported query days may change;outside retained; "
                  "unknown returned is retained asunknown;same original stage head/support",
        "numeric_controls": "Reused ownsplit refit base18/base_expert/base_blocks plus same-descriptor nearest-class distance. "
                            "Same training labels are available to numeric baselines; generator examples are additional test-time class context.",
        "gate": "Each main repetition strictly improves target dailyF1 and net correct vs expert and target_only; "
                ">=2benefiting records;weakR no loss;agreement-controls zero correct loss. "
                "Main not below fixed in either repeat and strictly above in at leastone; "
                "not below nearest-class numeric control;at leastone repeat above no-numeric. "
                "If any necessary condition fails stop fixed pilot, no seed/retrieval/window/prompt/query rescue.",
        "reference": "Historical controlled labels;author has seen development results;train-only remote exemplars do not restore blindness",
        "novelty": "Visual context/retrieval/contrastive examples already exist. GNSS balanced center-labelled temporal evidence requires efficacy.",
        "new_records": 0, "independent_confirmation": False})
    binding = {}
    sources = [Path(__file__), ROOT / "src/gnss_sim/landslide_contrastive_exemplars.py",
               ROOT / "tests/test_landslide_contrastive_exemplars.py",
               ROOT / "src/gnss_sim/landslide_feature_discovery.py",
               ROOT / "src/gnss_sim/landslide_visual.py", SOURCE / "reference-labels.csv",
               SOURCE / "protocol.json", SOURCE / "stage-xgboost.json",
               ANCHOR / "primary_process/validation-predictions.json",
               ANCHOR / "primary_process/validation-probabilities.json"]
    copies = {}
    for index, path in enumerate(sources):
        # A shallow archive avoids Windows' long-path limit for artifact inputs.
        copy = out / "source" / f"{index:02d}-{path.name}"
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
        binding[path.relative_to(ROOT).as_posix()] = sha(path)
        copies[path.relative_to(ROOT).as_posix()] = copy.relative_to(out).as_posix()
    export(out / "root-bindings.json", binding)
    export(out / "source-copy-map.json", copies)
    export(out / "pre-inference-freeze.json", {path.relative_to(out).as_posix(): sha(path)
                                              for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared", "new_api_calls": 0})
    print("Frozen", len(bank), "training examples;8queries,64requests", flush=True)


def parse_labels(payload):
    if (not isinstance(payload, dict) or set(payload) != {"labels"}
            or not isinstance(payload["labels"], list) or len(payload["labels"]) != 14
            or any(type(code) is not int or code not in (-1, 0, 1) for code in payload["labels"])):
        raise ValueError("Expected exactly fourteen original-day integer labels")
    return np.array(payload["labels"], dtype=int)


def apply_labels(packet, baseline, codes):
    activity = np.array(baseline["activity"])
    permitted = np.zeros(len(activity), dtype=bool)
    start, stop = packet["query"]
    permitted[start:stop] = True
    permitted &= np.array(packet["support"]) & (activity >= 0)
    reviewed = activity.copy()
    reviewed[start:stop] = codes
    activity[permitted] = reviewed[permitted]
    # The shared head is frozen once during preparation; a newly positive date
    # must not copy an old activity-gated unknown or trigger a model reload.
    stage = np.array(packet["stage_codes"])
    return learned_stages(stage, activity, np.array(packet["stage_support"])).to_dict()


def infer(out):
    verify_bindings(out, "pre-inference-freeze.json")
    for name, expected in read_json(out / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen source changed")
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Start only a never-run pilot")
    transport = VisualRequests(out / "requests", ROOT / ".env", "qwen3.8-flash", 64)
    packets = read_json(out / "packets.json")
    outputs, failures = {}, {}
    save_json(out / "state.json", {"status": "running", "budget": 64})
    started = time.perf_counter()
    for item in read_json(out / "schedule.json"):
        request_id, key, arm = item["request_id"], item["query_id"], item["arm"]
        packet = packets[key + ":" + arm]
        baseline = read_json(out / (key + "-baseline.json"))
        try:
            payload = transport.ask(request_id, packet["prompt"], [out / name for name in packet["images"]])
            codes = parse_labels(payload)
            outputs[request_id] = {"prediction": apply_labels(packet, baseline, codes),
                                   "labels": codes.tolist(), "status": "success"}
        except (ValueError, RuntimeError, TypeError) as error:
            outputs[request_id] = {"prediction": baseline, "labels": None, "status": "failed"}
            failures[request_id] = {"error_type": type(error).__name__, "message": str(error)}
        save_json(out / "outputs.json", outputs)
        save_json(out / "failures.json", failures)
        print(request_id, outputs[request_id]["status"], flush=True)
    save_json(out / "state.json", {"status": "inference_complete", "attempts": 64,
                                   "failures": len(failures), "loop_seconds": time.perf_counter() - started})
    export(out / "inference-freeze.json", {path.relative_to(out).as_posix(): sha(path)
                                          for path in out.rglob("*") if path.is_file()
                                          and path.name not in {"inference-freeze.json", "state.json"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "infer"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out.resolve())
    else:
        infer(args.out.resolve())
