"""Prepare and execute a bounded factual-claim review mechanism pilot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from probe_landslide_contrastive_exemplars import ANCHOR, SOURCE, block_packet, raw_values
from probe_landslide_endpoint_fit import ROOT, export
from probe_landslide_feature_discovery import expert_features
from probe_landslide_ordered_trajectory import render_time
from probe_landslide_visual_invariance import calibration_labels
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import save_json
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_claim_review import check_block_claims, parse_block_claims
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_visual_invariance import invariance_inputs

TASK = """Review sustained raw GNSS displacement on42 original dates-21..20.
Return only JSON with exactly two keys: {"labels":[42 integers],"changes":[12 entries]}.
Labels:1=continuing displacement,0=no continuing displacement,-1=unknown.
First require continuing raw displacement; elevated plateaus, single steps,
isolated spikes and derivative responses alone do not confirm motion. Weak
cumulative motion counts when supported. Keep original missing dates unknown.

The image shows original NEU measurements and thirteen14-day median dots.
Original slot j has center-84.5+14*j days; each median requires8 complete days.
The shaded42-day query covers slots5/6/7. Axes use daily-difference noise
proxy units, not confidence or standardized physical importance. There is no
detrending, time compression, interpolation or hidden reference label.

For changes[j], report the signs of median[j+1]-median[j] as[N,E,U], each
integer-1,0,1. Return null only if an original pair is missing or you cannot
read its signs. These are numerical direction claims, not activity or stages.
A small signed noise change does NOT establish sustained motion. Correct
arithmetic does NOT prove daily motion or an exact boundary. Do not infer
intermediate observed dates from a line between medians. Do not force labels.
"""

FOLLOWUP = """One final review of the same original dates and image. Your preceding
answer and auxiliary material follow. Reassess the motion labels using raw
observations; keep unknown when the boundary or continuing motion is unclear.
Numerical signs describe observed medians only. A sign correction is not an
activity label, and no rule supplies the answer. Return the same exact schema.
"""


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    train = read_json(SOURCE / "protocol.json")["split"]["train"]
    pairs = {cid: load_case_features(SOURCE, cid) for cid in train}
    labels = calibration_labels(SOURCE, pairs)
    choice = read_json(ANCHOR / "selection.json")["primary_process"]["base_expert"]
    model = XGBClassifier()
    model_path = ANCHOR / "primary_process" / f"base_expert-depth{choice['depth']}.json"
    model.load_model(model_path)
    candidates = []
    diagnostics = {}
    for cid, (case, features) in pairs.items():
        blocks, available, _, _ = discovery_block_matrix(raw_values(case), features.noise)
        _, base, support = invariance_inputs(case, features)
        probability = model.predict_proba(np.column_stack((base, expert_features(blocks))))[:, 1]
        prediction = probability >= choice["threshold"]
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        weak = np.array([int(row.get("weak_signal", 0)) == 1 for row in labels[cid]])
        diagnostics[cid] = {"probability": probability.tolist(), "supported": support.tolist()}
        for day in range(98, len(actual) - 98, 14):
            start, stop = day - 21, day + 21
            observed = features.observed[start:stop]
            if observed.sum() < 35 or not support[start:stop][observed].all():
                continue
            if available[day].sum() < 10 or not available[day, 5:8].all():
                continue
            truth, guess = actual[start:stop], prediction[start:stop]
            fp = int((observed & (truth == 0) & guess).sum())
            fn = int((observed & (truth == 1) & ~guess).sum())
            correct = int((observed & (truth == guess)).sum())
            roles = []
            if fp >= 7:
                roles.append(("numeric_false_positive", fp))
            if fn >= 7 and (observed & (truth == 1) & weak[start:stop]).sum() >= 7:
                roles.append(("numeric_weak_false_negative", fn))
            if np.all(truth[observed] == 1) and correct == observed.sum():
                roles.append(("correct_active_control", int(observed.sum())))
            if np.all(truth[observed] == 0) and correct == observed.sum():
                roles.append(("correct_inactive_control", int(observed.sum())))
            for role, priority in roles:
                candidates.append({"case_id": cid, "day": day, "query": [start, stop],
                                   "role": role, "priority": priority, "fp": fp, "fn": fn})
    roles = ("numeric_false_positive", "numeric_weak_false_negative",
             "correct_active_control", "correct_inactive_control")
    selected = []
    used = set()
    for role in roles:
        eligible = sorted((row for row in candidates if row["role"] == role and row["case_id"] not in used),
                          key=lambda row: (-row["priority"], row["case_id"], row["day"]))
        if not eligible:
            export(out / "preflight-failure.json", {"missing_role": role, "new_calls": 0})
            raise ValueError("No eligible training window for " + role)
        selected.append(eligible[0])
        used.add(eligible[0]["case_id"])
    packets = {}
    for index, row in enumerate(selected):
        key = f"q{index + 1}"
        case, features = pairs[row["case_id"]]
        packet = block_packet(case, features, row["day"], "TARGET", None)
        render_time(case, features, row["day"], out / (key + "-time.png"))
        export(out / "inputs" / (row["case_id"] + ".json"), case.model_dump(mode="json"))
        packets[key] = {"image": key + "-time.png", "table": packet,
                        "observed": features.observed[slice(*row["query"])].tolist()}
    export(out / "selected-training-queries.json", selected)
    export(out / "training-numerical-diagnostics.json", diagnostics)
    export(out / "packets.json", packets)
    rng = np.random.default_rng(20261019)
    schedule = []
    for repetition in range(2):
        for key in rng.permutation(list(packets)).tolist():
            schedule.append({"query_id": key, "repetition": repetition})
    export(out / "schedule.json", schedule)
    export(out / "prompts.json", {"initial": TASK, "followup": FOLLOWUP})
    export(out / "protocol.json", {
        "scope": "Training-only error-enriched mechanism check; not fair validation or new reference",
        "selection": "Fixed one strongest FP, one strongest weak FN, two correct controls, different train records",
        "model": "qwen3.8-flash", "repetitions": 2, "max_requests": 40, "retries": 0,
        "mechanism": "Audit explicit adjacent raw median sign claims; at most one corrective reobservation",
        "trigger": "Any concrete false arithmetic or missing-slot claim; unread null is not false; no activity threshold",
        "arms": ["image_initial", "numeric_first", "conditional_fact_review", "matched_repeat", "fixed_all_facts"],
        "fact_tolerance": 1e-12, "fact_limit": "Signs are facts, not evidence sufficiency or activity truth",
        "gate": "Both repeats net correct >0 versus initial and matched repeat, weak recall no loss, "
                "correct controls zero loss; not below fixed all facts F1, fewer followups at least once; "
                "at least one real conflict and one corrected activity error",
        "failure": "Stop fixed pilot, no altered claims, windows, seed, thresholds or prompt rescue",
        "references": "Exposed training latent labels used only for selection/scoring; immutable",
        "validation_rows_used_for_selection": False, "reference_csv_contains_other_splits": True,
        "independent_confirmation": False,
    })
    sources = [Path(__file__), ROOT / "src/gnss_sim/landslide_claim_review.py",
               ROOT / "tests/test_landslide_claim_review.py", SOURCE / "protocol.json",
               SOURCE / "reference-labels.csv", model_path, ANCHOR / "selection.json",
               ROOT / "scripts/probe_landslide_ordered_trajectory.py",
               ROOT / "scripts/probe_landslide_contrastive_exemplars.py"]
    bindings = {}
    for index, path in enumerate(sources):
        copy = out / "source" / f"{index:02d}-{path.name}"
        copy.parent.mkdir(exist_ok=True)
        copy.write_bytes(path.read_bytes())
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    export(out / "root-bindings.json", bindings)
    export(out / "pre-inference-freeze.json",
           {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared", "new_calls": 0})
    print("Prepared four fixed training queries; maximum40 calls", flush=True)


def infer(out):
    verify_bindings(out, "pre-inference-freeze.json")
    for name, expected in read_json(out / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen source changed")
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Run a never-started pilot only")
    packets = read_json(out / "packets.json")
    transport = VisualRequests(out / "requests", ROOT / ".env", "qwen3.8-flash", 40)
    outputs = {}
    prompts = read_json(out / "prompts.json")
    started = time.perf_counter()
    save_json(out / "state.json", {"status": "running", "budget": 40})

    def ask(request_id, prompt, packet):
        try:
            payload = transport.ask(request_id, prompt, [out / packet["image"]])
            codes, claims = parse_block_claims(payload)
            codes[~np.array(packet["observed"])] = -1
            facts, conflicts = check_block_claims(packet["table"]["normalized_neu_blocks"], claims)
            value = {"status": "success", "labels": codes.tolist(), "payload": payload,
                     "facts": facts, "conflicts": conflicts}
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            value = {"status": "failed", "labels": [-1] * 42, "error": str(exc)}
        outputs[request_id] = value
        save_json(out / "outputs.json", outputs)
        print(request_id, value["status"], flush=True)
        return value

    for row in read_json(out / "schedule.json"):
        key, repetition = row["query_id"], row["repetition"]
        packet = packets[key]
        prefix = f"{key}-r{repetition}"
        initial = ask(prefix + "-image_initial", prompts["initial"], packet)
        ask(prefix + "-numeric_first", prompts["initial"] + "\nRaw block table:" + json.dumps(packet["table"]), packet)
        initial_text = json.dumps(initial.get("payload", {"status": "failed"}), separators=(",", ":"))
        context = prompts["initial"] + "\n" + prompts["followup"] + "\nPreceding answer:" + initial_text
        # Failed claims remain unknown. A parse failure does not trigger an
        # arithmetic repair or hidden retry of the same request.
        trigger = initial["status"] == "success" and bool(initial["conflicts"])
        if trigger:
            ask(prefix + "-conditional_fact_review", context + "\nChecked factual conflicts only:"
                + json.dumps(initial["conflicts"], separators=(",", ":")), packet)
            ask(prefix + "-matched_repeat", context + "\nNo additional numerical observations.", packet)
        facts, _ = check_block_claims(packet["table"]["normalized_neu_blocks"], [None] * 12)
        ask(prefix + "-fixed_all_facts", context + "\nAll adjacent raw block differences:"
            + json.dumps(facts, separators=(",", ":")), packet)
        save_json(out / "state.json", {"status": "running", "budget": 40,
                                      "completed_queries": len(outputs), "elapsed_seconds": time.perf_counter() - started})
    save_json(out / "state.json", {"status": "inference_complete", "requests": len(outputs),
                                  "wall_seconds": time.perf_counter() - started})
    export(out / "inference-freeze.json",
           {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()})
    print("Inference complete", len(outputs), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "infer"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out)
    else:
        infer(args.out)
