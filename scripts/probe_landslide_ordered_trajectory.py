"""Freeze a four-training-window mechanism test before any new remote response."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from probe_landslide_contrastive_exemplars import SOURCE, block_packet, raw_values
from probe_landslide_endpoint_fit import ROOT, export
from probe_landslide_visual_invariance import calibration_labels
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import save_json
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_ordered_trajectory import parse_query_labels, render_trajectory
from gnss_sim.landslide_visual import VisualRequests

ARMS = ("time_only", "time_ordered_path", "time_gray_path")
TASK = """Review continuing raw GNSS displacement on the42 original queried dates.
Return exactly JSON {"labels":[42 integers]}, in consecutive original date order:
1 continuing displacement,0 no continuing displacement,-1 unknown. No extra keys.
First confirm continuing displacement within the query. An elevated plateau,
isolated spike or one displacement step is not sustained activity. Weak cumulative
motion counts if supported; do not turn derivative response into confirmed motion.

The first image always shows the same original NEU time observations. Any second
image projects the same thirteen14-day median blocks into NE/NU/EU coordinates,
each axis divided by that record's daily-difference noise proxy. A constant
reference is removed. No detrending, rotation, filling or compressed time.
Original block slot j has center-84.5+14*j days. Missing slots are absent and not
connected. Black rings mark the three blocks within queried dates-21..20; each
median summarizes14 original days. A trajectory line connects adjacent block
medians; it does NOT prove continuous movement between them or give daily
boundaries. Gray/color changes only visual order encoding, not measurements;
original slot indices and numeric block tables retain dates in all arms.
Color is chronology, not activity, confidence, velocity or a reference label.
All axes use noise units descriptively, not calibrated confidence or physical
equivalence of axis noise. If daily boundaries cannot be resolved, keep unknown.
"""


def render_time(case, features, day, path):
    values = raw_values(case)
    normalized, available, _, medians = discovery_block_matrix(values, features.noise)
    first = np.flatnonzero(available[day])[0]
    local = (values[day - 91:day + 91] - medians[day, first]) / features.noise
    figure, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for axis, panel in enumerate(axes):
        panel.plot(np.arange(-91, 91), local[:, axis], color="#197c78", linewidth=.8)
        panel.plot(-84.5 + 14 * np.arange(13), normalized[day, :, axis],
                   "o", color="#555555", markersize=4)
        panel.axvspan(-21, 20, color="#5289ba", alpha=.16)
        panel.set_ylabel(("N", "E", "U")[axis] + " / noise proxy")
        panel.grid(alpha=.2)
    axes[-1].set_xlim(-91, 90)
    axes[-1].set_xlabel("Original relative date | shaded42days queried")
    figure.suptitle("Original displacement | same time image in all arms")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    split = read_json(SOURCE / "protocol.json")["split"]
    pairs = {cid: load_case_features(SOURCE, cid) for cid in split["train"]}
    labels = calibration_labels(SOURCE, pairs)
    candidates = []
    for cid, (case, features) in pairs.items():
        blocks, availability, _, _ = discovery_block_matrix(raw_values(case), features.noise)
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        weak = np.array([int(row.get("weak_signal", 0)) == 1 for row in labels[cid]])
        for day in range(98, len(actual) - 98, 14):
            start, stop = day - 21, day + 21
            observed = features.observed[start:stop]
            codes = actual[start:stop][observed]
            if len(codes) < 35 or len(set(codes.tolist())) != 1 or codes[0] < 0:
                continue
            if availability[day].sum() < 10 or not availability[day, 5:8].all():
                continue
            changes = np.linalg.norm(np.diff(blocks[day], axis=0), axis=1)
            before, after = changes[5:7]
            max_outer = float(np.nanmax(np.r_[changes[:5], changes[7:]]))
            if codes[0] == 1 and weak[start:stop].sum() >= 14:
                role = "weak_active_training"
                priority = -float(min(before, after))
            elif codes[0] == 0 and max_outer >= 5 and max(before, after) <= 2:
                role = "post_change_inactive_training"
                priority = max_outer
            else:
                continue
            candidates.append({"case_id": cid, "day": day, "query": [start, stop],
                               "role": role, "training_center_class": int(codes[0]),
                               "priority": priority, "weak_days": int(weak[start:stop].sum())})
        export(out / "inputs" / (cid + ".json"), case.model_dump(mode="json"))
    selected, used = [], set()
    for role in ("weak_active_training", "post_change_inactive_training"):
        eligible = sorted([row for row in candidates if row["role"] == role],
                          key=lambda row: (-row["priority"], row["case_id"], row["day"]))
        for row in eligible:
            if row["case_id"] in used:
                continue
            selected.append(row)
            used.add(row["case_id"])
            if sum(value["role"] == role for value in selected) == 2:
                break
        if sum(value["role"] == role for value in selected) != 2:
            raise ValueError("Training evidence does not contain two eligible examples per role")
    export(out / "selected-training-queries.json", selected)
    packets = {}
    for index, row in enumerate(selected):
        key = f"q{index + 1}"
        cid, day = row["case_id"], row["day"]
        case, features = pairs[cid]
        table = block_packet(case, features, day, "TARGET", None)
        render_time(case, features, day, out / (key + "-time.png"))
        blocks = np.array(table["normalized_neu_blocks"], dtype=float)
        for color, suffix in ((True, "ordered"), (False, "gray")):
            render_trajectory(blocks, out / (key + "-" + suffix + ".png"), color)
        prompt = (TASK + "\nOriginal queried dates:" + json.dumps(list(range(*row["query"])))
                  + "\nSame original measurements:" + json.dumps(table, separators=(",", ":")))
        for arm in ARMS:
            images = [key + "-time.png"]
            if arm != "time_only":
                suffix = "ordered" if arm == "time_ordered_path" else "gray"
                images.append(key + "-" + suffix + ".png")
            packets[key + ":" + arm] = {"images": images, "prompt": prompt,
                                        "observed": features.observed[row["query"][0]:row["query"][1]].tolist()}
    export(out / "packets.json", packets)
    rng = np.random.default_rng(20261018)
    schedule = []
    for repetition in range(2):
        for key in rng.permutation(["q1", "q2", "q3", "q4"]).tolist():
            for arm in rng.permutation(ARMS).tolist():
                schedule.append({"query_id": key, "arm": arm, "repetition": repetition,
                                 "request_id": f"{key}-{arm}-r{repetition}"})
    export(out / "schedule.json", schedule)
    export(out / "protocol.json", {
        "scope": "Training-only mechanism value check, not validation or independent efficacy",
        "budget": 24, "model": "qwen3.8-flash", "repetitions": 2, "arms": list(ARMS),
        "selection": "Two known homogeneous weak active/two inactive after visible outer change; training labels only",
        "mechanism": "Additional space trajectory of existing raw blocks; same first image and numerical table",
        "gray_ablation": "Remove color order only; slot labels/table retain temporal information",
        "boundary_limit": "14day medians cannot prove all intervening daily boundaries; unknown allowed",
        "gate": "Each repeat net definite correct versus time_only >0; weak recall no loss; "
                "inactive controls zero definite correct loss; ordered above gray once and not below either repeat",
        "stop": "If any necessary condition fails stop fixed pilot without window/seed/color/prompt/example rescue",
        "references": "Already exposed training latent labels; not new gold standard; review never changes them",
        "new_records": 0, "validation_records_accessed": 0, "independent_confirmation": False,
    })
    sources = [Path(__file__), ROOT / "src/gnss_sim/landslide_ordered_trajectory.py",
               ROOT / "tests/test_landslide_ordered_trajectory.py", SOURCE / "protocol.json",
               SOURCE / "reference-labels.csv", ROOT / "src/gnss_sim/landslide_feature_discovery.py",
               ROOT / "src/gnss_sim/landslide_visual.py"]
    bindings = {}
    for index, path in enumerate(sources):
        copy = out / "source" / f"{index:02d}-{path.name}"
        copy.parent.mkdir(exist_ok=True)
        copy.write_bytes(path.read_bytes())
        bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    export(out / "root-bindings.json", bindings)
    export(out / "pre-inference-freeze.json",
           {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared", "new_api_calls": 0})
    print("Prepared four training windows, three arms,24 requests", flush=True)


def infer(out):
    verify_bindings(out, "pre-inference-freeze.json")
    for name, expected in read_json(out / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen source differs")
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Run only a never-started pilot")
    packets = read_json(out / "packets.json")
    transport = VisualRequests(out / "requests", ROOT / ".env", "qwen3.8-flash", 24)
    outputs = {}
    started = time.perf_counter()
    save_json(out / "state.json", {"status": "running", "budget": 24})
    for row in read_json(out / "schedule.json"):
        packet = packets[row["query_id"] + ":" + row["arm"]]
        request_id = row["request_id"]
        try:
            payload = transport.ask(request_id, packet["prompt"], [out / name for name in packet["images"]])
            codes = parse_query_labels(payload)
            codes[~np.array(packet["observed"])] = -1
            outputs[request_id] = {"status": "success", "labels": codes.tolist()}
        except (ValueError, RuntimeError, TypeError) as error:
            outputs[request_id] = {"status": "failed", "labels": [-1] * 42,
                                   "error_type": type(error).__name__, "error": str(error)}
        save_json(out / "outputs.json", outputs)
        print(request_id, outputs[request_id]["status"], flush=True)
    save_json(out / "state.json", {"status": "inference_complete", "attempts": 24,
                                   "loop_seconds": time.perf_counter() - started})
    export(out / "inference-freeze.json",
           {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*")
            if path.is_file() and path.name not in {"inference-freeze.json", "state.json"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "infer"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out.resolve())
    else:
        infer(args.out.resolve())
