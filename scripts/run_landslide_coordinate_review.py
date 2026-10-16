"""Freeze and execute an equal-cost coordinate-reflection mechanism pilot."""

from __future__ import annotations

import argparse
import csv
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence, review_priorities
from gnss_sim.landslide_coordinate_review import render_coordinate_view
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import WINDOW_TASK, parse_window_activity

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
CASE_IDS = [f"case_{index:04d}" for index in (109, 111, 113, 115, 117, 119)]
ARMS = ("raw", "raw_repeat", "reflected")
NOTICE = (
    "\nThe coordinate convention is printed in each image title. Displayed N/E/U may be "
    "+1 or -1 times the original three coordinates. This changes only the coordinate signs, "
    "not time, physical amplitudes, observations or the sustained-activity definition. "
    "Judge continuing displacement independently of the coordinate orientation."
)


def prepare(out):
    prior = BASE / "cross-seed-confirmation-v1"
    data = BASE / "cross-seed-confirmation-data-v1"
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    bindings = {}
    prior_predictions = read_json(prior / "analysis-v1/predictions.json")
    numeric = {method: {cid: values[cid] for cid in CASE_IDS}
               for method, values in prior_predictions.items() if not method.endswith(("r0", "r1"))
               and "__" not in method}
    if len(numeric) != 9:
        raise ValueError("Expected the nine common calibrated numerical controls")
    controls = {method: {cid: values[cid] for cid in CASE_IDS}
                for method, values in prior_predictions.items() if method.startswith("direct_visual")
                or method.startswith("adaptive_no_feedback") or method.startswith("fixed_image_review")}
    save_json(out / "prior-controls.json", {**controls, **numeric})
    shutil.copyfile(data / "stage-xgboost.json", out / "stage-xgboost.json")
    with (prior / "reference-subset.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        reference = [row for row in reader if row["case_id"] in CASE_IDS]
    with (out / "reference-subset.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(reference)
    plans, choices, rendering = [], {}, {}
    order = np.random.default_rng(2026101074)
    for cid in CASE_IDS:
        shutil.copytree(prior / "inputs" / cid, out / "inputs" / cid)
        case, feature = load_case_features(out, cid)
        long = raw_motion_evidence(case, feature)
        windows = regular_windows(len(case.dates), 180, 90)
        for sign, arm in ((1, "raw"), (-1, "reflected")):
            name = f"{cid}-{arm}-global.png"
            rendering[name] = render_coordinate_view(case, feature.noise, 0, len(case.dates),
                                                     sign, out / "images" / name)
        for rep in (0, 1):
            initial = np.array(controls[f"direct_visual_r{rep}"][cid]["activity"])
            priorities, selected = review_priorities(initial, feature, long, windows, ReviewPolicy())
            if selected != read_json(prior / "selections.json")[f"{cid}-r{rep}"]["selected"]:
                raise ValueError("The coordinate pilot must preserve the original observation selection")
            neighbourhood = conflict_neighbourhood(initial, feature, long, ReviewPolicy())[1]
            permission = target_scopes(neighbourhood, feature.observed, selected)[0]
            choices[f"{cid}-r{rep}"] = {"priorities": priorities, "selected": selected,
                                        "permission": np.flatnonzero(permission).tolist()}
            for target in selected:
                index, start, stop = [target[name] for name in ("index", "start", "stop")]
                for sign, arm in ((1, "raw"), (-1, "reflected")):
                    name = f"{cid}-{arm}-window-{index:02d}.png"
                    if name not in rendering:
                        rendering[name] = render_coordinate_view(case, feature.noise, start, stop,
                                                                 sign, out / "images" / name)
                for arm in order.permutation(ARMS):
                    display = "reflected" if arm == "reflected" else "raw"
                    plans.append({"request_id": f"{cid}-window-{index:02d}-{arm}-r{rep}",
                                  "case_id": cid, "repeat": rep, "arm": str(arm), **target,
                                  "prompt": WINDOW_TASK.format(start=start, end=stop - 1) + NOTICE,
                                  "images": [f"{cid}-{display}-global.png",
                                             f"{cid}-{display}-window-{index:02d}.png"]})
    for folder in (ROOT / "scripts", ROOT / "src/gnss_sim"):
        for source in folder.glob("*.py"):
            destination = out / "source" / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            bindings[str(source.resolve())] = sha(source)
    for source in (prior / "analysis-v1/predictions.json", prior / "reference-subset.csv",
                   prior / "selections.json", data / "stage-xgboost.json", data / "manifest.json"):
        bindings[str(source.resolve())] = sha(source)
    save_json(out / "bindings.json", bindings)
    save_json(out / "rendering.json", rendering)
    save_json(out / "selections.json", choices)
    save_json(out / "packets.json", plans)
    save_json(out / "protocol.json", {
        "date": "2026-10-10", "case_ids": CASE_IDS, "primary": "coordinate_consensus",
        "scope": "Six exposed controlled records, first member of each family in fixed family order; development only",
        "prior": str(prior), "data": str(data), "model": "qwen3.8-flash",
        "repetitions": 2, "maximum_requests": len(plans), "parallel_requests": 2, "retries": 0,
        "mechanism": "Raw and exactly sign-reflected global/local readings must agree for a definite revision; disagreement unknown",
        "selection": "Unchanged21/42 raw conflict,14 support days,<=2 disjoint180-day targets,49-day revision scope",
        "controls": "Single matched raw, reflected alone, two ordinary raw readings with identical consensus rule; original direct/fixed/adaptive and nine numeric controls",
        "deploy_calls": "One initial global plus up to four local calls; full receipt costs reported, extra compute controlled by ordinary-repeat consensus",
        "stage": "Unchanged commontrain24 stage head after activity; sign-reflected stage prediction is not used",
        "continue_gate": "Both repeats primaryF1>=single_rawF1, primaryWeakRecall>=single_rawWeakRecall, fewer wrong definite decisions and positive net definite correction versus single_raw. PrimaryF1>=repeat_consensus both and greater in at least one. No threshold or prompt search if failed",
        "unknown": "Unknown is neither TN credit nor a free removal of missed positives; full scoring denominators remain unchanged",
        "new_records": 0, "field_reference": "Still absent; no expert labels imputed",
    })
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path)
                                   for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared"})
    print({"prepared": str(out), "requests": len(plans)}, flush=True)


def verify(out):
    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError(f"Coordinate pilot frozen artifact changed: {name}")
    for name, expected in read_json(out / "bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Coordinate pilot source/control changed: {name}")


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Run only once from prepared state")
    config = read_json(out / "protocol.json")
    packets = read_json(out / "packets.json")
    records = {cid: load_case_features(out, cid) for cid in config["case_ids"]}
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], config["maximum_requests"])
    save_json(out / "state.json", {"status": "running"})
    reviews, failures = {}, {}

    def ask(packet):
        feature = records[packet["case_id"]][1]
        try:
            payload = transport.ask(packet["request_id"], packet["prompt"],
                                    [out / "images" / name for name in packet["images"]])
            activity = parse_window_activity(payload, feature.observed, packet["start"], packet["stop"])
            return packet, activity, "success", None
        except (RuntimeError, ValueError) as error:
            return packet, np.full(len(feature.observed), -1), "failed", str(error)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(ask, packet) for packet in packets]
        for count, future in enumerate(as_completed(futures), 1):
            packet, activity, status, error = future.result()
            key = packet["request_id"]
            reviews[key] = {"activity": activity.tolist(), "status": status}
            if error is not None:
                failures[key] = error
            save_json(out / "local-predictions.json", reviews)
            save_json(out / "failures.json", failures)
            print(f"Coordinate review {count}/{len(packets)}: {key} {status}", flush=True)
    save_json(out / "prediction-freeze.json", {
        name: sha(out / name) for name in ("local-predictions.json", "failures.json")})
    save_json(out / "state.json", {"status": "predictions_complete", "requests": len(packets)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out)
    else:
        execute(args.out)


if __name__ == "__main__":
    main()
