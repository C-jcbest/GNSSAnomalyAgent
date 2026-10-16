"""Pair retrieved training-background plots with fixed examples and repeated raw."""

from __future__ import annotations

import argparse
import csv
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_landslide_background_benchmark import verify_data
from probe_landslide_revision_scope import scoped_activity, target_scopes
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_context_observation import BASE, CASE_IDS
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import raw_motion_evidence, review_prompt
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_normal_contrast import (
    anchored_blocks,
    anchored_series,
    select_normal_examples,
)
from gnss_sim.landslide_seasonal_reference import observed_values
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import parse_window_activity

NORMAL_IDS = ["case_2001", "case_2007", "case_2037", "case_2043"]
NOTICE = (
    "\nThe third image compares the target raw curve with supporting curves, as identified by panel titles. "
    "They are either REPEATED QUERY curves or observed TRAIN BACKGROUND examples with no target injection. "
    "All are aligned by relative window day and subtract ONLY their first30-day observed median constant; "
    "physical amplitudes and missing gaps are preserved. They are separate time periods, not simultaneous "
    "measurements, an exact counterfactual for this query, or target labels. Training background can show "
    "ordinary seasonal changes and correlated fluctuations. For this task, identify additional sustained "
    "displacement activity rather than labelling every visible background fluctuation as activity. "
    "Compare continuing N/E/U change with the range and shape of background examples when supplied. "
    "Similarity alone does not certify no activity, and dissimilarity alone does not establish it: verify "
    "continuing cumulative change in the preserved raw target and use uncertain when indistinguishable. "
    "Repeated query panels supply no background evidence. Do not copy exemplar extrema or their class "
    "into target decisions. Output ONLY original query indices in the requested180-day target."
)


def render_contrast(case, feature, target, references, normal_cases, bank, arm, path):
    start, stop = target["start"], target["stop"]
    query = anchored_series(observed_values(case)[0], start, stop)
    all_normal = [anchored_series(observed_values(normal_cases[item["case_id"]])[0], item["start"], item["stop"])
                  for item in bank]
    limits = []
    for axis in range(3):
        values = np.concatenate([query[:, axis]] + [normal[:, axis] for normal in all_normal])
        values = values[np.isfinite(values)]
        low, high = values.min(), values.max()
        middle = (low + high) / 2
        span = max(high - low, 6 * feature.noise[axis], 10)
        limits.append((middle - .7 * span, middle + .7 * span))
    fig, panels = plt.subplots(3, 3, figsize=(13, 7), sharex=True)
    for column in range(3):
        reference = references[column] if len(references) == 3 else None
        if arm == "raw_repeated" or reference is None:
            example = query
            title = "REPEATED QUERY | not a background example"
        else:
            example = anchored_series(observed_values(normal_cases[reference["case_id"]])[0],
                                      reference["start"], reference["stop"])
            title = reference["case_id"] + f" | own days {reference['start']}..{reference['stop']-1}\nTRAIN BACKGROUND, no target injection"
        panels[0, column].set_title(title, fontsize=8)
        for axis, name in enumerate(("N", "E", "U")):
            panel = panels[axis, column]
            panel.plot(np.arange(start, stop), query[:, axis], color="#147d72", linewidth=.9, label="Query")
            panel.plot(np.arange(start, stop), example[:, axis], color="#c78635", linewidth=.8, label="Supporting curve")
            panel.set_ylim(*limits[axis])
            panel.set_xlim(start, stop - 1)
            panel.set_ylabel(name + " change (mm)")
            panel.grid(alpha=.2)
        panels[-1, column].set_xticks(np.r_[np.arange(start, stop, 60), stop-1])
        panels[-1, column].set_xlabel("Query original day; example relative days aligned", fontsize=7)
    panels[0, 0].legend(fontsize=7)
    fig.suptitle(f"{case.case_id} | target {start}..{stop-1} | constant alignment, NO amplitude scaling")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def prepare(out):
    data = BASE / "background-benchmark-data-v1"
    prior = BASE / "context-observation-development-v1"
    verify_data(data)
    for name, expected in read_json(prior / "freeze.json").items():
        if sha(prior / name) != expected:
            raise ValueError("The completed context probe input changed")
    split = read_json(data / "manifest.json")["split"]
    if not set(NORMAL_IDS).issubset(split["train"]):
        raise ValueError("Normal examples must come entirely from the common train split")
    with (data / "reference-labels.csv").open(encoding="utf-8", newline="") as handle:
        training_reference = [row for row in csv.DictReader(handle) if row["case_id"] in NORMAL_IDS]
    if {row["case_id"] for row in training_reference} != set(NORMAL_IDS):
        raise ValueError("Training examples lack reference labels")
    if any(int(row["activity_label"]) == 1 for row in training_reference):
        raise ValueError("A normal training example contains injected activity")
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    shutil.copytree(prior / "cached", out / "cached")
    bindings = {}
    for cid in CASE_IDS + NORMAL_IDS:
        target = out / "inputs" / cid
        target.mkdir(parents=True)
        for name in ("input.json", "motion-31.json", "motion-61.json", "motion-91.json"):
            source = data / "inputs" / cid / name
            shutil.copyfile(source, target / name)
            bindings[str(source.resolve())] = sha(source)
    normal_cases = {cid: load_case_features(out, cid)[0] for cid in NORMAL_IDS}
    bank = []
    for background, cid in zip((0, 1, 6, 7), NORMAL_IDS):
        case = normal_cases[cid]
        values = observed_values(case)[0]
        for index, (start, stop) in enumerate(regular_windows(len(case.dates), 180, 90)):
            descriptor = anchored_blocks(values, start, stop)
            if descriptor is not None:
                bank.append({"case_id": cid, "background": background, "index": index,
                             "start": start, "stop": stop, "descriptor": descriptor.tolist()})
    selections = read_json(prior / "selections.json")
    packets, retrieval = [], {}
    for cid in CASE_IDS:
        case, feature = load_case_features(out, cid)
        long = raw_motion_evidence(case, feature)
        for rep in (0, 1):
            for target in selections[f"{cid}-r{rep}"]["raw"]:
                query = anchored_blocks(observed_values(case)[0], target["start"], target["stop"])
                retrieved, status = select_normal_examples(query, bank, feature.noise)
                fixed = [next(item for item in bank if item["case_id"] == normal_id and item["index"] == target["index"])
                         for normal_id in NORMAL_IDS[:3]]
                target_key = f"{cid}-window-{target['index']:02d}-r{rep}"
                retrieval[target_key] = {"retrieved": retrieved, "fixed": fixed, "support_status": status,
                                         "query_descriptor": None if query is None else query.tolist()}
                order = ["normal_retrieved", "normal_fixed", "raw_repeated"]
                rotation = (target["index"] + rep) % 3
                order = order[rotation:] + order[:rotation]
                for arm in order:
                    name = f"{cid}-window-{target['index']:02d}-{arm}.png"
                    references = retrieved if arm == "normal_retrieved" else fixed
                    if not (out / "images" / name).exists():
                        render_contrast(case, feature, target, references, normal_cases, bank, arm, out / "images" / name)
                    images = [f"{cid}-raw.png", f"{cid}-window-{target['index']:02d}.png", name]
                    for raw in images[:2]:
                        if not (out / "images" / raw).exists():
                            shutil.copyfile(prior / "images" / raw, out / "images" / raw)
                    packets.append({"request_id": f"{cid}-window-{target['index']:02d}-{arm}-r{rep}",
                        "case_id": cid, "repeat": rep, "index": target["index"], "arm": arm,
                        "start": target["start"], "stop": target["stop"], "images": images,
                        "prompt": review_prompt(feature, long, target["start"], target["stop"], False) + NOTICE,
                        "support_status": status if arm == "normal_retrieved" else "available"})
    for name in ("initial-predictions.json", "selections.json", "stage-xgboost.json"):
        shutil.copyfile(prior / name, out / name)
        bindings[str((prior / name).resolve())] = sha(prior / name)
    legacy = prior / "analysis-v1/predictions-with-controls.json"
    old = read_json(legacy)
    save_json(out / "legacy-predictions.json", {
        f"legacy_{stem}_r{rep}": old[f"{stem}_r{rep}"] for rep in (0, 1) for stem in ("adaptive_local", "fixed_local")
    })
    bindings[str(legacy.resolve())] = sha(legacy)
    bindings[str((data / "reference-labels.csv").resolve())] = sha(data / "reference-labels.csv")
    save_json(out / "bank.json", bank)
    save_json(out / "retrieval.json", retrieval)
    save_json(out / "packets.json", packets)
    save_json(out / "protocol.json", {
        "date": "2026-10-10", "scope": "Six exposed development records, two shared backgrounds, not independent confirmation",
        "data": str(data.resolve()), "prior": str(prior.resolve()), "case_ids": CASE_IDS, "normal_ids": NORMAL_IDS,
        "primary": "normal_retrieved", "model": "qwen3.8-flash", "maximum_requests": len(packets),
        "parallel_requests": 2, "repetitions": 2, "retries": 0, "new_records": 0,
        "change": "Actual observed normal train examples instead of fitted annual curve; retrieve three distinct "
        "training backgrounds by six30-day raw block medians anchored to first block, distance scaled only by "
        "query axis daily-MAD noise proxy; no amplitude rescaling, DTW, target labels, trained approval or vote",
        "conditional_task": "Same known-injected activity recovery. Matched raw/fixed/retrieved prompt explicitly "
        "distinguishes expected background from additional sustained movement; old answers retained as legacy controls",
        "normal_scope": "Four common-train records with no target injection, not certified field no-motion. "
        "Offline grouped train; some dates later than queries, no online forecasting claim",
        "selection": "Unchanged originalraw21/42 conflict14 support, at most2 disjoint180-day targets and49 permission",
        "fixed_control": "First three normal IDs at query's same regular-window index; fixed examples, NOT fixed query windows",
        "all_images": "Same global/local raw plus three-column supplemental query/example contrast. All supplement axis "
        "limits matched across arms against complete train bank, constant first-block alignment only",
        "missing_support": "Unavailable query descriptor supplies no normal reference; paired panel becomes repeated "
        "query explicitly titled, support failure retained. No fallback to a different selector",
        "continue_gate": "Both repeats mainF1>matchedraw, net correct definite days vs matchedraw>0 and at least2 "
        "records helped; F1>=fixedexample both and strict gain in at least1. Weak/event/boundary/stage/cost reported",
        "fairness": "Original twelve numeric controls trained on the same train24 labels including four zero-injection "
        "records; no additional observations or target labels. If retrieval helps, full prototype adds explicitly "
        "retrieval-derived numerical rule/XGB with same train/validation tuning, before independent confirmation",
        "stage": "Same commontrain24 numeric stage head, activity gated, no nativeVLM ASD contribution claim",
        "novelty": "Normal reference conditioning is established in TAMA/Anlu. Only GNSS observed increment "
        "retrieval coupled to bounded review might differentiate; this pilot does not establish novelty/efficacy",
    })
    sources = [path for folder in (ROOT / "scripts", ROOT / "src/gnss_sim") for path in folder.glob("*.py")]
    sources.append(ROOT / "tests/test_landslide_normal_contrast.py")
    for source in sources:
        destination = out / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        bindings[str(source.resolve())] = sha(source)
    save_json(out / "execution-bindings.json", bindings)
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared"})
    print({"prepared": str(out), "bank_windows": len(bank), "requests": len(packets)}, flush=True)


def verify(out):
    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Frozen normal contrast input differs")
    for name, expected in read_json(out / "execution-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Executing normal contrast source or baseline differs")


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Only a fresh normal contrast experiment can execute; no retries")
    config = read_json(out / "protocol.json")
    records = {cid: load_case_features(out, cid) for cid in config["case_ids"]}
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], config["maximum_requests"])
    local, failures = {}, {}
    save_json(out / "state.json", {"status": "running"})

    def ask(packet):
        key = packet["request_id"]
        feature = records[packet["case_id"]][1]
        try:
            payload = transport.ask(key, packet["prompt"], [out / "images" / name for name in packet["images"]])
            activity = parse_window_activity(payload, feature.observed, packet["start"], packet["stop"])
            status, error = "success", None
        except (RuntimeError, ValueError) as exception:
            activity, status, error = np.full(len(feature.observed), -1), "failed", str(exception)
        return key, {**packet, "activity": activity.tolist(), "status": status}, error

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(ask, packet) for packet in read_json(out / "packets.json")]
        for future in as_completed(pending):
            key, value, error = future.result()
            local[key] = value
            if error is not None:
                failures[key] = error
            save_json(out / "local-predictions.json", local)
            save_json(out / "failures.json", failures)
            print(f"{len(local)}/{len(pending)} {key} {value['status']}", flush=True)
    predictions = read_json(out / "legacy-predictions.json")
    initial, selections = [read_json(out / name) for name in ("initial-predictions.json", "selections.json")]
    old_used = read_json(Path(config["prior"]) / "used-requests.json")
    used = {f"legacy_{stem}_r{rep}": old_used[f"{stem}_r{rep}"]
            for rep in (0, 1) for stem in ("adaptive_local", "fixed_local")}
    head = XGBClassifier()
    head.load_model(out / "stage-xgboost.json")
    permissions = {}
    for cid, (_, feature) in records.items():
        codes = head.predict(feature.matrix).astype(int) + 1
        for rep in (0, 1):
            baseline = np.array(initial[f"direct_visual_r{rep}"][cid]["activity"])
            global_key = f"{cid}-direct-r{rep}"
            method = f"direct_visual_r{rep}"
            predictions.setdefault(method, {})[cid] = initial[method][cid]
            used.setdefault(method, []).append(global_key)
            choice = selections[f"{cid}-r{rep}"]
            neighbourhood = np.zeros(len(baseline), dtype=bool)
            neighbourhood[choice["neighbourhood"]] = True
            targets = choice["raw"]
            authority = target_scopes(neighbourhood, feature.observed, targets)[0]
            for arm in ("normal_retrieved", "normal_fixed", "raw_repeated"):
                method = f"{arm}_r{rep}"
                activity, keys = baseline.copy(), [global_key]
                for target in targets:
                    key = f"{cid}-window-{target['index']:02d}-{arm}-r{rep}"
                    permission = target_scopes(neighbourhood, feature.observed, [target])[0]
                    activity = scoped_activity(activity, np.array(local[key]["activity"]), permission, feature.observed)
                    keys.append(key)
                predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                used.setdefault(method, []).extend(keys)
                permissions.setdefault(method, {})[cid] = np.flatnonzero(authority).tolist()
    for name, value in (("predictions.json", predictions), ("used-requests.json", used), ("permissions.json", permissions)):
        save_json(out / name, value)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in
        ("local-predictions.json", "predictions.json", "used-requests.json", "permissions.json", "failures.json")})
    save_json(out / "state.json", {"status": "predictions_complete", "requests": len(local), "failures": len(failures)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "execute"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out.resolve())
    else:
        execute(args.out.resolve())
