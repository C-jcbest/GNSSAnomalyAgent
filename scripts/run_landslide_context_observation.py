"""Pair a middle-scale raw view with the existing bounded visual review."""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from analyze_landslide_background_benchmark import verify_data
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json, render_window
from xgboost import XGBClassifier

from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    raw_motion_evidence,
    review_priorities,
    review_prompt,
)
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import parse_window_activity

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
CASE_IDS = ["case_2025", "case_2032", "case_2027", "case_2034", "case_2029", "case_2036"]
EXTRA_VIEW = (
    "\nThe third image is an additional raw-only view at its printed original day indices. "
    "Use it to check continuity around the target. It does not expand the requested output interval."
)


def prepare(out):
    data = BASE / "background-benchmark-data-v1"
    original = BASE / "background-review-v1"
    complete = BASE / "utility-prototype-development-v1"
    verify_data(data)
    for folder in (original, complete):
        for name, expected in read_json(folder / "prediction-freeze.json").items():
            if sha(folder / name) != expected:
                raise ValueError("An existing prediction was changed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    (out / "cached/images").mkdir(parents=True)
    (out / "cached/requests").mkdir()
    initial = read_json(original / "predictions.json")
    local = {**read_json(complete / "cached-locals.json"), **read_json(complete / "new-locals.json")}
    selections, packets, cached_local, bindings = {}, [], {}, {}
    for cid in CASE_IDS:
        target = out / "inputs" / cid
        target.mkdir(parents=True)
        for name in ("input.json", "motion-31.json", "motion-61.json", "motion-91.json"):
            source = data / "inputs" / cid / name
            shutil.copyfile(source, target / name)
            bindings[str(source.resolve())] = sha(source)
        case, feature = load_case_features(out, cid)
        windows = regular_windows(len(case.dates), 180, 90)
        long = raw_motion_evidence(case, feature)
        centers = np.linspace(0, len(case.dates), 4)[1:-1]
        fixed_indices = sorted({min(range(len(windows)), key=lambda i: abs(sum(windows[i]) / 2 - center))
                                for center in centers})
        fixed = [{"index": i, "start": windows[i][0], "stop": windows[i][1]} for i in fixed_indices]
        for rep in (0, 1):
            baseline = np.array(initial[f"direct_visual_r{rep}"][cid]["activity"])
            priorities, selected = review_priorities(baseline, feature, long, windows, ReviewPolicy())
            permission = conflict_neighbourhood(baseline, feature, long, ReviewPolicy())[1]
            selections[f"{cid}-r{rep}"] = {
                "priorities": priorities, "raw": selected, "fixed": fixed,
                "neighbourhood": np.flatnonzero(permission).tolist(),
            }
            needed = sorted({row["index"] for row in selected} | set(fixed_indices))
            cached_keys = [f"{cid}-direct-r{rep}"]
            for index in needed:
                start, stop = windows[index]
                key = f"{cid}-window-{index:02d}-image-r{rep}"
                if key not in local:
                    raise ValueError("The paired 180-day control has no cached answer")
                cached_local[key] = local[key]
                cached_keys.append(key)
                for variant in ("context360", "duplicate180"):
                    if variant == "duplicate180" and index not in {row["index"] for row in selected}:
                        continue
                    context_start = max(0, min(start - 90, len(case.dates) - 360))
                    context_stop = context_start + 360
                    extra_name = f"{cid}-window-{index:02d}-{variant}.png"
                    destination = out / "images" / extra_name
                    if not destination.exists():
                        if variant == "context360":
                            render_window(case, feature.noise, context_start, context_stop, destination,
                                          {"single_plot_inches": [10.24, 6.3], "single_plot_dpi": 150})
                        else:
                            shutil.copyfile(complete / "images" / f"{cid}-window-{index:02d}.png", destination)
                    packets.append({
                        "request_id": f"{cid}-window-{index:02d}-{variant}-r{rep}",
                        "case_id": cid, "repeat": rep, "index": index, "variant": variant,
                        "start": start, "stop": stop,
                        "context_start": context_start if variant == "context360" else start,
                        "context_stop": context_stop if variant == "context360" else stop,
                        "prompt": review_prompt(feature, long, start, stop, False) + EXTRA_VIEW,
                        "images": [f"{cid}-raw.png", f"{cid}-window-{index:02d}.png", extra_name],
                    })
            for key in cached_keys:
                owner = original if (original / "requests" / f"{key}.response.json").exists() else complete
                for suffix in ("started", "response"):
                    source = owner / "requests" / f"{key}.{suffix}.json"
                    shutil.copyfile(source, out / "cached/requests" / source.name)
                    bindings[str(source.resolve())] = sha(source)
                receipt = read_json(owner / "requests" / f"{key}.started.json")
                for name, digest in receipt["image_sha256"].items():
                    source = complete / "images" / name
                    if sha(source) != digest:
                        raise ValueError("The existing image control no longer matches its payload")
                    for folder in (out / "images", out / "cached/images"):
                        if not (folder / name).exists():
                            shutil.copyfile(source, folder / name)
    for source in (original / "predictions.json", complete / "cached-locals.json", complete / "new-locals.json",
                   data / "reference-labels.csv", data / "stage-xgboost.json"):
        bindings[str(source.resolve())] = sha(source)
    shutil.copyfile(data / "stage-xgboost.json", out / "stage-xgboost.json")
    save_json(out / "initial-predictions.json", {
        f"direct_visual_r{rep}": {cid: initial[f"direct_visual_r{rep}"][cid] for cid in CASE_IDS} for rep in (0, 1)
    })
    save_json(out / "cached-locals.json", cached_local)
    save_json(out / "selections.json", selections)
    save_json(out / "packets.json", packets)
    save_json(out / "protocol.json", {
        "date": "2026-10-10", "scope": "Six exposed development records on two shared residual backgrounds. "
        "One per family, alternating backgrounds by predefined family order; not independent confirmation",
        "case_ids": CASE_IDS, "data": str(data.resolve()), "original": str(original.resolve()),
        "complete": str(complete.resolve()), "primary": "adaptive_context", "model": "qwen3.8-flash",
        "change": "Add a 360-day raw supporting view to unchanged global+180-day view; same target and prompt "
        "as a duplicate180 supporting-image control. No numerical answer, payoff model or learned approval",
        "window_basis": "Exactly twice the existing180-day extent, centered when possible, shifted inside "
        "calendar at ends, no search. Same physical raw units, rendering/noise-floor and missing gaps",
        "selection": "Existing21/42 raw conflicts,14-day minimum, at most2 disjoint180-day targets",
        "revision": "Original49-day conflict-neighbourhood authority intersected with the target; VLM assigns states",
        "ablations": ["direct_visual", "adaptive_local", "fixed_local", "fixed_context", "adaptive_duplicate"],
        "continue_gate": "Both repeats contextualF1>adaptive_localF1, net correct definite days versus "
        "adaptive_local>0 and >=2 positive-net records; F1>=duplicate both and >duplicate in at least1 repeat. "
        "Direct, weak/event/boundary/stage and cost tradeoffs reported, no all-metric superiority requirement",
        "stage": "Existing commontrain24 numerical stage head gated by final activity, not native VLM ASD",
        "numerical": "Twelve original numerical/direction/scale controls, shared train24/validation12 and fixed models",
        "model_requests_per_deployed_record": "one cached global + up to2 local; image count differs, not call count",
        "repetitions": 2, "maximum_requests": len(packets), "parallel_requests": 2, "retries": 0,
        "references": "Known injected activity only, residual background not certified motion-free",
        "new_records": 0, "calibration_branch": "Stopped; no reuse of Ridge approval or numerical vote as contribution",
    })
    for folder in (ROOT / "scripts", ROOT / "src/gnss_sim"):
        for source in folder.glob("*.py"):
            target = out / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            bindings[str(source.resolve())] = sha(source)
    save_json(out / "execution-bindings.json", bindings)
    save_json(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared"})
    print({"prepared": str(out), "new_request_ceiling": len(packets), "records": len(CASE_IDS)}, flush=True)


def verify(out):
    for name, expected in read_json(out / "freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Frozen contextual inputs differ")
    for name, expected in read_json(out / "execution-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError("Context experiment source or control differs")


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Execute only a fresh experiment; no retry or overwrite")
    config = read_json(out / "protocol.json")
    records = {cid: load_case_features(out, cid) for cid in config["case_ids"]}
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"], config["maximum_requests"])
    save_json(out / "state.json", {"status": "running"})
    local, failures = {}, {}

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

    with ThreadPoolExecutor(max_workers=config["parallel_requests"]) as pool:
        pending = [pool.submit(ask, packet) for packet in read_json(out / "packets.json")]
        for future in as_completed(pending):
            key, result, error = future.result()
            local[key] = result
            if error is not None:
                failures[key] = error
            save_json(out / "local-predictions.json", local)
            save_json(out / "failures.json", failures)
            print(f"{len(local)}/{len(pending)} {key} {result['status']}", flush=True)
    initial = read_json(out / "initial-predictions.json")
    cached = read_json(out / "cached-locals.json")
    selections = read_json(out / "selections.json")
    head = XGBClassifier()
    head.load_model(out / "stage-xgboost.json")
    predictions, used, permissions = {}, {}, {}
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
            for stem in ("adaptive_local", "fixed_local", "adaptive_context", "fixed_context", "adaptive_duplicate"):
                method = f"{stem}_r{rep}"
                targets = choice["fixed" if stem.startswith("fixed") else "raw"]
                activity = baseline.copy()
                authority = target_scopes(neighbourhood, feature.observed, targets)[0]
                keys = [global_key]
                for target in targets:
                    index = target["index"]
                    if stem.endswith("local"):
                        key = f"{cid}-window-{index:02d}-image-r{rep}"
                        reviewed = cached[key]
                    else:
                        variant = "duplicate180" if stem.endswith("duplicate") else "context360"
                        key = f"{cid}-window-{index:02d}-{variant}-r{rep}"
                        reviewed = local[key]
                    permission = target_scopes(neighbourhood, feature.observed, [target])[0]
                    activity = scoped_activity(activity, np.array(reviewed["activity"]), permission, feature.observed)
                    keys.append(key)
                predictions.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                used.setdefault(method, []).extend(keys)
                permissions.setdefault(method, {})[cid] = np.flatnonzero(authority).tolist()
    for name, value in (("predictions.json", predictions), ("used-requests.json", used), ("permissions.json", permissions)):
        save_json(out / name, value)
    save_json(out / "prediction-freeze.json", {name: sha(out / name) for name in
        ("local-predictions.json", "predictions.json", "used-requests.json", "permissions.json", "failures.json")})
    save_json(out / "state.json", {"status": "predictions_complete", "requests": len(local), "failures": len(failures)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "execute"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out.resolve())
    else:
        execute(args.out.resolve())


if __name__ == "__main__":
    main()
