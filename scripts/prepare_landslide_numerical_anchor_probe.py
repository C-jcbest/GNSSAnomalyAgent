"""Freeze a numerical-first weak-motion review and inspect available image receipts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import save_json, sha
from run_landslide_context_observation import BASE, CASE_IDS
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence, review_priorities
from gnss_sim.landslide_error_audit import regular_windows


def prepare(out):
    data = BASE / "background-benchmark-data-v1"
    complete = BASE / "utility-prototype-development-v1"
    cached = {**read_json(complete / "cached-locals.json"), **read_json(complete / "new-locals.json")}
    numeric = read_json(complete / "numeric-predictions.json")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    selections, missing = {}, []
    for cid in CASE_IDS:
        case, features = load_case_features(complete, cid)
        baseline = np.array(numeric["xgboost_84"][cid]["activity"])
        long = raw_motion_evidence(case, features)
        windows = regular_windows(len(baseline), 180, 90)
        priorities, _ = review_priorities(baseline, features, long, windows, ReviewPolicy())
        ranked = sorted(priorities, key=lambda row: (-row["missed_motion_days"], row["start"]))
        selected = []
        for row in ranked:
            if row["missed_motion_days"] < ReviewPolicy().minimum_conflict_days:
                break
            if any(row["start"] < other["stop"] and other["start"] < row["stop"] for other in selected):
                continue
            selected.append(row)
            if len(selected) == 2:
                break
        selected.sort(key=lambda row: row["start"])
        centers = np.linspace(0, len(baseline), 4)[1:-1]
        fixed_indices = sorted({min(range(len(windows)), key=lambda i: abs(sum(windows[i]) / 2-center)) for center in centers})
        fixed = [{"index": i, "start": windows[i][0], "stop": windows[i][1]} for i in fixed_indices]
        neighbourhood = conflict_neighbourhood(baseline, features, long, ReviewPolicy())[1]
        authority = target_scopes(neighbourhood, features.observed, selected)[0] & (baseline != 1)
        selections[cid] = {"numerical_initial": baseline.tolist(), "priorities": priorities,
            "adaptive": selected, "fixed": fixed, "editable_indices": np.flatnonzero(authority).tolist(),
            "neighbourhood": np.flatnonzero(neighbourhood).tolist()}
        for rep in (0, 1):
            for index in sorted({row["index"] for row in selected} | set(fixed_indices)):
                key = f"{cid}-window-{index:02d}-image-r{rep}"
                if key not in cached:
                    missing.append({"request_id": key, "case_id": cid, "repeat": rep,
                        "index": index, "start": windows[index][0], "stop": windows[index][1]})
    save_json(out / "protocol.json", {
        "date": "2026-10-10", "scope": "Six exposed development records. New numerical-first mechanism, not independent confirmation",
        "case_ids": CASE_IDS, "data": str(data.resolve()), "complete": str(complete.resolve()),
        "primary": "numerical_anchor_visual_supplement", "numeric_initial": "Existing commontrain24/validation12 XGB84, parameters unchanged",
        "difference": "Replace noisy visual global initial by high-precision numerical initial; preserve every numerical positive, "
        "rank only raw21/42 missed-motion days, and let image-only VLM revise nonpositive dates within original49 authority",
        "selection": "Existing14-day support, two disjoint180-day views; no threshold search, numerical prior approval or learned payoff",
        "controls": "Numerical initial alone; matched fixed-window VLM supplement; raw numerical-only supplement in same authority; "
        "original direct/fixed/adaptive VLM and twelve strong numeric controls. VLM must add real recovery beyond numerical-only supplementation",
        "cache": "Historical image-only WINDOW_TASK depends on raw images and target, not earlier initial states. Reuse only identical "
        "prompt/images, charge all selected views; absent outputs are missing, never favourable fallback decisions",
        "continue_gate": "Both repeats mainF1>numeric initial and raw-only supplement, weakR>numeric initial, net definite correction "
        "versus numeric initial>0 in at least2 records; at least one strictF1 gain over fixed-window supplement. All tradeoffs reported",
        "stage": "Original common numerical head gated by final activity", "model": "qwen3.8-flash", "maximum_reviews_per_record": 2,
        "preflight_new_api": 0, "maximum_missing_view_requests": len(missing), "retries": 0,
        "fixed_scope": "Fixed target authority uses same49 neighbourhood and preserves original numeric positives",
        "references": "Known injections only, background field motion not certified", "new_records": 0,
    })
    save_json(out / "selections.json", selections)
    save_json(out / "missing-requests.json", missing)
    save_json(out / "bindings.json", {str(path.resolve()): sha(path) for path in
        (Path(__file__), complete / "cached-locals.json", complete / "new-locals.json", complete / "numeric-predictions.json")})
    save_json(out / "state.json", {"status": "prepared", "cached_coverage_complete": not missing,
                                   "missing_views": len(missing), "new_api": 0})
    print({"prepared": str(out), "missing_views": len(missing), "new_api": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out.resolve())
