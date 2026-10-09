"""Post-inference derivative consistency diagnostics, never reference labels or scores."""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_landslide_frozen_stage_heads import read_json, verify_hashes  # noqa: E402

from gnss_sim.artifacts import sha, write_json  # noqa: E402
from gnss_sim.landslide_frozen_stage_heads import METHODS  # noqa: E402


def finite(values):
    return [value for value in values if value is not None and np.isfinite(value)]


def median(values):
    values = finite(values)
    return float(np.median(values)) if values else None


def main():
    run = ROOT / "artifacts/landslide-frozen-stage-heads-2026-10-08/run-v1"
    public = run / "public"
    manifest = read_json(run / "manifest.json")
    verify_hashes(run, manifest)
    records = read_json(public / "source-manifest.json")["cases"]
    totals = {method: {str(window): Counter() for window in (31, 61, 91)} for method in METHODS}
    cases = []
    for record in records:
        case_id = record["case_id"]
        motions = {window: read_json(public / record["diagnostics"][str(window)]["data"])
                   for window in (31, 61, 91)}
        method_counts = {}
        for method in METHODS:
            rows = read_json(public / f"daily/{case_id}-{method}.json")
            method_counts[method] = {}
            for window, motion in motions.items():
                counts = Counter()
                for row, value in zip(rows, motion["tangential_acceleration_mm_day2"], strict=True):
                    if row["feature"] not in {"acceleration", "deceleration"}:
                        continue
                    if value is None or not np.isfinite(value):
                        counts["no_finite_tangent"] += 1
                        continue
                    counts["finite_A_D_days"] += 1
                    if row["feature"] == "acceleration" and value < 0:
                        counts["A_negative_tangent"] += 1
                    elif row["feature"] == "deceleration" and value > 0:
                        counts["D_positive_tangent"] += 1
                    elif value == 0:
                        counts["zero_tangent"] += 1
                method_counts[method][str(window)] = dict(counts)
                totals[method][str(window)].update(counts)
        intervals = []
        stage_file = public / f"stage-reviews/{case_id}.json"
        if stage_file.exists():
            for span in read_json(stage_file)["stages"]:
                windows = {}
                for window, motion in motions.items():
                    blocks = []
                    for start in range(span["start"], span["stop"], 15):
                        stop = min(start + 15, span["stop"])
                        blocks.append({"start": start, "stop": stop,
                                       "speed_median": median(motion["speed_mm_day"][start:stop]),
                                       "tangent_median": median(motion["tangential_acceleration_mm_day2"][start:stop]),
                                       "velocity_E_median": median([
                                           v[1] if v is not None else None
                                           for v in motion["velocity_mm_day"][start:stop]])})
                    windows[str(window)] = blocks
                intervals.append({**span, "15_day_block_statistics": windows})
        cases.append({"case_id": case_id, "final_derivative_sign_counts": method_counts,
                      "visual_native_span_statistics": intervals})
    report = {"provenance": "Post-inference author diagnostic; no independent stage reference",
              "meaning": "Instantaneous fitted sign versus interval labels; NOT error counts or accuracy",
              "limitations": "Noise, centered smoothing, steps and transition tolerance can cause sign disagreement. "
                             "No significance threshold calibrated; do not auto-relabel or rank methods.",
              "inference_freeze_sha256": sha(run / "inference-freeze.json"),
              "script_sha256": sha(Path(__file__)), "totals": totals, "cases": cases}
    source_copy = run / "posthoc-source" / Path(__file__).name
    source_copy.parent.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__), source_copy)
    write_json(public / "author-diagnostics.json", report)
    verify_hashes(run, manifest)
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
