"""Observation-derived turning proposals, never stage labels or physical boundaries."""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.signal import find_peaks

from gnss_sim.landslide_stage_numeric_experiment import numeric_summary

TURNING_NOTE = """

SAME-SOURCE INTERNAL TURNING PROPOSALS (NOT REFERENCE STAGES)
The JSON below proposes blocks where cached 3D speed has a local maximum/minimum.
Block ranges are coarse descriptive locations, NOT confidence intervals, stage
boundaries, physical onset or failure timing. Windows share observations and may
include future/platform/step effects. A speed extremum may reflect noise or a
measurement artifact; it does not confirm nonzero movement or any stage.
Use raw displacement and the existing multi-window plots to accept, reject or
leave each proposal unresolved. If warranted, split an activity into multiple
stages instead of carrying its first class through a later rate reversal.
You may choose different boundaries; never force a split at a proposal block.
No proposal does NOT imply a single stage, steady motion or stationary activity.
Keep activity fixed and retain uncertainty. The original output schema, stage
definitions, sparse spans and support masking remain unchanged.
Each fit lists preceding/current/following block speed medians in mm/day, availability
counts and raw NEU block medians in mm. Values are descriptive, not independent
validation or calibrated confidence. Missing/unsupported blocks are not bridged.
"""


def finite_runs(values):
    start = None
    for index, value in enumerate([*values, None]):
        if value is not None and start is None:
            start = index
        elif value is None and start is not None:
            yield start, index
            start = None


def turning_summary(case, review, motions, parameters):
    summary = numeric_summary(case, review, motions)
    if summary["bin_days"] != parameters["bin_days"]:
        raise ValueError("Proposal bins must match the frozen 30-day summary interface")
    activities = []
    for activity in summary["activities"]:
        blocks = activity["blocks"]
        proposals = []
        for window in (31, 61, 91):
            rates = []
            for block in blocks:
                fit = block["fits"][str(window)]
                observed, valid, _ = fit["n"]
                duration = block["bin"][1] - block["bin"][0]
                supported = (
                    observed >= math.ceil(parameters["minimum_observed_fraction"] * duration)
                    and valid >= max(3, math.ceil(parameters["minimum_fit_fraction"] * observed))
                    and fit["q"] is not None
                )
                rates.append(float(fit["q"][1]) if supported else None)
            for start, stop in finite_runs(rates):
                if stop - start < 3:
                    continue
                values = np.array(rates[start:stop])
                spread = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
                if spread <= 1e-12:
                    continue
                for kind, sign in (("local_maximum", 1), ("local_minimum", -1)):
                    peaks, properties = find_peaks(
                        values * sign,
                        distance=parameters["minimum_separation_bins"],
                        prominence=parameters["relative_prominence"] * spread,
                    )
                    for index, prominence in zip(peaks, properties["prominences"], strict=True):
                        number = start + int(index)
                        center_block = blocks[number]
                        neighbors = blocks[number - 1:number + 2]
                        proposals.append({
                            "kind": kind, "window_days": window, "block": center_block["bin"],
                            "center_day": sum(center_block["bin"]) / 2,
                            "relative_prominence": float(prominence / spread),
                            "context_blocks": [block["bin"] for block in neighbors],
                            "speed_medians_mm_day": [block["fits"][str(window)]["q"][1] for block in neighbors],
                            "availability_counts": [block["fits"][str(window)]["n"] for block in neighbors],
                            "raw_NEU_block_medians_mm": [block["raw_NEU_mm"] for block in neighbors],
                        })
        groups = []
        for proposal in sorted(proposals, key=lambda item: (item["kind"], item["center_day"], item["window_days"])):
            compatible = next((group for group in groups if group[0]["kind"] == proposal["kind"]
                               and proposal["center_day"] - min(p["center_day"] for p in group)
                               <= parameters["window_match_days"]
                               and proposal["window_days"] not in {p["window_days"] for p in group}), None)
            if compatible is None:
                groups.append([proposal])
            else:
                compatible.append(proposal)
        candidates = []
        for group in groups:
            if len(group) < parameters["minimum_windows"]:
                continue
            candidates.append({
                "kind": group[0]["kind"],
                "coarse_block_range": [min(p["block"][0] for p in group), max(p["block"][1] for p in group)],
                "windows": sorted(p["window_days"] for p in group),
                "fits": sorted(group, key=lambda p: p["window_days"]),
            })
        candidates.sort(key=lambda item: (-len(item["windows"]),
                                         -max(p["relative_prominence"] for p in item["fits"])))
        retained = candidates[:parameters["maximum_candidates_per_interior"]]
        retained.sort(key=lambda item: item["coarse_block_range"][0])
        activities.append({"interior": activity["interior"], "candidates": retained,
                           "single_window_proposals": len(proposals),
                           "unretained_multiscale_candidates": len(candidates) - len(retained)})
    return {"case_id": summary["case_id"], "input_sha256": summary["input_sha256"],
            "cache_sha256": summary["cache_sha256"], "parameters": parameters,
            "activities": activities,
            "meaning": "coarse cached speed turning proposals only; not stage predictions"}


def turning_prompt(original, summary):
    return original + TURNING_NOTE + json.dumps(summary, separators=(",", ":"), allow_nan=False)
