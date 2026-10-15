"""Competing visual explanations on large, fixed development and test cohorts."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exp14_fingerprint import apply, free_graph
from exp15_vector import nearest, vectors
from experiment_exp08 import data_records
from experiment_exp26 import PROMPT as CF_PROMPT
from experiment_exp26 import check_frozen, curve_context, draw_panels, finite_rows, infer
from experiment_exp27 import PROMPT as EDGE_PROMPT
from experiment_exp27 import draw_edges, request_payload, trend_vectors

from gnss_research.files import read_json, sha256, write_json
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.vision import SYSTEM

PROTOCOL = Path("docs/experiment-exp28.md")
DEPENDENCIES = [Path(__file__), PROTOCOL, Path("scripts/exp28_data_baseline.py"),
                Path("scripts/report_exp28.py"), Path("scripts/experiment_exp26.py"),
                Path("scripts/experiment_exp27.py"), Path("scripts/exp14_fingerprint.py"),
                Path("scripts/exp15_vector.py"), Path("scripts/experiment_exp25.py")]
MODES = {
    "cf_flash_image": ("qwen3.8-flash", "cf", CF_PROMPT),
    "edge_flash_image": ("qwen3.8-flash", "edge", EDGE_PROMPT),
    "trend_flash_image": ("qwen3.8-flash", "trend", EDGE_PROMPT),
    "ce_flash_image": ("qwen3.8-flash", "image", None),
    "ce_flash_cfonly": ("qwen3.8-flash", "cf", None),
    "ce_flash_raw": ("qwen3.8-flash", "raw", None),
    "ce_flash_noimage": ("qwen3.8-flash", "noimage", None),
    "ce_flash_text": ("qwen3.8-flash", "text", None),
    "ce_vl_image": ("qwen3-vl-plus", "image", None),
    "ce_vl_text": ("qwen3-vl-plus", "text", None),
    "ce_vl_noimage": ("qwen3-vl-plus", "noimage", None),
}
PROMPT = """Determine which candidate GNSS jump pairs are a genuine TEMPORARY displacement
offset and its return. Reply with one JSON object {"pair_ids":[],"evidence":"one short
observational sentence"}. Select only supplied P IDs; selected pairs cannot share endpoints.
An empty list is valid. Other numerical events are handled independently.

Compare competing explanations instead of assuming every pair is temporary:
(1) A sustained observation offset: several available interior observations follow a
shifted trajectory and the later edge reverses that same shift across N/E/U.
(2) An independent permanent step, possibly combined with continuous physical movement.
(3) An isolated outlying observation followed by ordinary observations or a separate step.
A one-day spike and another nearby step do NOT together establish a temporary platform.
A smooth bend, acceleration or slowing of physical motion is also not a temporary offset.
Use observations throughout the proposed interval, not just two conspicuous points.

When images are supplied, first inspect the original context, then compare each candidate's
original curve with three hypothetical operations: subtract the onset's local offset only
inside [start,end]; subtract it permanently from start onward; mask only the single onset
sample without filling it. Columns share centers and scales. Gray is the ORIGINAL reference,
blue is the displayed hypothesis. These are diagnostic alternatives, not corrected truth.
Only an onset estimate is used: the return is never fitted to force closure. Prefer a
temporary explanation only when the original supports a persistent shifted interval and
its correction improves BOTH boundaries while preserving unrelated steps and smooth motion.
If masking one isolated point removes the apparent excursion but a separate step remains,
do not force a temporary pairing. A gap after masking cannot establish unseen continuity.
The independent permanent step may lie before, inside or after a true temporary interval;
do not require absolute levels outside the interval to match or flatten genuine motion.

The onset estimate may be distorted by motion or short adjacent segments. Do not decide
solely from residual smoothness, historical extrapolation, a single axis or estimator name.
Text, when supplied, contains the same original points and offset estimates and permits
the identical hypothetical operations. Missing samples remain unavailable. Never invent
directions, impose a universal amplitude/speed/duration cutoff or assume an anomaly count.
If observations do not resolve the competing explanations, leave the pair unselected.
With no curve observations, select none rather than inventing visual evidence.
"""


def counterfactual_views(raw, pair, offset):
    temporary = raw.copy()
    temporary[pair["start"]:pair["end"] + 1] -= offset
    permanent = raw.copy()
    permanent[pair["start"]:] -= offset
    masked = raw.copy()
    masked[pair["start"]] = np.nan
    return raw, temporary, permanent, masked


def draw_competition(evidence, graph, pair, offset, path):
    raw = evidence.values / evidence.noise
    left = max(evidence.background_end + 1, pair["start"] - 30)
    right = min(len(raw), pair["end"] + 32)
    views = counterfactual_views(raw, pair, offset)
    titles = ("Original observations", "Temporary-offset hypothesis", "Permanent-step hypothesis", "Isolated-point hypothesis")
    figure, panels = plt.subplots(3, 4, figsize=(17, 7.8), dpi=150, sharex=True, sharey="row")
    times = np.arange(left, right)
    for axis in range(3):
        center = np.nanmedian(raw[left:min(left + 10, right), axis])
        for column, values in enumerate(views):
            panel = panels[axis, column]
            panel.plot(times, raw[left:right, axis] - center, ".-", color="#bcbcbc", linewidth=.65, markersize=2)
            panel.plot(times, values[left:right, axis] - center, ".-", color="#246e8c", linewidth=.9, markersize=2.5)
            for jump in graph["jumps"]:
                if left <= jump["index"] < right:
                    target = jump["index"] in (pair["start"], pair["end"] + 1)
                    panel.axvline(jump["index"], color="#ad503e" if target else "#999999", linestyle=":", linewidth=.9)
                    panel.text(jump["index"], .94, jump["id"], transform=panel.get_xaxis_transform(), fontsize=7)
            panel.grid(alpha=.2)
            panel.set_ylabel(f"{'NEU'[axis]} / station MAD")
            if axis == 0:
                panel.set_title(titles[column], fontsize=10)
            if axis == 2:
                panel.set_xlabel("Original daily index; gaps unfilled")
    figure.suptitle(f"{pair['id']} [{pair['start']},{pair['end']}]: compare alternatives; hypotheses are not observations")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def prepare(baseline: Path, output: Path, phase: str):
    config, dataset, manifest, records = data_records(baseline)
    if phase != "pilot" and not (baseline / "execution-complete.json").exists():
        raise ValueError("complete numerical predictions first")
    if config["split"] == "test":
        freeze = read_json(Path("docs/exp28-test-freeze.json"))
        if any(sha256(Path(path)) != digest for path, digest in freeze["frozen_files"].items()):
            raise ValueError("method changed after development freeze")
    if phase == "pilot":
        output.mkdir(parents=True, exist_ok=False)
    else:
        if not (output / "pilot-completed.json").exists():
            raise ValueError("finish technical pilot before expansion")
        check_frozen(read_json(output / "pilot-registration.json"))
    records = sorted(records, key=lambda record: record["station_id"])
    records = records[:60] if phase == "pilot" else records[60:]
    for record in records:
        for method in config["methods"]:
            path = baseline / "predictions" / method / record["station_id"] / "result.json"
            if not path.exists() or read_json(path)["status"] != "success":
                raise ValueError("complete all selected-station numerical predictions first")
    jobs = []
    for number, record in enumerate(records, 1):
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        numeric = read_json(baseline / "predictions/segment_numeric" / station / "result.json")
        graph = free_graph(numeric)
        local_vectors = vectors(observation, graph) if graph["pairs"] else []
        inputs = output / "inputs" / station
        inputs.mkdir(parents=True)
        candidates = {"graph": graph, "numerical_prediction": numeric}
        write_json(inputs / "candidates.json", {**candidates, "vectors": local_vectors})
        write_json(output / "final/vector_nearest" / station / "result.json",
                   apply(observation, numeric, graph, nearest(graph, local_vectors)))
        image_sets = {variant: [] for variant in ("cf", "edge", "trend", "raw", "image")}
        metadata = {"graph": graph, "correction_definition":
                    "Subtract onset local median delta on inclusive [start,end]; unchanged elsewhere"}
        ce_metadata = {"graph": graph, "hypotheses": {
            "temporary": "Subtract onset local median delta on inclusive [start,end]",
            "permanent": "Subtract same onset delta from start onward",
            "isolated": "Mask only the onset sample; no interpolation"}}
        text_data = None
        if graph["pairs"]:
            evidence, left, right = curve_context(observation, graph)
            missing = np.flatnonzero(~evidence.observed).tolist()
            metadata["missing_indices"] = missing
            ce_metadata["missing_indices"] = missing
            original_image = inputs / "original.png"
            draw_panels(observation, graph, original_image)
            for images in image_sets.values():
                images.append(original_image)
            offset_map = {item["id"]: np.asarray(item["delta"]) for item in local_vectors}
            trend_map = {item["id"]: np.asarray(item["delta"], dtype=float) for item in trend_vectors(observation, graph)}
            for pair in graph["pairs"]:
                offset = offset_map[pair["first_id"]]
                old_cf, zoom = inputs / f"{pair['id']}-cf.png", inputs / f"{pair['id']}-edges.png"
                competition = inputs / f"{pair['id']}-competition.png"
                draw_panels(observation, graph, old_cf, pair, offset)
                draw_edges(observation, graph, pair, offset, zoom)
                draw_competition(evidence, graph, pair, offset, competition)
                image_sets["cf"].append(old_cf)
                image_sets["edge"].extend([old_cf, zoom])
                image_sets["image"].append(competition)
                trend_offset = trend_map[pair["first_id"]]
                if np.isfinite(trend_offset).all():
                    trend_cf, trend_zoom = inputs / f"{pair['id']}-trend-cf.png", inputs / f"{pair['id']}-trend-edges.png"
                    draw_panels(observation, graph, trend_cf, pair, trend_offset)
                    draw_edges(observation, graph, pair, trend_offset, trend_zoom)
                    image_sets["trend"].extend([trend_cf, trend_zoom])
                else:
                    raw_edges = inputs / f"{pair['id']}-raw-edges.png"
                    draw_edges(observation, graph, pair, None, raw_edges)
                    image_sets["trend"].append(raw_edges)
            text_data = {"start_index": left, "end_index_exclusive": right,
                "original_station_normalized": finite_rows(evidence.values[left:right] / evidence.noise),
                "historical_residual_station_normalized": finite_rows(evidence.residual[left:right] / evidence.noise),
                "onset_and_return_vectors": local_vectors}
        for mode, (model, variant, prompt) in MODES.items():
            work = output / "predictions" / mode / station
            work.mkdir(parents=True)
            write_json(work / "candidates.json", candidates)
            if not graph["pairs"]:
                write_json(work / "result.json", {"status": "success", "request_count": 0,
                           "pair_decision": {"pair_ids": [], "evidence": "No free candidate pairs"}})
                write_json(output / "final" / mode / station / "result.json", numeric)
                continue
            is_competing = mode.startswith("ce_")
            payload = request_payload(model, PROMPT if is_competing else prompt,
                ce_metadata if is_competing else metadata,
                image_sets.get(variant, []), text_data if variant == "text" else None)
            write_json(work / "request.json", payload)
            jobs.append({"station": station, "mode": mode, "model": model, "kind": "pair",
                         "request_sha256": sha256(work / "request.json")})
        rf = read_json(baseline / "predictions/rf_split_temporal" / station / "result.json")
        work = output / "predictions/sdr_flash" / station
        payload = prepare_pair_focus(observation, rf, work, "qwen3.8-flash", SYSTEM, "slow_subtracted")
        if payload is None:
            write_json(output / "final/sdr_flash" / station / "result.json",
                       repair_return_pairs(observation, read_json(work / "result.json"), read_json(work / "candidates.json")))
        else:
            jobs.append({"station": station, "mode": "sdr_flash", "model": "qwen3.8-flash", "kind": "sdr",
                         "request_sha256": sha256(work / "request.json")})
        if graph["pairs"] or number % 20 == 0:
            print("prepared", number, "/", len(records), station, len(graph["pairs"]), "pairs", flush=True)
    registration = {"experiment_id": "GNSS-VLM-EXP28", "phase": phase, "split": config["split"],
                    "baseline": str(baseline), "dataset": str(dataset), "seed": manifest["seed"],
                    "cohort_seeds": manifest["cohort_seeds"], "records": records, "jobs": jobs,
                    "frozen_files": {str(path): sha256(path) for path in DEPENDENCIES}}
    write_json(output / f"{phase}-registration.json", registration)
    frozen = output / "frozen-source"
    frozen.mkdir(exist_ok=True)
    for path in DEPENDENCIES:
        if not (frozen / path.name).exists():
            shutil.copyfile(path, frozen / path.name)
    print("registered", len(jobs), "requests", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "infer"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "remaining"), default="pilot")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.baseline, args.output, args.phase)
    else:
        infer(args.output, args.phase)
