"""Compile explicitly authored Codex decisions with the existing reference contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_diagnostics import LandslideDiagnostics
from gnss_sim.landslide_reference_interface import parse_partition_draft, partition_sha256
from gnss_sim.landslide_stage_reference import compile_reference, reference_template


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite_median(values):
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else None


def numerical_inventory(context, segments):
    """Describe cached rates; this function never assigns a label."""
    inventory = []
    for segment in segments:
        start, stop = segment["start"], segment["stop"]
        window_statistics = {}
        for window, motion in context["motions"].items():
            speed = np.array([np.nan if value is None else value
                              for value in motion.speed_mm_day[start:stop]])
            acceleration = np.array([np.nan if value is None else value for value in
                                     motion.tangential_acceleration_mm_day2[start:stop]])
            third = max(1, len(speed) // 3)
            window_statistics[str(window)] = {
                "speed_valid_days": int(np.isfinite(speed).sum()),
                "tangential_valid_days": int(np.isfinite(acceleration).sum()),
                "speed_thirds_mm_day": [finite_median(speed[:third]),
                                        finite_median(speed[third:-third]),
                                        finite_median(speed[-third:])],
            }
        inventory.append({"start": start, "stop": stop, "windows": window_statistics})
    return inventory


def render_review(case, rows, target):
    values = np.asarray([value if value is not None else [np.nan] * 3
                         for value in case.displacement_mm])
    feature_codes = {"unknown": 0, "none": 1, "acceleration": 2,
                     "steady_motion": 3, "deceleration": 4}
    activity_codes = {-1: 0, 0: 1, 1: 2}
    activity = [activity_codes[row["activity_label"]] for row in rows]
    stage = [feature_codes[row["feature"]] for row in rows]
    for day, value in enumerate(case.displacement_mm):
        if value is None:
            activity[day] = 3
            stage[day] = 5
    figure, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True,
                               gridspec_kw={"height_ratios": [3, 3, 3, 1, 1]})
    for component, name in enumerate(("N", "E", "U")):
        axes[component].plot(np.arange(len(values)), values[:, component], linewidth=.7)
        axes[component].set_ylabel(name + " (mm)")
        axes[component].grid(alpha=.2)
    bands = ((activity, ["#edc46a", "#e0e7e0", "#407ea0", "#777777"], "Activity"),
             (stage, ["#edc46a", "#e0e7e0", "#c96055", "#498774", "#6581b7", "#777777"], "Stage"))
    for panel, (codes, colors, name) in zip(axes[3:], bands, strict=True):
        panel.imshow([codes], aspect="auto", interpolation="nearest", vmin=0,
                     vmax=len(colors) - 1, cmap=ListedColormap(colors),
                     extent=(-.5, len(values) - .5, .5, -.5))
        panel.set_yticks([])
        panel.set_ylabel(name)
    axes[-1].set_xlim(0, len(values) - 1)
    axes[-1].set_xlabel("Original day index; half-open authored intervals")
    figure.suptitle(case.case_id + " | Codex assisted reference, DEVELOPMENT ONLY")
    figure.tight_layout(rect=(0, .06, 1, 1))
    figure.text(.10, .02, "Activity: blue=confirmed, pale=stationary, amber=unknown, gray=missing\n"
                "Stage: red=A, green=S, blue=D; all bands use the exported daily rows", fontsize=9)
    figure.savefig(target, dpi=120)
    plt.close(figure)
    return {"activity_codes": activity, "stage_codes": stage}


def compile_review(review_directory):
    out = review_directory / "compiled"
    if out.exists():
        raise FileExistsError("Review already compiled; preserve it and use a new version")
    for name, digest in read_json(review_directory / "preparation-manifest.json")["files"].items():
        if sha(review_directory / name) != digest:
            raise ValueError("Prepared evidence changed: " + name)
    activity_freeze = read_json(review_directory / "activity-freeze.json")
    if sha(review_directory / "activity-decisions.json") != activity_freeze["activity_decisions_sha256"]:
        raise ValueError("Activity changed after its pre-stage freeze")
    activity_notes = read_json(review_directory / "activity-decisions.json")
    stage_notes = read_json(review_directory / "stage-decisions.json")
    if list(activity_notes["cases"]) != list(stage_notes["cases"]):
        raise ValueError("Activity and stage case inventories differ")
    contexts = []
    for case_id, spans in activity_notes["cases"].items():
        folder = review_directory / case_id
        case = LandslideInput.model_validate_json((folder / "input.json").read_bytes())
        canonical_sha = hashlib.sha256(case.model_dump_json().encode("utf-8")).hexdigest()
        record = {"input_sha256": canonical_sha, "views": [
            {"image_sha256": sha(folder / name)} for name in ("raw-global.png", "raw-local.png")]}
        payload = {"spans": [dict(zip(("start", "stop", "state", "raw_evidence"), span, strict=True))
                             for span in spans], "activity_notes": activity_notes["activity_notes"]}
        review, rows = parse_partition_draft(payload, case, record, "Current Codex session")
        review["reviewer"] = activity_notes["reviewer"]
        motions = {window: LandslideDiagnostics.model_validate_json(
            (folder / f"motion-{window}.json").read_bytes()) for window in (31, 61, 91)}
        if any(motion.input_sha256 != canonical_sha for motion in motions.values()):
            raise ValueError("Cached motion belongs to different observations")
        contexts.append({"case": case, "review": review, "rows": rows,
                         "activity_sha256": partition_sha256(review), "motions": motions})

    source_sha = sha(review_directory / "preparation-manifest.json")
    packet = reference_template(contexts, source_sha)
    packet["reviewer"] = {
        "name": "Codex / current session direct visual and numerical review / 2026-10-11",
        "background": "AI辅助复核，已见历史方法、历史生成标签及通用设计；本批具体预测与过程标签未读取。非独立专家。",
        "reviewer_type": "ai", "prior_predictions_seen": True,
        "prior_generation_labels_seen": True, "participated_in_activity_review": True,
    }
    reviewed_at = datetime.now(timezone.utc).isoformat()
    numeric = {}
    for entry, context in zip(packet["cases"], contexts, strict=True):
        segments = [dict(zip(stage_notes["columns"], values, strict=True))
                    for values in stage_notes["cases"][entry["case_id"]]]
        used = []
        for interior in entry["interiors"]:
            interior_segments = [segment for segment in segments
                                 if interior["start"] <= segment["start"] < interior["stop"]]
            used.extend(interior_segments)
            span = next(span for span in context["review"]["spans"]
                        if span["start"] == interior["start"])
            interior.update(review_status="reviewed", raw_activity_status="confirmed",
                            raw_activity_evidence=span["raw_evidence"], raw_views_seen=True,
                            auxiliary_views_seen=True, reviewed_at=reviewed_at,
                            segments=interior_segments)
        if used != segments:
            raise ValueError("Stage decision lies outside a frozen activity interior")
        numeric[entry["case_id"]] = numerical_inventory(context, segments)
    compiled_packet, daily, summary, support = compile_reference(packet, contexts, source_sha)
    summary["activity_counts"] = dict(Counter(row["activity_label"] for row in daily))
    summary["calendar_rows"] = len(daily)
    summary["missing_days"] = sum(value is None for c in contexts
                                  for value in c["case"].displacement_mm)
    out.mkdir()
    write_json(out / "activity-reviews.json", [context["review"] for context in contexts])
    write_json(out / "review.json", compiled_packet)
    write_json(out / "daily-reference.json", daily)
    write_json(out / "summary.json", summary)
    write_json(out / "support-inventory.json", support)
    write_json(out / "rate-checks.json", numeric)
    with (out / "daily-reference.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(daily[0]))
        writer.writeheader()
        writer.writerows(daily)
    plot_data = {}
    for context in contexts:
        case = context["case"]
        rows = [row for row in daily if row["case_id"] == case.case_id]
        plot_data[case.case_id] = render_review(case, rows, out / f"{case.case_id}-review.png")
    write_json(out / "plot-data.json", plot_data)
    inspected = {f"{case_id}/{name}": sha(review_directory / case_id / name)
                 for case_id in activity_notes["cases"]
                 for name in ("raw-global.png", "raw-local.png", "auxiliary.png")}
    write_json(out / "inspection-ledger.json", {
        "images_actually_inspected_sha256": inspected, "raw_images": 24,
        "auxiliary_images": 12, "process": "raw global/local + raw blocks -> activity freeze -> auxiliary + cached rate checks -> stage decisions",
        "reviewer": packet["reviewer"], "new_remote_model_requests": 0,
        "raw_local_display": "2160x840 resized by image tool to 2048x796",
        "input_labels_read_for_this_review": False,
        "post_visual_numeric_corrections": stage_notes["post_visual_numeric_corrections"],
    })
    shutil.copyfile(Path(__file__), out / "source.py")
    write_json(out / "reference-freeze.json", {
        "reviewed_at": reviewed_at, "activity_decisions_sha256": sha(review_directory / "activity-decisions.json"),
        "activity_freeze_sha256": sha(review_directory / "activity-freeze.json"),
        "stage_decisions_sha256": sha(review_directory / "stage-decisions.json"),
        "source_manifest_sha256": source_sha, "formal_reference_verified": False,
        "files": {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()},
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_directory", type=Path)
    compile_review(parser.parse_args().review_directory.resolve())
