"""Measure whether a local observation intervention also alters distant pixels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_paired_persistence import render_encoder_chart
from gnss_sim.landslide_visual_invariance import chart_numeric_descriptors, raw_chart_patches


def probe(data, pairs, out):
    verify_manifest(pairs, "encoding-freeze.json")
    selected = read_json(pairs / "selected-training-pairs.json")
    cases = {}
    patches = {}
    rows = []
    for row in selected:
        cid, day = row["case_id"], row["day"]
        if cid not in cases:
            case, features = load_case_features(data, cid)
            cases[cid] = case
            patches[cid] = raw_chart_patches(case, features.noise)
        original = patches[cid][day]
        with np.load(pairs / "pairs" / (row["pair_id"] + ".npz")) as archive:
            negative = archive["local_image"]
            negative_values = archive["local_observations"]
        raw = np.array(
            [value if value is not None else [np.nan] * 3 for value in cases[cid].displacement_mm]
        )
        np.testing.assert_array_equal(negative_values[: day - 21], raw[: day - 21])
        np.testing.assert_array_equal(negative_values[day + 22 :], raw[day + 22 :])
        # A 3-day slot centered at either endpoint may contain a changed
        # neighbor. Include all such slots in the permitted intervention area.
        inside = np.abs(np.arange(-90, 91, 3)) <= 21
        original_descriptor = chart_numeric_descriptors(original[None])[0][:183].reshape(3, 61)
        negative_descriptor = chart_numeric_descriptors(negative[None])[0][:183].reshape(3, 61)
        available = np.isfinite(original_descriptor) & np.isfinite(negative_descriptor)
        np.testing.assert_array_equal(original[3:], negative[3:])
        difference = np.abs(original_descriptor - negative_descriptor) * 31
        outside = available & ~inside[None, :]
        trace_difference = np.abs(original[:3] - negative[:3])
        total = float(trace_difference.sum())
        outer = float(trace_difference[:, :, ~inside].sum())
        rows.append(
            dict(
                pair_id=row["pair_id"],
                case_id=cid,
                day=day,
                family=row["family"],
                outside_observation_changed=0,
                outside_supported_axis_slots=int(outside.sum()),
                outside_axis_slots_changed=int((outside & (difference > 1e-6)).sum()),
                outside_max_pixel_shift=float(difference[outside].max()),
                outside_mean_pixel_shift=float(difference[outside].mean()),
                outside_trace_l1=outer,
                total_trace_l1=total,
                outside_trace_fraction=None if total == 0 else outer / total,
            )
        )
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "pair-context.csv", rows)
    fractions = np.array(
        [row["outside_trace_fraction"] for row in rows if row["outside_trace_fraction"] is not None]
    )
    summary = dict(
        pairs=len(rows),
        outside_observation_changed=0,
        pairs_with_outside_pixel_change=sum(row["outside_axis_slots_changed"] > 0 for row in rows),
        median_outside_trace_fraction=float(np.median(fractions)),
        median_outside_mean_pixel_shift=float(
            np.median([row["outside_mean_pixel_shift"] for row in rows])
        ),
        max_outside_pixel_shift=max(row["outside_max_pixel_shift"] for row in rows),
        interpretation="Training-side rendering diagnostic, not recognition benefit or permission to revive CP-PP",
        test_truth_opened=False,
        parent_manifest_sha256=sha(pairs / "encoding-freeze.json"),
    )
    write_json(out / "summary.json", summary)
    examples = sorted(rows, key=lambda row: row["outside_trace_fraction"], reverse=True)[:2]
    for row in examples:
        original = patches[row["case_id"]][row["day"]]
        with np.load(pairs / "pairs" / (row["pair_id"] + ".npz")) as archive:
            negative = archive["local_image"]
        figure, panels = plt.subplots(1, 2, figsize=(8, 4))
        for panel, patch, label in zip(
            panels, (original, negative), ("Original", "Local replacement, independently scaled")
        ):
            panel.imshow(render_encoder_chart(patch))
            panel.set_title(label)
            panel.axis("off")
        figure.suptitle(f"{row['pair_id']} | unchanged outside observations, altered outer pixels")
        figure.tight_layout()
        figure.savefig(out / (row["pair_id"] + ".png"), dpi=150)
        plt.close(figure)
    write_json(
        out / "example-plan.json",
        dict(
            pair_ids=[row["pair_id"] for row in examples],
            selection="Two largest outside trace fractions; TRAIN only, no score selection",
        ),
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(
        out / "freeze.json", {path.name: sha(path) for path in out.iterdir() if path.is_file()}
    )
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "pairs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    probe(args.data.resolve(), args.pairs.resolve(), args.out.resolve())
