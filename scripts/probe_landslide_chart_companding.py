"""Diagnose plotted target evidence on TRAIN only before classifier experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_companded_chart import (
    chart_ordinates,
    companded_chart_patches,
    raw_window_medians,
)
from gnss_sim.landslide_paired_persistence import render_encoder_chart
from gnss_sim.landslide_visual_invariance import raw_chart_patches


def probe(data, existing, out):
    out.mkdir(parents=True, exist_ok=False)
    ids = read_json(existing / "protocol.json")["split"]["train"]
    write_json(
        out / "protocol.json",
        {
            "scope": "All48 TRAIN only; exposure diagnostics do not select labels, samples, parameters or claims of classifier improvement",
            "change": "Asinh scaled by3 raw difference MAD proxy; central14 raw median anchor; same181-day footprint,61 three-day median slots,32 raster rows, original missing calendar; no clipping or jump removal",
            "control": "Same asinh/noise transform anchored to global window median; original linear raw chart",
            "measure": "Absolute signed component plotted difference between columns27/34 (~day-9/+12), pixels58/31; report max NEU only for descriptive display diagnostics",
            "noise_limit": "Any amplification also affects noise; pixel gain is not a detection gain or independent significance",
            "test_truth_opened": False,
            "validation_truth_used": False,
            "data_manifest_sha256": sha(data / "data-freeze.json"),
        },
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    all_rows = []
    examples = []
    for cid in ids:
        case, features = load_case_features(data, cid)
        medians, anchor = raw_window_medians(case)
        original = raw_chart_patches(case, features.noise)
        local = companded_chart_patches(medians, anchor, features.noise)
        global_part = companded_chart_patches(
            medians, anchor, features.noise, target_anchored=False
        )
        if not np.array_equal(original[features.observed, 3:], local[features.observed, 3:]):
            raise ValueError("Target companding changed original available columns")
        with np.load(existing / "train" / (cid + ".npz")) as archive:
            selected, activity = archive["selected"], archive["activity"]
        ordinates = [chart_ordinates(patches) for patches in (original, local, global_part)]
        differences = [np.abs(value[:, :, 34] - value[:, :, 27]) * 58 / 31 for value in ordinates]
        valid = selected & np.stack([np.isfinite(value).all(axis=1) for value in differences]).all(
            axis=0
        )
        for day in np.flatnonzero(valid):
            changes = [float(value[day].max()) for value in differences]
            all_rows.append(
                {
                    "case_id": cid,
                    "day": int(day),
                    "activity": int(activity[day]),
                    "original_pixels": changes[0],
                    "target_companded_pixels": changes[1],
                    "global_companded_pixels": changes[2],
                }
            )
        # The largest pixel gain per TRAIN record is a visualization example,
        # not a selected training example or evidence of anomaly accuracy.
        positive = np.flatnonzero(valid & (activity == 1))
        if len(positive):
            gain = (differences[1][positive].max(axis=1) + 0.1) / (
                differences[0][positive].max(axis=1) + 0.1
            )
            index = int(np.argmax(gain))
            examples.append(
                (float(gain[index]), cid, int(positive[index]), original, local, global_part)
            )
            examples.sort(key=lambda row: row[0], reverse=True)
            examples = examples[:2]
    write_csv(out / "train-display-measures.csv", all_rows)
    summary = {}
    for label in (0, 1):
        rows = [row for row in all_rows if row["activity"] == label]
        summary[str(label)] = {"centers": len(rows)}
        for key in ("original_pixels", "target_companded_pixels", "global_companded_pixels"):
            values = np.array([row[key] for row in rows])
            summary[str(label)][key] = {
                "median": float(np.median(values)),
                "q10": float(np.quantile(values, 0.1)),
                "q90": float(np.quantile(values, 0.9)),
                "below_one_pixel": int((values < 1).sum()),
            }
    write_json(out / "display-summary.json", summary)
    example_rows = []
    for gain, cid, day, original, local, global_part in examples:
        for name, patches in (("original", original), ("target", local), ("global", global_part)):
            render_encoder_chart(patches[day]).save(out / f"{cid}-day{day}-{name}.png")
        example_rows.append(
            {
                "case_id": cid,
                "day": day,
                "display_gain": gain,
                "selection": "Post-measure TRAIN illustration only; no classifier scores exist",
            }
        )
    write_json(out / "examples.json", example_rows)
    write_json(
        out / "freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(summary, flush=True)
    print(example_rows, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "existing", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    probe(args.data.resolve(), args.existing.resolve(), args.out.resolve())
