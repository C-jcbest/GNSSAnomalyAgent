"""Reproduce the training-only strict-counterevidence coverage preflight."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from probe_landslide_contrastive_exemplars import SOURCE
from probe_landslide_endpoint_fit import export
from probe_landslide_visual_invariance import calibration_labels
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide_counterevidence_review import raw_counterevidence


def probe(out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    split = read_json(SOURCE / "protocol.json")["split"]
    pairs = {cid: load_case_features(SOURCE, cid) for cid in split["train"]}
    labels = calibration_labels(SOURCE, pairs)
    started = time.perf_counter()
    rows = []
    for cid, (case, features) in pairs.items():
        actual = np.array([int(row["activity_label"]) for row in labels[cid]])
        raw = raw_counterevidence(case, features.noise)
        active_conflict = (actual == 1) & raw["plateau"]
        inactive_conflict = (actual == 0) & raw["motion"]
        conflict = active_conflict | inactive_conflict
        candidates = []
        for day in range(98, len(actual) - 98, 14):
            start, stop = day - 49, day + 49
            known = (actual[start:stop] >= 0) & features.observed[start:stop]
            candidates.append((int(conflict[start:stop].sum()), day, int(known.sum())))
        count, day, known = min(candidates, key=lambda value: (-value[0], value[1]))
        rows.append({"case_id": cid, "day": day, "query": [day - 49, day + 49],
                     "conflict_days": count, "known_days": known,
                     "all_active_plateau_conflicts": int(active_conflict.sum()),
                     "all_inactive_motion_conflicts": int(inactive_conflict.sum())})
    export(out / "train-only-selection.json", rows)
    export(out / "summary.json", {
        "source": str(SOURCE), "new_api_calls": 0, "validation_rows_used_for_selection": False,
        "reference_csv_contains_other_splits": True, "seconds": time.perf_counter() - started,
        "selected_conflicts": sum(row["conflict_days"] for row in rows),
    })
    print("Training-only conflict days:", sum(row["conflict_days"] for row in rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    probe(args.out.resolve())
