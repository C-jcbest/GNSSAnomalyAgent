"""Save post-score frame, raw-observation and auxiliary-signal diagnostics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from run_landslide_window_optimization import load_case_features, read_json
from train_landslide_shared_frame import MAIN

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_paired_persistence import render_encoder_chart
from gnss_sim.landslide_visual_invariance import raw_chart_patches


def intervals(values):
    selected = np.asarray(values) == 1
    edges = np.diff(np.r_[False, selected, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def analyze(data, pairs, parent_pairs, run, report, out):
    for folder, manifest in (
        (pairs, "encoding-freeze.json"),
        (parent_pairs, "encoding-freeze.json"),
        (run, "freeze.json"),
        (report, "freeze.json"),
    ):
        verify_manifest(folder, manifest)
    out.mkdir(parents=True, exist_ok=False)
    selected = read_json(pairs / "selected-training-pairs.json")
    plan = read_json(report / "analysis-plan.json")
    predictions = read_json(run / "predictions.json")
    probabilities = read_json(run / "probabilities.json")
    augmentation = "shared_visual_numeric_augmentation"
    independent = "independent_visual_numeric_pair"
    raw_rows = []
    for cid in plan["review_ids"]:
        case = load_case_features(data, cid)[0]
        values = np.array([v if v is not None else [np.nan] * 3 for v in case.displacement_mm])
        limits = (126, 308) if cid == "case_0385" else (518, 630)
        blocks = []
        for first in range(limits[0], limits[1], 14):
            last = min(first + 14, limits[1])
            observed = np.isfinite(values[first:last]).all(axis=1)
            median = (
                np.median(values[first:last][observed], axis=0) if observed.any() else [np.nan] * 3
            )
            blocks.append(
                dict(
                    first=first,
                    last=last,
                    observed=int(observed.sum()),
                    median_neu=[float(v) if np.isfinite(v) else None for v in median],
                )
            )
        raw_rows.append(
            dict(
                case_id=cid,
                blocks=blocks,
                predictions={
                    name: intervals(predictions[name][cid]["activity"])
                    for name in (MAIN, augmentation, independent, "visual_numeric", "xgb_raw84")
                },
            )
        )
    write_json(out / "raw-observation-blocks.json", raw_rows)
    # These same TRAIN examples were selected before this model experiment,
    # by outer-pixel diagnostic magnitude rather than validation performance.
    diagnostic_plan = read_json(
        data.parent / "local-persistence-context-probe-v1/example-plan.json"
    )
    for pair_id in diagnostic_plan["pair_ids"]:
        row = next(item for item in selected if item["pair_id"] == pair_id)
        case, features = load_case_features(data, row["case_id"])
        original = raw_chart_patches(case, features.noise)[row["day"]]
        with np.load(parent_pairs / "pairs" / (pair_id + ".npz")) as archive:
            independent_negative = archive["local_image"]
        with np.load(pairs / "pairs" / (pair_id + ".npz")) as archive:
            shared_positive = archive["positive_image"]
            shared_negative = archive["negative_image"]
        figure, panels = plt.subplots(1, 4, figsize=(13, 4))
        for panel, patch, title in zip(
            panels,
            (original, independent_negative, shared_positive, shared_negative),
            ("Independent positive", "Independent negative", "Shared positive", "Shared negative"),
        ):
            panel.imshow(render_encoder_chart(patch))
            panel.set_title(title)
            panel.axis("off")
        figure.suptitle(pair_id + " | same physical local replacement; shared outer pixels equal")
        figure.tight_layout()
        figure.savefig(out / (pair_id + "-frames.png"), dpi=130)
        plt.close(figure)
    with np.load(pairs / "pair-bank.npz") as archive:
        bank = {key: archive[key] for key in archive.files}
    distances = {}
    for frame in ("independent", "shared"):
        left = bank[frame + "_positive_embeddings"]
        right = bank[frame + "_negative_embeddings"]
        distances[frame] = dict(
            mean_l2=float(np.linalg.norm(left - right, axis=1).mean()),
            median_l2=float(np.median(np.linalg.norm(left - right, axis=1))),
        )
    changed = sum(
        np.count_nonzero(
            np.array(predictions[MAIN][cid]["activity"])
            != np.array(predictions[independent][cid]["activity"])
        )
        for cid in plan.get("validation", read_json(run / "protocol.json")["split"]["validation"])
    )
    score_difference = np.concatenate(
        [
            np.array(probabilities[MAIN][cid]) - np.array(probabilities[independent][cid])
            for cid in probabilities[MAIN]
        ]
    )
    write_json(
        out / "diagnostic.json",
        dict(
            frame_pair_embedding_distances=distances,
            main_vs_independent_changed_days=int(changed),
            mean_absolute_probability_difference=float(np.abs(score_difference).mean()),
            max_absolute_probability_difference=float(np.abs(score_difference).max()),
            interpretation="Post-score description, not evidence that global normalization caused the old failure or that one pair frame helps recognition",
            test_truth_opened=False,
            created_unix=time.time(),
            evidence_bindings={
                str(p.resolve()): sha(p)
                for p in (
                    pairs / "encoding-freeze.json",
                    parent_pairs / "encoding-freeze.json",
                    run / "freeze.json",
                    report / "freeze.json",
                )
            },
        ),
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(read_json(out / "diagnostic.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "pairs", "parent-pairs", "run", "report", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    analyze(
        args.data.resolve(),
        args.pairs.resolve(),
        args.parent_pairs.resolve(),
        args.run.resolve(),
        args.report.resolve(),
        args.out.resolve(),
    )
