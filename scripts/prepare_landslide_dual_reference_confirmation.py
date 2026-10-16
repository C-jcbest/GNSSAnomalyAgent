"""Freeze target-matched baselines and create unseen raw-first dual-reference records."""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fit_landslide_raw_persistence_baselines import fit as fit_raw_controls
from prepare_landslide_ai_reference_holdout import TRAIN, VALIDATION, write_csv
from prepare_landslide_codex_observation_review import prepare as prepare_views
from prepare_landslide_motion_probe import controlled_case
from run_landslide_comparison import ROOT, fit_baselines, load_labels, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide import LandslideInput, LandslideRequest
from gnss_sim.landslide_diagnostics import derive_motion

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
AI_TRAINING = BASE / "ai-observable-calibration-v3"
TEST_IDS = [f"case_{index:04d}" for index in range(145, 157)]


def verify_training(directory):
    for name, digest in read_json(directory / "training-freeze.json").items():
        if sha(directory / name) != digest:
            raise ValueError("Frozen training changed: " + name)


def train_process(out):
    """Fit the same grids on the same old observations with the process target."""
    verify_training(AI_TRAINING)
    old = BASE / "confirmation-v1"
    split = {"train": TRAIN, "validation": VALIDATION, "mechanism_test": []}
    out.mkdir(parents=True, exist_ok=False)
    save_json(
        out / "protocol.json",
        {
            "schema": "process-target-matched-calibration-v1",
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "reference": "Existing controlled latent process labels of0049-0060, not new test labels",
            "source_labels_sha256": sha(old / "reference-labels.csv"),
            "ai_training_freeze_sha256": sha(AI_TRAINING / "training-freeze.json"),
            "fairness": "Same12 input records, same6/6 split and established grids/objective; primary AI models remain frozen; every process numerical control and common stage head is refitted",
            "selection": "Validation dailyF1+eventF1; no new test inputs exist; no extra threshold search",
        },
    )
    with (old / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["case_id"] in TRAIN + VALIDATION]
    write_csv(out / "reference-labels.csv", rows)
    cases = {}
    for cid in TRAIN + VALIDATION:
        case = LandslideInput.model_validate_json(
            (old / "inputs" / cid / "input.json").read_bytes()
        )
        matched = LandslideInput.model_validate_json(
            (AI_TRAINING / "inputs" / cid / "input.json").read_bytes()
        )
        if case != matched:
            raise ValueError("The two target calibrations must use identical observations")
        cases[cid] = case
        folder = out / "inputs" / cid
        folder.mkdir(parents=True)
        for name in ("input.json", "motion-31.json", "motion-61.json", "motion-91.json"):
            shutil.copyfile(AI_TRAINING / "inputs" / cid / name, folder / name)
    labels = load_labels(out / "reference-labels.csv", cases)
    sources = [
        Path(__file__),
        ROOT / "scripts/run_landslide_comparison.py",
        ROOT / "scripts/fit_landslide_raw_persistence_baselines.py",
    ]
    (out / "source").mkdir()
    for source in sources:
        shutil.copyfile(source, out / "source" / source.name)
    save_json(
        out / "pre-fit-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    features = {cid: load_case_features(out, cid)[1] for cid in TRAIN + VALIDATION}
    primary = out / "primary-baselines"
    primary.mkdir()
    fit_baselines(features, labels, split, primary)
    shutil.copyfile(primary / "stage-xgboost.json", out / "stage-xgboost.json")
    save_json(
        out / "manifest.json",
        {
            "split": split,
            "status": "primary_models_frozen_no_test_inputs",
            "files": {
                str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()
            },
        },
    )
    fit_raw_controls(out, out / "raw-baselines")
    save_json(
        out / "training-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print("Frozen all six process-target numerical controls and the common stage head")


def generate(process_training, out):
    verify_training(AI_TRAINING)
    verify_training(process_training)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "unseen-dual-reference-confirmation-v1",
        "date": "2026-10-11",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "case_ids": TEST_IDS,
        "generation": {
            "seed": 2026101181,
            "permutation_seed": 2026101182,
            "stream_indices": list(range(144, 156)),
            "count": 12,
            "days": 730,
            "allocation": "Two of each existing six controlled families; IDs permuted; no resampling",
        },
        "primary_target": "Observed raw-displacement author-AI reference, activity frozen before auxiliary stage review and before predictions; unknowns not filled",
        "secondary_target": "All observed controlled latent-process labels; independently refitted target-matched numerical models and common stage head on identical old6/6 records",
        "reviewer": "Current Codex method author, user-authorized actual visual plus numeric review; historical methods and0133-0144 process labels exposed; this batch-specific truth/predictions unseen during review; not independent expert",
        "candidate": "Unchanged original raw21/42 three-block conflict placement, at most2 nonoverlapping180-day views;14-day admission,49-day revision scope; no stopped protection extensions",
        "comparison": "All original34 methods in both target protocols; two repeats, direct/fixed/no-feedback/numeric-feedback, sameK early/late and same-response scope/center; common stage head within each target",
        "registration": {
            "primary": "Placement: original main versus sameK early and late; both-repeat dailyF1 gain, nondecreasing prespecified weak recall, positive net correct definite decisions. Failure retained; no default policy changes",
            "secondary": "Repeat those descriptive comparisons under process-target calibration; no choosing a target by outcome",
            "not_claimed": "Universal numerical superiority, scope uniqueness, nativeVLM stage improvement, independent expert/field accuracy",
        },
        "evaluation": "Same activityP/R/F1,eventIoU.3/.5,boundary errors,internal and end-to-endASD,unknown coverage,cost; report all methods/repeats and per-record results",
        "cost": {
            "maximum_unique_remote_requests": 168,
            "retries": 0,
            "model": "qwen3.8-flash",
            "sharing": "Identical visual replies for both targets; no duplicated API calls for second stage head",
        },
        "training": {
            "primary": str(AI_TRAINING),
            "secondary": str(process_training),
            "primary_freeze_sha256": sha(AI_TRAINING / "training-freeze.json"),
            "secondary_freeze_sha256": sha(process_training / "training-freeze.json"),
        },
        "no_test_selection": "No prompt/policy/threshold/seed/repeat changes, no best-target selection or stopped-module rescue",
        "formal_reference_verified": False,
    }
    save_json(out / "protocol.json", protocol)
    files = [
        Path(__file__),
        ROOT / "scripts/prepare_landslide_motion_probe.py",
        ROOT / "scripts/run_landslide_ai_reference_holdout.py",
        ROOT / "scripts/run_landslide_adaptive_review.py",
        ROOT / "scripts/probe_landslide_revision_scope.py",
        ROOT / "src/gnss_sim/landslide_adaptive_review.py",
        ROOT / "src/gnss_sim/landslide_evaluation.py",
        ROOT / "src/gnss_sim/landslide_visual.py",
    ]
    save_json(
        out / "source-bindings.json", {str(path.relative_to(ROOT)): sha(path) for path in files}
    )
    save_json(
        out / "pre-generation-freeze.json",
        {path.name: sha(path) for path in out.iterdir() if path.is_file()},
    )
    pack = out / "raw-review-pack"
    pack.mkdir()
    request = LandslideRequest(seed=protocol["generation"]["seed"], count=12, days=730)
    order = np.random.default_rng(protocol["generation"]["permutation_seed"]).permutation(
        protocol["generation"]["stream_indices"]
    )
    sealed_rows, sealed_metadata = [], []
    for cid, stream_index in zip(TEST_IDS, order, strict=True):
        source_case, labels, truth = controlled_case(request, int(stream_index))
        case = LandslideInput.model_validate(dict(source_case.model_dump(), case_id=cid))
        folder = out / "inputs" / cid
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        observation = pack / "observations" / cid
        observation.mkdir(parents=True)
        shutil.copyfile(folder / "input.json", observation / "input.json")
        for window in (31, 61, 91):
            motion = derive_motion(case, window)
            (folder / f"motion-{window}.json").write_text(
                motion.model_dump_json(indent=2), encoding="utf-8"
            )
        sealed_rows.extend(dict(row, case_id=cid) for row in labels)
        sealed_metadata.append({"case_id": cid, "stream_index": int(stream_index), "truth": truth})
    sealed = out / "mechanism-sealed"
    sealed.mkdir()
    write_csv(sealed / "reference-labels.csv", sealed_rows)
    save_json(sealed / "generation-ledger.json", sealed_metadata)
    save_json(
        pack / "manifest.json",
        {
            "case_ids": TEST_IDS,
            "source": "Observations only, no truth/predictions",
            "files": {
                str(path.relative_to(pack)): sha(path) for path in pack.rglob("*") if path.is_file()
            },
        },
    )
    prepare_views(pack, out / "codex-review")
    save_json(
        out / "data-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    save_json(
        out / "state.json", {"status": "raw_review_ready_no_predictions", "new_remote_requests": 0}
    )
    print(
        "Prepared twelve new records after both target calibrations froze; sealed truth not read for review"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train-process", "generate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--process-training", type=Path)
    args = parser.parse_args()
    if args.mode == "train-process":
        train_process(args.out.resolve())
    else:
        if args.process_training is None:
            parser.error("generate requires --process-training")
        generate(args.process_training.resolve(), args.out.resolve())
