"""Freeze the retained core method and make one complete, unseen confirmation batch."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import write_csv
from prepare_landslide_codex_observation_review import prepare as prepare_views
from prepare_landslide_motion_probe import FAMILIES, controlled_case
from probe_landslide_revision_scope import PROTOCOL as SCOPE_PROTOCOL
from run_landslide_adaptive_review import prepare as prepare_run
from run_landslide_comparison import observation_features
from run_landslide_evidence_display_confirmation import numerical_predictions
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide import LandslideInput, LandslideRequest
from gnss_sim.landslide_diagnostics import derive_motion

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
MODEL_SOURCE = BASE / "evidence-display-confirmation-v2"
MODEL_FILES = (
    "stage-xgboost.json",
    "activity-xgboost.json",
    "original-selection.json",
    "raw-selection.json",
    "activity-xgboost-raw-persistence.json",
    "long-selection.json",
    "activity-xgboost-84.json",
    "matched-selection.json",
    "activity-xgboost-42-matched-support.json",
)


def prepare(out):
    ids = [f"case_{index:04d}" for index in range(169, 193)]
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "date": "2026-10-11",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "case_ids": ids,
        "primary_method": "adaptive_no_feedback__conflict_scope",
        "method": SCOPE_PROTOCOL["policy"],
        "revision_neighbourhood_days": 49,
        "generation": {
            "seed": 2026101113,
            "permutation_seed": 2026101114,
            "stream_indices": list(range(168, 192)),
            "days": 730,
            "count": 24,
            "families": list(FAMILIES),
            "records_per_family": 4,
            "sampling": "One complete batch, no replacement, screening, duration or noise selection",
        },
        "question": "Evaluate the retained original core method, its matched-cost placement mechanism and recognition limits on a final unseen random realization. Do not retest or promote stopped extensions.",
        "primary_reference": "Exact independently generated controlled velocity/activity and ASD; missing raw dates unscored. This proves designed-motion recovery only, not observable field accuracy.",
        "secondary_review_ids": ids[:12],
        "secondary_reference": "First twelve opaque IDs fixed before generation; current Codex raw visual and numerical review frozen before model inference or process-reference access. Developer knows historical results, not an independent expert. Report separately and never substitute for the full primary population.",
        "baselines": "Nine original process-target numerical controls from train24/validation12, identical distribution and original calibration opportunities; no new fitting/tuning. Direct VLM, fixed two-window review, matched K early/late, whole-target/49-day/size-matched-center authority, and numerical-text review share original responses. Secondary author-AI subset will use its own frozen6/6 target-matched models.",
        "gate": "In both repeats original core activityF1 exceeds direct, full fixed whole-target and matched-K early conflict-scope; positive net definite correction against each, with at least two benefited records against direct. Mean weak recall across both repeats not below matched-K early. Main deployment calls below full fixed. Report all numerical/event/boundary/stage differences, both repeats and every family; not required to beat every numeric method.",
        "interpretation": "This one trial adds independent records for the unchanged retained core. It cannot erase any old primary failure, re-establish the old rejected strict scope claim, or rescue sequential/partition/fusion/other stopped modules. Descriptive record bootstrap only; correlated days/repeats are not independent samples.",
        "failure": "No seed, threshold, prompt, repetition, subset, best-arm replacement or new confirmation batch to salvage this trial. Failure limits the claimed core contribution; retain all output and unknowns.",
        "inference": {
            "model": "qwen3.8-flash",
            "repeats": 2,
            "retries": 0,
            "maximum_unique_requests": 336,
            "hard_timeout_seconds": 3600,
        },
        "formal_field_accuracy_verified": False,
        "default_changed": False,
    }
    write_json(out / "protocol.json", protocol)
    bindings = {}
    for name in MODEL_FILES:
        path = MODEL_SOURCE / name
        shutil.copyfile(path, out / name)
        bindings[str(path)] = sha(path)
    write_json(out / "training-bindings.json", bindings)
    # Reuse the project's existing frozen-source convention, without a new harness.
    sources = list((ROOT / "src/gnss_sim").glob("*.py"))
    sources += list((ROOT / "scripts").glob("*.py"))
    for path in sources:
        destination = out / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    config = read_json(BASE / "budget-selection-confirmation-v1/protocol.json")
    config.update(
        date=protocol["date"],
        case_ids=ids,
        source_bundle=out.relative_to(ROOT).as_posix(),
        maximum_requests=336,
        confirmation_protocol=str(out / "protocol.json"),
        run_name="core-lockbox-confirmation-v1",
        scope="One registered24-record controlled recovery confirmation; retained core only, no rescue of stopped modules",
    )
    write_json(out / "runner-config.json", config)
    write_json(
        out / "pre-generation-freeze.json",
        {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    started = time.perf_counter()
    request = LandslideRequest(seed=protocol["generation"]["seed"], count=24, days=730)
    order = np.random.default_rng(protocol["generation"]["permutation_seed"]).permutation(
        protocol["generation"]["stream_indices"]
    )
    records, rows, generation = {}, [], []
    pack = out / "raw-review-pack"
    pack.mkdir()
    for cid, index in zip(ids, order, strict=True):
        original, labels, truth = controlled_case(request, int(index))
        case = LandslideInput.model_validate(dict(original.model_dump(), case_id=cid))
        folder = out / "inputs" / cid
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        motions = {}
        for window in (31, 61, 91):
            motion = derive_motion(case, window)
            motions[window] = motion
            (folder / f"motion-{window}.json").write_text(
                motion.model_dump_json(indent=2), encoding="utf-8"
            )
        records[cid] = (case, observation_features(case, motions))
        rows.extend(dict(row, case_id=cid) for row in labels)
        generation.append({"case_id": cid, "stream_index": int(index), "truth": truth})
        if cid in protocol["secondary_review_ids"]:
            observation = pack / "observations" / cid
            observation.mkdir(parents=True)
            shutil.copyfile(folder / "input.json", observation / "input.json")
        print(f"Generated registered observation {cid}", flush=True)
    sealed = out / "mechanism-sealed"
    sealed.mkdir()
    write_csv(sealed / "reference-labels.csv", rows)
    write_json(sealed / "generation-ledger.json", generation)
    prediction_started = time.perf_counter()
    write_json(out / "numeric-predictions.json", numerical_predictions(out, records))
    write_json(
        out / "numeric-cost.json",
        {
            "nine_controls_seconds": time.perf_counter() - prediction_started,
            "scope": "Model loading and nine predictions plus long raw features; excludes initial observation-feature construction and historical fitting; not per-method CPU timing",
        },
    )
    write_json(
        pack / "manifest.json",
        {
            "case_ids": protocol["secondary_review_ids"],
            "source": "Fixed observation-only subset; no predictions or generation metadata",
            "files": {
                p.relative_to(pack).as_posix(): sha(p) for p in pack.rglob("*") if p.is_file()
            },
        },
    )
    prepare_views(pack, out / "codex-review")
    for name, expected in read_json(out / "pre-generation-freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("A pre-generation algorithm, model or protocol changed")
    write_json(
        out / "data-freeze.json",
        {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    write_json(
        out / "state.json",
        {
            "status": "raw_review_ready_no_vlm_predictions",
            "new_records": 24,
            "new_remote_requests": 0,
            "seconds": time.perf_counter() - started,
            "process_reference_opened_by_reviewer": False,
        },
    )
    print(
        "24 registered records ready; complete fixed12-record developer reference before inference",
        flush=True,
    )


def prepare_inference(data, run):
    from run_landslide_ai_reference_holdout import verify_bindings

    verify_bindings(data, "data-freeze.json")
    reference = data / "codex-review/compiled/reference-freeze.json"
    if not reference.exists():
        raise ValueError("Freeze the registered developer subset before inference")
    for filename, expected in read_json(reference)["files"].items():
        if sha(reference.parent / filename) != expected:
            raise ValueError("Frozen developer reference changed: " + filename)
    # The old executor only copies labels during prepare; it never reads them in run.
    # Keep the exact saved process reference for that existing interface.
    shutil.copyfile(data / "mechanism-sealed/reference-labels.csv", data / "reference-labels.csv")
    prepare_run(data / "runner-config.json", run)
    write_json(
        run / "developer-reference-binding.json", {"path": str(reference), "sha256": sha(reference)}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "prepare-inference"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out.resolve())
    else:
        if args.run is None:
            parser.error("prepare-inference requires --run")
        prepare_inference(args.out.resolve(), args.run.resolve())
