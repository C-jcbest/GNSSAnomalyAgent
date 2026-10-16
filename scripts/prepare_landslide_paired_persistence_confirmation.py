"""Register one new complete batch before generating observations or opening truth."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import write_csv
from prepare_landslide_codex_observation_review import prepare as prepare_views
from prepare_landslide_core_lockbox import MODEL_FILES, MODEL_SOURCE
from prepare_landslide_motion_probe import FAMILIES, controlled_case
from probe_landslide_endpoint_fit import BASE, ROOT, export
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import observation_features, save_json
from run_landslide_evidence_display_confirmation import numerical_predictions
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha
from gnss_sim.landslide import LandslideInput, LandslideRequest
from gnss_sim.landslide_diagnostics import derive_motion


def prepare(out):
    calibration = BASE / "paired-persistence-calibration-v1"
    development = BASE / "paired-persistence-development-v1"
    verify_bindings(calibration, "calibration-freeze.json")
    verify_bindings(development, "result-freeze.json")
    if read_json(development / "decision.json")["decision"] != "PROCEED_PAIRED_PERSISTENCE_CONFIRMATION":
        raise ValueError("Confirmation requires the complete registered development gate")
    out.mkdir(parents=True, exist_ok=False)
    ids = [f"case_{index:04d}" for index in range(193, 217)]
    protocol = {
        "registered_at": datetime.now(timezone.utc).isoformat(), "date": "2026-10-11",
        "case_ids": ids, "primary_method": "visual_numeric_pair",
        "generation": {"seed": 2026101121, "permutation_seed": 2026101122,
                       "stream_indices": list(range(192, 216)), "days": 730,
                       "families": list(FAMILIES), "records_per_family": 4},
        "sampling": "One complete unfiltered batch; no replacement or regeneration after results",
        "selection": read_json(calibration / "selection.json"),
        "calibration": calibration.relative_to(ROOT).as_posix(),
        "refitting": False, "independent_of_method_selection_records": True,
        "primary_reference": "Complete24 controlled process recovery only; missing observations remain unknown",
        "secondary_review_ids": ids[:12],
        "secondary_reference": "Current Codex raw visual/numeric review, activity frozen before auxiliary stages; prior design and historical labels exposed, no new specific truth or prediction access; not independent expert/field truth",
        "secondary_models": "Process-trained methods evaluated as target-transfer sensitivity; separately report frozen observable-target six controls. Do not call process heads calibrated on author-AI labels",
        "baselines": "Seven frozen neural arms, three same-augmentation XGB, nine original process numerical controls, direct/fixed/original bounded Agent two repeats and cached cost-matched variants. Original train24/validation12 unchanged, same generated distribution",
        "gate": "Same ten development conditions: nondecrease F1 vs augmentation/shuffled; main-plain>=.005; strict F1 or event gain vs augmentation; weakR>=plain/augmentation; visual F1>=numeric_pair or event gain with daily loss<=.01; positive net and >=2 benefited records vs plain and augmentation",
        "failure": "Do not change weights, seed, threshold, width, margin, plots or substitute best ablation. Keep all outcomes; no second batch to salvage this test",
        "statistics": "Descriptive complete-record paired bootstrap, not independent dates/repeats or confirmatory significance",
        "vlm": {"model": "qwen3.8-flash", "repeats": 2, "retries": 0,
                "maximum_requests": 336, "hard_timeout_seconds": 3600},
        "field_accuracy_verified": False,
    }
    export(out / "protocol.json", protocol)
    for name in MODEL_FILES:
        shutil.copyfile(MODEL_SOURCE / name, out / name)
    config = read_json(BASE / "core-lockbox-data-v1/runner-config.json")
    config.update(case_ids=ids, source_bundle=out.relative_to(ROOT).as_posix(),
                  confirmation_protocol=str(out / "protocol.json"),
                  run_name="paired-persistence-confirmation-v1",
                  scope="New paired-persistence controlled confirmation; all original workflows frozen")
    export(out / "runner-config.json", config)
    sources = list((ROOT / "src/gnss_sim").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    for source in sources:
        destination = out / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    bindings = {path.relative_to(ROOT).as_posix(): sha(path)
                for path in calibration.rglob("*") if path.is_file()}
    bindings.update({(MODEL_SOURCE / name).relative_to(ROOT).as_posix(): sha(MODEL_SOURCE / name)
                     for name in MODEL_FILES})
    export(out / "training-bindings.json", bindings)
    export(out / "pre-generation-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()})
    started = time.perf_counter()
    request = LandslideRequest(seed=protocol["generation"]["seed"], count=24, days=730)
    order = np.random.default_rng(protocol["generation"]["permutation_seed"]).permutation(
        protocol["generation"]["stream_indices"])
    records, labels, generation = {}, [], []
    pack = out / "raw-review-pack"
    pack.mkdir()
    for cid, index in zip(ids, order, strict=True):
        original, rows, truth = controlled_case(request, int(index))
        case = LandslideInput.model_validate(dict(original.model_dump(), case_id=cid))
        folder = out / "inputs" / cid
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        motions = {}
        for window in (31, 61, 91):
            motion = derive_motion(case, window)
            motions[window] = motion
            (folder / f"motion-{window}.json").write_text(motion.model_dump_json(indent=2), encoding="utf-8")
        records[cid] = case, observation_features(case, motions)
        labels.extend(dict(row, case_id=cid) for row in rows)
        generation.append({"case_id": cid, "stream_index": int(index), "truth": truth})
        if cid in protocol["secondary_review_ids"]:
            observation = pack / "observations" / cid
            observation.mkdir(parents=True)
            shutil.copyfile(folder / "input.json", observation / "input.json")
        print("Generated opaque observation", cid, flush=True)
    sealed = out / "mechanism-sealed"
    sealed.mkdir()
    write_csv(sealed / "reference-labels.csv", labels)
    export(sealed / "generation-ledger.json", generation)
    before = time.perf_counter()
    export(out / "numeric-predictions.json", numerical_predictions(out, records))
    export(out / "numeric-cost.json", {"nine_controls_seconds": time.perf_counter() - before,
        "scope": "Model loading/nine predictions, excludes initial feature extraction, no per-method timing"})
    export(pack / "manifest.json", {"case_ids": protocol["secondary_review_ids"],
        "source": "Only observations; no specific process labels or predictions",
        "files": {path.relative_to(pack).as_posix(): sha(path) for path in pack.rglob("*") if path.is_file()}})
    prepare_views(pack, out / "codex-review")
    verify_bindings(out, "pre-generation-freeze.json")
    export(out / "data-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "raw_review_ready", "new_records": 24,
        "new_api_calls": 0, "seconds": time.perf_counter() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out.resolve())
