"""Register the unchanged censored-endpoint mechanism before fresh data exist."""

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import write_csv
from prepare_landslide_codex_observation_review import prepare as prepare_views
from prepare_landslide_motion_probe import controlled_case
from run_landslide_ai_reference_holdout import verify_bindings

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide import LandslideInput, LandslideRequest
from gnss_sim.landslide_diagnostics import derive_motion

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"


def generate(out):
    primary = BASE / "ai-observable-calibration-v3"
    secondary = BASE / "dual-reference-process-calibration-v1"
    for training in (primary, secondary):
        verify_bindings(training, "training-freeze.json")
    out.mkdir(parents=True, exist_ok=False)
    ids = [f"case_{index:04d}" for index in range(157, 169)]
    protocol = {
        "schema": "fresh-sequential-censored-confirmation-v1",
        "date": "2026-10-11", "frozen_at": datetime.now(timezone.utc).isoformat(), "case_ids": ids,
        "generation": {"seed": 2026101191, "permutation_seed": 2026101192,
            "stream_indices": list(range(156, 168)), "days": 730, "count": 12,
            "allocation": "Two per existing six controlled families; permuted IDs; no resampling"},
        "primary_target": "Author-AI raw-observable reference frozen before predictions; unknown retained",
        "secondary_target": "Observed latent process, opened only after all predictions frozen; secondary cannot rescue primary",
        "reviewer": "Current Codex method author, user-authorized visual+numerical review; historical methods and0133-0156 process labels exposed; new specific labels/predictions unseen; not independent expert",
        "candidate": "Original0/1/2 budget; first highest-conflict180-day original selection; right-edge activity and>=14 original conflicts in following49 days select centered98-day second view; else original other target. Existing21/42 evidence,49 revision authority,strict-majority overlap,images,prompts and model unchanged.",
        "comparison": "All34 original methods+5 follow-up arms*2 repeats+first-only*2,46 methods on each of2 targets. Pure VLM,fixed,original,same-count location/scope,numerical text feedback,strong rules/PELT/XGBoost retained.",
        "registration": {
            "primary": "Across these12 AI records, average over both repeats of dailyF1 differences must be>0 against original conflict-scope,matched-trigger fixed98 and first-only; net definite correction summed over both repeats>0 against each; mean weak recall must not decrease versus original. All individual outcomes retained; no default automatically changed.",
            "first_only_reason": "Exposed development averageF1 was-.00521 versus first-only; cheaper control is mandatory, not posthoc removable.",
            "secondary": "Same descriptive gate with process-target matched models; not an alternative primary",
            "failure": "Report failure/zero triggers/no gain without threshold,seed,prompt,repeat or arm rescue; no more confirmation batches to salvage this registered trial",
        },
        "evaluation": "Same activityP/R/F1,eventIoU.3/.5,boundaryMAE,weakRecall,coverage/unknown,internal and end-to-endASD,full per-record comparisons,cost. Activity before stages; no missing decisions.",
        "cost": {"model": "qwen3.8-flash", "repetitions": 2, "retries": 0,
            "maximum_unique_remote_requests": 288, "hard_timeout_seconds": 3600,
            "sharing": "Fresh same-run questions byte-identical per record/repeat shared across arms and targets; first answer precedes followup selection; no historical inference cache"},
        "training": {"primary": str(primary), "secondary": str(secondary),
            "primary_freeze_sha256": sha(primary / "training-freeze.json"),
            "secondary_freeze_sha256": sha(secondary / "training-freeze.json"),
            "fairness": "Unchanged controlled distribution and two target definitions; same original6/6 inputs and splits,each model set already target-matched and froze before generation; no test retraining or new tuning"},
        "sequential_executor_sha256": sha(ROOT / "scripts/run_landslide_sequential_followup.py"),
        "selector_sha256": sha(ROOT / "src/gnss_sim/landslide_censored_followup.py"),
        "scorer_sha256": sha(ROOT / "scripts/analyze_landslide_sequential_followup.py"),
        "formal_reference_verified": False, "default_changed": False,
    }
    write_json(out / "protocol.json", protocol)
    sources = [Path(__file__), ROOT / "scripts/run_landslide_sequential_followup.py",
        ROOT / "scripts/analyze_landslide_sequential_followup.py",
        ROOT / "scripts/prepare_landslide_motion_probe.py",
        ROOT / "scripts/run_landslide_ai_reference_holdout.py",
        ROOT / "scripts/run_landslide_adaptive_review.py",
        ROOT / "scripts/probe_landslide_revision_scope.py",
        ROOT / "scripts/analyze_landslide_censored_followup_v2.py",
        ROOT / "src/gnss_sim/landslide_censored_followup.py",
        ROOT / "src/gnss_sim/landslide_adaptive_review.py",
        ROOT / "src/gnss_sim/landslide_visual.py", ROOT / "src/gnss_sim/landslide_evaluation.py"]
    (out / "source").mkdir()
    for source in sources:
        shutil.copyfile(source, out / "source" / source.name)
    write_json(out / "source-bindings.json", {str(path.relative_to(ROOT)): sha(path) for path in sources})
    write_json(out / "pre-generation-freeze.json", {str(path.relative_to(out)): sha(path)
        for path in out.rglob("*") if path.is_file()})
    pack = out / "raw-review-pack"
    pack.mkdir()
    request = LandslideRequest(seed=protocol["generation"]["seed"], count=12, days=730)
    order = np.random.default_rng(protocol["generation"]["permutation_seed"]).permutation(
        protocol["generation"]["stream_indices"])
    rows, metadata = [], []
    for cid, index in zip(ids, order, strict=True):
        source, labels, truth = controlled_case(request, int(index))
        case = LandslideInput.model_validate(dict(source.model_dump(), case_id=cid))
        folder = out / "inputs" / cid
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        observation = pack / "observations" / cid
        observation.mkdir(parents=True)
        shutil.copyfile(folder / "input.json", observation / "input.json")
        for window in (31, 61, 91):
            (folder / f"motion-{window}.json").write_text(derive_motion(case, window).model_dump_json(indent=2),
                                                         encoding="utf-8")
        rows.extend(dict(row, case_id=cid) for row in labels)
        metadata.append({"case_id": cid, "stream_index": int(index), "truth": truth})
    sealed = out / "mechanism-sealed"
    sealed.mkdir()
    write_csv(sealed / "reference-labels.csv", rows)
    write_json(sealed / "generation-ledger.json", metadata)
    write_json(pack / "manifest.json", {"case_ids": ids, "source": "Observations only,no truth/predictions",
        "files": {str(path.relative_to(pack)): sha(path) for path in pack.rglob("*") if path.is_file()}})
    prepare_views(pack, out / "codex-review")
    write_json(out / "data-freeze.json", {str(path.relative_to(out)): sha(path)
        for path in out.rglob("*") if path.is_file()})
    write_json(out / "state.json", {"status": "raw_review_ready_no_predictions", "new_remote_requests": 0,
        "new_records": 12, "truth_opened_by_reviewer": False})
    print("Generated12 registered raw records; review before inference; sealed truth unread", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    generate(parser.parse_args().out.resolve())
