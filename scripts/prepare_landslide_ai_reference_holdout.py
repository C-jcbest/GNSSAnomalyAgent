"""Calibrate existing baselines on AI-observable references, then seal new inputs.

This is a new-data evaluation with a disclosed same-session AI reviewer, not an
independent human reference or field benchmark. No held-out predictions or
mechanism labels are supplied to the raw-first review packet.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from analyze_landslide_codex_observation_reference import scoring_inputs
from fit_landslide_raw_persistence_baselines import fit as fit_raw_controls
from prepare_landslide_codex_observation_review import prepare as prepare_views
from prepare_landslide_motion_probe import controlled_case
from run_landslide_comparison import ROOT, fit_baselines, save_json, sha
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.landslide import LandslideInput, LandslideRequest
from gnss_sim.landslide_diagnostics import derive_motion

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
TRAIN = ["case_0049", "case_0051", "case_0053", "case_0056", "case_0058", "case_0060"]
VALIDATION = ["case_0050", "case_0052", "case_0054", "case_0055", "case_0057", "case_0059"]
TEST_IDS = [f"case_{index:04d}" for index in range(133, 145)]


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_manifest(directory, filename="manifest.json"):
    for name, expected in read_json(directory / filename)["files"].items():
        if sha(directory / name) != expected:
            raise ValueError("Frozen artifact changed: " + name)


def train(out):
    reference = BASE / "codex-observation-review-v1"
    verify_manifest(reference, "delivery-manifest.json")
    out.mkdir(parents=True, exist_ok=False)
    split = {"train": TRAIN, "validation": VALIDATION, "mechanism_test": []}
    protocol = {
        "schema": "ai-observable-reference-calibration-v1", "date": "2026-10-10", "split": split,
        "reference": "Frozen same-session raw-first AI reference, 0049-0060; developer-exposed labels",
        "reference_manifest_sha256": sha(reference / "delivery-manifest.json"),
        "purpose": "Fairly adapt all numerical activity/stage baselines to the SAME observation definition before new holdout inputs exist",
        "split_basis": "Paired original records separated across train/validation; existing AI labels have all A/S/D in BOTH sets; this is calibration design, not blinded training data",
        "selection": "Exact existing robust/PELT/XGBoost grids and validation dailyF1+eventF1; existing stage-rule grid; common120-tree depth2 stage head; raw21/42 rule and persistence XGBoost use exact original grids",
        "unknown": "Exclude AI-reference unknown from fitting and validation metric denominators; keep prediction unknown as FN and without TN credit",
        "visual_policy": "Original raw21/42 .03mm/day and noise ratio2,14 conflict days,two180-day views,49-day permission; no re-selection, no failed extra modules",
        "visual_stages": "Same newly fitted stage head for direct/fixed/adaptive visual activity to isolate activity mechanism",
        "scope": "Small six-record training/six-record validation on AI labels; no independent expert or field claim",
        "preflight_correction": "v1 stopped before fitting because compiled reference omits episode IDs; reuse established conservative-core scoring inputs. v2 primary fit completed but raw fitter reads UTF8 headers without BOM; v3 exports UTF8 consistent with that existing contract. Labels/split/grids unchanged; failed versions preserved",
        "new_remote_model_requests": 0,
    }
    save_json(out / "protocol.json", protocol)
    for source in (Path(__file__), ROOT / "scripts/run_landslide_comparison.py",
                   ROOT / "scripts/fit_landslide_raw_persistence_baselines.py",
                   ROOT / "src/gnss_sim/landslide_baselines.py", ROOT / "src/gnss_sim/landslide_evaluation.py"):
        target = out / "source" / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
    labels, _, _ = scoring_inputs(reference)
    write_csv(out / "reference-labels.csv", [row for case_id in TRAIN + VALIDATION for row in labels[case_id]])
    for case_id in TRAIN + VALIDATION:
        folder = out / "inputs" / case_id
        folder.mkdir(parents=True)
        for name in ("input.json", "motion-31.json", "motion-61.json", "motion-91.json"):
            shutil.copyfile(reference / case_id / name, folder / name)
    save_json(out / "pre-fit-freeze.json", {str(path.relative_to(out)): sha(path)
                                           for path in out.rglob("*") if path.is_file()})
    features = {case_id: load_case_features(out, case_id)[1] for case_id in TRAIN + VALIDATION}
    stage_codes = {"acceleration": 1, "steady_motion": 2, "deceleration": 3}
    support = {}
    for name, case_ids in split.items():
        if not case_ids:
            continue
        counts = {code: 0 for code in (1, 2, 3)}
        for case_id in case_ids:
            for day, row in enumerate(labels[case_id]):
                code = stage_codes.get(row["feature"])
                if code is not None and features[case_id].supported[day]:
                    counts[code] += 1
        if not all(counts.values()):
            raise ValueError("Both calibration splits must have supported A/S/D references")
        support[name] = counts
    save_json(out / "eligible-stage-inventory.json", support)
    primary = out / "primary-baselines"
    primary.mkdir()
    choices, _, _ = fit_baselines(features, labels, split, primary)
    shutil.copyfile(primary / "stage-xgboost.json", out / "stage-xgboost.json")
    save_json(out / "manifest.json", {"split": split, "status": "primary_models_fitted_no_test_inputs",
                                      "files": {str(path.relative_to(out)): sha(path) for path in out.rglob("*")
                                                if path.is_file() and path.name != "manifest.json"}})
    fit_raw_controls(out, out / "raw-baselines")
    save_json(out / "training-freeze.json", {str(path.relative_to(out)): sha(path)
                                             for path in out.rglob("*") if path.is_file()})
    print({"status": "all_six_activity_controls_and_common_stage_head_frozen",
           "train": TRAIN, "validation": VALIDATION, "supported_stages": support,
           "primary_selection": choices})


def generate(training, out):
    for name, expected in read_json(training / "training-freeze.json").items():
        if sha(training / name) != expected:
            raise ValueError("AI-calibrated model changed before holdout generation")
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": "ai-observable-new-record-holdout-v1", "date": "2026-10-10", "case_ids": TEST_IDS,
        "generation": {"seed": 2026101081, "stream_indices": list(range(132, 144)), "days": 730,
                       "count": 12, "family_allocation": "Existing six families, two each, IDs permuted in sealed generation ledger"},
        "candidate": "Original finite raw21/42 conflict observation,49-day revision permission; no conditional numeric extension",
        "training_freeze_sha256": sha(training / "training-freeze.json"),
        "reference_order": "Current Codex reads raw global/local and raw30-day descriptive blocks, freezes activity, then inspects31/61/91 auxiliary and freezes stages. NO test predictions or mechanism labels before reference freeze",
        "reviewer": "Same-session AI method author; familiar with generic simulation design and historical methods, not an independent expert; new-record predictions/truth unseen during reference construction",
        "independence": "New independent RNG records never used in method or parameter choice; reference reviewer is NOT independent; controlled distribution, not field transfer",
        "comparison": "Two repeats of direct/fixed/adaptive-image/adaptive-numeric; same-count fixed placement and same-response scope/center controls; six newly calibrated numerical activity controls; common stage head",
        "evaluation": "Same known-observation activityP/R/F1,eventIoU.3/.5,conservative-core boundaries,activity/internalASD,unknown exposure and deployment cost; no reference filling",
        "no_test_selection": "Report complete two repetitions regardless of sign; no threshold/prompt/model/split changes, no next-seed rescue, no replacement best arm",
        "cost": {"maximum_unique_remote_requests": 168, "retries": 0, "model": "qwen3.8-flash"},
        "formal_reference_verified": False,
    }
    save_json(out / "protocol.json", protocol)
    bindings = {str(path.relative_to(ROOT)): sha(path) for path in (
        Path(__file__), ROOT / "scripts/prepare_landslide_motion_probe.py",
        ROOT / "scripts/run_landslide_adaptive_review.py", ROOT / "scripts/probe_landslide_revision_scope.py",
        ROOT / "src/gnss_sim/landslide.py", ROOT / "src/gnss_sim/landslide_adaptive_review.py",
        ROOT / "src/gnss_sim/landslide_evaluation.py", ROOT / "src/gnss_sim/landslide_visual.py",
    )}
    save_json(out / "source-bindings.json", bindings)
    save_json(out / "pre-generation-freeze.json", {path.name: sha(path) for path in out.iterdir() if path.is_file()})
    pack = out / "raw-review-pack"
    pack.mkdir()
    request = LandslideRequest(seed=protocol["generation"]["seed"], count=12, days=730)
    order = np.random.default_rng(2026101082).permutation(protocol["generation"]["stream_indices"])
    sealed_rows, sealed_metadata = [], []
    for case_id, stream_index in zip(TEST_IDS, order, strict=True):
        source_case, labels, truth = controlled_case(request, int(stream_index))
        values = source_case.model_dump()
        values["case_id"] = case_id
        case = LandslideInput.model_validate(values)
        folder = out / "inputs" / case_id
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        observation = pack / "observations" / case_id
        observation.mkdir(parents=True)
        shutil.copyfile(folder / "input.json", observation / "input.json")
        for window in (31, 61, 91):
            motion = derive_motion(case, window)
            (folder / f"motion-{window}.json").write_text(motion.model_dump_json(indent=2), encoding="utf-8")
        sealed_rows.extend(dict(row, case_id=case_id) for row in labels)
        sealed_metadata.append({"case_id": case_id, "stream_index": int(stream_index), "truth": truth})
    sealed = out / "mechanism-sealed"
    sealed.mkdir()
    write_csv(sealed / "reference-labels.csv", sealed_rows)
    save_json(sealed / "generation-ledger.json", sealed_metadata)
    save_json(pack / "manifest.json", {"case_ids": TEST_IDS, "source": "New observations only; no predictions/truth",
                                       "files": {str(path.relative_to(pack)): sha(path) for path in pack.rglob("*") if path.is_file()}})
    prepare_views(pack, out / "codex-review")
    save_json(out / "data-freeze.json", {str(path.relative_to(out)): sha(path)
                                         for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "raw_review_ready_no_predictions", "case_ids": TEST_IDS,
                                    "new_remote_model_requests": 0})
    print("Prepared12 new raw-only records; no model predictions; sealed process labels not read for review")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "generate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.out.resolve())
    else:
        if args.training is None:
            parser.error("generate requires --training")
        generate(args.training.resolve(), args.out.resolve())
