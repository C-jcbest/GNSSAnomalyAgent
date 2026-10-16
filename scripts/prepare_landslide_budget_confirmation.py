"""Register the existing selector and matched-count controls before new data exist."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from prepare_landslide_motion_probe import FAMILIES, controlled_case
from probe_landslide_revision_scope import PROTOCOL as SCOPE_PROTOCOL
from run_landslide_comparison import ROOT, observation_features, save_json, sha
from run_landslide_evidence_display_confirmation import numerical_predictions
from run_landslide_window_optimization import read_json

from gnss_sim.landslide import LandslideRequest
from gnss_sim.landslide_diagnostics import derive_motion

BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"


def prepare(out):
    indices = list(range(120, 132))
    ids = [f"case_{index + 1:04d}" for index in indices]
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "date": "2026-10-10", "schema": "budget-matched-selection-confirmation-v1", "case_ids": ids,
        "primary": "Unchanged adaptive_no_feedback + conflict_scope",
        "generation": {"seed": 2026101075, "indices_zero_based": indices, "days": 730,
                       "families": list(FAMILIES), "records_per_family": 2},
        "scope": "New controlled random realizations; observation selector and matched-count controls frozen before input generation. No field/expert reference claim",
        "motivation": "Exposed twelve-record equal-count replay supports a placement effect. Validate that effect with the existing method, without adding a mechanism or rescuing the earlier failed scope condition",
        "candidate_source_sha256": sha(ROOT / "scripts/probe_landslide_revision_scope.py"),
        "method": SCOPE_PROTOCOL["policy"], "revision": "Original49-day conflict permission, unchanged",
        "primary_ablation": "same_count_fixed_early", "sensitivity_ablation": "same_count_fixed_late",
        "matched_count": "K=0/1/2 per record/repeat from original raw selector; fixed targets already at existing equal-spacing locations. EarliestK primary, latestK sensitivity, from the earlier prefix convention",
        "comparisons": "37 configurations: original29 including pure/fixed/numeric-text and nine numeric controls, plus two matched fixed orders x two scopes x two repeats",
        "selection_gate": "Both repeats and both whole-target/conflict-scope variants adaptive activityF1>matched_early, weakRecall>=matched_early, and net correct definite decisions>matched_early. LatestK sensitivity must be reported; cannot become a replacement primary",
        "additional_reporting": "Direct/full fixed, all TP/FP gains/losses, event IoU.3/.5, onset/offset MAE, unknown coverage, internal/end-to-end ASD, all families and deployment/actual costs",
        "scope_claim": "The earlier two-repeat FP-reduction scope condition remains unconfirmed. This trial does not count as another chance to pass that failed gate",
        "training": "Same controlled distribution, originaltrain24/validation12 fitted and calibrated nine numeric controls; no new label training or tuning",
        "references": "Known controlled latent motion/stages with observed gaps. Validates controlled recovery only, not observable field activity or site transfer",
        "repetitions": 2, "maximum_requests": 168, "retries": 0,
        "failure_action": "No prompt/threshold/scope changes, no candidate switch or next-seed rescue. If matched placement fails, report location mechanism unconfirmed",
    }
    save_json(out / "protocol.json", protocol)
    model_folder = BASE / "evidence-display-confirmation-v2"
    files = ("stage-xgboost.json", "activity-xgboost.json", "original-selection.json", "raw-selection.json",
             "activity-xgboost-raw-persistence.json", "long-selection.json", "activity-xgboost-84.json",
             "matched-selection.json", "activity-xgboost-42-matched-support.json")
    bindings = {}
    for name in files:
        source = model_folder / name
        shutil.copyfile(source, out / name)
        bindings[str(source)] = sha(source)
    save_json(out / "training-bindings.json", bindings)
    for folder in (ROOT / "scripts", ROOT / "src/gnss_sim"):
        for source in folder.glob("*.py"):
            target = out / "source" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    config = {
        "schema_version": "landslide-adaptive-review-v1", "date": "2026-10-10",
        "scope": protocol["scope"], "source_bundle": out.relative_to(ROOT).as_posix(),
        "case_ids": ids, "model": "qwen3.8-flash", "repetitions": 2,
        "window_days": 180, "stride_days": 90, "policy": SCOPE_PROTOCOL["policy"],
        "maximum_requests": 168, "hard_timeout_seconds": 3600, "retries": 0,
        "primary_candidate": protocol["primary"], "stage_policy": "Unchanged original trained stage head",
        "confirmation_protocol": str(out / "protocol.json"), "run_name": "budget-selection-confirmation-v1",
    }
    save_json(out / "runner-config.json", config)
    save_json(out / "pre-generation-freeze.json", {str(path.relative_to(out)).replace("\\", "/"): sha(path)
                                                  for path in out.rglob("*") if path.is_file()})
    request = LandslideRequest(seed=2026101075, count=132, days=730)
    records, rows, families = {}, [], []
    for index in indices:
        case, labels, truth = controlled_case(request, index)
        family = truth["family"]
        if family != FAMILIES[index % len(FAMILIES)]:
            raise ValueError("Generated actual family differs from the registered ordering")
        families.append(family)
        folder = out / "inputs" / case.case_id
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        reference = out / "mechanism-reference" / case.case_id
        reference.mkdir(parents=True)
        save_json(reference / "truth.json", truth)
        motions = {}
        for window in (31, 61, 91):
            motion = derive_motion(case, window)
            motions[window] = motion
            (folder / f"motion-{window}.json").write_text(motion.model_dump_json(indent=2), encoding="utf-8")
        records[case.case_id] = case, observation_features(case, motions)
        rows.extend(labels)
        print(f"Prepared {case.case_id}: {family}", flush=True)
    population = dict(Counter(families))
    if len(population) != 6 or set(population.values()) != {2}:
        raise ValueError("Actual generated population is not six families with two records each")
    save_json(out / "actual-population.json", population)
    with (out / "reference-labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(out / "numeric-predictions.json", numerical_predictions(out, records))
    for name, expected in read_json(out / "pre-generation-freeze.json").items():
        if sha(out / name) != expected:
            raise ValueError("Registered method/scorer changed during generation")
    save_json(out / "manifest.json", {"files": {str(path.relative_to(out)).replace("\\", "/"): sha(path)
                                               for path in out.rglob("*") if path.is_file()}})
    print({"records": len(records), "population": population, "new_api": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.out.resolve())
