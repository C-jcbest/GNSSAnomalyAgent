"""Register an unchanged published-design task adapter on an already-scored population."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from finalize_landslide_adaptive_evidence import verify_manifest
from run_landslide_adaptive_review import verify_freeze
from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_tama_adapter import prepare
from run_landslide_window_optimization import read_json


def register(run, tama, registration):
    verify_freeze(run)
    verify_manifest(run / "analysis-v1")
    earlier = run.parent / "tama-gnss-confirmation-v1"
    adapter = ROOT / "scripts/run_landslide_tama_adapter.py"
    if sha(adapter) != sha(earlier / "source/run_landslide_tama_adapter.py"):
        raise ValueError("The old tested task adapter must remain unchanged")
    if registration.exists() or tama.exists():
        raise ValueError("Register only into new paths, without overwriting previous trials")
    registration.mkdir(parents=True)
    protocol = {"date": "2026-10-10", "status": "registered before comparator inference",
        "purpose": "Append the existing TAMA-inspired GNSS baseline to the same-data primary comparison",
        "already_exposed_population": read_json(run / "protocol.json")["case_ids"],
        "new_records": 0, "independence": "Same evaluated population; not a new method confirmation or field reference",
        "adapter": "Unchanged run_landslide_tama_adapter from the earlier 0061..0072 study",
        "author_tree": "ChongKaKam/TAMA db99a75a83d0ef05caf7ff0d199b75b594e98584",
        "adaptations": "GNSS continuing motion/unknown, three-component raw plots, common Qwen and stage head, "
                        "training-only references, flattened dialogue with retained images, no retries/JSON repair",
        "controller": "Three quiet common-training references; one pattern call per repeat; "
                      "one full detection; parsed nonempty activity triggers one first/last two-thirds review",
        "training": "Original controlled train24/validation12 models unchanged; no added training or tuning",
        "primary_method": "Original raw21/42, conflict_scope, no numeric text; not selected by comparator results",
        "evaluation": "Append four adapter arms to all original37; retain full population, all failures, two repeats, "
                      "activity P/R/F1, weak recall, event IoU.3/.5, boundaries, coverage, internal/end-to-end ASD, "
                      "all six families and gained/lost decisions",
        "costs": "Actual unique receipts and per-method deployment; include one normal-pattern call per repeat",
        "model_budget": {"maximum_requests": 50, "repetitions": 2, "retries": 0},
        "interpretation": "Descriptive nearest-design task adaptation, not the original published model/benchmark",
        "failure_action": "Keep unknown and costs, do not change prompts/windows/thresholds, choose repeats or rescue by seed",
        "field_accuracy_confirmed": False, "scope_condition_retested": False, "goal_status": "active"}
    save_json(registration / "protocol.json", protocol)
    prepare(run, tama)
    paths = [adapter, Path(__file__), ROOT / "scripts/analyze_landslide_tama_budget_comparison.py",
             run / "analysis-v1/manifest.json", run / "prediction-freeze.json", tama / "freeze.json",
             ROOT / "docs/landslide-adaptive-review-manuscript-draft.md"]
    for folder in (ROOT / "src/gnss_sim", ROOT / "scripts"):
        paths.extend(folder.rglob("*.py"))
    paths = sorted(set(path.resolve() for path in paths))
    save_json(registration / "bindings.json", {str(path): sha(path) for path in paths})
    shutil.copyfile(Path(__file__), registration / "source.py")
    shutil.copyfile(adapter, registration / "adapter.py")
    shutil.copyfile(ROOT / "scripts/analyze_landslide_tama_budget_comparison.py", registration / "scorer.py")
    save_json(registration / "manifest.json", {"files": {path.name: sha(path)
                                                        for path in registration.iterdir() if path.is_file()}})
    print({"registered": str(registration), "maximum_requests": 50, "new_records": 0}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tama", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    arguments = parser.parse_args()
    register(arguments.run, arguments.tama, arguments.registration)
