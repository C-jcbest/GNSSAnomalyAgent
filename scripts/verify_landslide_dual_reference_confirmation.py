"""Replay numerical predictions and both target scores without remote inference."""

from pathlib import Path

from analyze_landslide_dual_reference_confirmation import BASE, analyze, prepare_process
from prepare_landslide_ai_reference_holdout import verify_manifest
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha, write_json


def verify():
    original = BASE / "dual-reference-confirmation-analysis-v1"
    numerical = BASE / "dual-reference-process-predictions-v1"
    replay_predictions = BASE / "dual-reference-process-predictions-replay-v1"
    replay_analysis = BASE / "dual-reference-confirmation-analysis-replay-v1"
    output = BASE / "dual-reference-confirmation-verification-v1"
    output.mkdir(parents=True, exist_ok=False)
    prepare_process(replay_predictions)
    for filename in ("numeric-predictions.json", "stage-codes.json"):
        if sha(numerical / filename) != sha(replay_predictions / filename):
            raise ValueError("Process-target numerical replay differs")
    analyze(replay_predictions, replay_analysis)
    identical = []
    for target in ("primary_ai", "secondary_process"):
        for filename in ("predictions.json", "reference-labels.json", "results.json", "bootstrap.json",
                "registration.json", "summary.csv", "per-record.csv", "corrections.csv"):
            relative = target + "/" + filename
            if sha(original / relative) != sha(replay_analysis / relative):
                raise ValueError("Target score replay differs: " + relative)
            identical.append(relative)
    for filename in ("decision.json", "reference-agreement.json", "reference-agreement.csv"):
        if sha(original / filename) != sha(replay_analysis / filename):
            raise ValueError("Reference/decision replay differs")
        identical.append(filename)
    for directory in (original, replay_analysis, BASE / "dual-reference-confirmation-errors-v1"):
        verify_manifest(directory)
    audit = read_json(original / "audit.json")
    write_json(output / "audit.json", {"status": "passed", "exact_score_files": identical,
        "exact_score_file_count": len(identical), "exact_numerical_files": 2,
        "source_score_audit": audit, "new_remote_requests": 0,
        "all_primary34_metrics_checked_against_original_analysis": True,
        "process_models_rerun_without_refitting": True})
    (output / "source.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    verify()
