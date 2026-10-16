"""Record actual review provenance without rewriting frozen reference labels."""

from datetime import datetime, timezone
from pathlib import Path

from run_landslide_window_optimization import read_json
from verify_landslide_codex_observation_review import verify

from gnss_sim.artifacts import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts/landslide-adaptive-review-2026-10-09/dual-reference-confirmation-data-v1"


def finalize():
    review = DATA / "codex-review"
    target = review / "delivery-manifest.json"
    if target.exists():
        raise FileExistsError("Preserve the completed review provenance")
    audit = verify(review, DATA / "raw-review-pack")
    write_json(review / "verification.json", audit)
    write_json(review / "actual-review-provenance.json", {
        "review_date_local": "2026-10-11",
        "reviewer": "Current Codex method author; actual visual and numerical review",
        "historical_generation_labels_seen": "0133-0144 process reference read in v30; generic design known",
        "specific_new_generation_labels_seen_during_review": False,
        "specific_new_predictions_seen_during_review": False,
        "independent_reviewer": False,
        "formal_reference_verified": False,
        "compiled_metadata_limitation": "Shared compiler retained its hardcoded2026-10-10 reviewer date and prior_generation_labels_seen=false. Those fields do not describe historical exposure for this batch. This explicit sidecar and the pre-generation protocol disclose actual exposure; frozen rows are not changed.",
        "compiled_reference_freeze_sha256": sha(review / "compiled/reference-freeze.json"),
        "activity_first_freeze_sha256": sha(review / "activity-freeze.json"),
        "primary_prepare_source_change": "Only run date now comes from the data protocol and stage_policy describes previously frozen target-matched models; no selection, prompts, model, thresholds or inference logic change",
        "pre_generation_prepare_source_sha256": read_json(DATA / "source-bindings.json")[str(Path("scripts/run_landslide_ai_reference_holdout.py"))],
        "actual_prepare_source_sha256": sha(ROOT / "scripts/run_landslide_ai_reference_holdout.py"),
    })
    write_json(target, {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "specific_new_predictions_seen_during_review": False,
        "specific_new_generation_labels_seen_during_review": False,
        "generic_design_and_historical_results_known": True,
        "historical0133_0144_process_labels_seen": True,
        "independent_reviewer": False,
        "actual_raw_images_inspected": 24,
        "actual_auxiliary_images_inspected": 12,
        "compiled_images_visually_checked": ["case_0148", "case_0152", "case_0155"],
        "files": {path.relative_to(review).as_posix(): sha(path)
                  for path in sorted(review.rglob("*")) if path.is_file()},
    })


if __name__ == "__main__":
    finalize()
