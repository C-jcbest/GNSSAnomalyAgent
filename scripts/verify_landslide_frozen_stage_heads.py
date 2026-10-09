"""Read-only replay of delivered frozen stages, plots and response associations."""
from __future__ import annotations

import csv
import json
import sys
from itertools import combinations
from pathlib import Path

from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_landslide_frozen_stage_heads import read_json, verify_hashes  # noqa: E402
from build_landslide_raw_adjudication import verify as verify_activity_run  # noqa: E402

from gnss_sim.artifacts import sha  # noqa: E402
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_baselines import observation_features, rule_stages  # noqa: E402
from gnss_sim.landslide_diagnostics import LandslideDiagnostics  # noqa: E402
from gnss_sim.landslide_evaluation import ActivityPrediction  # noqa: E402
from gnss_sim.landslide_frozen_stage_heads import (  # noqa: E402
    METHODS,
    apply_stage_support,
    compile_stage_answer,
    numeric_stage_rows,
    stage_pair_counts,
    unclassified_rows,
    validate_activity,
)
from gnss_sim.visual import _unique_object  # noqa: E402


def main():
    run = ROOT / "artifacts/landslide-frozen-stage-heads-2026-10-08/run-v1"
    public = run / "public"
    manifest = read_json(run / "manifest.json")
    freeze = verify_hashes(run, manifest)
    author = ROOT / manifest["config"]["activity_run"]
    verify_activity_run(author)
    entries = {r["case_id"]: r for r in read_json(author / "public/manifest.json")["cases"]}
    frozen_activity = read_json(public / "activity-freeze.json")
    records = read_json(public / "source-manifest.json")["cases"]
    packets = {p["case_id"]: p for p in read_json(public / "packets.json")}
    summary = read_json(public / "results.json")
    results = {r["case_id"]: r for r in summary["results"]}
    model = XGBClassifier()
    model.load_model(run / "models/stage-xgboost.json")
    csv_rows = []
    for record in records:
        case_id = record["case_id"]
        case = LandslideInput.model_validate_json((public / record["input"]).read_bytes())
        review = read_json(public / "activity-source" / entries[case_id]["review"])
        rows = read_json(public / "activity-source" / entries[case_id]["daily"])
        digest = frozen_activity["reviews"][case_id]["activity_sha256"]
        activity = validate_activity(case, review, rows, digest)
        diagnostics = {w: LandslideDiagnostics.model_validate_json(
            (public / record["diagnostics"][str(w)]["data"]).read_bytes()) for w in (31, 61, 91)}
        features = observation_features(case, diagnostics)
        predictions = {"rule": rule_stages(features, activity, manifest["config"]["rule_relative_change_60d"])}
        stage = model.predict(features.matrix).astype(int) + 1
        stage[(activity != 1) | ~features.supported] = -1
        stage[activity == 0] = 0
        predictions["xgboost"] = ActivityPrediction(activity.copy(), stage)
        native, final = {}, {}
        for method in ("rule", "xgboost"):
            native[method] = numeric_stage_rows(rows, predictions[method])
            final[method], _ = apply_stage_support(native[method], rows, case, diagnostics[61])
        packet = packets.get(case_id)
        if packet:
            receipt = read_json(public / f"requests/{packet['request_id']}.response.json")
            payload = json.loads(receipt["output"], object_pairs_hook=_unique_object)
            _, native["visual"], final["visual"], _ = compile_stage_answer(
                payload, case, review, rows, digest, diagnostics[61])
        else:
            native["visual"] = unclassified_rows(rows)
            final["visual"] = unclassified_rows(rows)
        for method in METHODS:
            if native[method] != read_json(public / f"native/{case_id}-{method}.json"):
                raise ValueError("Native replay changed")
            if final[method] != read_json(public / f"daily/{case_id}-{method}.json"):
                raise ValueError("Final replay changed")
            csv_rows.extend({"method": method, **row} for row in final[method])
        for a, b in combinations(METHODS, 2):
            if stage_pair_counts(final[a], final[b]) != results[case_id]["pairs"][f"{a}_vs_{b}"]:
                raise ValueError("Pair replay changed")
    with (public / "daily-stage-heads.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    if saved != [{k: str(v) for k, v in row.items()} for row in csv_rows] or len(saved) != 39420:
        raise ValueError("CSV replay changed")
    plots = read_json(public / "presentation/plot-audits.json")
    for name, audit in plots.items():
        if sha(public / name) != audit["sha256"]:
            raise ValueError("Rendered plot changed")
    delivery = read_json(run / "presentation-audit.json")
    for path, key in ((public / "analysis.json", "analysis_sha256"), (public / "index.html", "page_sha256"),
                      (public / "presentation/plot-audits.json", "plot_audits_sha256"),
                      (run / "author-findings.json", "author_findings_sha256")):
        if sha(path) != delivery[key]:
            raise ValueError("Presentation changed")
    diagnostic = read_json(public / "author-diagnostics.json")
    if diagnostic["script_sha256"] != sha(ROOT / "scripts/diagnose_landslide_frozen_stage_heads.py"):
        raise ValueError("Post-inference diagnostic source changed")
    if diagnostic["inference_freeze_sha256"] != sha(run / "inference-freeze.json"):
        raise ValueError("Post-inference diagnostic uses another inference freeze")
    print(json.dumps({"cases_replayed": 12, "csv_rows_replayed": len(saved),
                      "visual_answers_replayed": 9, "plots_hash_verified": len(plots),
                      "inference_files_hash_verified": len(freeze["evidence_sha256"]),
                      "activity_unchanged": True, "performance_scores": None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
