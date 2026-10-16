"""Measure frozen utility forecast errors by background; no refitting or API calls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_comparison import ROOT, load_labels, save_json, sha
from run_landslide_utility_prototype import verify
from run_landslide_window_optimization import load_case_features, read_json
from summarize_landslide_missed_motion_repair import write_rows

from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_review_utility import correction_utility


def diagnose(run, out):
    verify(run)
    config = read_json(run / "protocol.json")
    source = Path(config["source"])
    data = ROOT / read_json(source / "protocol.json")["source_bundle"]
    initial_saved = read_json(source / "predictions.json")
    locals_saved = {**read_json(run / "cached-locals.json"), **read_json(run / "new-locals.json")}
    selections = read_json(run / "selections.json")
    records = {cid: load_case_features(run, cid) for cid in config["case_ids"]}
    labels = load_labels(source / "analysis-v1/reference-subset.csv", {cid: pair[0] for cid, pair in records.items()})
    rows = []
    for rep in (0, 1):
        for cid, (case, features) in records.items():
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            initial = np.array(initial_saved[f"direct_visual_r{rep}"][cid]["activity"])
            neighbourhood = conflict_neighbourhood(initial, features, raw_motion_evidence(case, features), ReviewPolicy())[1]
            truth = read_json(data / "mechanism-reference" / cid / "truth.json")
            for selection in selections[f"{cid}-r{rep}"]["calibrated_utility"]["selected"]:
                scope = target_scopes(neighbourhood, features.observed, [selection])[0]
                reviewed = np.array(locals_saved[f"{cid}-window-{selection['index']:02d}-image-r{rep}"]["activity"])
                utility = correction_utility(actual, initial, reviewed, scope)
                negative = scope & (actual == 0)
                positive = scope & (actual == 1)
                rows.append({"case_id": cid, "repeat": rep, "background": truth["background_group"],
                             "family": truth["family"], **selection, **utility,
                             "forecast_error_days": selection["expected_net_corrections"] - utility["net_correct"],
                             "new_false_active_days": int((negative & (initial != 1) & (reviewed == 1)).sum()),
                             "lost_true_active_days": int((positive & (initial == 1) & (reviewed != 1)).sum())})
    groups = []
    for rep in (0, 1):
        for group in sorted({row["background"] for row in rows}):
            values = [row for row in rows if row["repeat"] == rep and row["background"] == group]
            groups.append({"repeat": rep, "background": group, "selected_views": len(values),
                           "harmful_views": sum(row["net_correct"] < 0 for row in values),
                           "beneficial_views": sum(row["net_correct"] > 0 for row in values),
                           "expected_net_correct": sum(row["expected_net_corrections"] for row in values),
                           "actual_net_correct": sum(row["net_correct"] for row in values),
                           "new_false_active_days": sum(row["new_false_active_days"] for row in values),
                           "lost_true_active_days": sum(row["lost_true_active_days"] for row in values)})
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out / "selected-view-utilities.csv", rows)
    write_rows(out / "background-calibration.csv", groups)
    save_json(out / "state.json", {"status": "complete", "selected_views": len(rows), "background_groups": len(groups),
                                   "new_requests": 0, "new_records": 0,
                                   "interpretation": "Post-outcome descriptive development diagnostics; not an independent test "
                                                     "or proof that raw/numerical forecast errors caused VLM errors"})
    save_json(out / "source-bindings.json", {str(Path(__file__).resolve()): sha(Path(__file__))})
    print(groups, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.run, args.out)
