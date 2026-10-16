"""Separate global-contract recovery from recognition using frozen per-case results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from run_landslide_window_optimization import load_case_features, read_json

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores
from gnss_sim.landslide_review_utility import correction_utility


def diagnose(data, analysis, out):
    verify_manifest(analysis)
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    sensitivities, numerical = [], []
    for target in ("primary_process", "secondary_developer"):
        folder = analysis / target
        results = read_json(folder / "results.json")
        predictions = read_json(folder / "predictions.json")
        labels = read_json(folder / "reference-labels.json")
        for rep, failed in ((0, "case_0174"), (1, "case_0177")):
            main = f"adaptive_no_feedback_r{rep}__conflict_scope"
            ids = [cid for cid in labels if cid != failed]
            main_f1 = aggregate_scores([results[main]["per_case"][cid] for cid in ids])["daily"][
                "f1"
            ]
            for control in (
                f"direct_visual_r{rep}",
                f"fixed_image_review_r{rep}__whole_target",
                f"same_count_fixed_early_r{rep}__conflict_scope",
            ):
                control_f1 = aggregate_scores([results[control]["per_case"][cid] for cid in ids])[
                    "daily"
                ]["f1"]
                net = 0
                for cid in ids:
                    actual = np.array([int(row["activity_label"]) for row in labels[cid]])
                    net += correction_utility(
                        actual,
                        np.array(predictions[control][cid]["activity"]),
                        np.array(predictions[main][cid]["activity"]),
                        actual >= 0,
                    )["net_correct"]
                sensitivities.append(
                    {
                        "target": target,
                        "repeat": rep,
                        "excluded_global_contract_failure": failed,
                        "records": len(ids),
                        "control": control,
                        "treatment": main,
                        "main_f1": main_f1,
                        "control_f1": control_f1,
                        "f1_difference": main_f1 - control_f1,
                        "net_correct": net,
                        "scope": "Post-score failure sensitivity; no replacement of complete registered gate/population",
                    }
                )
            for control in predictions:
                if any(f"_r{i}" in control for i in (0, 1)):
                    continue
                totals = {
                    key: 0
                    for key in ("eligible_days", "gained_correct", "lost_correct", "net_correct")
                }
                for cid in labels:
                    actual = np.array([int(row["activity_label"]) for row in labels[cid]])
                    counts = correction_utility(
                        actual,
                        np.array(predictions[control][cid]["activity"]),
                        np.array(predictions[main][cid]["activity"]),
                        actual >= 0,
                    )
                    for key in totals:
                        totals[key] += counts[key]
                numerical.append(
                    {"target": target, "control": control, "treatment": main, **totals}
                )
    write_csv(out / "global-failure-sensitivity.csv", sensitivities)
    write_csv(out / "all-numeric-paired-corrections.csv", numerical)
    block_plans = {
        "case_0169": [(330, 360), (360, 390), (390, 410)],
        "case_0171": [(270, 300), (330, 360), (390, 420)],
        "case_0174": [(510, 540), (540, 570), (600, 630), (660, 685)],
        "case_0178": [(480, 510), (540, 570), (600, 630), (635, 665)],
    }
    blocks = {}
    for cid, intervals in block_plans.items():
        case, feature = load_case_features(data, cid)
        raw = np.asarray([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        blocks[cid] = [
            {
                "start": start,
                "stop": stop,
                "observed_days": int(feature.observed[start:stop].sum()),
                "median_NEU_mm": np.nanmedian(raw[start:stop], axis=0).tolist(),
            }
            for start, stop in intervals
        ]
    write_json(out / "raw-inspection-blocks.json", blocks)
    write_json(
        out / "audit.json",
        {
            "global_failure_sensitivities": len(sensitivities),
            "numeric_comparisons": len(numerical),
            "new_remote_requests": 0,
            "post_score_descriptive_only": True,
            "registered_gate_unchanged": True,
            "filter_note": "Frozen scorer paired list used '_r' substring, which also occurs in rule/raw names; all nine/six controls were scored correctly. This supplement adds all30 numerical pairs; existing metrics and gate are unchanged.",
        },
    )
    write_json(
        out / "manifest.json",
        {
            "files": {
                path.relative_to(out).as_posix(): sha(path)
                for path in sorted(out.rglob("*"))
                if path.is_file()
            }
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "analysis", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.data, args.analysis, args.out)
