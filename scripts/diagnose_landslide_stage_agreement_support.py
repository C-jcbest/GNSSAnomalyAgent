"""Check window availability on jointly abstained development-reference days."""

from __future__ import annotations

import math
from pathlib import Path

from landslide_stage_reference import DEFAULT_KIT, read_json, verify_bundle

from gnss_sim.artifacts import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/landslide-stage-agreement-2026-10-08/analysis-v1"


def finite_velocity(vector):
    return vector is not None and all(math.isfinite(value) for value in vector)


def main():
    source = RUN / "public/case-analysis.json"
    freeze = read_json(RUN / "analysis-freeze.json")
    if sha(source) != freeze["public_files_sha256"]["case-analysis.json"]:
        raise ValueError("Frozen comparison data changed")
    contexts, source_digest = verify_bundle(DEFAULT_KIT)
    by_id = {context["case"].case_id: context for context in contexts}
    records = []
    totals = {str(window): {"velocity_days": 0, "tangent_days": 0} for window in (31, 61, 91)}
    for case in read_json(source):
        predictions = case["prediction_features"]["final"]
        days = [day for day, evaluable in enumerate(case["reference_evaluable"])
                if evaluable and all(predictions[method][day] == "unknown"
                                     for method in ("rule", "xgboost", "visual"))]
        if not days:
            continue
        windows = {}
        for window, motion in by_id[case["case_id"]]["motions"].items():
            velocity = [day for day in days if finite_velocity(motion.velocity_mm_day[day])]
            tangent = [day for day in days if motion.tangential_acceleration_mm_day2[day] is not None
                       and math.isfinite(motion.tangential_acceleration_mm_day2[day])]
            windows[str(window)] = {"velocity_days": len(velocity), "tangent_days": len(tangent),
                                    "finite_velocity_day_indices": velocity,
                                    "finite_tangent_day_indices": tangent}
            totals[str(window)]["velocity_days"] += len(velocity)
            totals[str(window)]["tangent_days"] += len(tangent)
        records.append({"case_id": case["case_id"], "joint_abstention_days": days, "windows": windows})
    result = {
        "reference_use": "development_only", "performance_scores": None,
        "comparison_data_sha256": sha(source), "source_manifest_sha256": source_digest,
        "implementation_sha256": sha(Path(__file__)), "records": records, "totals": totals,
        "joint_abstention_days": sum(len(record["joint_abstention_days"]) for record in records),
        "meaning": "finite estimate availability only; does not establish correct alternate stages",
        "old_prediction_or_support_changed": False,
    }
    write_json(RUN / "support-diagnosis.json", result)
    print({"joint_abstention_days": result["joint_abstention_days"], "totals": totals})


if __name__ == "__main__":
    main()
