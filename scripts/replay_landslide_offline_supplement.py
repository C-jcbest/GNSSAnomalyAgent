"""Recompute the archived 41-arm comparison using only the portable supplement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scoring import evaluate_population, verify_metrics

from gnss_sim.landslide_evaluation import aggregate_scores
from gnss_sim.landslide_review_utility import correction_utility


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay(bundle, out):
    manifest = read_json(bundle / "manifest.json")
    for name, expected in manifest["files"].items():
        path = bundle / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Supplement file differs from its freeze: {name}")
    predictions = read_json(bundle / "data/predictions.json")
    families = read_json(bundle / "data/families.json")
    labels = {cid: [] for cid in families}
    with (bundle / "data/reference.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["case_id"]].append(row)
    records = {}
    for cid, rows in labels.items():
        case = read_json(bundle / "data/inputs" / f"{cid}.json")
        values = np.asarray([value if value is not None else [np.nan] * 3
                             for value in case["displacement_mm"]], dtype=float)
        observed = np.isfinite(values).all(axis=1)
        if len(rows) != len(case["dates"]):
            raise ValueError(f"Incomplete reference calendar: {cid}")
        for day, row in enumerate(rows):
            if int(row["day_index"]) != day or row["date"] != case["dates"][day]:
                raise ValueError(f"Reference date mismatch: {cid}, {day}")
            if not observed[day] and int(row["activity_label"]) != -1:
                raise ValueError(f"Missing observation became a scored reference: {cid}, {day}")
        records[cid] = (case, SimpleNamespace(observed=observed))
    if len(predictions) != 41 or any(set(cases) != set(labels) for cases in predictions.values()):
        raise ValueError("All 41 methods must contain the same twelve records")
    results, summary, per_case = evaluate_population(predictions, labels)
    internal = verify_metrics(predictions, results, labels, records)
    usage = read_json(bundle / "data/usage.json")
    family_rows = []
    for row in summary:
        method = row["method"]
        row["internal_stage_macro_f1"] = internal[method]["macro_f1"]
        row.update(usage[method])
        for family in sorted(set(families.values())):
            scores = [results[method]["per_case"][cid]
                      for cid in labels if families[cid] == family]
            aggregate = aggregate_scores(scores)
            family_rows.append({"method": method, "family": family, **aggregate["daily"],
                                "weak_recall": aggregate["weak_recall"],
                                "coverage": aggregate["coverage"]})
    corrections = []
    for repeat in (0, 1):
        comparisons = [
            (f"adaptive_no_feedback_r{repeat}__conflict_scope",
             f"tama_gnss_positive_review_r{repeat}"),
            (f"adaptive_no_feedback_r{repeat}__conflict_scope", f"tama_gnss_no_review_r{repeat}"),
            (f"tama_gnss_positive_review_r{repeat}", f"tama_gnss_no_review_r{repeat}"),
        ]
        for method, control in comparisons:
            for cid, rows in labels.items():
                actual = np.array([int(row["activity_label"]) for row in rows])
                before = np.array(predictions[control][cid]["activity"])
                after = np.array(predictions[method][cid]["activity"])
                corrections.append({"method": method, "control": control, "case_id": cid,
                    "family": families[cid], **correction_utility(actual, before, after, actual >= 0),
                    "gained_tp": int(((actual == 1) & (before != 1) & (after == 1)).sum()),
                    "lost_tp": int(((actual == 1) & (before == 1) & (after != 1)).sum()),
                    "added_fp": int(((actual == 0) & (before != 1) & (after == 1)).sum()),
                    "removed_fp": int(((actual == 0) & (before == 1) & (after != 1)).sum())})
    if results != read_json(bundle / "expected/results.json"):
        raise ValueError("Recomputed daily, stage or event/boundary result differs from the archive")
    if internal != read_json(bundle / "expected/internal-stages.json"):
        raise ValueError("Independent sklearn stage results differ from the archive")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "results.json", results)
    write_json(out / "internal-stages.json", internal)
    for name, rows in (("summary.csv", summary), ("per-case.csv", per_case),
                       ("family-results.csv", family_rows), ("corrections.csv", corrections)):
        write_csv(out / name, rows)
        if (out / name).read_bytes() != (bundle / "expected" / name).read_bytes():
            raise ValueError(f"Recomputed table differs from the archive: {name}")
    daily_path = out / "daily.csv"
    count = 0
    with daily_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "case_id", "day_index",
            "actual_activity", "actual_stage", "predicted_activity", "predicted_stage"])
        writer.writeheader()
        for method, cases in predictions.items():
            for cid, value in cases.items():
                for day, row in enumerate(labels[cid]):
                    writer.writerow({"method": method, "case_id": cid, "day_index": day,
                        "actual_activity": row["activity_label"], "actual_stage": row["feature"],
                        "predicted_activity": value["activity"][day],
                        "predicted_stage": value["stage"][day]})
                    count += 1
    expected_daily = read_json(bundle / "expected/daily-digest.json")
    if hashlib.sha256(daily_path.read_bytes()).hexdigest() != expected_daily["sha256"]:
        raise ValueError("Recomputed complete daily export differs from the archive")
    checks = {"methods": len(results), "records": len(labels), "daily_rows": count,
              "family_rows": len(family_rows), "correction_rows": len(corrections),
              "results_and_internal_stages_exact": True, "csv_exports_exact": True,
              "new_model_requests": 0, "usage": "Archived deployment usage; no new inference",
              "reference": "Controlled mechanism reference, not independently reviewed field truth"}
    write_json(out / "checks.json", checks)
    print(json.dumps(checks, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay(Path(__file__).resolve().parent, args.out)


if __name__ == "__main__":
    main()
