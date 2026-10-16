"""Measure reference sensitivity of existing frozen activity predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from check_landslide_reference_process_agreement import read_json, write_csv
from sklearn.metrics import precision_recall_fscore_support

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import score_counts

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
ORIGINAL = BASE / "ai-observable-holdout-comparison-v1/analysis"


def run(out):
    protocol = read_json(out / "protocol.json")
    for name, digest in protocol["bindings"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen evidence changed: " + name)
    if (out / "summary.csv").exists():
        raise FileExistsError("Preserve the completed sensitivity analysis")
    write_json(
        out / "analysis-start.json",
        {"source_sha256": sha(Path(__file__)), "protocol_sha256": sha(out / "protocol.json")},
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    with (BASE / "reference-process-agreement-v1/daily-agreement.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    with (ORIGINAL / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        original = {row["method"]: row for row in csv.DictReader(stream)}
    predictions = read_json(ORIGINAL / "predictions.json")
    if set(predictions) != set(original):
        raise ValueError("Do not omit any original method")
    case_ids = sorted({row["case_id"] for row in rows})
    grouped = {cid: [row for row in rows if row["case_id"] == cid] for cid in case_ids}
    observed = np.array([row["observed"] == "True" for row in rows])
    ai = np.array([int(row["ai_activity"]) for row in rows])
    process = np.array([int(row["process_activity"]) for row in rows])
    common = observed & (ai >= 0)
    variants = {
        "ai_definite": (ai, common),
        "process_same_mask": (process, common),
        "process_all_observed": (process, observed),
    }
    summaries, records, unknown_records = [], [], []
    for method, cases in predictions.items():
        if sorted(cases) != case_ids:
            raise ValueError("Every method must use all twelve records")
        activity = np.array([value for cid in case_ids for value in cases[cid]["activity"]])
        if len(activity) != len(rows) or not np.isin(activity, [-1, 0, 1]).all():
            raise ValueError("Invalid activity calendar or codes")
        if np.any(activity[~observed] != -1):
            raise ValueError("Missing day gained a prediction")
        for variant, (actual, mask) in variants.items():
            counts = score_counts(
                np.count_nonzero(mask & (actual == 1) & (activity == 1)),
                np.count_nonzero(mask & (actual == 0) & (activity == 1)),
                np.count_nonzero(mask & (actual == 1) & (activity != 1)),
            )
            p, r, f, _ = precision_recall_fscore_support(
                actual[mask], activity[mask], labels=[1], zero_division=0
            )
            for key, independent in (("precision", p[0]), ("recall", r[0]), ("f1", f[0])):
                if counts[key] is not None and not np.isclose(counts[key], independent):
                    raise ValueError("Independent activity metric differs")
            if variant == "ai_definite":
                for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
                    if not np.isclose(counts[key], float(original[method][key])):
                        raise ValueError("Original reference score did not replay")
            summaries.append(
                {
                    "method": method,
                    "variant": variant,
                    **counts,
                    "evaluation_days": int(mask.sum()),
                    "positive_days": int((mask & (actual == 1)).sum()),
                    "coverage": float((activity[mask] >= 0).mean()),
                }
            )
            start = 0
            for cid in case_ids:
                stop = start + len(grouped[cid])
                a, pred, eligible = actual[start:stop], activity[start:stop], mask[start:stop]
                records.append(
                    {
                        "method": method,
                        "variant": variant,
                        "case_id": cid,
                        **score_counts(
                            np.count_nonzero(eligible & (a == 1) & (pred == 1)),
                            np.count_nonzero(eligible & (a == 0) & (pred == 1)),
                            np.count_nonzero(eligible & (a == 1) & (pred != 1)),
                        ),
                    }
                )
                start = stop
        unknown = observed & (ai == -1)
        unknown_records.append(
            {
                "method": method,
                **{
                    f"process_{actual}_prediction_{pred}": int(
                        (unknown & (process == actual) & (activity == pred)).sum()
                    )
                    for actual in (0, 1)
                    for pred in (-1, 0, 1)
                },
            }
        )
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "per-record.csv", records)
    write_csv(out / "unknown-reference-decisions.csv", unknown_records)
    write_json(
        out / "audit.json",
        {
            "status": "passed",
            "methods": len(predictions),
            "variants": len(variants),
            "original_methods_replayed": len(predictions),
            "sklearn_comparisons": len(summaries),
            "new_model_calls": 0,
            "new_training": False,
            "independent_confirmation": False,
            "scope": protocol["purpose"],
        },
    )
    write_json(
        out / "manifest.json",
        {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()},
    )
    print(json.dumps(read_json(out / "audit.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output.resolve())
