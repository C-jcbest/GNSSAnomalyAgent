"""Measure duration-filter fragmentation without changing frozen models or scores."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_window_optimization import read_json
from score_landslide_claim_review import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import runs
from gnss_sim.landslide_local_experiment import learned_stages


def duration_across_raw_missing(probability, threshold, supported, observed):
    """Count positive evidence across short raw gaps; never label a missing day.

    Observed negatives and observed unsupported days remain separators. The
    minimum is14 supported positive observations, rather than14 calendar days
    padded with absence. The seven-day cap is the existing event-mask cap.
    """
    positive = (np.asarray(probability) >= threshold) & supported
    linked = positive.copy()
    for start, stop in runs(~observed):
        if stop - start <= 7 and start > 0 and stop < len(linked):
            if positive[start - 1] and positive[stop]:
                linked[start:stop] = True
    activity = np.where(supported, 0, -1)
    for start, stop in runs(linked):
        if positive[start:stop].sum() >= 14:
            selected = np.flatnonzero(positive[start:stop]) + start
            activity[selected] = 1
    return activity


def probe(data, encoding, dq, paired, out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "protocol.json",
        {
            "scope": "Exposed24 validation diagnostic, not a new contribution or recalibrated comparison",
            "change": "Only count positive evidence across<=7 raw missing days; minimum14 observed positives; no negative gap, no filled labels, same saved thresholds/models/support/stages",
            "fairness": "Apply to all7DQ+9PP learned probability outputs, not just the proposed method; any useful correction must enter future strong baselines too",
            "decision_use": "Diagnose source of event fragmentation; no rescue of stopped versions and no reserved labels",
            "new_training": False,
            "new_api_calls": 0,
            "test_truth_opened": False,
            "upstream": {
                str(folder): sha(folder / manifest)
                for folder, manifest in (
                    (data, "data-freeze.json"),
                    (encoding, "encoding-freeze.json"),
                    (dq, "calibration-freeze.json"),
                    (paired, "freeze.json"),
                )
            },
        },
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    ids = read_json(encoding / "protocol.json")["split"]["validation"]
    labels = {cid: [] for cid in ids}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(dq / "stage-xgboost.json")
    banks = {}
    for cid in ids:
        with np.load(encoding / "validation" / (cid + ".npz")) as archive:
            banks[cid] = {key: archive[key] for key in archive.files}
    predictions = {}
    for prefix, folder in (("dq", dq), ("paired", paired)):
        probabilities = read_json(folder / "probabilities.json")
        choices = read_json(folder / "selection.json")
        old = read_json(folder / "predictions.json")
        for name, cases in probabilities.items():
            key = prefix + ":" + name
            predictions[key + ":strict"] = old[name]
            predictions[key + ":raw_missing_link"] = {}
            for cid in ids:
                probability = cases[cid]["activity"] if prefix == "dq" else cases[cid]
                bank = banks[cid]
                activity = duration_across_raw_missing(
                    probability, choices[name]["threshold"], bank["supported"], bank["observed"]
                )
                codes = stage.predict(bank["stage_matrix"]).astype(int) + 1
                predictions[key + ":raw_missing_link"][cid] = learned_stages(
                    codes, activity, bank["stage_supported"]
                ).to_dict()
    write_json(out / "predictions.json", predictions)
    observed = {cid: banks[cid]["observed"] for cid in ids}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in ids}
    results, summary, per_case = score_target(labels, observed, weak, predictions, {}, {})
    write_json(out / "results.json", results)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", per_case)
    table = {row["method"]: row for row in summary}
    deltas = []
    for key in predictions:
        if not key.endswith(":strict"):
            continue
        other = key.removesuffix(":strict") + ":raw_missing_link"
        before, after = table[key], table[other]
        deltas.append(
            {
                "method": key.removesuffix(":strict"),
                "strict_f1": before["f1"],
                "linked_f1": after["f1"],
                "strict_event_f1": before["event_f1_iou03"],
                "linked_event_f1": after["event_f1_iou03"],
                "tp_added": after["tp"] - before["tp"],
                "fp_added": after["fp"] - before["fp"],
            }
        )
    write_csv(out / "paired-change.csv", deltas)
    write_json(
        out / "freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print(
        [
            row
            for row in deltas
            if row["method"]
            in ("dq:xgb_raw84", "paired:visual_numeric_pair", "paired:xgb_raw84_augmentation")
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoding", "dq", "paired", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    probe(
        args.data.resolve(),
        args.encoding.resolve(),
        args.dq.resolve(),
        args.paired.resolve(),
        args.out.resolve(),
    )
