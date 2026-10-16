"""Diagnose common duration filtering on all frozen deformation methods."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_baselines import pelt_segment_rates
from gnss_sim.landslide_evaluation import aggregate_scores
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_observed_duration import activity_across_missing


def probe(data, encoded, run, out):
    verify_manifest(run, "freeze.json")
    protocol = read_json(run / "protocol.json")
    split = protocol["split"]
    banks = load_banks(encoded, split)
    choices = read_json(run / "selection.json")
    probability = read_json(run / "probabilities.json")
    labels = {cid: [] for cid in split["validation"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(run / "stage-xgboost.json")
    codes = {cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1 for cid in labels}
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in labels[cid]]) for cid in labels}
    families = {
        row["case_id"]: row["truth"]["family"]
        for row in read_json(data / "training-generation-ledger.json")
        if row["case_id"] in labels
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "protocol.json",
        dict(
            parent_manifest_sha256=sha(run / "freeze.json"),
            minimum_observations=[1, 7, 14],
            no_retraining=True,
            no_threshold_selection=True,
            scope="post-score common decoder diagnostic on exposed24; not method confirmation",
            all_methods=True,
            test_truth_opened=False,
        ),
    )
    predictions, support = {}, {}
    for name, choice in choices.items():
        support[name] = {}
        for cid in labels:
            bank = banks[cid]
            if name == "xgb_raw12":
                available = bank["stage_supported"]
            elif name in {"robust_rule", "pelt_rule"}:
                feature = load_case_features(data, cid)[1]
                available = feature.supported
                if name == "pelt_rule":
                    available = available & np.isfinite(
                        pelt_segment_rates(feature, choice["parameter"])
                    )
            else:
                available = bank["supported"]
            support[name][cid] = available
        for minimum in (1, 7, 14):
            method = f"{name}__minimum{minimum}"
            predictions[method] = {}
            for cid in labels:
                bank = banks[cid]
                mask = np.array(probability[name][cid]) >= choice["threshold"]
                activity = activity_across_missing(
                    mask, support[name][cid], bank["observed"], minimum_observations=minimum
                )
                predictions[method][cid] = learned_stages(
                    codes[cid], activity, bank["stage_supported"]
                ).to_dict()
    results, summary, _ = score_target(labels, observed, weak, predictions, {}, {})
    for name in choices:
        if predictions[name + "__minimum14"] != read_json(run / "predictions.json")[name]:
            raise ValueError("Common14 diagnostic did not replay original activity/stage outputs")
    write_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)
    rows = []
    for name in predictions:
        for family in sorted(set(families.values())):
            score = aggregate_scores(
                [results[name]["per_case"][cid] for cid in labels if families[cid] == family]
            )
            rows.append(
                dict(method=name, family=family, **score["daily"], event_f1=score["events"]["f1"])
            )
    write_csv(out / "family-summary.csv", rows)
    write_json(out / "predictions.json", predictions)
    write_json(out / "results.json", results)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(
        [
            {key: row[key] for key in ("method", "f1", "event_f1_iou03", "fp", "fn")}
            for row in summary
            if row["method"].startswith(
                ("visual_numeric__", "xgb_raw84__", "raw_evidence_review__")
            )
        ],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    probe(args.data.resolve(), args.encoded.resolve(), args.run.resolve(), args.out.resolve())
