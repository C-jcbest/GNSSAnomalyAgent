"""Independently reconstruct the frozen daily-authority probe without API calls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import write_csv
from run_landslide_diversity_local_review import verify_sources
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_evaluation import runs
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_observed_duration import activity_across_missing
from gnss_sim.landslide_visual import parse_activity


def reconstruct(initial, reviews, observed, features, evidence, policy, targets, codes, masks):
    """Rebuild evidence, permission, votes, abstention and propagation separately."""
    valid_short = np.isfinite(features.progressive_rate) & observed
    valid_long = np.isfinite(evidence["rate"]) & observed
    motion_short = (features.progressive_rate >= policy.minimum_rate) & (
        features.signal_to_noise >= policy.minimum_signal_to_noise
    )
    motion_long = (evidence["rate"] >= policy.minimum_rate) & (
        evidence["signal_to_noise"] >= policy.minimum_signal_to_noise
    )
    motion = (valid_short & motion_short) | (valid_long & motion_long)
    conflict = (valid_short | valid_long) & (motion != (initial == 1))
    neighborhood = np.zeros(len(initial), dtype=bool)
    for start, stop in runs(conflict):
        neighborhood[max(0, start - 49):min(len(initial), stop + 49)] = True
    neighborhood &= observed
    daily = np.zeros(len(initial), dtype=bool)
    expanded = daily.copy()
    matched = daily.copy()
    target_days = daily.copy()
    details = []
    for target in targets:
        start, stop = target["start"], target["stop"]
        if target_days[start:stop].any():
            raise ValueError("Overlapping revision targets")
        target_days[start:stop] = True
        daily[start:stop] = conflict[start:stop]
        expanded[start:stop] = neighborhood[start:stop]
        budget = int(conflict[start:stop].sum())
        available = np.flatnonzero(observed[start:stop]) + start
        middle = (start + stop - 1) / 2
        ordered = sorted(available, key=lambda day: (abs(day - middle), day))
        matched[ordered[:budget]] = True
        details.append(dict(start=start, stop=stop, editable_observed_days=budget,
                            target_observed_days=int(observed[start:stop].sum())))

    coverage = np.zeros(len(initial), dtype=int)
    positive = coverage.copy()
    negative = coverage.copy()
    failures = coverage.copy()
    for review in reviews:
        start, stop = review["start"], review["stop"]
        values = np.array(review["activity"])
        coverage[start:stop] += 1
        positive[start:stop] += values[start:stop] == 1
        negative[start:stop] += values[start:stop] == 0
        if review["status"] == "failed":
            if np.any(values != -1):
                raise ValueError("Failure gained a definite answer")
            failures[start:stop] += 1
    strict = initial.copy()
    strict[coverage > 0] = -1
    strict[2 * positive > coverage] = 1
    strict[2 * negative > coverage] = 0
    strict[~observed] = -1
    retained = observed & (coverage > 0) & (positive + negative == 0) & (failures == 0)
    partial = strict.copy()
    partial[retained] = initial[retained]
    baseline = activity_across_missing(initial == 1, masks["activity"] & (initial >= 0), observed)
    outputs, audits = {}, {}
    variants = (
        ("whole_partial", observed, partial),
        ("neighborhood_partial", expanded, partial),
        ("daily_partial", daily, partial),
        ("matched_partial", matched, partial),
        ("daily_strict", daily, strict),
        ("no_local_visual", daily, np.where(initial == 1, 0, 1)),
    )
    for name, permission, answer in variants:
        raw = initial.copy()
        raw[permission] = answer[permission]
        activity = activity_across_missing(raw == 1, masks["activity"] & (raw >= 0), observed)
        outputs[name] = learned_stages(codes, activity, masks["stage"]).to_dict()
        # Preserve the probe's diagnostic definition even for variants that do
        # not apply retention; the report must distinguish available from used.
        audits[name] = dict(editable_days=int(permission.sum()),
                           predecoder_changes=int((raw != initial).sum()),
                           propagated_outside_permission=int(((activity != baseline) & ~permission).sum()),
                           retained_successful_abstentions=int((retained & permission).sum()))
    return outputs, audits, details


def verify(run, out):
    verify_manifest(run, "freeze.json")
    protocol = read_json(run / "protocol.json")
    for path, expected in protocol["source_bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Probe source binding differs: " + path)
    source = Path(protocol["source"])
    parent_protocol = verify_sources(source)
    verify_manifest(source, "prediction-freeze.json")
    data = Path(parent_protocol["data"])
    policy = ReviewPolicy(**parent_protocol["policy"])
    predictions = read_json(Path(parent_protocol["parent"]) / "predictions.json")
    predictions.update(read_json(source / "predictions.json"))
    used = read_json(source / "used.json")
    selection = read_json(source / "selections.json")
    reviews_saved = read_json(source / "window-outputs.json")
    numeric = read_json(source / "numeric-initial.json")
    codes = read_json(source / "stage-codes.json")
    masks_saved = read_json(source / "support-masks.json")
    receipts = {p.name.removesuffix(".response.json"): read_json(p)
                for p in (source / "requests").glob("*.response.json")}
    scopes, audits = {}, []
    for rep in range(parent_protocol["repetitions"]):
        for cid in parent_protocol["ids"]:
            case, features = load_case_features(data, cid)
            evidence = raw_motion_evidence(case, features)
            key = f"{cid}-r{rep}"
            receipt = receipts[key + "-global"]
            if receipt["status"] == "returned_json":
                try:
                    initial_visual = parse_activity(json.loads(receipt["output"]), features.observed)
                except ValueError:
                    initial_visual = np.full(len(case.dates), -1)
            else:
                initial_visual = np.full(len(case.dates), -1)
            masks = {name: np.array(value) for name, value in masks_saved[cid].items()}
            for mode, initial in (("visual", initial_visual), ("numeric", np.array(numeric[cid]["activity"]))):
                selected = selection[key][mode + "_selected"]
                targets = [selection[key][mode + "_priorities"][index] for index in selected]
                reviews = [reviews_saved[key][str(index)] for index in selected]
                outputs, audit, details = reconstruct(
                    initial, reviews, features.observed, features, evidence, policy,
                    targets, np.array(codes[cid]), masks,
                )
                scopes[mode + ":" + key] = details
                for variant, prediction in outputs.items():
                    method = f"{mode}_{variant}_r{rep}"
                    predictions.setdefault(method, {})[cid] = prediction
                    if variant == "no_local_visual":
                        request_ids = [key + "-global"] if mode == "visual" else []
                    else:
                        request_ids = used[f"adaptive_{mode}_r{rep}"][cid]
                    used.setdefault(method, {})[cid] = request_ids
                    audits.append(dict(method=method, case_id=cid, **audit[variant]))
    for name, value in (("predictions", predictions), ("used", used), ("scopes", scopes)):
        if read_json(run / (name + ".json")) != value:
            raise ValueError("Exact reconstruction differs: " + name)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "authority-audit.csv", audits)
    if (out / "authority-audit.csv").read_bytes() != (run / "authority-audit.csv").read_bytes():
        raise ValueError("Propagation or permission audit differs")
    labels = {cid: [] for cid in parent_protocol["ids"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    observed = {cid: load_case_features(data, cid)[1].observed for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in rows]) for cid, rows in labels.items()}
    results, summary, per_case = score_target(labels, observed, weak, predictions, used, receipts)
    for name, value in (("results", results), ("summary", summary)):
        if read_json(run / (name + ".json")) != value:
            raise ValueError("Unified score differs: " + name)
    for name, value in (("summary", summary), ("per-case", per_case)):
        write_csv(out / (name + ".csv"), value)
        if (out / (name + ".csv")).read_bytes() != (run / (name + ".csv")).read_bytes():
            raise ValueError("Score CSV differs: " + name)
    write_json(out / "verification.json", dict(
        status="VERIFIED", methods=len(predictions), outputs=len(predictions) * len(labels),
        exact_conflicts_permissions_center_budgets_votes_abstentions_outputs_scores=True,
        requests_inherited=len(receipts), new_api_calls=0,
        probe_manifest_sha256=sha(run / "freeze.json"), reserved_content_loaded=False,
        independent_scientific_validation=False,
    ))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir()})
    print(read_json(out / "verification.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.out.resolve())
