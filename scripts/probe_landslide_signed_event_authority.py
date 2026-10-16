"""Freeze a signed candidate-event authority experiment on all cached development records."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT, write_csv
from probe_landslide_revision_scope import conflict_neighbourhood, target_scopes
from run_landslide_daily_authority_expansion import verify_prepared
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import ReviewPolicy, combine_reviews, raw_motion_evidence
from gnss_sim.landslide_evaluation import aggregate_scores
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_observed_duration import activity_across_missing
from gnss_sim.landslide_partial_review import combine_partial_reviews
from gnss_sim.landslide_signed_event_revision import (
    candidate_event_authority,
    revise_with_signed_authority,
)
from gnss_sim.landslide_visual import parse_activity
from gnss_sim.visual import _unique_object

VARIANTS = ("signed_event", "whole_event", "signed_no_event", "signed_center", "signed_strict", "signed_no_local")


def outputs_for_record(case, features, initial, reviews, targets, policy, masks, codes):
    evidence = raw_motion_evidence(case, features)
    conflict = conflict_neighbourhood(initial, features, evidence, policy)[0]
    daily = target_scopes(conflict, features.observed, targets)[0]
    partial, retained = combine_partial_reviews(initial, reviews, features.observed)
    strict = combine_reviews(initial, reviews, features.observed)
    additions, unfiltered, centered, coverage, details = candidate_event_authority(
        case, features.noise, initial, partial.activity, daily, features.observed, targets,
    )
    # The strict arm uses the same precomputed definite candidate authority;
    # only successful abstention handling changes.
    configurations = (
        ("signed_event", partial.activity, daily, additions),
        ("whole_event", partial.activity, coverage, additions),
        ("signed_no_event", partial.activity, daily, unfiltered),
        ("signed_center", partial.activity, daily, centered),
        ("signed_strict", strict.activity, daily, additions),
    )
    baseline = activity_across_missing(initial == 1, masks["activity"] & (initial >= 0), features.observed)
    outputs, audit = {}, {}
    for name, local, withdrawals, positive in configurations:
        raw, edits = revise_with_signed_authority(initial, local, features.observed, withdrawals, positive)
        permission = withdrawals | positive
        if np.any((raw != initial) & ~permission):
            raise ValueError("Signed pre-decoder revision escaped authority")
        activity = activity_across_missing(raw == 1, masks["activity"] & (raw >= 0), features.observed)
        outputs[name] = learned_stages(codes, activity, masks["stage"]).to_dict()
        audit[name] = dict(
            editable_days=int(permission.sum()), withdrawal_permission_days=int(withdrawals.sum()),
            addition_permission_days=int(positive.sum()), predecoder_changes=int(edits.sum()),
            retained_successful_abstentions=int((retained & withdrawals).sum()),
            propagated_outside_permission=int(((activity != baseline) & ~permission).sum()),
        )
    numerical = initial.copy()
    numerical[daily] = np.where(initial[daily] == 1, 0, 1)
    activity = activity_across_missing(numerical == 1, masks["activity"] & (numerical >= 0), features.observed)
    outputs["signed_no_local"] = learned_stages(codes, activity, masks["stage"]).to_dict()
    audit["signed_no_local"] = dict(
        editable_days=int(daily.sum()), withdrawal_permission_days=int(daily.sum()),
        addition_permission_days=int((daily & (initial != 1)).sum()),
        predecoder_changes=int((numerical != initial).sum()), retained_successful_abstentions=0,
        propagated_outside_permission=int(((activity != baseline) & ~daily).sum()),
    )
    details["daily_permission_days"] = int(daily.sum())
    return outputs, audit, details


def probe(source, replay, out):
    protocol = verify_prepared(source)
    verify_manifest(source, "prediction-freeze.json")
    verify_manifest(source / "analysis-v1", "freeze.json")
    verify_manifest(replay, "freeze.json")
    receipt = read_json(replay / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["analysis_manifest_sha256"] != sha(source / "analysis-v1/freeze.json"):
        raise ValueError("Matching completed full-development replay required")
    out.mkdir(parents=True, exist_ok=False)
    paths = (Path(__file__), ROOT / "src/gnss_sim/landslide_signed_event_revision.py",
             ROOT / "src/gnss_sim/landslide_event_motion_protection.py",
             ROOT / "src/gnss_sim/landslide_partial_review.py", ROOT / "src/gnss_sim/landslide_observed_duration.py",
             ROOT / "scripts/score_landslide_claim_review.py", ROOT / "tests/test_landslide_signed_event_revision.py")
    experiment = dict(
        name="SERA / Signed Event Revision Authority;development candidate",
        registered_utc=datetime.now(timezone.utc).isoformat(), principal="signed_event;not selected after scoring",
        source=str(source), source_manifest_sha256=sha(source / "prediction-freeze.json"),
        replay_manifest_sha256=sha(replay / "freeze.json"), variants=VARIANTS,
        scope="Same48TRAIN/24exposedVALIDATION;two retained cached replies;no independent records",
        mechanism="Withdrawals retain exact daily conflict authority;new activity needs a local-positive candidate event with at least one novel conflict seed and existing accumulated ordered non-concentrated raw movement support",
        prior_art="Existing event support formerly protects INITIAL positive denials;existing daily double evidence restricts additions. Neither routine is claimed novel. New test is directional authority with support evaluated inside LOCAL proposals",
        evidence="Reuse14day/8observed medians;6blocks;net/noise2,rate.03,Kendall.6,3positive changes,max share.5;no refit or threshold sweep",
        controls="Wholewithdrawals with same event additions;noevent restriction;matched-center addition day budget;remove abstention;remove localVLM through daily numerical flip;retain36original methods",
        continuation="Both repeats:main dayF1 and positive net correction above olddaily/wholeevent/noevent/center/nolocal;eventIoU.3>=olddaily and wholepartial;eventIoU.5>=old49neighborhood;pure nuisanceFP<=olddaily;mixed recall>=olddaily",
        failure="Any failure stops fixed candidate;no threshold/event-length/rank/seed/prompt rescue and no reserved evaluation",
        costs="No new API;event evidence overhead timed separately;current full48TRAIN stage head and decoder unchanged",
        source_bindings={str(p): sha(p) for p in paths}, reserved_content_loaded=False,
        independent_validation=False, new_api_calls=0,
    )
    write_json(out / "protocol.json", experiment)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "pre-probe-freeze.json", {p.name: sha(p) for p in out.iterdir()})
    predictions = read_json(Path(protocol["parent"]) / "predictions.json")
    predictions.update(read_json(source / "predictions.json"))
    used = read_json(source / "used.json")
    selections = read_json(source / "selections.json")
    windows = read_json(source / "window-outputs.json")
    codes, masks = read_json(source / "stage-codes.json"), read_json(source / "support-masks.json")
    receipts = {}
    for folder in (Path(protocol["cached_source"]), source):
        receipts.update({p.name.removesuffix(".response.json"): read_json(p)
                         for p in (folder / "requests").glob("*.response.json")})
    policy = ReviewPolicy(**protocol["policy"])
    data = Path(protocol["data"])
    details_saved, audits, timings = {}, [], []
    for rep in range(2):
        for cid in protocol["ids"]:
            case, features = load_case_features(data, cid)
            key = f"{cid}-r{rep}"
            record = receipts[key + "-global"]
            if record["status"] != "returned_json":
                initial = np.full(len(case.dates), -1)
            else:
                try:
                    initial = parse_activity(json.loads(record["output"], object_pairs_hook=_unique_object), features.observed)
                except ValueError:
                    initial = np.full(len(case.dates), -1)
            selected = selections[key]["visual_selected"]
            targets = [selections[key]["visual_priorities"][index] for index in selected]
            reviews = [windows[key][str(index)] for index in selected]
            started = time.perf_counter()
            outputs, audit, details = outputs_for_record(
                case, features, initial, reviews, targets, policy,
                {name: np.array(value) for name, value in masks[cid].items()}, np.array(codes[cid]),
            )
            timings.append(dict(case_id=cid, repetition=rep, seconds=time.perf_counter() - started))
            details_saved[key] = details
            for variant, value in outputs.items():
                method = f"sera_{variant}_r{rep}"
                predictions.setdefault(method, {})[cid] = value
                used.setdefault(method, {})[cid] = [key + "-global"] if variant == "signed_no_local" else used[f"adaptive_visual_r{rep}"][cid]
                audits.append(dict(method=method, case_id=cid, **audit[variant]))
    for name, value in (("predictions", predictions), ("used", used), ("event-authority", details_saved)):
        write_json(out / (name + ".json"), value)
    write_csv(out / "authority-audit.csv", audits)
    write_csv(out / "timings.csv", timings)
    write_json(out / "prediction-freeze.json", {p.name: sha(p) for p in out.iterdir()})
    labels = {cid: [] for cid in protocol["ids"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    observed = {cid: load_case_features(data, cid)[1].observed for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in values]) for cid, values in labels.items()}
    results, summary, per_case = score_target(labels, observed, weak, predictions, used, receipts)
    for name, value in (("results", results), ("summary", summary)):
        write_json(out / (name + ".json"), value)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", per_case)
    rows = {row["method"]: row for row in summary}
    families = {row["case_id"]: row["truth"]["family"]
                for row in read_json(data / "training-generation-ledger.json") if row["case_id"] in labels}
    negatives = [cid for cid, family in families.items() if family in {"quiet", "step_only", "spike_only"}]
    mixed = [cid for cid, family in families.items() if family in {"deformation_step", "deformation_spike"}]
    family_rows = []
    for method in predictions:
        for family in sorted(set(families.values())):
            subset = aggregate_scores([results[method]["per_case"][cid] for cid in labels if families[cid] == family])
            family_rows.append(dict(method=method, family=family, **subset["daily"], event_f1=subset["events"]["f1"]))
    write_csv(out / "family-summary.csv", family_rows)
    checks, nets = {}, {}
    for rep in range(2):
        main = f"sera_signed_event_r{rep}"
        old = f"visual_daily_partial_r{rep}"
        controls = [old] + [f"sera_{variant}_r{rep}" for variant in
                           ("whole_event", "signed_no_event", "signed_center", "signed_no_local")]
        candidate = rows[main]
        nets[str(rep)] = {name: rows[name]["fp"] + rows[name]["fn"] - candidate["fp"] - candidate["fn"] for name in controls}
        checks[str(rep)] = dict(
            daily_gain=candidate["f1"] > max(rows[name]["f1"] for name in controls),
            positive_net=min(nets[str(rep)].values()) > 0,
            event03_nonloss=candidate["event_f1_iou03"] >= max(rows[old]["event_f1_iou03"], rows[f"visual_whole_partial_r{rep}"]["event_f1_iou03"]),
            event05_recovered=candidate["event_f1_iou05"] >= rows[f"visual_neighborhood_partial_r{rep}"]["event_f1_iou05"],
            nuisance_nonloss=aggregate_scores([results[main]["per_case"][cid] for cid in negatives])["daily"]["fp"]
                             <= aggregate_scores([results[old]["per_case"][cid] for cid in negatives])["daily"]["fp"],
            mixed_nonloss=aggregate_scores([results[main]["per_case"][cid] for cid in mixed])["daily"]["recall"]
                          >= aggregate_scores([results[old]["per_case"][cid] for cid in mixed])["daily"]["recall"],
        )
    passed = all(value for repeat in checks.values() for value in repeat.values())
    write_json(out / "decision.json", dict(
        status="SERA_NEW_RECORD_CONFIRMATION_PENDING_REPLAY" if passed else "STOP_SIGNED_EVENT_AUTHORITY",
        checks=checks, net_correct=nets, methods=len(predictions), outputs=len(predictions) * len(labels),
        new_api_calls=0, new_records=0, reserved_content_loaded=False, stable_contribution_established=False,
    ))
    for path, expected in experiment["source_bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Signed-authority source changed")
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir()})
    print(read_json(out / "decision.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "replay", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    probe(args.source.resolve(), args.replay.resolve(), args.out.resolve())
