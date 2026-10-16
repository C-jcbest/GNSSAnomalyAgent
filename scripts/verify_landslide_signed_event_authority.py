"""Reconstruct signed permissions independently of the candidate implementation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from run_landslide_daily_authority_expansion import verify_prepared
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import ReviewPolicy, raw_motion_evidence
from gnss_sim.landslide_evaluation import event_masks
from gnss_sim.landslide_event_motion_protection import event_motion_support
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_observed_duration import activity_across_missing
from gnss_sim.landslide_visual import parse_activity
from gnss_sim.visual import _unique_object


def reconstruct(case, features, initial, reviews, targets, policy, codes, masks):
    observed = features.observed
    evidence = raw_motion_evidence(case, features)
    short = np.isfinite(features.progressive_rate) & observed
    long = np.isfinite(evidence["rate"]) & observed
    moving = short & (features.progressive_rate >= policy.minimum_rate)
    moving &= features.signal_to_noise >= policy.minimum_signal_to_noise
    moving |= long & (evidence["rate"] >= policy.minimum_rate) & (
        evidence["signal_to_noise"] >= policy.minimum_signal_to_noise
    )
    conflict = (short | long) & (moving != (initial == 1))
    target_mask = np.zeros(len(initial), dtype=bool)
    for target in targets:
        target_mask[target["start"]:target["stop"]] = True
    daily = conflict & target_mask
    coverage = np.zeros(len(initial), dtype=int)
    positive, negative, failed = coverage.copy(), coverage.copy(), coverage.copy()
    for review in reviews:
        start, stop = review["start"], review["stop"]
        answer = np.array(review["activity"])
        coverage[start:stop] += 1
        positive[start:stop] += answer[start:stop] == 1
        negative[start:stop] += answer[start:stop] == 0
        if review["status"] == "failed":
            failed[start:stop] += 1
    strict = initial.copy()
    strict[coverage > 0] = -1
    strict[positive * 2 > coverage] = 1
    strict[negative * 2 > coverage] = 0
    strict[~observed] = -1
    retained = observed & (coverage > 0) & (positive + negative == 0) & (failed == 0)
    local = strict.copy()
    local[retained] = initial[retained]
    proposal = np.where(observed, 0, -1)
    proposal[target_mask & observed] = local[target_mask & observed]
    # Reuse the frozen pre-existing evidence routine. This replay independently
    # verifies its new ownership/seed/direction logic, not its physical validity.
    support, details = event_motion_support(case, features.noise, proposal)
    seed = daily & (initial != 1) & (proposal == 1)
    seeded = np.zeros(len(initial), dtype=bool)
    for event in event_masks(proposal):
        if np.any(event & seed):
            seeded[event] = True
    novel = target_mask & observed & (initial != 1) & (local == 1)
    additions = novel & support & seeded
    centered = np.zeros(len(initial), dtype=bool)
    for target in targets:
        start, stop = target["start"], target["stop"]
        available = np.flatnonzero(novel[start:stop]) + start
        ordered = sorted(available, key=lambda day: (abs(day - (start + stop - 1) / 2), day))
        centered[ordered[:int(additions[start:stop].sum())]] = True
    diagnostics = dict(
        proposed_addition_days=int(novel.sum()), seeded_event_days=int((novel & seeded).sum()),
        supported_addition_days=int(additions.sum()), centered_addition_days=int(centered.sum()),
        event_checks=details, daily_permission_days=int(daily.sum()),
    )
    configurations = (
        ("signed_event", local, daily, additions),
        ("whole_event", local, target_mask & observed, additions),
        ("signed_no_event", local, daily, novel),
        ("signed_center", local, daily, centered),
        ("signed_strict", strict, daily, additions),
    )
    outputs, audits = {}, {}
    baseline = activity_across_missing(initial == 1, masks["activity"] & (initial >= 0), observed)
    for name, answer, withdrawals, positive_scope in configurations:
        raw = initial.copy()
        withdrawal = withdrawals & (initial == 1) & (answer != 1)
        unresolved = withdrawals & (initial != 1) & (answer == -1)
        addition = positive_scope & (initial != 1) & (answer == 1)
        raw[withdrawal | unresolved] = answer[withdrawal | unresolved]
        raw[addition] = 1
        permission = withdrawals | positive_scope
        if np.any((raw != initial) & ~permission) or np.any(raw[~observed] != -1):
            raise ValueError("Independent pre-decoder authority violation")
        activity = activity_across_missing(raw == 1, masks["activity"] & (raw >= 0), observed)
        outputs[name] = learned_stages(codes, activity, masks["stage"]).to_dict()
        audits[name] = dict(
            editable_days=int(permission.sum()), withdrawal_permission_days=int(withdrawals.sum()),
            addition_permission_days=int(positive_scope.sum()),
            predecoder_changes=int((withdrawal | unresolved | addition).sum()),
            retained_successful_abstentions=int((retained & withdrawals).sum()),
            propagated_outside_permission=int(((activity != baseline) & ~permission).sum()),
        )
    raw = initial.copy()
    raw[daily] = np.where(initial[daily] == 1, 0, 1)
    activity = activity_across_missing(raw == 1, masks["activity"] & (raw >= 0), observed)
    outputs["signed_no_local"] = learned_stages(codes, activity, masks["stage"]).to_dict()
    audits["signed_no_local"] = dict(
        editable_days=int(daily.sum()), withdrawal_permission_days=int(daily.sum()),
        addition_permission_days=int((daily & (initial != 1)).sum()),
        predecoder_changes=int((raw != initial).sum()), retained_successful_abstentions=0,
        propagated_outside_permission=int(((activity != baseline) & ~daily).sum()),
    )
    return outputs, audits, diagnostics


def verify(run, replay, out):
    verify_manifest(run, "freeze.json")
    protocol = read_json(run / "protocol.json")
    for path, expected in protocol["source_bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Frozen mechanism changed: " + path)
    source = Path(protocol["source"])
    upstream = verify_prepared(source)
    verify_manifest(replay, "freeze.json")
    if sha(replay / "freeze.json") != protocol["replay_manifest_sha256"]:
        raise ValueError("Upstream replay identity differs")
    if read_json(replay / "verification.json")["status"] != "VERIFIED":
        raise ValueError("Verified original payloads required")
    predictions = read_json(Path(upstream["parent"]) / "predictions.json")
    predictions.update(read_json(source / "predictions.json"))
    used = read_json(source / "used.json")
    windows, selections = read_json(source / "window-outputs.json"), read_json(source / "selections.json")
    codes, masks = read_json(source / "stage-codes.json"), read_json(source / "support-masks.json")
    receipts = {}
    for folder in (Path(upstream["cached_source"]), source):
        for path in (folder / "requests").glob("*.response.json"):
            receipts[path.name.removesuffix(".response.json")] = read_json(path)
    audit_rows, details = [], {}
    policy = ReviewPolicy(**upstream["policy"])
    data = Path(upstream["data"])
    for rep in range(2):
        for cid in upstream["ids"]:
            case, features = load_case_features(data, cid)
            key = f"{cid}-r{rep}"
            receipt = receipts[key + "-global"]
            initial = np.full(len(case.dates), -1)
            if receipt["status"] == "returned_json":
                try:
                    initial = parse_activity(json.loads(receipt["output"], object_pairs_hook=_unique_object), features.observed)
                except ValueError:
                    pass
            selected = selections[key]["visual_selected"]
            targets = [selections[key]["visual_priorities"][index] for index in selected]
            reviews = [windows[key][str(index)] for index in selected]
            outputs, audits, details[key] = reconstruct(
                case, features, initial, reviews, targets, policy, np.array(codes[cid]),
                {name: np.array(value) for name, value in masks[cid].items()},
            )
            for name, value in outputs.items():
                method = f"sera_{name}_r{rep}"
                predictions.setdefault(method, {})[cid] = value
                used.setdefault(method, {})[cid] = (
                    [key + "-global"] if name == "signed_no_local" else used[f"adaptive_visual_r{rep}"][cid]
                )
                audit_rows.append(dict(method=method, case_id=cid, **audits[name]))
    if predictions != read_json(run / "predictions.json") or used != read_json(run / "used.json"):
        raise ValueError("Reconstructed outputs/request use differ")
    if details != read_json(run / "event-authority.json"):
        raise ValueError("Candidate evidence/seed diagnostics differ")
    with (run / "authority-audit.csv").open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    if [{key: str(value) for key, value in row.items()} for row in audit_rows] != saved:
        raise ValueError("Independent directional authority audit differs")
    labels = {cid: [] for cid in upstream["ids"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    observed = {cid: load_case_features(data, cid)[1].observed for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in values]) for cid, values in labels.items()}
    results, summary, _ = score_target(labels, observed, weak, predictions, used, receipts)
    if results != read_json(run / "results.json") or summary != read_json(run / "summary.json"):
        raise ValueError("Full unified scores differ")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "verification.json", dict(
        status="VERIFIED", methods=len(predictions), outputs=len(predictions) * len(labels),
        exact_output_authority_seed_center_abstention_decoder_stage_score=True,
        event_evidence="Existing frozen helper reused;authority independently reconstructed",
        original_request_binding="Inherited completed upstream exact payload/image replay",
        new_api_calls=0, reserved_content_loaded=False, independent_effectiveness_established=False,
        probe_manifest_sha256=sha(run / "freeze.json"),
    ))
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "freeze.json", {path.name: sha(path) for path in out.iterdir()})
    print(read_json(out / "verification.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--upstream-replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    verify(arguments.run.resolve(), arguments.upstream_replay.resolve(), arguments.out.resolve())
