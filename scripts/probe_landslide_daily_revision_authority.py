"""Test original-day revision authority using frozen local replies and old controls."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT, write_csv
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_diversity_local_review import NUMERIC, verify_sources
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import ReviewPolicy, combine_reviews, raw_motion_evidence
from gnss_sim.landslide_evaluation import aggregate_scores
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_observed_duration import activity_across_missing
from gnss_sim.landslide_partial_review import combine_partial_reviews
from gnss_sim.landslide_visual import parse_activity

VARIANTS = ("whole_partial", "neighborhood_partial", "daily_partial", "matched_partial",
            "daily_strict", "no_local_visual")


def authority_outputs(initial, reviews, observed, support, stage_support, codes, conflict, neighborhood, targets):
    """Reuse old abstention and scope rules, isolating the original-day permission.

    The numerical conflict is a routing descriptor, not evidence of true motion
    or stationarity. The principal arm requires a definite successful local
    answer. Failed observations stay unknown only within permitted dates.
    A common duration decoder can remove a short fragment outside permission;
    those propagated changes are separately reported rather than concealed.
    """
    strict = combine_reviews(initial, reviews, observed)
    partial, retained = combine_partial_reviews(initial, reviews, observed)
    daily, matched, details = target_scopes(conflict, observed, targets)
    expanded = target_scopes(neighborhood, observed, targets)[0]
    raw_only = initial.copy()
    raw_only[daily] = np.where(initial[daily] == 1, 0, 1)
    raw_outputs = dict(
        whole_partial=partial.activity,
        neighborhood_partial=scoped_activity(initial, partial.activity, expanded, observed),
        daily_partial=scoped_activity(initial, partial.activity, daily, observed),
        matched_partial=scoped_activity(initial, partial.activity, matched, observed),
        daily_strict=scoped_activity(initial, strict.activity, daily, observed),
        no_local_visual=raw_only,
    )
    outputs, diagnostics = {}, {}
    permissions = dict(whole_partial=observed, neighborhood_partial=expanded,
                       daily_partial=daily, matched_partial=matched, daily_strict=daily,
                       no_local_visual=daily)
    for name, raw in raw_outputs.items():
        permission = permissions[name]
        if np.any(raw[~permission] != initial[~permission]):
            raise ValueError("Pre-decoder edit escaped its authority")
        activity = activity_across_missing(raw == 1, support & (raw >= 0), observed)
        outputs[name] = learned_stages(codes, activity, stage_support).to_dict()
        baseline = activity_across_missing(initial == 1, support & (initial >= 0), observed)
        diagnostics[name] = dict(
            editable_days=int(permission.sum()), predecoder_changes=int((raw != initial).sum()),
            propagated_outside_permission=int(((activity != baseline) & ~permission).sum()),
            retained_successful_abstentions=int((retained & permission).sum()),
        )
    return outputs, diagnostics, details


def probe(source, verified, out):
    protocol = verify_sources(source)
    verify_manifest(source, "prediction-freeze.json")
    verify_manifest(source / "analysis-v1", "freeze.json")
    verify_manifest(verified, "freeze.json")
    receipt = read_json(verified / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["prediction_manifest_sha256"] != sha(source / "prediction-freeze.json"):
        raise ValueError("A matching frozen local-review replay is required")
    out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), ROOT / "src/gnss_sim/landslide_partial_review.py",
                    ROOT / "scripts/probe_landslide_revision_scope.py",
                    ROOT / "src/gnss_sim/landslide_observed_duration.py",
                    ROOT / "scripts/score_landslide_claim_review.py"]
    experiment = dict(
        name="RCRA / Raw-Conflict Revision Authority;original-day candidate",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        scope="Same eight-family16TRAIN/8exposedVALIDATION;development mechanism probe;no independent record",
        source=str(source), source_manifest_sha256=sha(source / "prediction-freeze.json"),
        source_replay_sha256=sha(verified / "freeze.json"), variants=VARIANTS,
        principal="daily_partial;two prior modes tested separately;neither is renamed after seeing results",
        mechanism="Only exact supported raw-conflict dates within original targets may receive local edits;no +/-49dilation;old successful-abstention retention reused",
        controls="Whole partial;old49day neighborhood;equal-observed-day centered permission;remove abstention retention;remove localVLM and invert the conflict prior numerically",
        comparison="Retain40original methods;all reuse same answers,stage/support masks14/7decoder;no fitting or new thresholds",
        visual_continuation="Both repeats:dailyF1 strictly above direct/strict/wholepartial/49neighborhood/matched/nolocal controls;net correction positive vs all;event nonloss;pure nuisanceFP no increase",
        numeric_continuation="Both repeats:same gain/net/event tests including unchanged strong numeric prior;pure nuisanceFP no increase;mixed recall no loss",
        failure="Any failed condition stops that fixed prior mode;no margin,policy,prompt,threshold or repetition rescue",
        propagation="Pre-decoder permission exact;common14observation decoding can suppress new short fragments outside it;report every such date count",
        source_bindings={str(p): sha(p) for p in source_paths}, new_api_calls=0,
        reserved_content_loaded=False,
    )
    write_json(out / "protocol.json", experiment)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "pre-probe-freeze.json", {p.name: sha(p) for p in out.iterdir()})
    policy = ReviewPolicy(**protocol["policy"])
    data = Path(protocol["data"])
    numeric = read_json(source / "numeric-initial.json")
    codes, masks = read_json(source / "stage-codes.json"), read_json(source / "support-masks.json")
    selection = read_json(source / "selections.json")
    window_outputs = read_json(source / "window-outputs.json")
    original_used = read_json(source / "used.json")
    receipts = {p.name.removesuffix(".response.json"): read_json(p)
                for p in (source / "requests").glob("*.response.json")}
    predictions = read_json(Path(protocol["parent"]) / "predictions.json")
    predictions.update(read_json(source / "predictions.json"))
    used, audits, scopes = dict(original_used), [], {}
    for rep in range(protocol["repetitions"]):
        for cid in protocol["ids"]:
            case, features = load_case_features(data, cid)
            long_evidence = raw_motion_evidence(case, features)
            key = f"{cid}-r{rep}"
            global_receipt = receipts[key + "-global"]
            if global_receipt["status"] != "returned_json":
                visual_initial = np.full(len(case.dates), -1)
            else:
                try:
                    visual_initial = parse_activity(json.loads(global_receipt["output"]), features.observed)
                except ValueError:
                    visual_initial = np.full(len(case.dates), -1)
            for mode, initial in (("visual", visual_initial), ("numeric", np.array(numeric[cid]["activity"]))):
                conflict, neighborhood = conflict_neighbourhood(initial, features, long_evidence, policy)
                chosen = selection[key][mode + "_selected"]
                priorities = selection[key][mode + "_priorities"]
                targets = [priorities[index] for index in chosen]
                reviews = [window_outputs[key][str(index)] for index in chosen]
                outputs, diagnostics, details = authority_outputs(
                    initial, reviews, features.observed, np.array(masks[cid]["activity"]),
                    np.array(masks[cid]["stage"]), np.array(codes[cid]), conflict, neighborhood, targets,
                )
                scopes[mode + ":" + key] = details
                for variant, value in outputs.items():
                    name = f"{mode}_{variant}_r{rep}"
                    predictions.setdefault(name, {})[cid] = value
                    if variant == "no_local_visual":
                        requests = [key + "-global"] if mode == "visual" else []
                    else:
                        requests = original_used[f"adaptive_{mode}_r{rep}"][cid]
                    used.setdefault(name, {})[cid] = requests
                    audits.append(dict(method=name, case_id=cid, **diagnostics[variant]))
    # Labels are read only after every candidate/control output is formed.
    labels = {cid: [] for cid in protocol["ids"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    observed = {cid: load_case_features(data, cid)[1].observed for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in rows]) for cid, rows in labels.items()}
    results, summary, per_case = score_target(labels, observed, weak, predictions, used, receipts)
    for name, value in (("predictions", predictions), ("results", results), ("summary", summary),
                        ("used", used), ("scopes", scopes)):
        write_json(out / (name + ".json"), value)
    for name, value in (("summary", summary), ("per-case", per_case), ("authority-audit", audits)):
        write_csv(out / (name + ".csv"), value)
    rows = {row["method"]: row for row in summary}
    families = {row["case_id"]: row["truth"]["family"]
                for row in read_json(data / "training-generation-ledger.json") if row["case_id"] in labels}
    negatives = [cid for cid, family in families.items() if family in {"quiet", "step_only", "spike_only"}]
    mixed = [cid for cid, family in families.items() if family in {"deformation_step", "deformation_spike"}]

    def subset(method, ids):
        return aggregate_scores([results[method]["per_case"][cid] for cid in ids])["daily"]

    decisions = {}
    for mode in ("visual", "numeric"):
        checks, net = {}, {}
        for rep in range(2):
            main = f"{mode}_daily_partial_r{rep}"
            prior = f"direct_visual_r{rep}" if mode == "visual" else NUMERIC
            controls = [prior, f"adaptive_{mode}_r{rep}"] + [
                f"{mode}_{variant}_r{rep}" for variant in
                ("whole_partial", "neighborhood_partial", "matched_partial", "no_local_visual")
            ]
            candidate = rows[main]
            net[str(rep)] = {name: rows[name]["fp"] + rows[name]["fn"] - candidate["fp"] - candidate["fn"]
                             for name in controls}
            checks[str(rep)] = dict(
                daily_gain=candidate["f1"] > max(rows[name]["f1"] for name in controls),
                positive_net=min(net[str(rep)].values()) > 0,
                event_nonloss=candidate["event_f1_iou03"] >= max(rows[name]["event_f1_iou03"] for name in controls),
                nuisance_nonloss=subset(main, negatives)["fp"] <= subset(prior, negatives)["fp"],
                mixed_nonloss=subset(main, mixed)["recall"] >= subset(prior, mixed)["recall"],
            )
        decisions[mode] = dict(checks=checks, net_correct=net,
                               status="DAILY_AUTHORITY_EXPANSION_PENDING_REPLAY" if all(value for repeat in checks.values() for value in repeat.values())
                               else "STOP_DAILY_AUTHORITY_" + mode.upper())
    write_json(out / "decision.json", dict(prior_modes=decisions, methods=len(predictions),
                                          new_api_calls=0, reserved_content_loaded=False,
                                          stable_contribution_established=False))
    for filename, expected in experiment["source_bindings"].items():
        if sha(Path(filename)) != expected:
            raise ValueError("Probe source changed")
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(read_json(out / "decision.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "verified", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    probe(args.source.resolve(), args.verified.resolve(), args.out.resolve())
