"""Expand frozen visual-prior RCRA to all exposed development records."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_daily_revision_authority import authority_outputs
from probe_landslide_nuisance_evidence import ROOT, write_csv
from probe_landslide_revision_scope import conflict_neighbourhood
from run_landslide_comparison import save_json
from run_landslide_diversity_local_review import PLOT, assemble
from run_landslide_visual_comparison import render_raw
from run_landslide_window_optimization import load_case_features, read_json, render_window
from score_landslide_claim_review import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    raw_motion_evidence,
    review_priorities,
    review_prompt,
)
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import aggregate_scores
from gnss_sim.landslide_visual import ACTIVITY_TASK, VisualRequests, parse_activity
from gnss_sim.landslide_window_experiment import parse_window_activity
from gnss_sim.visual import _unique_object


def prepare(root, out):
    source = root / "diversity-local-review-v2"
    pilot = root / "daily-authority-probe-v1"
    replay = root / "daily-authority-replay-v1"
    data, encoded = root / "data-v1", root / "quality-encoding-v1"
    parent, stage_source = root / "motion-mass-probe-v1", root / "quality-calibration-v1"
    bindings = ((source, "prediction-freeze.json"), (pilot, "freeze.json"),
                (replay, "freeze.json"), (data, "data-freeze.json"),
                (encoded, "encoding-freeze.json"), (parent, "freeze.json"),
                (stage_source, "calibration-freeze.json"))
    for folder, manifest in bindings:
        verify_manifest(folder, manifest)
    receipt = read_json(replay / "verification.json")
    if receipt["status"] != "VERIFIED" or receipt["probe_manifest_sha256"] != sha(pilot / "freeze.json"):
        raise ValueError("Matching pilot replay required")
    if read_json(pilot / "decision.json")["prior_modes"]["visual"]["status"] != "DAILY_AUTHORITY_EXPANSION_PENDING_REPLAY":
        raise ValueError("Visual-prior pilot must pass its frozen conditions")
    split = read_json(encoded / "protocol.json")["split"]
    if split != read_json(parent / "protocol.json")["split"]:
        raise ValueError("Strong controls require the same full48/24 split")
    old = read_json(source / "protocol.json")
    ids = split["validation"]
    reused_ids = old["ids"]
    new_ids = sorted(set(ids) - set(reused_ids))
    out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), ROOT / "scripts/probe_landslide_daily_revision_authority.py",
                    ROOT / "scripts/run_landslide_diversity_local_review.py",
                    ROOT / "scripts/probe_landslide_revision_scope.py",
                    ROOT / "src/gnss_sim/landslide_partial_review.py",
                    ROOT / "src/gnss_sim/landslide_adaptive_review.py",
                    ROOT / "src/gnss_sim/landslide_observed_duration.py",
                    ROOT / "scripts/score_landslide_claim_review.py"]
    protocol = dict(
        candidate="RCRA / Raw-Conflict Revision Authority;visual prior only",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        data=str(data), encoded=str(encoded), parent=str(parent), stage_source=str(stage_source),
        cached_source=str(source), pilot=str(pilot), ids=ids, split=split,
        reused_ids=reused_ids, new_inference_ids=new_ids,
        scope="48TRAIN/24already-exposed VALIDATION;not independent;no reserved records",
        principal="visual_daily_partial;never replace with a better control",
        numerical_prior_mode="Stopped in pilot;not expanded",
        repetitions=2, model=old["model"], policy=old["policy"], retries=0,
        maximum_requests=2 * len(new_ids) * 5, hard_timeout_seconds=3600,
        mechanism="Unchanged global/local prompts;original supported daily conflicts authorize revision;no49day dilation;at most2 disjoint adaptive crops",
        controls="Direct/fixed/old adaptive/wholepartial/49day/matched day budget/strict abstention/no local visual;18refit numerical and visual feature controls retained",
        stage_policy="Common original48TRAIN stage head;same activity/stage support and14observed/7raw missing decoder",
        numerical_feedback="No numerical summary supplied to local VLM in any arm;feedback is raw-evidence routing and revision permission",
        continuation="Both repeats:main dailyF1 and positive net correction above direct/fixed/oldadaptive/wholepartial/neighborhood/matched/nolocal;event nonloss against all;pure nuisanceFP<=direct;mixed recall>=direct. Report all strong numerical gaps;no superiority over numerical models assumed",
        failure="Any condition failure stops full expansion;no prompt/threshold/seed or repeated-run rescue;keep reserved unopened",
        input_bindings={str(folder / manifest): sha(folder / manifest) for folder, manifest in bindings},
        source_bindings={str(p): sha(p) for p in source_paths}, reserved_content_loaded=False,
    )
    write_json(out / "protocol.json", protocol)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    (out / "images").mkdir()
    model = XGBClassifier()
    model.load_model(stage_source / "stage-xgboost.json")
    codes, masks = {}, {}
    for cid in ids:
        case, features = load_case_features(data, cid)
        if cid not in reused_ids:
            render_raw(case, out / "images" / (cid + "-global.png"))
            for index, (start, stop) in enumerate(regular_windows(len(case.dates), 180, 90)):
                render_window(case, features.noise, start, stop,
                              out / "images" / f"{cid}-window-{index:02d}.png", PLOT)
        with np.load(encoded / "validation" / (cid + ".npz")) as bank:
            codes[cid] = (model.predict(bank["stage_matrix"]).astype(int) + 1).tolist()
            masks[cid] = dict(activity=bank["supported"].tolist(), stage=bank["stage_supported"].tolist())
    # The strongest full-data controls must use this very phase head, not a
    # small-pilot model. Equality is checked on every definite stage output.
    controls = read_json(parent / "predictions.json")
    for cases in controls.values():
        for cid, prediction in cases.items():
            stages = np.array(prediction["stage"])
            definite = stages > 0
            if np.any(stages[definite] != np.array(codes[cid])[definite]):
                raise ValueError("Full-data control uses a different common stage head")
    write_json(out / "stage-codes.json", codes)
    write_json(out / "support-masks.json", masks)
    write_json(out / "prepared-freeze.json", {
        str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()
    })
    save_json(out / "state.json", dict(status="prepared", reserved_content_loaded=False))
    print(dict(prepared=True, records=len(ids), new_records=0, new_inference_records=len(new_ids),
               reused=len(reused_ids), maximum_new_requests=protocol["maximum_requests"]), flush=True)


def verify_prepared(out):
    verify_manifest(out, "prepared-freeze.json")
    protocol = read_json(out / "protocol.json")
    for key in ("source_bindings", "input_bindings"):
        for path, expected in protocol[key].items():
            if sha(Path(path)) != expected:
                raise ValueError("Frozen expansion binding changed: " + path)
    return protocol


def execute(out):
    protocol = verify_prepared(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Only one execution of a prepared protocol is allowed")
    data, cached = Path(protocol["data"]), Path(protocol["cached_source"])
    codes, masks = read_json(out / "stage-codes.json"), read_json(out / "support-masks.json")
    old_selection, old_reviews = read_json(cached / "selections.json"), read_json(cached / "window-outputs.json")
    transport = VisualRequests(out / "requests", ROOT / ".env", protocol["model"], protocol["maximum_requests"])
    policy = ReviewPolicy(**protocol["policy"])
    predictions, selections, windows_saved, used, failures, audits, scopes = {}, {}, {}, {}, {}, [], {}
    save_json(out / "state.json", dict(status="running", reserved_content_loaded=False))
    started = time.monotonic()
    for rep in range(2):
        for cid in protocol["ids"]:
            if time.monotonic() - started > protocol["hard_timeout_seconds"]:
                raise RuntimeError("Frozen expansion wall-time budget exceeded")
            case, features = load_case_features(data, cid)
            evidence = raw_motion_evidence(case, features)
            windows = regular_windows(len(case.dates), 180, 90)
            key = f"{cid}-r{rep}"
            global_id = key + "-global"
            prompt = ACTIVITY_TASK.format(last=len(case.dates) - 1) + (
                '\nReturn exactly JSON: {"activity":[[start,end]],"uncertain":[[start,end]]}. '
                "Adjacent intervals never share endpoints; use [100,149],[150,200]."
            )
            try:
                if cid in protocol["reused_ids"]:
                    receipt = read_json(cached / "requests" / (global_id + ".response.json"))
                    if receipt["status"] != "returned_json":
                        raise RuntimeError("Cached transport failure retained")
                    payload = json.loads(receipt["output"], object_pairs_hook=_unique_object)
                else:
                    payload = transport.ask(global_id, prompt, [out / "images" / (cid + "-global.png")])
                initial = parse_activity(payload, features.observed)
                initial_status = "success"
            except (ValueError, RuntimeError) as error:
                failures[global_id] = str(error)
                initial, initial_status = np.full(len(case.dates), -1), "failed"
            priorities, targets = review_priorities(initial, features, evidence, windows, policy)
            centers = np.linspace(0, len(case.dates), 4)[1:-1]
            fixed = sorted({min(range(len(windows)), key=lambda index: abs(sum(windows[index]) / 2 - center))
                            for center in centers})
            selection = dict(fixed=fixed, visual_selected=[row["index"] for row in targets],
                             visual_priorities=priorities, numeric_selected=[])
            if cid in protocol["reused_ids"]:
                for name in ("fixed", "visual_selected", "visual_priorities"):
                    if selection[name] != old_selection[key][name]:
                        raise ValueError("Cached observation routing changed")
                reviews = {index: old_reviews[key][str(index)] for index in sorted(set(fixed + selection["visual_selected"]))}
            else:
                reviews = {}
                for index in sorted(set(fixed + selection["visual_selected"])):
                    start, stop = windows[index]
                    local_id = key + f"-window-{index:02d}"
                    try:
                        payload = transport.ask(local_id, review_prompt(features, evidence, start, stop, False),
                                                [out / "images" / (cid + "-global.png"),
                                                 out / "images" / f"{cid}-window-{index:02d}.png"])
                        activity = parse_window_activity(payload, features.observed, start, stop)
                        status = "success"
                    except (ValueError, RuntimeError) as error:
                        failures[local_id] = str(error)
                        activity, status = np.full(len(case.dates), -1), "failed"
                    reviews[index] = dict(start=start, stop=stop, activity=activity.tolist(),
                                          status=status, request_id=local_id)
            selections[key] = selection
            windows_saved[key] = {str(index): value for index, value in reviews.items()}
            outputs = assemble(initial, initial_status, initial, np.array(codes[cid]), features.observed,
                               np.array(masks[cid]["activity"]), np.array(masks[cid]["stage"]), reviews, selection)
            for arm in ("direct_visual", "fixed_visual", "adaptive_visual"):
                method = f"{arm}_r{rep}"
                predictions.setdefault(method, {})[cid] = outputs[arm]
                if arm == "direct_visual":
                    indices = []
                elif arm == "fixed_visual":
                    indices = fixed
                else:
                    indices = selection["visual_selected"]
                used.setdefault(method, {})[cid] = [global_id] + [reviews[index]["request_id"] for index in indices]
            conflict, neighborhood = conflict_neighbourhood(initial, features, evidence, policy)
            variants, diagnostics, details = authority_outputs(
                initial, [reviews[index] for index in selection["visual_selected"]], features.observed,
                np.array(masks[cid]["activity"]), np.array(masks[cid]["stage"]), np.array(codes[cid]),
                conflict, neighborhood, targets,
            )
            scopes[key] = details
            for variant, value in variants.items():
                method = f"visual_{variant}_r{rep}"
                predictions.setdefault(method, {})[cid] = value
                used.setdefault(method, {})[cid] = [global_id] if variant == "no_local_visual" else used[f"adaptive_visual_r{rep}"][cid]
                audits.append(dict(method=method, case_id=cid, **diagnostics[variant]))
            for name, value in (("predictions", predictions), ("selections", selections), ("window-outputs", windows_saved),
                                ("used", used), ("failures", failures), ("authority-audit", audits), ("scopes", scopes)):
                save_json(out / (name + ".json"), value)
            print(f"{key}: reused={cid in protocol['reused_ids']} adaptive={selection['visual_selected']} fixed={fixed}; "
                  f"new requests={len(list((out / 'requests').glob('*.started.json')))}", flush=True)
    verify_prepared(out)
    write_json(out / "prediction-freeze.json", {
        str(p.relative_to(out)): sha(p) for p in out.rglob("*.json")
        if p.name not in {"state.json", "prepared-freeze.json"}
    })
    save_json(out / "state.json", dict(status="predictions_complete", seconds=time.monotonic() - started,
                                       reserved_content_loaded=False))


def score(out):
    protocol = verify_prepared(out)
    verify_manifest(out, "prediction-freeze.json")
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("All outputs must be frozen before loading references")
    data = Path(protocol["data"])
    labels = {cid: [] for cid in protocol["ids"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    observed = {cid: load_case_features(data, cid)[1].observed for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in rows]) for cid, rows in labels.items()}
    predictions = read_json(Path(protocol["parent"]) / "predictions.json")
    predictions.update(read_json(out / "predictions.json"))
    receipts = {}
    for folder in (Path(protocol["cached_source"]), out):
        receipts.update({p.name.removesuffix(".response.json"): read_json(p)
                         for p in (folder / "requests").glob("*.response.json")})
    used = read_json(out / "used.json")
    results, summary, per_case = score_target(labels, observed, weak, predictions, used, receipts)
    analysis = out / "analysis-v1"
    analysis.mkdir(exist_ok=False)
    for name, value in (("results", results), ("summary", summary)):
        write_json(analysis / (name + ".json"), value)
    for name, value in (("summary", summary), ("per-case", per_case),
                        ("authority-audit", read_json(out / "authority-audit.json"))):
        write_csv(analysis / (name + ".csv"), value)
    families = {row["case_id"]: row["truth"]["family"]
                for row in read_json(data / "training-generation-ledger.json") if row["case_id"] in labels}
    negatives = [cid for cid, family in families.items() if family in {"quiet", "step_only", "spike_only"}]
    mixed = [cid for cid, family in families.items() if family in {"deformation_step", "deformation_spike"}]
    family_rows = []
    for name in predictions:
        for family in sorted(set(families.values())):
            subset = aggregate_scores([results[name]["per_case"][cid] for cid in labels if families[cid] == family])
            family_rows.append(dict(method=name, family=family, **subset["daily"], event_f1=subset["events"]["f1"]))
    write_csv(analysis / "family-summary.csv", family_rows)
    rows = {row["method"]: row for row in summary}
    checks, nets = {}, {}
    for rep in range(2):
        main = f"visual_daily_partial_r{rep}"
        candidate = rows[main]
        controls = [f"{name}_r{rep}" for name in ("direct_visual", "fixed_visual", "adaptive_visual",
                    "visual_whole_partial", "visual_neighborhood_partial", "visual_matched_partial", "visual_no_local_visual")]
        nets[str(rep)] = {name: rows[name]["fp"] + rows[name]["fn"] - candidate["fp"] - candidate["fn"] for name in controls}
        nuisance_main = aggregate_scores([results[main]["per_case"][cid] for cid in negatives])["daily"]
        nuisance_prior = aggregate_scores([results[f"direct_visual_r{rep}"]["per_case"][cid] for cid in negatives])["daily"]
        mixed_main = aggregate_scores([results[main]["per_case"][cid] for cid in mixed])["daily"]
        mixed_prior = aggregate_scores([results[f"direct_visual_r{rep}"]["per_case"][cid] for cid in mixed])["daily"]
        checks[str(rep)] = dict(
            daily_gain=candidate["f1"] > max(rows[name]["f1"] for name in controls),
            positive_net=min(nets[str(rep)].values()) > 0,
            event_nonloss=candidate["event_f1_iou03"] >= max(rows[name]["event_f1_iou03"] for name in controls),
            nuisance_nonloss=nuisance_main["fp"] <= nuisance_prior["fp"],
            mixed_nonloss=mixed_main["recall"] >= mixed_prior["recall"],
        )
    passed = all(value for repeat in checks.values() for value in repeat.values())
    new_receipts = list((out / "requests").glob("*.response.json"))
    write_json(analysis / "decision.json", dict(
        status="RCRA_CONFIRMATION_PENDING_REPLAY" if passed else "STOP_RCRA_FULL_DEVELOPMENT",
        checks=checks, net_correct=nets, methods=len(predictions), outputs=len(predictions) * len(labels),
        new_api_calls=len(new_receipts), new_tokens=sum(read_json(p).get("usage", {}).get("total_tokens", 0) for p in new_receipts),
        failures=read_json(out / "failures.json"), reserved_content_loaded=False,
        stable_contribution_established=False, new_independent_records=0,
    ))
    write_json(analysis / "freeze.json", {p.name: sha(p) for p in analysis.iterdir()})
    save_json(out / "state.json", dict(status="complete", reserved_content_loaded=False))
    print(read_json(analysis / "decision.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "score"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root.resolve(), args.out.resolve())
    elif args.mode == "execute":
        execute(args.out.resolve())
    else:
        score(args.out.resolve())
