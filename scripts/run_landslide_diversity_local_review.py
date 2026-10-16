"""Test existing bounded raw-visual review on the diverse development population."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_comparison import save_json
from run_landslide_visual_comparison import render_raw
from run_landslide_window_optimization import load_case_features, read_json, render_window
from score_landslide_claim_review import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_priorities,
    review_prompt,
)
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_observed_duration import activity_across_missing
from gnss_sim.landslide_visual import ACTIVITY_TASK, VisualRequests, parse_activity
from gnss_sim.landslide_window_experiment import parse_window_activity

NUMERIC = "pilot_global_mass_xgb"
ARMS = ("direct_visual", "fixed_visual", "adaptive_visual", "fixed_numeric", "adaptive_numeric")
PLOT = {"single_plot_inches": [10.24, 6.3], "single_plot_dpi": 150}


def verify_sources(out):
    verify_manifest(out, "prepared-freeze.json")
    protocol = read_json(out / "protocol.json")
    for filename, expected in protocol["source_bindings"].items():
        if sha(Path(filename)) != expected:
            raise ValueError("Prepared source changed: " + filename)
    for filename, expected in protocol["input_bindings"].items():
        if sha(Path(filename)) != expected:
            raise ValueError("Prepared input changed: " + filename)
    return protocol


def prepare(data, encoded, parent, out):
    for folder, manifest in (
        (data, "data-freeze.json"), (encoded, "encoding-freeze.json"), (parent, "freeze.json"),
    ):
        verify_manifest(folder, manifest)
    split = read_json(encoded / "protocol.json")["split"]
    ids = split["validation"]
    stage_source = data.parent / "spatial-pilot-calibration-v1"
    verify_manifest(stage_source, "calibration-freeze.json")
    source_paths = [
        Path(__file__), ROOT / "src/gnss_sim/landslide_adaptive_review.py",
        ROOT / "src/gnss_sim/landslide_visual.py", ROOT / "src/gnss_sim/visual.py",
        ROOT / "src/gnss_sim/landslide_window_experiment.py",
        ROOT / "src/gnss_sim/landslide_local_experiment.py",
        ROOT / "src/gnss_sim/landslide_evaluation.py",
        ROOT / "src/gnss_sim/landslide_observed_duration.py",
        ROOT / "scripts/run_landslide_window_optimization.py",
        ROOT / "scripts/run_landslide_visual_comparison.py",
        ROOT / "scripts/score_landslide_claim_review.py",
    ]
    inputs = [data / "data-freeze.json", encoded / "encoding-freeze.json",
              parent / "freeze.json", stage_source / "calibration-freeze.json"]
    protocol = dict(
        purpose="Existing local-observation correction feasibility;not a new validated innovation",
        data=str(data), encoded=str(encoded), parent=str(parent), stage_source=str(stage_source),
        ids=ids, repetitions=2, maximum_requests=112, hard_timeout_seconds=3600,
        model="qwen3.8-flash", retries=0, window_days=180, stride_days=90,
        numeric_initial=NUMERIC, maximum_reviews=2,
        policy=dict(maximum_reviews=2, minimum_conflict_days=14, minimum_rate=.03,
                    minimum_signal_to_noise=2.),
        selection="Frozen existing raw-conflict policy;no activity labels in routing;fixed two equal-spaced targets",
        mechanism="Unchanged global/raw-local images and old prompts;shared local replies for both priors",
        scope="Eight previously exposed development records,eight families;no new or reserved records",
        continuation="Numeric review only if both repeats beat numerical prior and fixed numeric review on dailyF1,positive net correction,event nonloss,pure-nuisance FP nonloss and mixed recall nonloss;adaptive visual only if both repeats beat fixed visual and direct visual with positive net correction and no event loss",
        failure="Stop this exact workflow if gates fail;no prompt,policy,threshold,repetition rescue",
        stage_policy="Same frozen16TRAIN stage head;unknown/missing/activity contracts unchanged",
        decoder="Common14positive supported observations;only bridge<=7raw missing;original bank activity support and separate stage support;no uncertain filling",
        source_bindings={str(p): sha(p) for p in source_paths},
        input_bindings={str(p): sha(p) for p in inputs},
        reserved_content_loaded=False,
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "source").mkdir()
    for path in source_paths:
        (out / "source" / path.name).write_bytes(path.read_bytes())
    write_json(out / "protocol.json", protocol)
    (out / "images").mkdir()
    numeric = read_json(parent / "predictions.json")[NUMERIC]
    write_json(out / "numeric-initial.json", {cid: numeric[cid] for cid in ids})
    stage = XGBClassifier()
    stage.load_model(stage_source / "stage-xgboost.json")
    codes, masks = {}, {}
    for cid in ids:
        case, features = load_case_features(data, cid)
        render_raw(case, out / "images" / (cid + "-global.png"))
        windows = regular_windows(len(case.dates), 180, 90)
        for index, (start, stop) in enumerate(windows):
            render_window(case, features.noise, start, stop,
                          out / "images" / f"{cid}-window-{index:02d}.png", PLOT)
        with np.load(encoded / "validation" / (cid + ".npz")) as bank:
            codes[cid] = (stage.predict(bank["stage_matrix"]).astype(int) + 1).tolist()
            masks[cid] = dict(activity=bank["supported"].tolist(), stage=bank["stage_supported"].tolist())
    write_json(out / "stage-codes.json", codes)
    write_json(out / "support-masks.json", masks)
    write_json(out / "prepared-freeze.json", {
        str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()
    })
    write_json(out / "state.json", dict(status="prepared", reserved_content_loaded=False))
    print("Prepared eight-family pilot;112-request hard maximum;no reference access in routing", flush=True)


def assemble(initial, initial_status, numeric, codes, observed, support, stage_support, reviews, selection):
    gates = {"direct_visual": ActivityPrediction(
        initial, np.where(initial == 0, 0, -1), status=initial_status,
    )}
    for arm, prior, selected in (
        ("fixed_visual", initial, selection["fixed"]),
        ("adaptive_visual", initial, selection["visual_selected"]),
        ("fixed_numeric", numeric, selection["fixed"]),
        ("adaptive_numeric", numeric, selection["numeric_selected"]),
    ):
        gates[arm] = combine_reviews(prior, [reviews[index] for index in selected], observed)
    result = {}
    for arm, gate in gates.items():
        if gate.status == "failed":
            prediction = gate
        else:
            activity = activity_across_missing(gate.activity == 1, support & (gate.activity >= 0), observed)
            prediction = learned_stages(codes, activity, stage_support)
        result[arm] = prediction.to_dict()
    return result


def execute(out):
    protocol = verify_sources(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Execute only a prepared run;no retry or overwrite")
    data = Path(protocol["data"])
    numeric = read_json(out / "numeric-initial.json")
    codes = read_json(out / "stage-codes.json")
    masks = read_json(out / "support-masks.json")
    transport = VisualRequests(out / "requests", ROOT / ".env", protocol["model"], protocol["maximum_requests"])
    policy = ReviewPolicy(**protocol["policy"])
    predictions, selections, windows_saved, used, failures = {}, {}, {}, {}, {}
    save_json(out / "state.json", dict(status="running", reserved_content_loaded=False))
    started = time.monotonic()
    for rep in range(protocol["repetitions"]):
        for cid in protocol["ids"]:
            if time.monotonic() - started > protocol["hard_timeout_seconds"]:
                raise RuntimeError("Frozen wall-time budget exceeded")
            case, features = load_case_features(data, cid)
            evidence = raw_motion_evidence(case, features)
            windows = regular_windows(len(case.dates), 180, 90)
            global_image = out / "images" / (cid + "-global.png")
            key = f"{cid}-r{rep}"
            request_id = key + "-global"
            prompt = ACTIVITY_TASK.format(last=len(case.dates) - 1) + (
                '\nReturn exactly JSON: {"activity":[[start,end]],"uncertain":[[start,end]]}. '
                "Adjacent intervals never share endpoints; use [100,149],[150,200]."
            )
            try:
                initial = parse_activity(transport.ask(request_id, prompt, [global_image]), features.observed)
                status = "success"
            except (ValueError, RuntimeError) as error:
                failures[request_id] = str(error)
                initial, status = np.full(len(case.dates), -1), "failed"
            prior = np.array(numeric[cid]["activity"])
            visual_priorities, visual_targets = review_priorities(initial, features, evidence, windows, policy)
            numeric_priorities, numeric_targets = review_priorities(prior, features, evidence, windows, policy)
            centers = np.linspace(0, len(case.dates), 4)[1:-1]
            fixed = sorted({min(range(len(windows)), key=lambda index: abs(sum(windows[index]) / 2 - center))
                            for center in centers})
            selection = dict(fixed=fixed, visual_selected=[row["index"] for row in visual_targets],
                             numeric_selected=[row["index"] for row in numeric_targets],
                             visual_priorities=visual_priorities, numeric_priorities=numeric_priorities)
            selections[key] = selection
            save_json(out / "selections.json", selections)
            reviews = {}
            targets = sorted(set(fixed + selection["visual_selected"] + selection["numeric_selected"]))
            for index in targets:
                start, stop = windows[index]
                local_id = key + f"-window-{index:02d}"
                local_image = out / "images" / f"{cid}-window-{index:02d}.png"
                try:
                    payload = transport.ask(local_id, review_prompt(features, evidence, start, stop, False),
                                            [global_image, local_image])
                    activity = parse_window_activity(payload, features.observed, start, stop)
                    review_status = "success"
                except (ValueError, RuntimeError) as error:
                    failures[local_id] = str(error)
                    activity, review_status = np.full(len(case.dates), -1), "failed"
                reviews[index] = dict(start=start, stop=stop, activity=activity.tolist(),
                                      status=review_status, request_id=local_id)
            windows_saved[key] = {str(index): value for index, value in reviews.items()}
            outputs = assemble(initial, status, prior, np.array(codes[cid]), features.observed,
                               np.array(masks[cid]["activity"]), np.array(masks[cid]["stage"]), reviews, selection)
            for arm, value in outputs.items():
                method = arm + f"_r{rep}"
                predictions.setdefault(method, {})[cid] = value
                if arm == "direct_visual":
                    indices = []
                elif arm.startswith("fixed"):
                    indices = selection["fixed"]
                elif arm == "adaptive_visual":
                    indices = selection["visual_selected"]
                else:
                    indices = selection["numeric_selected"]
                # Numeric-prior deployment needs only these local calls. The global
                # request remains shared experiment preparation, and is reported.
                requests = [] if arm.endswith("numeric") else [request_id]
                requests += [reviews[index]["request_id"] for index in indices]
                used.setdefault(method, {})[cid] = requests
            for name, value in (("predictions", predictions), ("used", used),
                                ("window-outputs", windows_saved), ("failures", failures)):
                save_json(out / (name + ".json"), value)
            print(f"{key}: visual{selection['visual_selected']} numeric{selection['numeric_selected']} "
                  f"fixed{fixed}; unique requests{len(list((out / 'requests').glob('*.started.json')))}", flush=True)
    verify_sources(out)
    write_json(out / "prediction-freeze.json", {
        str(p.relative_to(out)): sha(p) for p in out.rglob("*.json")
        if p.name not in {"state.json", "prepared-freeze.json"}
    })
    save_json(out / "state.json", dict(status="predictions_complete", seconds=time.monotonic() - started,
                                       reserved_content_loaded=False))


def score(out):
    protocol = verify_sources(out)
    verify_manifest(out, "prediction-freeze.json")
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("Require completed inference before opening references")
    labels = {cid: [] for cid in protocol["ids"]}
    with (Path(protocol["data"]) / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    features = {cid: load_case_features(Path(protocol["data"]), cid)[1] for cid in labels}
    observed = {cid: value.observed for cid, value in features.items()}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in rows]) for cid, rows in labels.items()}
    predictions = read_json(Path(protocol["parent"]) / "predictions.json")
    predictions.update(read_json(out / "predictions.json"))
    receipts = {p.name.removesuffix(".response.json"): read_json(p)
                for p in (out / "requests").glob("*.response.json")}
    results, summary, per_case = score_target(labels, observed, weak, predictions, read_json(out / "used.json"), receipts)
    analysis = out / "analysis-v1"
    analysis.mkdir(exist_ok=False)
    write_json(analysis / "results.json", results)
    write_json(analysis / "summary.json", summary)
    write_csv(analysis / "summary.csv", summary)
    write_csv(analysis / "per-case.csv", per_case)
    rows = {row["method"]: row for row in summary}
    families = {row["case_id"]: row["truth"]["family"]
                for row in read_json(Path(protocol["data"]) / "training-generation-ledger.json") if row["case_id"] in labels}
    negative = [cid for cid, family in families.items() if family in {"quiet", "step_only", "spike_only"}]
    mixed = [cid for cid, family in families.items() if family in {"deformation_step", "deformation_spike"}]

    def subset(method, ids):
        return aggregate_scores([results[method]["per_case"][cid] for cid in ids])["daily"]

    checks = {}
    for rep in range(protocol["repetitions"]):
        main = "adaptive_numeric_r" + str(rep)
        reference, candidate = rows[NUMERIC], rows[main]
        fixed_numeric = rows["fixed_numeric_r" + str(rep)]
        fixed, visual, direct = (rows[name + f"_r{rep}"] for name in ("fixed_visual", "adaptive_visual", "direct_visual"))
        checks[str(rep)] = dict(
            numeric_gain=candidate["f1"] > reference["f1"],
            numeric_net=reference["fp"] + reference["fn"] > candidate["fp"] + candidate["fn"],
            numeric_event=candidate["event_f1_iou03"] >= reference["event_f1_iou03"],
            numeric_fixed_gain=candidate["f1"] > fixed_numeric["f1"],
            numeric_fixed_net=candidate["fp"] + candidate["fn"] < fixed_numeric["fp"] + fixed_numeric["fn"],
            numeric_fixed_event=candidate["event_f1_iou03"] >= fixed_numeric["event_f1_iou03"],
            numeric_nuisance=subset(main, negative)["fp"] <= subset(NUMERIC, negative)["fp"],
            numeric_mixed=subset(main, mixed)["recall"] >= subset(NUMERIC, mixed)["recall"],
            visual_gain=visual["f1"] > max(fixed["f1"], direct["f1"]),
            visual_net=visual["fp"] + visual["fn"] < min(fixed["fp"] + fixed["fn"], direct["fp"] + direct["fn"]),
            visual_event=visual["event_f1_iou03"] >= max(fixed["event_f1_iou03"], direct["event_f1_iou03"]),
        )
    numerical = all(value for repeat in checks.values() for key, value in repeat.items() if key.startswith("numeric"))
    visual = all(value for repeat in checks.values() for key, value in repeat.items() if key.startswith("visual"))
    write_json(analysis / "decision.json", dict(
        checks=checks, numerical_support=numerical, visual_support=visual,
        status="LOCAL_REVIEW_EXPANSION_PENDING_REPLAY" if numerical or visual else "STOP_DIVERSITY_LOCAL_REVIEW",
        requests=len(receipts), reported_tokens=sum(row.get("usage", {}).get("total_tokens", 0) for row in receipts.values()),
        failures=read_json(out / "failures.json"), new_records=0, reserved_content_loaded=False,
        stable_contribution_established=False,
    ))
    write_json(analysis / "freeze.json", {p.name: sha(p) for p in analysis.iterdir()})
    save_json(out / "state.json", dict(status="complete", reserved_content_loaded=False))
    print(read_json(analysis / "decision.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "score"))
    parser.add_argument("--out", type=Path, required=True)
    for name in ("data", "encoded", "parent"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.data.resolve(), args.encoded.resolve(), args.parent.resolve(), args.out.resolve())
    elif args.mode == "execute":
        execute(args.out.resolve())
    else:
        score(args.out.resolve())
