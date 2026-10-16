"""Train a small query-grounded visual/numerical detector with matched ablations."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_comparison import calibration_objective, save_json
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_claim_review import score_target
from train_landslide_deformation_quality import THRESHOLDS
from train_landslide_spatial_readout import decode
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_query_shape_grounding import (
    grounded_chart_model,
    shape_grounding_targets,
)
from gnss_sim.landslide_visual_invariance import (
    chart_numeric_descriptors,
    fit_numeric_scaling,
    numeric_input,
    raw_chart_patches,
)

SEED = 20261014
ARMS = {
    "query_shape_grounded": dict(image=True, numeric=True, supervision="query"),
    "query_shape_plain": dict(image=True, numeric=True, supervision="none"),
    "query_shape_global": dict(image=True, numeric=True, supervision="global"),
    "query_shape_shuffled": dict(image=True, numeric=True, supervision="shuffled"),
    "query_shape_visual": dict(image=True, numeric=False, supervision="query"),
    "query_shape_visual_plain": dict(image=True, numeric=False, supervision="none"),
    "query_shape_numeric": dict(image=False, numeric=True, supervision="none"),
}


def probabilities(model, images, numbers, target_scaling):
    model.eval()
    activity, auxiliary = [], []
    with torch.inference_mode():
        for start in range(0, len(images), 256):
            logits, shape = model(
                torch.from_numpy(images[start:start + 256]),
                torch.from_numpy(numbers[start:start + 256]),
            )
            activity.append(torch.sigmoid(logits).numpy())
            if shape is not None:
                auxiliary.append(
                    shape.numpy() * np.array(target_scaling["scale"]) + np.array(target_scaling["mean"])
                )
    return np.concatenate(activity).astype(float), np.concatenate(auxiliary) if auxiliary else None


def fit(images, numbers, actual, query, global_shape, scaling, arm, width):
    torch.manual_seed(SEED)
    model = grounded_chart_model(width, numbers.shape[1], arm["image"], arm["numeric"])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    classification = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(actual) - actual.sum()) / actual.sum(), dtype=torch.float32)
    )
    selected_target = global_shape if arm["supervision"] == "global" else query
    selected_target = selected_target.copy()
    if arm["supervision"] == "shuffled":
        selected_target = selected_target[np.random.default_rng(SEED + 1).permutation(len(query))]
    mean, scale = np.array(scaling["mean"]), np.array(scaling["scale"])
    target = (selected_target - mean) / scale
    available = np.isfinite(target)
    target = np.nan_to_num(target).astype(np.float32)
    random = np.random.default_rng(SEED)
    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(12):
        primary_total = secondary_total = 0.0
        order = random.permutation(len(actual))
        for start in range(0, len(order), 128):
            index = order[start:start + 128]
            logits, auxiliary = model(torch.from_numpy(images[index]), torch.from_numpy(numbers[index]))
            primary = classification(logits, torch.from_numpy(actual[index].astype(np.float32)))
            secondary = torch.tensor(0.0)
            if arm["supervision"] != "none":
                valid = torch.from_numpy(available[index])
                if valid.any():
                    errors = torch.nn.functional.smooth_l1_loss(
                        auxiliary, torch.from_numpy(target[index]), reduction="none"
                    )
                    secondary = errors[valid].mean()
            optimizer.zero_grad()
            (primary + secondary).backward()
            optimizer.step()
            primary_total += float(primary.detach()) * len(index)
            secondary_total += float(secondary.detach()) * len(index)
        history.append(dict(epoch=epoch + 1, classification=primary_total / len(actual),
                            shape=secondary_total / len(actual)))
    return model, history, time.perf_counter() - started


def run(data, encoded, parent, out):
    for folder, manifest in ((data, "data-freeze.json"), (encoded, "encoding-freeze.json"), (parent, "freeze.json")):
        verify_manifest(folder, manifest)
    split = read_json(encoded / "protocol.json")["split"]
    if set(split) != {"train", "validation"} or split != read_json(parent / "protocol.json")["split"]:
        raise ValueError("Exact existing48/24 development split required")
    stage_source = data.parent / "quality-calibration-v1"
    verify_manifest(stage_source, "calibration-freeze.json")
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    (out / "targets").mkdir()
    sources = (
        Path(__file__), ROOT / "src/gnss_sim/landslide_query_shape_grounding.py",
        ROOT / "src/gnss_sim/landslide_visual_invariance.py",
        ROOT / "src/gnss_sim/landslide_observed_duration.py",
        ROOT / "scripts/score_landslide_claim_review.py",
        ROOT / "tests/test_landslide_query_shape_grounding.py",
    )
    protocol = dict(
        candidate="QSG-VN / Query-Shape-Grounded Visual-Numeric learning;not a language VLM or Agent",
        registered_utc=datetime.now(timezone.utc).isoformat(), data=str(data), encoded=str(encoded),
        parent=str(parent), stage_source=str(stage_source), split=split, arms=ARMS,
        mechanism="One label-free local plot-shape loss supervises only the image encoder;activity head keeps original raw numeric observations",
        raw_view="Existing6channel32x61 raster,3day medians>=2complete,missing columns masked;all inputs regenerated and checked against cached366chart descriptors",
        query_teacher="Median plotted N/E/U ordinates at columns23:28,28:33,33:38,at least4of5available;two signed differences. Plot geometry target,not physical magnitude removed by raster scaling",
        global_teacher="Same6outputs from groups0:5,28:33,56:61;TRAIN own scaling;same head/loss/sample/order",
        missing_teacher="NaN excluded from shape loss only;activity supervision/population never filtered",
        numerical_shortcut="Shape head takes image160only;numeric values cannot produce the teacher",
        fitting="Same all10187 encoded known TRAIN centers;12epochs,Adam.001/wd.0001,batch128,seed20261014,2CPUthreads;shape loss SmoothL1 coefficient1;no seed/lambda/epoch search",
        calibration="Hidden widths16/32 and19thresholds .05..95;38opportunities for each7arms. Stable first max dailyF1+eventIoU.3F1",
        support="Shared original48TRAIN numerical stage head,bank observed/activity/stage support;14positive observed,bridge<=7raw missing;no unknown filling",
        controls="Same visual head,no shape loss;global shape;shuffled TRAIN teacher;pure visual with/without loss;numeric only;retain18same48/24refit strong baselines incl chart384/querymass",
        continuation=[
            "Main dailyF1>plain,global,shuffled and positive net correction versus plain",
            "Main IoU.3>=plain-.01",
            "Visual grounding F1>visual plain OR IoU.5>visual plain with F1loss<=.01",
            "Main F1>numeric OR IoU.5>numeric with F1loss<=.01 OR precision>numeric with recallloss<=.02",
            "Pure nuisance FP<=plain",
            "Mixed recall>=plain-.02",
            "Main queried shape MAE<plain and shuffled;diagnostic not activity score",
        ],
        failure="Stop fixed candidate if any condition fails;no loss/seed/epoch/teacher/window/threshold rescue;reserved inputs/references remain unopened",
        scope="48TRAIN24exposedVALIDATION,not independent;0new records/API;all eight activity/nuisance families,not weak-only",
        prior_art="Auxiliary/self-supervised visual reconstruction and fusion are existing ideas. New test is query-relative signed shape supervision without numerical shortcut;not first auxiliary loss or claimed novelty before nearest-neighbor check",
        source_bindings={str(path.resolve()): sha(path) for path in sources},
        input_bindings={str(folder / manifest): sha(folder / manifest) for folder, manifest in
                        ((data, "data-freeze.json"), (encoded, "encoding-freeze.json"),
                         (parent, "freeze.json"), (stage_source, "calibration-freeze.json"))},
        reserved_content_loaded=False, new_api_calls=0,
    )
    write_json(out / "protocol.json", protocol)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "pre-fit-freeze.json", {path.name: sha(path) for path in out.iterdir() if path.is_file()})
    save_json(out / "state.json", dict(status="running", reserved_content_loaded=False))
    torch.set_num_threads(2)
    banks = load_banks(encoded, split)
    images, query, global_shape = {}, {}, {}
    preparation_started = time.perf_counter()
    for cid, bank in banks.items():
        case, features = load_case_features(data, cid)
        images[cid] = raw_chart_patches(case, features.noise)
        descriptor = chart_numeric_descriptors(images[cid])
        np.testing.assert_allclose(descriptor, bank["chart_descriptors"], atol=2e-7, rtol=2e-7, equal_nan=True)
        query[cid] = shape_grounding_targets(images[cid])
        global_shape[cid] = shape_grounding_targets(images[cid], "global")
        np.savez_compressed(out / "targets" / (cid + ".npz"), query=query[cid], global_shape=global_shape[cid])
    prepare_seconds = time.perf_counter() - preparation_started
    train_numbers = np.concatenate([banks[cid]["numbers"][banks[cid]["selected"]] for cid in split["train"]])
    numeric_scaling = fit_numeric_scaling(train_numbers)
    numbers = {cid: numeric_input(bank["numbers"], numeric_scaling) for cid, bank in banks.items()}
    training_query = np.concatenate([query[cid][banks[cid]["selected"]] for cid in split["train"]])
    training_global = np.concatenate([global_shape[cid][banks[cid]["selected"]] for cid in split["train"]])
    scaling = dict(query=fit_numeric_scaling(training_query), global_shape=fit_numeric_scaling(training_global))
    write_json(out / "scaling.json", dict(numbers=numeric_scaling, **scaling))
    train_images = np.concatenate([images[cid][banks[cid]["selected"]] for cid in split["train"]])
    train_actual = np.concatenate([banks[cid]["activity"][banks[cid]["selected"]] for cid in split["train"]])
    stage = XGBClassifier()
    stage.load_model(stage_source / "stage-xgboost.json")
    codes = {cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1 for cid in split["validation"]}
    write_json(out / "stage-codes.json", {cid: value.tolist() for cid, value in codes.items()})
    parent_predictions = read_json(parent / "predictions.json")
    for method in parent_predictions.values():
        for cid, value in method.items():
            stage_values = np.array(value["stage"])
            active_stage = stage_values > 0
            if np.any(stage_values[active_stage] != codes[cid][active_stage]):
                raise ValueError("Original strong baseline stage head differs")
    labels = {cid: [] for cid in split["validation"]}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    trials, choices, predictions, probabilities_saved, auxiliary_saved, costs, fidelity_scores = [], {}, {}, {}, {}, [], {}
    for name, arm in ARMS.items():
        candidate_predictions, candidate_probability, candidate_auxiliary = {}, {}, {}
        teacher_scale = scaling["global_shape"] if arm["supervision"] == "global" else scaling["query"]
        for width in (16, 32):
            model, history, seconds = fit(
                train_images, np.concatenate([numbers[cid][banks[cid]["selected"]] for cid in split["train"]]),
                train_actual, training_query, training_global, teacher_scale, arm, width,
            )
            torch.save(model.state_dict(), out / "models" / f"{name}-width{width}.pt")
            write_json(out / "models" / f"{name}-width{width}.loss.json", history)
            probability, auxiliary = {}, {}
            for cid in split["validation"]:
                probability[cid], auxiliary[cid] = probabilities(model, images[cid], numbers[cid], teacher_scale)
            candidate_probability[width], candidate_auxiliary[width] = probability, auxiliary
            for threshold in THRESHOLDS:
                decoded = {cid: decode(probability[cid], threshold, banks[cid], codes[cid]) for cid in labels}
                aggregate = aggregate_scores([evaluate_case(labels[cid], decoded[cid]) for cid in labels])
                trials.append(dict(method=name, parameter=width, threshold=threshold,
                                   objective=calibration_objective(aggregate), validation=aggregate))
                candidate_predictions[width, threshold] = {cid: value.to_dict() for cid, value in decoded.items()}
            costs.append(dict(method=name, width=width, seconds=seconds, samples=len(train_actual),
                              parameters=sum(parameter.numel() for parameter in model.parameters())))
            save_json(out / "state.json", dict(status="running", method=name, width=width, completed_models=len(costs),
                                              reserved_content_loaded=False))
            print(dict(method=name, width=width, seconds=seconds), flush=True)
        choice = max((row for row in trials if row["method"] == name), key=lambda row: row["objective"])
        choices[name] = choice
        predictions[name] = candidate_predictions[choice["parameter"], choice["threshold"]]
        probabilities_saved[name] = {cid: value.tolist() for cid, value in candidate_probability[choice["parameter"]].items()}
        auxiliary_saved[name] = {}
        errors = []
        for cid, value in candidate_auxiliary[choice["parameter"]].items():
            if value is None:
                continue
            auxiliary_saved[name][cid] = value.tolist()
            valid = np.isfinite(query[cid]) & banks[cid]["supported"][:, None]
            errors.extend(np.abs(value - np.nan_to_num(query[cid]))[valid].tolist())
        fidelity_scores[name] = float(np.mean(errors)) if errors else None
    predictions.update(parent_predictions)
    for name, value in (("selection", choices), ("probabilities", probabilities_saved), ("auxiliary", auxiliary_saved),
                        ("predictions", predictions), ("trials", trials), ("shape-fidelity", fidelity_scores)):
        write_json(out / (name + ".json"), value)
    write_csv(out / "training-costs.csv", costs)
    write_json(out / "costs.json", dict(preparation_seconds=prepare_seconds, training_seconds=sum(row["seconds"] for row in costs),
                                      new_api_calls=0, new_records=0))
    write_json(out / "prediction-freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*")
                                              if path.is_file() and path.name != "state.json"})
    observed = {cid: banks[cid]["observed"] for cid in labels}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in values]) for cid, values in labels.items()}
    results, summary, per_case = score_target(labels, observed, weak, predictions, {}, {})
    write_json(out / "results.json", results)
    write_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", per_case)
    rows = {row["method"]: row for row in summary}
    main, plain, numeric = (rows["query_shape_" + suffix] for suffix in ("grounded", "plain", "numeric"))
    visual, visual_plain = rows["query_shape_visual"], rows["query_shape_visual_plain"]
    families = {row["case_id"]: row["truth"]["family"] for row in read_json(data / "training-generation-ledger.json")}
    negatives = [cid for cid in labels if families[cid] in {"quiet", "step_only", "spike_only"}]
    mixed = [cid for cid in labels if families[cid] in {"deformation_step", "deformation_spike"}]
    def subgroup(name, ids):
        return aggregate_scores([results[name]["per_case"][cid] for cid in ids])["daily"]
    checks = dict(
        matched_core_gain=main["f1"] > max(rows["query_shape_" + suffix]["f1"] for suffix in ("plain", "global", "shuffled"))
        and plain["fp"] + plain["fn"] > main["fp"] + main["fn"],
        event_nonloss=main["event_f1_iou03"] >= plain["event_f1_iou03"] - .01,
        visual_mechanism=visual["f1"] > visual_plain["f1"] or (
            visual["event_f1_iou05"] > visual_plain["event_f1_iou05"] and visual["f1"] >= visual_plain["f1"] - .01),
        numeric_increment=main["f1"] > numeric["f1"] or (
            main["event_f1_iou05"] > numeric["event_f1_iou05"] and main["f1"] >= numeric["f1"] - .01) or (
            main["precision"] > numeric["precision"] and main["recall"] >= numeric["recall"] - .02),
        nuisance_nonloss=subgroup("query_shape_grounded", negatives)["fp"] <= subgroup("query_shape_plain", negatives)["fp"],
        mixed_nonloss=subgroup("query_shape_grounded", mixed)["recall"] >= subgroup("query_shape_plain", mixed)["recall"] - .02,
        target_fidelity=fidelity_scores["query_shape_grounded"] < min(fidelity_scores["query_shape_plain"], fidelity_scores["query_shape_shuffled"]),
    )
    decision = dict(
        status="QUERY_SHAPE_CONFIRMATION_PENDING_REPLAY" if all(checks.values()) else "STOP_QUERY_SHAPE_GROUNDING",
        checks=checks, methods=len(predictions), outputs=len(predictions) * len(labels),
        new_api_calls=0, reserved_content_loaded=False, stable_contribution_established=False,
    )
    write_json(out / "decision.json", decision)
    for path, expected in protocol["source_bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Source changed during frozen training: " + path)
    save_json(out / "state.json", dict(status="complete", decision=decision["status"], reserved_content_loaded=False))
    write_json(out / "freeze.json", {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()})
    print(dict(decision=decision, main=main, shape_fidelity=fidelity_scores), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "parent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.data.resolve(), arguments.encoded.resolve(), arguments.parent.resolve(), arguments.out.resolve())
