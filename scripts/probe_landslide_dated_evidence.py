"""Test date-ordered motion evidence before developing another visual mechanism."""

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
from run_landslide_comparison import calibration_objective
from run_landslide_disagreement_review import load_banks
from run_landslide_window_optimization import read_json
from score_landslide_claim_review import score_target
from threadpoolctl import threadpool_limits
from train_landslide_deformation_quality import THRESHOLDS
from train_landslide_paired_persistence import probabilities
from train_landslide_spatial_readout import decode
from write_landslide_distribution_manuscript import score_table
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_dated_evidence import DatedEvidenceHead, dated_inputs, original_edges
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_visual_invariance import fit_numeric_scaling, numeric_input

ARMS = {
    "dated_numeric": "center",
    "permuted_dated_numeric": "permuted",
    "pooled_dated_numeric": "pooled",
    "flat_dated_numeric": "flat",
    "dated_visual_numeric": "visual",
}


def fit_head(training, target, width, kind):
    torch.manual_seed(20261020)
    model = DatedEvidenceHead(width, kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(target) - target.sum()) / target.sum())
    )
    inputs = torch.from_numpy(training)
    labels = torch.from_numpy(target.astype(np.float32))
    random = np.random.default_rng(20261020)
    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(12):
        order = random.permutation(len(training))
        total = 0.
        for first in range(0, len(order), 128):
            indices = order[first:first + 128]
            loss = criterion(model(inputs[indices]).squeeze(1), labels[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append(dict(epoch=epoch + 1, balanced_bce=total / len(training)))
    return model, history, time.perf_counter() - started


def probe(data, encoded, evidence_source, parent, out):
    for folder, filename in (
        (data, "data-freeze.json"), (encoded, "encoding-freeze.json"),
        (evidence_source, "freeze.json"), (parent, "freeze.json"),
    ):
        verify_manifest(folder, filename)
    split = read_json(encoded / "protocol.json")["split"]
    if split != read_json(parent / "protocol.json")["split"]:
        raise ValueError("Different original train/validation population")
    if split != read_json(evidence_source / "protocol.json")["split"]:
        raise ValueError("Dated evidence belongs to another population")
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__), ROOT / "src/gnss_sim/landslide_dated_evidence.py",
        ROOT / "tests/test_landslide_dated_evidence.py",
        ROOT / "scripts/train_landslide_spatial_readout.py",
    ]
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    write_json(
        out / "protocol.json",
        dict(
            registered_utc=datetime.now(timezone.utc).isoformat(), split=split,
            scope="Numerical chronology feasibility probe, not a declared novel Agent method",
            hypothesis="Reading target-adjacent features of ordered original block changes may improve localization over pooled or flattened representations",
            observation="13 existing14-day original NEU block medians;12 adjacent changes;no bridging missing blocks;same39numerical inputs and flags",
            order_control="Per-query seed2026101327 permutation of10outer edges;target5/6 unchanged;values and missing flags moved together;no new derivative after permutation",
            arms=ARMS, widths=[16, 32], epochs=12, seed=20261020, thresholds=THRESHOLDS,
            training="Same10187 supported known TRAIN stride3centers,class-balancedBCE,Adam.001/wd.0001;TRAIN-only scaling",
            selection="Same dailyF1+eventF1 objective,19thresholds per2widths per5arms=>190new opportunities",
            fairness="Center,permuted,pooled heads have identical parameter counts;flat/visual differing counts disclosed.29previous methods retained including all-info XGB",
            decoder="Unchanged14 observed positives and<=7raw missing bridge;original48TRAIN stage head",
            numerical_gate="DailyF1 above pooled,flat,permuted AND global_motion_mass;eventF1>=global;positive net correction versus each three readout controls",
            visual_gate="F1 above dated_numeric AND raw_visual_motion AND strongest old nonvisual;eventF1>=numeric and raw;positive net correction vs numeric and raw",
            failure="Stop exact probe version if gates fail;no seed,width,depth or threshold rescue;do not rename best control as contribution",
            reserved_content_loaded=False, new_api_calls=0, new_visual_encodings=0,
            source_bindings=bindings,
            input_bindings={str(folder.resolve()): sha(folder / filename) for folder, filename in (
                (data, "data-freeze.json"), (encoded, "encoding-freeze.json"),
                (evidence_source, "freeze.json"), (parent, "freeze.json"),
            )},
        ),
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    write_json(out / "pre-fit-freeze.json", {p.name: sha(p) for p in out.iterdir()})
    banks = load_banks(encoded, split)
    evidence = {}
    for cid in banks:
        with np.load(evidence_source / "evidence" / (cid + ".npz")) as archive:
            evidence[cid] = {key: archive[key] for key in archive.files}
    training_edges = np.concatenate([
        original_edges(evidence[cid])[banks[cid]["selected"]] for cid in split["train"]
    ])
    edge_scaling = fit_numeric_scaling(training_edges)
    scaling = read_json(parent / "numeric-scaling.json")
    write_json(out / "edge-scaling.json", edge_scaling)
    labels = {cid: [] for cid in banks}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in labels:
                labels[row["case_id"]].append(row)
    stage = XGBClassifier()
    stage.load_model(data.parent / "disagreement-calibration-v1/stage-xgboost.json")
    codes = {cid: stage.predict(banks[cid]["stage_matrix"]).astype(int) + 1 for cid in split["validation"]}
    target = np.concatenate([banks[cid]["activity"][banks[cid]["selected"]] for cid in split["train"]])
    predictions = read_json(parent / "predictions.json")
    choices, trials, saved_probability, costs = {}, [], {}, []
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    for name, kind in ARMS.items():
        inputs = {}
        for cid, bank in banks.items():
            numbers = numeric_input(
                np.column_stack((bank["numbers"], evidence[cid]["global_distribution"])), scaling
            )
            inputs[cid] = dated_inputs(evidence[cid], edge_scaling, numbers, bank["embeddings"], kind)
        training = np.concatenate([inputs[cid][banks[cid]["selected"]] for cid in split["train"]])
        candidates, by_width = {}, {}
        for width in (16, 32):
            model, history, seconds = fit_head(training, target, width, kind)
            torch.save(model.state_dict(), out / f"{name}-width{width}.pt")
            write_json(out / f"{name}-width{width}-loss.json", history)
            probability = {cid: probabilities(model, inputs[cid]) for cid in split["validation"]}
            by_width[width] = probability
            for threshold in THRESHOLDS:
                decoded = {cid: decode(probability[cid], threshold, banks[cid], codes[cid]) for cid in probability}
                score = aggregate_scores([evaluate_case(labels[cid], decoded[cid]) for cid in decoded])
                trials.append(dict(method=name, parameter=width, threshold=threshold,
                                   objective=calibration_objective(score), validation=score))
                candidates[width, threshold] = {cid: result.to_dict() for cid, result in decoded.items()}
            costs.append(dict(method=name, width=width, seconds=seconds, samples=len(training),
                              dimensions=training.shape[1],
                              trainable_parameters=sum(p.numel() for p in model.parameters())))
        choice = max((row for row in trials if row["method"] == name), key=lambda row: row["objective"])
        choices[name] = choice
        predictions[name] = candidates[choice["parameter"], choice["threshold"]]
        saved_probability[name] = {cid: value.tolist() for cid, value in by_width[choice["parameter"]].items()}
        print("calibrated", name, flush=True)
    write_json(out / "selection.json", choices)
    write_json(out / "trials.json", trials)
    write_json(out / "probabilities.json", saved_probability)
    write_json(out / "predictions.json", predictions)
    write_csv(out / "training-costs.csv", costs)
    validation = {cid: labels[cid] for cid in split["validation"]}
    observed = {cid: banks[cid]["observed"] for cid in validation}
    weak = {cid: np.array([bool(int(row["weak_signal"])) for row in validation[cid]]) for cid in validation}
    results, summary, per_case = score_target(validation, observed, weak, predictions, {}, {})
    write_json(out / "results.json", results)
    write_json(out / "summary.json", summary)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per-case.csv", per_case)
    table = {row["method"]: row for row in summary}
    main, visual = table["dated_numeric"], table["dated_visual_numeric"]
    controls = ("flat_dated_numeric", "permuted_dated_numeric", "pooled_dated_numeric")
    net = {name: table[name]["fp"] + table[name]["fn"] - main["fp"] - main["fn"] for name in controls}
    best_numeric = max(
        (table[name] for name in ("global_motion_mass", "query_motion_mass", "motion_numeric_mlp",
                                 "raw_motion_information_xgb", "mass_raw84", *controls, "dated_numeric")),
        key=lambda row: row["f1"],
    )
    numeric_checks = dict(
        gain_vs_readout_controls=main["f1"] > max(table[name]["f1"] for name in controls),
        gain_vs_global_tree=main["f1"] > table["global_motion_mass"]["f1"],
        event_nonloss=main["event_f1_iou03"] >= table["global_motion_mass"]["event_f1_iou03"],
        positive_net_correction=min(net.values()) > 0,
    )
    visual_controls = ("dated_numeric", "raw_visual_motion")
    visual_checks = dict(
        gain_vs_numeric_and_raw=visual["f1"] > max(table[name]["f1"] for name in visual_controls),
        gain_vs_strong_numeric=visual["f1"] > best_numeric["f1"],
        event_nonloss=visual["event_f1_iou03"] >= max(table[name]["event_f1_iou03"] for name in visual_controls),
        positive_net_correction=all(
            table[name]["fp"] + table[name]["fn"] > visual["fp"] + visual["fn"] for name in visual_controls
        ),
    )
    decision = dict(
        numerical_status="ELIGIBLE_DATED_EVIDENCE" if all(numeric_checks.values()) else "STOP_DATED_EVIDENCE",
        numerical_checks=numeric_checks, visual_checks=visual_checks,
        visual_status="ELIGIBLE_DATED_VISUAL_PENDING_CONFIRMATION" if all(visual_checks.values()) else "STOP_DATED_VISUAL",
        net_correct=net, best_nonvisual=best_numeric["method"], reserved_content_loaded=False,
        new_api_calls=0, stable_contribution_established=False,
    )
    write_json(out / "decision.json", decision)
    report = "# 原日期运动证据利用探针\n\n同48TRAIN／24暴露VALIDATION，五臂190配置、34统一方法；这是时序读取强对照与机制依据实验，普通卷积不能称本项目创新。无新记录／编码／API，保留24内容未加载。完整输出及模型冻结，重放另存。\n\n"
    report += score_table(summary, tuple(ARMS) + ("query_motion_mass", "global_motion_mass", "raw_visual_motion", "visual_numeric", "robust_rule", "pelt_rule"))
    report += "\n\n数值判据：" + str(numeric_checks) + "；视觉判据：" + str(visual_checks)
    report += "\n\n结论：" + decision["numerical_status"] + "／" + decision["visual_status"] + "。不修改本版参数救分，不以最佳普通对照更名为贡献。\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    for relative, expected in bindings.items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen probe source changed: " + relative)
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "encoded", "evidence-source", "parent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        probe(args.data.resolve(), args.encoded.resolve(), args.evidence_source.resolve(),
              args.parent.resolve(), args.out.resolve())
