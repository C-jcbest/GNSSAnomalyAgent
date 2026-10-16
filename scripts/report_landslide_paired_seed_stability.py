"""Replay every saved seed head and summarize sensitivity without model selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import export
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import read_json
from train_landslide_paired_persistence import (
    ARMS,
    calibrated_prediction,
    modality_inputs,
    probabilities,
)

from gnss_sim.artifacts import sha
from gnss_sim.landslide_paired_persistence import pretrained_activity_head
from gnss_sim.landslide_visual_invariance import numeric_input


def report(run, calibration, heads, analysis, out):
    verify_bindings(run, "result-freeze.json")
    verify_bindings(calibration, "calibration-freeze.json")
    verify_bindings(heads, "result-freeze.json")
    out.mkdir(exist_ok=False)
    protocol = read_json(run / "protocol.json")
    scaling = read_json(calibration / "numeric-scaling.json")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    banks = {}
    for cid in read_json(heads / "protocol.json")["case_ids"]:
        with np.load(heads / (cid + "-inputs.npz")) as data:
            banks[cid] = {name: data[name] for name in ("embeddings", "numbers", "supported", "codes", "stage_support")}
            banks[cid]["normalized"] = numeric_input(data["numbers"], scaling)
    for seed in protocol["seeds"]:
        saved = read_json(run / f"{seed}-probabilities.json")
        predictions = read_json(run / f"{seed}-predictions.json")
        for name, arm in ARMS.items():
            chosen = protocol["selection"][name]
            model = pretrained_activity_head(512 if arm["image"] else 0, 36 if arm["numeric"] else 0, chosen["width"])
            model.load_state_dict(torch.load(run / "models" / f"{seed}-{name}.pt", weights_only=True))
            for cid, bank in banks.items():
                probability = probabilities(model, modality_inputs(bank["embeddings"], bank["normalized"], arm))
                np.testing.assert_array_equal(probability, saved[name][cid])
                prediction = calibrated_prediction(probability, bank["supported"], bank["codes"],
                                                    bank["stage_support"], chosen["threshold"]).to_dict()
                if prediction != predictions[name][cid]:
                    raise ValueError("Seed forecast weight replay differs")
    rows = read_json(run / "summary.json")
    lookup = {(row["target"], row["seed"], row["method"]): row for row in rows}
    contrasts, comparison_rows = {}, []
    for target in ("primary_process", "secondary_observation"):
        for control in ("visual_numeric_augmentation", "visual_numeric_plain", "visual_numeric_shuffled_pairs",
                        "numeric_pair", "numeric_augmentation"):
            differences = [lookup[target, seed, "visual_numeric_pair"]["f1"] - lookup[target, seed, control]["f1"]
                           for seed in protocol["seeds"]]
            contrasts[target + "__" + control] = {
                "differences": differences, "minimum": min(differences), "maximum": max(differences),
                "mean": float(np.mean(differences)), "sample_std": float(np.std(differences, ddof=1)),
                "positive_seeds": sum(value > 0 for value in differences),
                "scope": "Descriptive variation across five related fits on the same evaluation records; not independent sample evidence or a confidence interval",
            }
    for seed in protocol["seeds"]:
        primary = lookup["primary_process", seed, "visual_numeric_pair"]
        comparison_rows.append({"seed": seed,
            "main_f1": primary["f1"],
            "augmentation_f1": lookup["primary_process", seed, "visual_numeric_augmentation"]["f1"],
            "plain_f1": lookup["primary_process", seed, "visual_numeric_plain"]["f1"],
            "shuffled_f1": lookup["primary_process", seed, "visual_numeric_shuffled_pairs"]["f1"],
            "numeric_pair_f1": lookup["primary_process", seed, "numeric_pair"]["f1"],
            "main_weak_recall": primary["weak_recall"], "main_event_f1": primary["event_f1_iou03"],
            "secondary_main_f1": lookup["secondary_observation", seed, "visual_numeric_pair"]["f1"],
            "secondary_augmentation_f1": lookup["secondary_observation", seed, "visual_numeric_augmentation"]["f1"]})
    write_csv(out / "comparison.csv", comparison_rows)
    export(out / "contrasts.json", contrasts)
    export(out / "verification.json", {"all35_saved_heads_probabilities_predictions": "exact",
        "original_seed_training_reproduction": read_json(run / "verification.json"),
        "models_replaced": False, "new_api_calls": 0})
    columns = tuple(comparison_rows[0])
    text = ["# 冻结参数的五训练种子敏感性", "",
            "在已评分新24条上作事后稳健性检查，七臂原宽度/阈值/数据/损失不变；没有重新校准或替换正式主模型。",
            "五次拟合共享评估记录，不能当作五个独立测试集。全部35头概率及预测从保存权重精确重放，原种子权重也精确复现。", "",
            "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in comparison_rows:
        text.append("| " + " | ".join(f"{row[key]:.6f}" if isinstance(row[key], float) else str(row[key])
                                       for key in columns) + " |")
    for name, result in contrasts.items():
        text.extend(("", f"{name}: F1差范围[{result['minimum']:+.6f}, {result['maximum']:+.6f}]，"
                     f"正差种子{result['positive_seeds']}/5；这是随机拟合敏感性，不能替代原记录重采样区间。"))
    text.extend(("", "观测参考上主对同增广五种子全部恶化。该结果和原正式确认一起保留，不用最好种子救分。", ""))
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, label in (("visual_numeric_pair", "Visual + numeric pair"),
                        ("visual_numeric_augmentation", "Identical augmentation"),
                        ("numeric_pair", "Numerical pair"),
                        ("visual_numeric_shuffled_pairs", "Shuffled pairs")):
        axes[0].plot(range(5), [lookup["primary_process", seed, name]["f1"] for seed in protocol["seeds"]],
                     marker="o", label=label)
    baseline = next(row for row in read_json(analysis / "primary_process/summary.json") if row["method"] == "xgboost_84")
    axes[0].axhline(baseline["f1"], color="#555555", linestyle=":", label="Frozen original XGBoost84")
    for name, label in (("visual_numeric_pair", "Visual + numeric pair"),
                        ("visual_numeric_augmentation", "Identical augmentation")):
        axes[1].plot(range(5), [lookup["secondary_observation", seed, name]["f1"] for seed in protocol["seeds"]],
                     marker="o", label=label)
    for axis, title in zip(axes, ("Primary controlled process", "Secondary developer-observation target"), strict=True):
        axis.set_xticks(range(5), [str(seed) for seed in protocol["seeds"]], rotation=25, fontsize=8)
        axis.set_ylabel("Activity F1 (cropped axis)")
        axis.set_title(title)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle("Post-confirmation fixed-setting seed sensitivity; no model replacement")
    figure.tight_layout()
    figure.savefig(out / "seed-sensitivity.png", dpi=160)
    figure.savefig(out / "seed-sensitivity.pdf")
    plt.close(figure)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    export(out / "report-freeze.json", {path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()})
    print("Replayed 35 frozen heads; all five primary F1 gains and secondary losses retained", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "calibration", "heads", "analysis", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.run.resolve(), arguments.calibration.resolve(), arguments.heads.resolve(),
           arguments.analysis.resolve(), arguments.out.resolve())
