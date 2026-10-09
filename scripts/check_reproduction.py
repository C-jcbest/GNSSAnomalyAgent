"""Run upstream examples and independent algorithm checks; never claim paper-table replication."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_example(script: Path, arguments: list[str], log_path: Path) -> dict:
    """Use runpy so the original example can be inspected after its unchanged entry point exits."""
    previous_directory = Path.cwd()
    previous_arguments = sys.argv
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(script.parent)
        sys.argv = [str(script), *arguments]
        with log_path.open("x", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                namespace = runpy.run_path(str(script), run_name="__main__")
        return namespace
    finally:
        os.chdir(previous_directory)
        sys.argv = previous_arguments


def check_pelt() -> dict:
    import ruptures as rpt

    # Compare penalized PELT with exhaustive segmentation, independently of its pruning logic.
    rng = np.random.default_rng(731)
    time_index = np.arange(36, dtype=float)
    values = np.r_[np.zeros(12), 0.6 * np.arange(12), np.full(12, 6.6)]
    values += rng.normal(0, 0.035, len(values))
    signal = np.column_stack((values, np.ones(len(values)), time_index))
    minimum_length = 6
    penalty = 0.5
    ends = rpt.Pelt(model="linear", min_size=minimum_length, jump=1).fit_predict(
        signal, pen=penalty
    )

    def objective(boundaries: list[int]) -> float:
        total = 0.0
        start = 0
        for end in boundaries:
            design = signal[start:end, 1:]
            observed = signal[start:end, 0]
            coefficients = np.linalg.lstsq(design, observed, rcond=None)[0]
            total += float(np.sum((observed - design @ coefficients) ** 2)) + penalty
            start = end
        return total

    best = float("inf")
    best_ends = []
    for count in range(len(values) // minimum_length):
        for internal in itertools.combinations(range(minimum_length, 31), count):
            candidate = [*internal, len(values)]
            if min(np.diff([0, *candidate])) < minimum_length:
                continue
            score = objective(candidate)
            if score < best:
                best = score
                best_ends = candidate
    actual = objective(ends)
    assert np.isclose(actual, best, atol=1e-8), (ends, best_ends, actual, best)
    flat = np.column_stack((np.zeros(80), np.ones(80), np.arange(80)))
    flat_ends = rpt.Pelt(model="linear", min_size=8, jump=1).fit_predict(flat, pen=1)
    assert flat_ends == [80], flat_ends
    shifted = signal.copy()
    shifted[:, 0] += 10000
    shifted_ends = rpt.Pelt(model="linear", min_size=minimum_length, jump=1).fit_predict(
        shifted, pen=penalty
    )
    assert shifted_ends == ends, (shifted_ends, ends)
    return {"status": "passed", "pelt_exclusive_ends": ends,
            "exhaustive_exclusive_ends": best_ends, "objective": actual,
            "exhaustive_objective": best, "flat_exclusive_ends": flat_ends,
            "offset_invariance": True, "scope": "algorithm correctness checks, not GNSS accuracy"}


def check_xgboost(work: Path, output: Path) -> dict:
    import xgboost as xgb

    script = work / "upstream/xgboost/demo/multiclass_classification/train.py"
    result = run_example(script, [], output / "official-example.log")
    estimator = result["bst"]
    probabilities = result["pred_prob"]
    assert np.isfinite(probabilities).all()
    assert probabilities.shape == (len(result["test_Y"]), 6)
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    checkpoint = output / "official-model.ubj"
    estimator.save_model(checkpoint)
    restored = xgb.Booster()
    restored.load_model(checkpoint)
    assert np.array_equal(restored.predict(result["xg_test"]), probabilities)
    labels = result["test_Y"].astype(int)
    training_labels = result["train_Y"].astype(int)
    majority = int(np.bincount(training_labels).argmax())
    accuracy = float(np.mean(probabilities.argmax(axis=1) == labels))
    majority_accuracy = float(np.mean(labels == majority))
    assert accuracy > majority_accuracy, (accuracy, majority_accuracy)
    return {"status": "passed", "train_samples": len(training_labels),
            "test_samples": len(labels), "accuracy": accuracy,
            "majority_accuracy": majority_accuracy, "serialization_identical": True,
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "scope": "unchanged official multiclass example; not original paper table or GNSS"}


def check_tcn(work: Path, output: Path) -> dict:
    import torch

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    np.random.seed(1111)  # The upstream --seed sets torch, but its mask generator also uses NumPy.
    upstream = work / "upstream/tcn"
    sys.path.insert(0, str(upstream))
    arguments = ["--cuda", "--epochs", "3", "--seq_len", "64", "--levels", "5",
                 "--nhid", "8", "--ksize", "3", "--batch_size", "128", "--seed", "1111"]
    script = upstream / "TCN/adding_problem/add_test.py"
    result = run_example(script, arguments, output / "official-example.log")
    model = result["model"].eval()
    with torch.no_grad():
        predicted = model(result["X_test"])
        mse = float(torch.mean((predicted - result["Y_test"]) ** 2))
        baseline = float(torch.mean((1 - result["Y_test"]) ** 2))
    assert np.isfinite(mse)
    from TCN.tcn import TemporalConvNet

    backbone = TemporalConvNet(3, [4, 4], kernel_size=3, dropout=0).eval()
    series = torch.randn(2, 3, 80)
    changed = series.clone()
    changed[:, :, 40:] += 100
    with torch.no_grad():
        before = backbone(series)
        after = backbone(changed)
    assert before.shape == (2, 4, 80)
    assert torch.equal(before[:, :, :40], after[:, :, :40])
    checkpoint = output / "official-model.pt"
    torch.save(model.state_dict(), checkpoint)
    from TCN.adding_problem.model import TCN

    restored = TCN(2, 1, [8] * 5, kernel_size=3, dropout=0).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        assert torch.equal(predicted, restored(result["X_test"]))
    return {"status": "passed", "arguments": arguments,
            "note": "Upstream --cuda is store_false: this flag selects CPU.",
            "train_samples": 50000, "test_samples": 1000, "mse": mse,
            "constant_one_mse": baseline, "beats_constant": mse < baseline,
            "causality_passed": True, "serialization_identical": True,
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "scope": "unchanged upstream example, reduced CLI configuration, not paper-table replication"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["pelt", "xgboost", "tcn"])
    parser.add_argument("--work", type=Path, default=ROOT / "artifacts/reproduction-2026-10-07")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write_json(output / "started.json", {"method": args.method, "argv": sys.argv})
    try:
        if args.method == "pelt":
            result = check_pelt()
        elif args.method == "xgboost":
            result = check_xgboost(args.work.resolve(), output)
        else:
            result = check_tcn(args.work.resolve(), output)
        result["seconds"] = time.monotonic() - start
        write_json(output / "result.json", result)
        print(json.dumps(result))
    except Exception as error:
        write_json(output / "result.json", {"status": "failed", "error": str(error),
                                           "seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
