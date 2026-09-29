"""Isolated upstream runner: requires the separately locked GutenTAG environment.

stdout is a JSONL stream; no detector, network or truth-based resampling is used.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import gutenTAG
import numpy as np
from gutenTAG.anomalies import Anomaly, Position
from gutenTAG.anomalies.types.extremum import AnomalyExtremum, AnomalyExtremumParameters
from gutenTAG.anomalies.types.mean import AnomalyMean, AnomalyMeanParameters
from gutenTAG.base_oscillations import Polynomial
from gutenTAG.consolidator import Consolidator
from gutenTAG.utils.types import GenerationContext

ROOT = Path(__file__).resolve().parents[1]


def verify_sources():
    lock_path = ROOT / "configs/generator-sources.json"
    lock = json.loads(lock_path.read_text())
    installed = Path(gutenTAG.__file__).resolve().parent.parent
    for package, base in (("gutentag", installed), ("tods", ROOT)):
        for relative, digest in lock[package]["files"].items():
            if hashlib.sha256((base / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Source hash mismatch: {relative}")
    for name, version in lock["dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Dependency version mismatch: {name}")
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


def load_tods(filename):
    spec = importlib.util.spec_from_file_location(filename, ROOT / "vendor/tods" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UnivariateDataGenerator


def tods_generator(cls, values):
    return cls(len(values), behavior=lambda length: values.copy(), behavior_config={})


def verify_tods_equivalence():
    original = load_tods("univariate_generator.py")
    adapted = load_tods("univariate_generator_position.py")
    base = np.random.default_rng(991).normal(size=365)
    # Cover both upstream random mode and explicit positions with the same RNG state.
    np.random.seed(314)
    positions = (np.random.rand(round(365 * (180 / 365) / 90)) * 365).astype(int)
    state = np.random.get_state()
    reference = tods_generator(original, base)
    np.random.seed(314)
    reference.collective_trend_outliers(180 / 365, 3 / 89, 45)
    default = tods_generator(adapted, base)
    np.random.seed(314)
    default.collective_trend_outliers(180 / 365, 3 / 89, 45)
    explicit = tods_generator(adapted, base)
    np.random.set_state(state)
    explicit.collective_trend_outliers(180 / 365, 3 / 89, 45, positions=positions)
    for candidate in (default, explicit):
        np.testing.assert_array_equal(reference.data, candidate.data)
        np.testing.assert_array_equal(reference.label, candidate.label)
    return adapted


def generate(plan, tods, lock_hash):
    seed, kind, axis, sign = (plan[k] for k in ("noise_seed", "case_type", "axis", "sign"))
    length = {"normal": 0, "global_extremum": 1, "trend": 90, "mean_shift": 14}[kind]
    start = int(np.random.default_rng(plan["position_seed"]).integers(60, 306 - max(length, 1)))
    anomalies = []
    parameters = {}
    source = "GutenTAG Consolidator / Polynomial"
    if kind in ("global_extremum", "mean_shift"):
        if kind == "global_extremum":
            operator = AnomalyExtremum(AnomalyExtremumParameters(min=sign < 0, local=False))
            parameters = {"min": sign < 0, "local": False, "context_window": 10}
            source = "GutenTAG AnomalyExtremum (unmodified)"
        else:
            operator = AnomalyMean(AnomalyMeanParameters(offset=sign * 3.0))
            parameters = {"offset": sign * 3.0}
            source = "GutenTAG AnomalyMean (unmodified)"
        parameters.update(anomaly_length=length, creeping_length=0, exact_position=start)
        anomalies = [Anomaly(Position.Middle, start, length, channel=axis,
                             creeping_length=0).set_anomaly(operator)]
    consolidator = Consolidator(
        [Polynomial(length=365, polynomial=[0], amplitude=1, variance=1, offset=0)
         for _ in range(3)], anomalies)
    observed, labels = consolidator.generate(GenerationContext(np.random.SeedSequence(seed)))
    noise = np.column_stack([bo.noise for bo in consolidator.consolidated_channels])
    # Each fresh consolidator draws all three noises before any anomaly RNG is consumed.
    axis_labels = np.zeros((365, 3), dtype=int)
    if kind == "trend":
        generator = tods_generator(tods, noise[:, axis])
        # Use the unchanged upstream sign draw; signed factor maps it to planned +/- direction.
        np.random.seed(plan["tods_seed"])
        upstream_sign = int(np.random.choice([-1, 1]))
        factor = sign * upstream_sign * 3 / 89
        np.random.seed(plan["tods_seed"])
        generator.collective_trend_outliers(90 / 365, factor, 45, positions=[start + 45])
        observed[:, axis] = generator.data
        labels = generator.label.astype(int)
        axis_labels[:, axis] = labels
        parameters = {"ratio": 90 / 365, "factor": factor, "radius": 45,
                      "center": start + 45, "upstream_sign": upstream_sign}
        source = "TODS collective_trend_outliers (explicit-position patch)"
    else:
        for protocol, channel in consolidator.generated_anomalies:
            if channel != axis or protocol.start != start or protocol.end != start + length:
                raise RuntimeError("Unexpected native anomaly interval")
            label = protocol.labels
            axis_labels[label.start:label.start + label.length, channel] = 1
    if not np.array_equal(axis_labels.max(axis=1), labels):
        raise RuntimeError("Native labels and channel protocol disagree")
    if int(labels.sum()) != length:
        raise RuntimeError("Native anomaly count/length mismatch")
    if kind == "global_extremum":
        others = np.delete(observed[:, axis], start)
        valid = observed[start, axis] > others.max() if sign > 0 else observed[start, axis] < others.min()
        if not valid:
            raise RuntimeError("Native extremum is not strictly extreme; no resampling")
    return {"plan": plan, "observed": observed.tolist(), "noise": noise.tolist(),
            "labels": labels.astype(int).tolist(), "axis_labels": axis_labels.tolist(),
            "start": start, "end": start + length - 1, "source": source,
            "native_parameters": parameters, "source_lock_sha256": lock_hash}


def main():
    lock_hash = verify_sources()
    tods = verify_tods_equivalence()
    if "--verify" in sys.argv:
        print(json.dumps({"source_lock_sha256": lock_hash, "native_sources": "match",
                          "tods_random_and_explicit_equivalence": "exact"}))
        return
    plans = json.loads(sys.stdin.readline())
    for plan in plans:
        print(json.dumps(generate(plan, tods, lock_hash), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
