"""Fit and replay daily artifact detection separately from deformation activity."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from encode_landslide_spatial_readout import verify_manifest
from probe_landslide_nuisance_evidence import ROOT, write_csv
from run_landslide_window_optimization import load_case_features, read_json
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_artifact_points import artifact_point_features

KINDS = {"": 0, "instrument_offset": 1, "spike": 2}
THRESHOLDS = [number / 20 for number in range(1, 20)]


def load_banks(data, split):
    ids = split["train"] + split["validation"]
    references = {cid: [] for cid in ids}
    with (data / "reference-labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case_id"] in references:
                references[row["case_id"]].append(row)
    banks = {}
    for cid in ids:
        case, features = load_case_features(data, cid)
        values = np.array(
            [row if row is not None else [np.nan] * 3 for row in case.displacement_mm]
        )
        matrix, observed, supported = artifact_point_features(values, features.noise)
        rows = references[cid]
        if len(rows) != len(values) or any(
            int(row["day_index"]) != i for i, row in enumerate(rows)
        ):
            raise ValueError("Artifact references differ from original calendar: " + cid)
        actual = np.array(
            [
                KINDS[row["quality_kind"]] if int(row["quality_event_label"]) >= 0 else -1
                for row in rows
            ]
        )
        activity = np.array([int(row["activity_label"]) for row in rows])
        if np.any((actual >= 0) & ~observed):
            raise ValueError("Missing observation has a known artifact label")
        banks[cid] = dict(
            matrix=matrix, observed=observed, supported=supported, actual=actual, activity=activity
        )
    return banks


def decode(probability, supported, threshold):
    result = np.full(len(supported), -1, dtype=int)
    result[supported] = 0
    positive = supported & (probability[:, 1:].sum(axis=1) >= threshold)
    result[positive] = np.argmax(probability[positive, 1:], axis=1) + 1
    return result


def scores(banks, predictions, probabilities):
    def subset(mask_kind, target):
        actual, predicted, ranked_actual, ranked_probability = [], [], [], []
        for cid, bank in banks.items():
            known = bank["observed"] & (bank["actual"] >= 0)
            if mask_kind == "deformation_days":
                known &= bank["activity"] == 1
            actual.extend(bank["actual"][known].tolist())
            predicted.extend(predictions[cid][known].tolist())
            supported = known & bank["supported"]
            truth = bank["actual"][supported]
            ranked_actual.extend((truth > 0 if target == 0 else truth == target).tolist())
            probability = np.asarray(probabilities[cid])
            ranked_probability.extend(
                (
                    probability[supported, 1:].sum(axis=1)
                    if target == 0
                    else probability[supported, target]
                ).tolist()
            )
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
        truth_positive = actual > 0 if target == 0 else actual == target
        predicted_positive = predicted > 0 if target == 0 else predicted == target
        tp = int(np.sum(truth_positive & predicted_positive))
        fp = int(np.sum(~truth_positive & predicted_positive))
        fn = int(np.sum(truth_positive & ~predicted_positive))
        return dict(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=tp / (tp + fp) if tp + fp else None,
            recall=tp / (tp + fn) if tp + fn else None,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            known_days=len(actual),
            unknown_on_known=int(np.sum(predicted < 0)),
            coverage=float(np.mean(predicted >= 0)) if len(actual) else None,
            auprc_known_supported=float(average_precision_score(ranked_actual, ranked_probability))
            if any(ranked_actual)
            else None,
        )

    return {
        group: {
            name: subset(group, code)
            for name, code in (("presence", 0), ("instrument_offset", 1), ("spike", 2))
        }
        for group in ("all", "deformation_days")
    }


def run(data, out):
    verify_manifest(data, "data-freeze.json")
    original = read_json(data / "protocol.json")
    split = {name: original["split"][name] for name in ("train", "validation")}
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        split=split,
        data_manifest_sha256=sha(data / "data-freeze.json"),
        depths=[2, 4],
        thresholds=THRESHOLDS,
        selection="presence daily F1, first tie",
        seed=20261007,
        n_estimators=120,
        learning_rate=0.05,
        weights="inverse TRAIN class frequency, mean1",
        task="original-day artifact presence/type",
        support="original observation and >=4 observations on each 7-day side",
        unknown="missing/unsupported prediction=-1; known unsupported positive remains FN",
        coexistence="independent of deformation; no veto or shared activity duration filter",
        scope="controlled exact-day labels; exposed validation only, no field claim",
        test_truth_opened=False,
        source_sha256=sha(Path(__file__)),
        feature_source_sha256=sha(ROOT / "src/gnss_sim/landslide_artifact_points.py"),
    )
    write_json(out / "protocol.json", protocol)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    (out / "feature-source.py").write_bytes(
        (ROOT / "src/gnss_sim/landslide_artifact_points.py").read_bytes()
    )
    write_json(out / "pre-fit-freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    started = time.perf_counter()
    banks = load_banks(data, split)
    for cid, bank in banks.items():
        np.savez_compressed(out / (cid + ".npz"), **bank)
    training = np.concatenate([banks[cid]["matrix"] for cid in split["train"]])
    actual = np.concatenate([banks[cid]["actual"] for cid in split["train"]])
    eligible = np.concatenate([banks[cid]["supported"] for cid in split["train"]]) & (actual >= 0)
    counts = np.bincount(actual[eligible], minlength=3)
    if np.any(counts == 0):
        raise ValueError("All artifact classes must occur in supported training days")
    weights = eligible.sum() / (3 * counts[actual[eligible]])
    validation = {cid: banks[cid] for cid in split["validation"]}
    trials, probability_by_depth, costs = [], {}, []
    for depth in protocol["depths"]:
        model = XGBClassifier(
            n_estimators=120,
            max_depth=depth,
            learning_rate=0.05,
            random_state=20261007,
            n_jobs=2,
            objective="multi:softprob",
        )
        fit_started = time.perf_counter()
        model.fit(training[eligible], actual[eligible], sample_weight=weights)
        model.save_model(out / f"artifact-depth{depth}.json")
        probability = {cid: model.predict_proba(bank["matrix"]) for cid, bank in validation.items()}
        probability_by_depth[depth] = probability
        costs.append(dict(depth=depth, seconds=time.perf_counter() - fit_started))
        for threshold in THRESHOLDS:
            predictions = {
                cid: decode(value, validation[cid]["supported"], threshold)
                for cid, value in probability.items()
            }
            result = scores(validation, predictions, probability)
            trials.append(
                dict(
                    depth=depth,
                    threshold=threshold,
                    scores=result,
                    objective=result["all"]["presence"]["f1"],
                )
            )
    choice = max(trials, key=lambda row: row["objective"])
    probability = probability_by_depth[choice["depth"]]
    predictions = {
        cid: decode(value, validation[cid]["supported"], choice["threshold"])
        for cid, value in probability.items()
    }
    write_json(out / "selection.json", choice)
    write_json(out / "validation-trials.json", trials)
    write_json(
        out / "probabilities.json", {cid: value.tolist() for cid, value in probability.items()}
    )
    write_json(
        out / "predictions.json", {cid: value.tolist() for cid, value in predictions.items()}
    )
    write_json(out / "scores.json", choice["scores"])
    write_json(
        out / "cost.json",
        dict(
            total_seconds=time.perf_counter() - started,
            training_known_supported=int(eligible.sum()),
            class_counts=counts.tolist(),
            dimensions=21,
            fits=costs,
        ),
    )
    write_csv(
        out / "summary.csv",
        [
            dict(group=group, task=name, **score)
            for group, rows in choice["scores"].items()
            for name, score in rows.items()
        ],
    )
    write_json(out / "freeze.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(choice, flush=True)


def verify(data, run_folder, out):
    verify_manifest(run_folder, "freeze.json")
    protocol = read_json(run_folder / "protocol.json")
    if sha(data / "data-freeze.json") != protocol["data_manifest_sha256"]:
        raise ValueError("Artifact data changed")
    if (
        sha(Path(__file__)) != protocol["source_sha256"]
        or sha(ROOT / "src/gnss_sim/landslide_artifact_points.py")
        != protocol["feature_source_sha256"]
    ):
        raise ValueError("Artifact implementation changed")
    banks = load_banks(data, protocol["split"])
    for cid, bank in banks.items():
        with np.load(run_folder / (cid + ".npz")) as saved:
            for key, value in bank.items():
                if not np.array_equal(value, saved[key], equal_nan=True):
                    raise ValueError("Raw artifact feature replay differs: " + cid + ":" + key)
    choice = read_json(run_folder / "selection.json")
    if choice != max(
        read_json(run_folder / "validation-trials.json"), key=lambda row: row["objective"]
    ):
        raise ValueError("Artifact selection differs")
    model = XGBClassifier()
    model.load_model(run_folder / f"artifact-depth{choice['depth']}.json")
    validation = {cid: banks[cid] for cid in protocol["split"]["validation"]}
    probability = {cid: model.predict_proba(bank["matrix"]) for cid, bank in validation.items()}
    predictions = {
        cid: decode(value, validation[cid]["supported"], choice["threshold"])
        for cid, value in probability.items()
    }
    if {cid: value.tolist() for cid, value in probability.items()} != read_json(
        run_folder / "probabilities.json"
    ):
        raise ValueError("Artifact probabilities differ")
    if {cid: value.tolist() for cid, value in predictions.items()} != read_json(
        run_folder / "predictions.json"
    ):
        raise ValueError("Artifact predictions differ")
    if scores(validation, predictions, probability) != read_json(run_folder / "scores.json"):
        raise ValueError("Artifact scoring differs")
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "verification.json",
        dict(
            status="VERIFIED",
            original_feature_records=len(banks),
            probability_values=sum(value.size for value in probability.values()),
            calendar_predictions=sum(len(value) for value in predictions.values()),
            all_scores_exact=True,
            test_truth_opened=False,
            run_manifest_sha256=sha(run_folder / "freeze.json"),
        ),
    )
    print(read_json(out / "verification.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "verify"])
    for name in ("data", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.data.resolve(), args.out.resolve())
    else:
        if args.run is None:
            parser.error("verify requires --run")
        verify(args.data.resolve(), args.run.resolve(), args.out.resolve())
