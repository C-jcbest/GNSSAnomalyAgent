"""Train independent spike specialists and retain the original visual role."""
from __future__ import annotations

import argparse
import copy
import shutil
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
from experiment_exp07 import rebuild, score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from summarize_exp06 import pooled_bootstrap

from gnss_research.features import morphology_features, numerical_evidence
from gnss_research.files import read_json, sha256, source_hashes, write_json
from gnss_research.schema import Prediction, event_masks
from gnss_research.segmentation import segment_proposals
from gnss_research.segmented_review import prepare_pair_focus, repair_return_pairs
from gnss_research.vision import SYSTEM, call_visual, credentials

CLASSIFIERS = ("binary_rf", "binary_et", "binary_rf_default")


def data_records(run: Path):
    registered = read_json(run / "registration.json")
    if registered["source_hashes"] != source_hashes():
        raise ValueError("production sources differ from the baseline")
    dataset = Path(registered["dataset"])
    manifest = read_json(dataset / "manifest.json")
    if sha256(dataset / "manifest.json") != registered["manifest_sha256"]:
        raise ValueError("baseline dataset changed")
    records = [record for record in manifest["records"]
               if record["station_id"] in registered["station_ids"]]
    return registered, dataset, manifest, records


def fit_spike_specialists(dataset: Path, records: list[dict], seed: int, names: list[str]):
    matrices = []
    targets = []
    training_ids = []
    for record in records:
        if record["split"] != "train":
            continue
        observation = read_json(dataset / record["observation"])
        evidence = numerical_evidence(observation)
        labels = read_json(dataset / record["truth"])
        spike_mask = event_masks(Prediction.model_validate({"events": labels["events"]}),
                                evidence.observed)["spike"]
        eligible = evidence.observed & (np.arange(len(evidence.values)) > evidence.background_end)
        matrices.append(morphology_features(evidence)[eligible])
        targets.append(spike_mask[eligible])
        training_ids.append(record["station_id"])
    matrix, target = np.concatenate(matrices), np.concatenate(targets)
    models = {}
    parameters = {}
    for name in names:
        estimator = ExtraTreesClassifier if name == "binary_et" else RandomForestClassifier
        weight = None if name == "binary_rf_default" else "balanced"
        forest = estimator(random_state=seed, class_weight=weight)
        model = make_pipeline(SimpleImputer(), forest)
        model.fit(matrix, target)
        models[name] = model
        parameters[name] = forest.get_params()
    return models, {"training_stations": training_ids, "training_rows": len(target),
                    "positive_spike_rows": int(target.sum()), "feature_count": matrix.shape[1],
                    "sklearn_version": version("scikit-learn"), "parameters": parameters,
                    "decision": "unmodified library predict; no score cutoff tuning",
                    "adjustments": ["single binary spike target instead of four-output target",
                                    "same existing 48 GNSS morphology features",
                                    "training-only median imputation",
                                    "balanced and random_state explicitly set; other parameters default"],
                    "paper_reproduction": False}


def replace_spikes(observation: dict, original: dict, mask: np.ndarray) -> dict:
    events = [copy.deepcopy(event) for event in original["prediction"]["events"]
              if event["kind"] != "spike"]
    for day in np.flatnonzero(mask):
        events.append({"kind": "spike", "start": int(day), "end": int(day),
                       "status": "confirmed", "phases": [],
                       "evidence": "Independent binary spike classifier library predict"})
    result = rebuild(observation, original, events)
    result.pop("type_scores", None)
    result.update(spike_ownership="independent binary classifier replaces original spike set")
    return result


def numeric_stage(run: Path, out: Path):
    registered, dataset, manifest, records = data_records(run)
    names = list(CLASSIFIERS)
    if registered["split"] == "test":
        names = [registered["experiment"]["selected_classifier"]]
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "experiment-source.py")
    write_json(out / "registration.json", {
        "baseline_registration_sha256": sha256(run / "registration.json"),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"), "classifiers": names,
        "split": registered["split"], "script_sha256": sha256(Path(__file__)),
        "source_hashes": source_hashes(), "protocol": registered["experiment"]})
    models, training = fit_spike_specialists(dataset, manifest["records"], registered["seed"], names)
    write_json(out / "training.json", training)
    predictions = {method: {} for method in ("segment_numeric", "rf_split_temporal", "rf_balanced",
                                             *[name + "_numeric" for name in names])}
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        evidence = numerical_evidence(observation)
        matrix = morphology_features(evidence)
        original = read_json(run / "predictions/rf_split_temporal" / station / "result.json")
        for name, model in models.items():
            mask = model.predict(matrix).astype(bool) & evidence.observed
            mask[:evidence.background_end + 1] = False
            source = replace_spikes(observation, original, mask)
            segmented = segment_proposals(observation, source, evidence)
            write_json(out / "predictions" / (name + "_source") / station / "result.json", source)
            write_json(out / "predictions" / (name + "_numeric") / station / "result.json", segmented)
            predictions[name + "_numeric"][station] = segmented
        for method in ("segment_numeric", "rf_split_temporal", "rf_balanced"):
            predictions[method][station] = read_json(run / "predictions" / method / station / "result.json")
        print(station, "specialists complete", flush=True)
    summary, rows = score(dataset, records, predictions)
    baseline = summary["segment_numeric"]
    selection_candidates = [method.removesuffix("_repair")
                            for method in registered["experiment"]["candidates"]]
    decisions = {}
    for name in names:
        values = summary[name + "_numeric"]
        difference = values["types_daily_macro_f1"] - baseline["types_daily_macro_f1"]
        worst = min(values["types"][kind]["daily_micro"]["f1"]
                    - baseline["types"][kind]["daily_micro"]["f1"] for kind in baseline["types"])
        decisions[name] = {"difference": difference, "worst_type_difference": worst,
                           "eligible_for_visual": difference >= .02 and worst >= -.05
                                                  and name in selection_candidates}
    write_json(out / "summary.json", {"methods": summary, "decisions": decisions,
               "split": registered["split"], "dataset_seed": manifest["seed"], "requests": 0})
    write_json(out / "per-station.json", rows)
    print({method: values["types_daily_macro_f1"] for method, values in summary.items()})
    print(decisions)


def prepare_stage(run: Path, numeric: Path, out: Path, env: Path):
    registered, dataset, _, records = data_records(run)
    summary = read_json(numeric / "summary.json")
    names = [name for name, decision in summary["decisions"].items() if decision["eligible_for_visual"]]
    if registered["split"] == "test":
        names = [registered["experiment"]["selected_classifier"]]
    if not names:
        raise ValueError("no candidate cleared the numeric gate; cancel integration without model requests")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "experiment-source.py")
    settings = credentials(env)
    parents = ["original", *names]
    pending = []
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        for parent in parents:
            folder = run / "predictions/rf_split_temporal" if parent == "original" else numeric / "predictions" / (parent + "_source")
            source = read_json(folder / station / "result.json")
            work = out / "predictions" / parent / station
            payload = prepare_pair_focus(observation, source, work, settings[2], SYSTEM)
            if payload is not None:
                pending.append({"station": station, "parent": parent,
                                "request_sha256": sha256(work / "request.json")})
    pilot = sorted({job["station"] for job in pending})[:6]
    write_json(out / "registration.json", {"baseline_run": str(run.resolve()),
        "dataset": str(dataset.resolve()), "split": registered["split"],
        "dataset_seed": read_json(dataset / "manifest.json")["seed"],
        "station_ids": registered["station_ids"], "parents": parents, "pending": pending,
        "pilot_ids": pilot, "model": settings[2], "source_hashes": source_hashes(),
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": sha256(Path(__file__).with_name("experiment_exp07.py")),
        "protocol_path": registered["experiment"]["protocol_path"],
        "protocol_sha256": sha256(Path(registered["experiment"]["protocol_path"])),
        "numeric_summary_sha256": sha256(numeric / "summary.json")})
    print({"parents": parents, "pilot_ids": pilot, "request_upper_bound": len(pending)})


def visual_stage(out: Path, env: Path, phase: str):
    registered = read_json(out / "registration.json")
    if registered["source_hashes"] != source_hashes() or registered["script_sha256"] != sha256(Path(__file__)):
        raise ValueError("registered source changed")
    if registered["helper_sha256"] != sha256(Path(__file__).with_name("experiment_exp07.py")):
        raise ValueError("registered helper changed")
    if registered["protocol_sha256"] != sha256(Path(registered["protocol_path"])):
        raise ValueError("registered protocol changed")
    sentinel = out / (phase + "-started.json")
    if sentinel.exists():
        raise ValueError("phase already started; no implicit retry")
    dataset = Path(registered["dataset"])
    records = {record["station_id"]: record for record in read_json(dataset / "manifest.json")["records"]}
    settings = credentials(env)
    if settings[2] != registered["model"]:
        raise ValueError("registered model changed")
    jobs = [job for job in registered["pending"]
            if (job["station"] in registered["pilot_ids"]) == (phase == "pilot")]
    write_json(sentinel, {"requests": len(jobs), "retry_count": 0})

    def infer(job):
        work = out / "predictions" / job["parent"] / job["station"]
        if sha256(work / "request.json") != job["request_sha256"]:
            raise ValueError("request changed")
        observation = read_json(dataset / records[job["station"]]["observation"])
        result = call_visual(observation, job["parent"], work, settings)
        print(job["station"], job["parent"], result["status"], flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(infer, jobs))
    write_json(out / (phase + "-completed.json"), {"requests": len(jobs)})


def summarize_stage(run: Path, numeric: Path, visual: Path, out: Path):
    registered, dataset, _, records = data_records(run)
    out.mkdir(parents=True, exist_ok=False)
    frozen = read_json(visual / "registration.json")
    parents = frozen["parents"]
    mapping = {parent: "pair_focus_repair" if parent == "original" else parent + "_repair"
               for parent in parents}
    predictions = {method: {} for method in mapping.values()}
    costs = {method: {"requests": 0, "failures": 0} for method in mapping.values()}
    for record in records:
        station = record["station_id"]
        observation = read_json(dataset / record["observation"])
        for parent, method in mapping.items():
            work = visual / "predictions" / parent / station
            response = read_json(work / "result.json")
            result = repair_return_pairs(observation, response, read_json(work / "candidates.json"))
            predictions[method][station] = result
            write_json(out / "predictions" / method / station / "result.json", result)
            costs[method]["requests"] += response.get("request_count", 0)
            costs[method]["failures"] += response["status"] != "success"
    summary, rows = score(dataset, records, predictions)
    numeric_rows = read_json(numeric / "per-station.json")
    numeric_summary = read_json(numeric / "summary.json")
    summary.update(numeric_summary["methods"])
    rows.update(numeric_rows)
    baseline = summary["pair_focus_repair"]
    decisions = {}
    for method in mapping.values():
        if method == "pair_focus_repair":
            continue
        values = summary[method]
        difference = values["types_daily_macro_f1"] - baseline["types_daily_macro_f1"]
        worst = min(values["types"][kind]["daily_micro"]["f1"]
                    - baseline["types"][kind]["daily_micro"]["f1"] for kind in baseline["types"])
        decisions[method] = {"difference": difference, "worst_type_difference": worst,
                            "eligible_for_new_test": difference >= .02 and worst >= -.05,
                            "paired_bootstrap": pooled_bootstrap(rows[method], rows["pair_focus_repair"], registered["seed"])}
    write_json(out / "summary.json", {"methods": summary, "decisions": decisions,
               "split": registered["split"], "dataset_seed": numeric_summary["dataset_seed"],
               "station_ids": registered["station_ids"], "costs": costs,
               "baseline_run": str(run.resolve()), "numeric_out": str(numeric.resolve()),
               "visual_run": str(visual.resolve())})
    write_json(out / "per-station.json", rows)
    shutil.copyfile(__file__, out / "experiment-source.py")
    print({method: values["types_daily_macro_f1"] for method, values in summary.items()})
    print(decisions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("numeric", "prepare", "pilot", "remaining", "summarize"))
    parser.add_argument("--run", type=Path, default=Path("runs/exp08-development-baseline-v1"))
    parser.add_argument("--numeric", type=Path, default=Path("results/exp08-development-numeric-v1"))
    parser.add_argument("--visual", type=Path, default=Path("runs/exp08-development-visual-v1"))
    parser.add_argument("--out", type=Path, default=Path("results/exp08-development-v1"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if args.stage == "numeric":
        numeric_stage(args.run, args.numeric)
    elif args.stage == "prepare":
        prepare_stage(args.run, args.numeric, args.visual, args.env_file)
    elif args.stage in ("pilot", "remaining"):
        visual_stage(args.visual, args.env_file, args.stage)
    else:
        summarize_stage(args.run, args.numeric, args.visual, args.out)


if __name__ == "__main__":
    main()
