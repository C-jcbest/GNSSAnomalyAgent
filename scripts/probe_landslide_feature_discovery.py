"""Review frozen feature proposals, then compare fixed pools on original splits."""

from __future__ import annotations

import argparse
import ast
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import BASE, ROOT, export
from probe_landslide_visual_invariance import TRAINING, calibration_labels
from probe_landslide_visual_residual import THRESHOLDS, numerical_model
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import calibration_objective, save_json
from run_landslide_window_optimization import load_case_features, read_json
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha
from gnss_sim.landslide_baselines import activity_from_mask
from gnss_sim.landslide_evaluation import aggregate_scores, evaluate_case
from gnss_sim.landslide_feature_discovery import discovery_block_matrix
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual_invariance import invariance_inputs

PROPOSALS = BASE / "visual-feature-proposals-v2"
NOTES = {
    ("visual_numeric-r0", "center_increment_relative_to_window_dispersion"):
        "Reject: skips unavailable increments then indexes compressed position5; "
        "wrong center after a gap and IndexError with3-5 increments.",
    ("visual_numeric-r0", "center_increment_consistency"):
        "Uses3->4 and5->6 cosine, not immediate predecessor; no calendar compression.",
    ("visual_only-r0", "center_increment_relative_to_plateau"):
        "Deletes center only for background median; original center index retained. "
        "Any remote missing increment makes background medianNaN; retained unchanged.",
    ("visual_only-r1", "center_relative_velocity_ratio"):
        "Semantic duplicate of first function in this pool; retained unchanged, "
        "reported as duplicate, not selected or removed by downstream score.",
    ("visual_numeric-r1", "center_slope_continuity"):
        "Requires block4 even though only5,6,7 used; retained unchanged.",
    ("visual_numeric-r1", "center_local_acceleration"):
        "Second-difference magnitude, not physical acceleration or an ASD label.",
    ("numeric_only-r1", "center_acceleration_projection"):
        "Signed unscaled second-difference projection, not physical acceleration.",
}


def review_definition(function):
    """Inspect these finite proposals before compiling any accepted definition."""
    code = function["code"]
    tree = ast.parse(code)
    definition = tree.body[0]
    if (len(tree.body) != 1 or not isinstance(definition, ast.FunctionDef)
            or definition.name != function["name"] or definition.decorator_list
            or definition.args.defaults or definition.args.kw_defaults
            or definition.args.vararg or definition.args.kwarg
            or definition.args.kwonlyargs or definition.args.posonlyargs
            or len(definition.args.args) != 1 or definition.args.args[0].arg != "m"):
        raise ValueError("Unexpected proposal signature")
    # An allowlist applies only to the definitions already read by the author.
    allowed_numpy = {"any", "isnan", "diff", "abs", "nanmedian", "max", "sign",
                     "sum", "dot", "mean", "array", "delete", "median", "linalg"}
    forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With,
                 ast.Try, ast.Raise, ast.ClassDef, ast.Lambda, ast.While,
                 ast.Delete, ast.AsyncFunctionDef, ast.Await, ast.Yield)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError("Unexpected executable proposal operation")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError("Private attribute in proposal")
            if isinstance(node.value, ast.Name) and node.value.id == "np":
                if node.attr not in allowed_numpy:
                    raise ValueError("Unexpected numpy operation")
            elif node.attr not in {"norm", "any", "append"}:
                raise ValueError("Unexpected method in proposal")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            assigned = {n.id for n in ast.walk(tree)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            if node.id not in assigned | {"m", "np", "float", "range", "len", "any"}:
                raise ValueError("Unexpected external proposal name")
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, (ast.Subscript, ast.Attribute)) for target in targets):
                raise ValueError("Proposal writes through an external object")
    reasons = []
    if len(code.splitlines()) > 12:
        reasons.append("exceeds_frozen_12_line_limit")
    if function["name"] == "center_increment_relative_to_window_dispersion":
        reasons.append("compressed_calendar_wrong_center_and_possible_index_error")
    return reasons


def compile_reviewed(pools):
    """Execute only the exact definitions that passed the pre-score review."""
    result = {}
    for pool, functions in pools.items():
        result[pool] = []
        for function in functions:
            if review_definition(function):
                raise ValueError("Unreviewed definition cannot execute")
            namespace = {"np": np, "__builtins__": {"float": float, "range": range,
                                                   "len": len, "any": any}}
            exec(compile(function["code"], "<reviewed-GNSS-feature>", "exec"), namespace)
            result[pool].append(namespace[function["name"]])
    return result


def probe_functions(pools, functions):
    """Check missing-center, scalar, translation and mutation contracts pre-score."""
    ramp = np.arange(13)[:, None] * np.array([[.3, -.7, .1]])
    step = np.zeros((13, 3))
    step[6:] = [3, -7, 1]
    recent = np.zeros((13, 3))
    recent[5:] = np.arange(8)[:, None] * np.array([[.3, -.7, .1]])
    shapes = {"constant": np.zeros((13, 3)), "linear": ramp, "step": step,
              "recent_ramp": recent, "quadratic": ramp * np.arange(13)[:, None]}
    for slot in (0, 4, 5, 6, 7, 12):
        matrix = ramp.copy()
        matrix[slot] = np.nan
        shapes[f"gap_slot{slot}"] = matrix
    shapes["all_missing"] = np.full((13, 3), np.nan)
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for pool, callables in functions.items():
            for definition, function in zip(pools[pool], callables, strict=True):
                for name, matrix in shapes.items():
                    before = matrix.copy()
                    value = function(matrix)
                    if not isinstance(value, float) or np.isinf(value):
                        raise ValueError("Feature must return a finite scalar orNaN")
                    if not np.array_equal(before, matrix, equal_nan=True):
                        raise ValueError("Feature mutated original blocks")
                    if name in {"gap_slot5", "gap_slot6", "all_missing"} and not np.isnan(value):
                        raise ValueError("Feature manufactured missing center evidence")
                    for modified in (matrix + [21., -17., 9.], matrix * [-1, 1, -1]):
                        observed = function(modified)
                        if not np.isclose(value, observed, rtol=1e-9, atol=1e-9, equal_nan=True):
                            raise ValueError("Feature offset/reflection contract changed")
                    results.append({"pool": pool, "feature": definition["name"], "shape": name,
                                    "value": value if np.isfinite(value) else None})
    return results


def expert_features(blocks):
    """Six fixed generic center statistics, chosen before detector fitting."""
    before = blocks[:, 6] - blocks[:, 5]
    after = blocks[:, 7] - blocks[:, 6]
    magnitude = np.linalg.norm(before, axis=1)
    next_magnitude = np.linalg.norm(after, axis=1)
    product = magnitude * next_magnitude
    direction = np.divide(np.sum(before * after, axis=1), product,
                          out=np.full(len(blocks), np.nan), where=product > 0)
    persistence = np.linalg.norm(blocks[:, 7] - blocks[:, 5], axis=1)
    curvature = np.linalg.norm(after - before, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        background = np.nanmedian(np.linalg.norm(np.diff(blocks, axis=1), axis=2), axis=1)
    return np.column_stack((magnitude, next_magnitude, direction, persistence, curvature, background))


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    verify_bindings(PROPOSALS, "pre-proposal-freeze.json")
    verify_bindings(PROPOSALS, "proposal-freeze.json")
    proposals = read_json(PROPOSALS / "proposals.json")
    pools, decisions = {}, []
    for pool, item in proposals.items():
        pools[pool] = []
        for definition in item["proposal"]["functions"]:
            reasons = review_definition(definition)
            decisions.append({"pool": pool, "feature": definition["name"],
                              "lines": len(definition["code"].splitlines()),
                              "accepted": not reasons, "reasons": reasons,
                              "note": NOTES.get((pool, definition["name"]), "Original definition retained"),
                              "code": definition["code"]})
            if not reasons:
                pools[pool].append(definition)
        if not pools[pool]:
            raise ValueError("No reviewed features in proposed pool")
    # The complete static review is saved before the first execution.
    export(out / "definition-review.json", decisions)
    export(out / "reviewed-pools.json", pools)
    functions = compile_reviewed(pools)
    export(out / "pre-score-probes.json", probe_functions(pools, functions))
    arms = {"base18": {"type": "base"}, "expert_only": {"type": "expert"},
            "base_expert": {"type": "expert", "union": True},
            "blocks_only": {"type": "blocks"}, "base_blocks": {"type": "blocks", "union": True}}
    for pool in pools:
        arms[pool + "_only"] = {"type": "generated", "pool": pool}
        arms[pool + "_union"] = {"type": "generated", "pool": pool, "union": True}
    export(out / "protocol.json", {
        "arms": arms, "generation": str(PROPOSALS.relative_to(ROOT)),
        "calibration_splits": {target: read_json(source / "protocol.json")["split"]
                               for target, source in TRAINING.items()},
        "training": "All originally supported known training days; no parity thinning/augmentation",
        "model": "XGB120trees,depth2/4,lr.05,seed20261007,n_jobs2",
        "selection": "19thresholds .05..95;38 opportunities/arm;first max dailyF1+eventF1",
        "expert_order": ["norm5to6", "norm6to7", "cos5to6_6to7", "norm5to7",
                         "norm_second_difference5to7", "median_contiguous_increment_norm"],
        "all_blocks": "39noise-normalized block values in original slot order plus13availability flags",
        "support": "Original raw84 activity support, original14-day persistence, original stage head/support",
        "missing": "No interpolation or imputation; XGB sees NaNs; generated-only may use missing patterns",
        "gate": "Both primary visual_numeric unions strictly improve dailyF1 and correction net vs base18, "
                "benefit >=2validation records, improve weakR/eventR orFP, within .02F1 of base_blocks; "
                "secondary author6 validation reported separately and must not worsen dailyF1 or net. "
                "Multimodal necessity requires each main union above paired visual_only andnumeric_only unions.",
        "failure": "Stop fixed pools without regeneration, feature subset, seed, window or threshold rescue",
        "independence": "Historical development splits; six remote training examples only; author not blind",
    })
    bindings = [Path(__file__), ROOT / "src/gnss_sim/landslide_feature_discovery.py",
                ROOT / "tests/test_landslide_feature_discovery.py",
                ROOT / "scripts/probe_landslide_visual_invariance.py",
                ROOT / "scripts/probe_landslide_visual_residual.py",
                ROOT / "src/gnss_sim/landslide_visual_invariance.py"]
    for source in TRAINING.values():
        bindings.extend([source / "protocol.json", source / "reference-labels.csv",
                         source / "stage-xgboost.json"])
        split = read_json(source / "protocol.json")["split"]
        for cid in split["train"] + split["validation"]:
            case, _ = load_case_features(source, cid)
            export(out / "inputs" / source.name / (cid + ".json"), case.model_dump(mode="json"))
    binding = {}
    for path in bindings:
        copy = out / "source" / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)
        binding[path.relative_to(ROOT).as_posix()] = sha(path)
    export(out / "root-bindings.json", binding)
    export(out / "freeze.json", {path.relative_to(out).as_posix(): sha(path)
                                for path in out.rglob("*") if path.is_file()})
    save_json(out / "state.json", {"status": "prepared", "new_api_calls": 0})
    print("Reviewed18 definitions; accepted", sum(map(len, pools.values())),
          "; froze17 arms before detector scores", flush=True)


def case_inputs(case, features, functions):
    _, base, support = invariance_inputs(case, features)
    values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
    blocks, available, _, _ = discovery_block_matrix(values, features.noise)
    inputs = {"base": base, "expert": expert_features(blocks),
              "blocks": np.column_stack((blocks.reshape(len(values), -1), available.astype(float)))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for pool, callables in functions.items():
            inputs[pool] = np.array([[function(matrix) for function in callables] for matrix in blocks])
    return inputs, support


def arm_matrix(inputs, arm):
    key = arm.get("pool", arm["type"])
    matrix = inputs[key]
    if arm.get("union"):
        matrix = np.column_stack((inputs["base"], matrix))
    if np.isinf(matrix).any():
        raise ValueError("Features contain infinities")
    return matrix


def correction(reference, before, after):
    known = reference >= 0
    gained = int((known & (before != reference) & (after == reference)).sum())
    lost = int((known & (before == reference) & (after != reference)).sum())
    return gained, lost


def calibrate(out):
    verify_bindings(out, "freeze.json")
    for name, expected in read_json(out / "root-bindings.json").items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen fitting source/reference changed")
    verify_bindings(PROPOSALS, "proposal-freeze.json")
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Run only a never-started fixed-pool calibration")
    protocol = read_json(out / "protocol.json")
    functions = compile_reviewed(read_json(out / "reviewed-pools.json"))
    save_json(out / "state.json", {"status": "training", "new_api_calls": 0})
    selections, trials, costs, corrections, contracts = {}, [], [], [], []
    for target, source in TRAINING.items():
        started = time.perf_counter()
        split = protocol["calibration_splits"][target]
        ids = split["train"] + split["validation"]
        pairs = {cid: load_case_features(source, cid) for cid in ids}
        for cid in ids:
            saved = out / "inputs" / source.name / (cid + ".json")
            if pairs[cid][0].model_dump(mode="json") != read_json(saved):
                raise ValueError("Frozen calendar inputs changed")
        labels = calibration_labels(source, pairs)
        inputs = {cid: case_inputs(*pairs[cid], functions) for cid in ids}
        actual = {cid: np.array([int(row["activity_label"]) for row in labels[cid]]) for cid in ids}
        eligible = {cid: inputs[cid][1] & (actual[cid] >= 0) for cid in split["train"]}
        train_y = np.concatenate([actual[cid][eligible[cid]] for cid in split["train"]])
        stage = XGBClassifier()
        stage.load_model(source / "stage-xgboost.json")
        codes = {cid: stage.predict(pairs[cid][1].matrix).astype(int) + 1 for cid in split["validation"]}
        folder = out / target
        folder.mkdir()
        export(folder / "preprocessing.json", {"wall_seconds": time.perf_counter() - started,
                                               "includes_unused_original_raster": True,
                                               "training_days": len(train_y)})
        predictions, probability_bank = {}, {}
        for name, arm in protocol["arms"].items():
            matrix = {cid: arm_matrix(inputs[cid][0], arm) for cid in ids}
            train_x = np.concatenate([matrix[cid][eligible[cid]] for cid in split["train"]])
            arm_trials = []
            chosen_outputs = chosen_probability = None
            chosen = None
            for depth in (2, 4):
                started = time.perf_counter()
                model = numerical_model(depth)
                model.fit(train_x, train_y)
                model.save_model(folder / f"{name}-depth{depth}.json")
                probability = {cid: model.predict_proba(matrix[cid])[:, 1] for cid in split["validation"]}
                costs.append({"target": target, "arm": name, "depth": depth,
                              "dimensions": train_x.shape[1], "training_days": len(train_y),
                              "fit_and_validation_seconds": time.perf_counter() - started})
                for threshold in THRESHOLDS:
                    outputs, scores = {}, []
                    weak_tp = weak_total = 0
                    for cid in split["validation"]:
                        activity = activity_from_mask(probability[cid] >= threshold, inputs[cid][1])
                        prediction = learned_stages(codes[cid], activity, pairs[cid][1].supported)
                        outputs[cid] = prediction.to_dict()
                        scores.append(evaluate_case(labels[cid], prediction))
                        weak = np.array([int(row.get("weak_signal", 0)) == 1
                                         and int(row["activity_label"]) == 1 for row in labels[cid]])
                        weak_tp += int((weak & (activity == 1)).sum())
                        weak_total += int(weak.sum())
                    score = aggregate_scores(scores)
                    row = {"target": target, "arm": name, "depth": depth, "threshold": threshold,
                           "dimensions": train_x.shape[1], "scores": score,
                           "objective": calibration_objective(score), "weak_tp": weak_tp,
                           "weak_total": weak_total,
                           "weak_recall": weak_tp / weak_total if weak_total else None}
                    arm_trials.append(row)
                    if chosen is None or row["objective"] > chosen["objective"]:
                        chosen, chosen_outputs, chosen_probability = row, outputs, probability
            selections.setdefault(target, {})[name] = chosen
            predictions[name] = chosen_outputs
            probability_bank[name] = {cid: value.tolist() for cid, value in chosen_probability.items()}
            trials.extend(arm_trials)
            for cid in split["validation"]:
                value = chosen_outputs[cid]
                activity, stages = np.array(value["activity"]), np.array(value["stage"])
                observed, support = pairs[cid][1].observed, inputs[cid][1]
                if (np.any(activity[~support] != -1) or np.any(stages[~observed] != -1)
                        or np.any(activity[~observed] != -1) or np.any(stages[activity != 1] > 0)
                        or np.any(stages[(activity == 1) & ~pairs[cid][1].supported] != -1)):
                    raise ValueError("Prediction violated original observation/activity/stage support")
                gained, lost = correction(actual[cid], np.array(predictions["base18"][cid]["activity"]), activity)
                corrections.append({"target": target, "arm": name, "case_id": cid,
                                    "gained": gained, "lost": lost, "net": gained - lost})
                contracts.append({"target": target, "arm": name, "case_id": cid,
                                  "days": len(activity), "passed": True})
            print(target, name, "F1=", round(chosen["scores"]["daily"]["f1"], 6), flush=True)
        export(folder / "validation-predictions.json", predictions)
        export(folder / "validation-probabilities.json", probability_bank)
    export(out / "selection.json", selections)
    export(out / "validation-trials.json", trials)
    write_csv(out / "training-cost.csv", costs)
    write_csv(out / "per-record-corrections.csv", corrections)
    export(out / "prediction-contracts.json", contracts)
    conditions = {}
    multimodal = {}
    for repetition in (0, 1):
        name = f"visual_numeric-r{repetition}_union"
        rows = [row for row in corrections if row["target"] == "primary_process" and row["arm"] == name]
        primary = selections["primary_process"]
        main, base, all_blocks = primary[name], primary["base18"], primary["base_blocks"]
        conditions[f"r{repetition}_primary_f1"] = main["scores"]["daily"]["f1"] > base["scores"]["daily"]["f1"]
        conditions[f"r{repetition}_primary_net"] = sum(row["net"] for row in rows) > 0
        conditions[f"r{repetition}_records"] = sum(row["net"] > 0 for row in rows) >= 2
        conditions[f"r{repetition}_target_error_gain"] = (
            main["weak_recall"] > base["weak_recall"]
            or main["scores"]["events"]["recall"] > base["scores"]["events"]["recall"]
            or main["scores"]["daily"]["fp"] < base["scores"]["daily"]["fp"])
        conditions[f"r{repetition}_information_control"] = (
            main["scores"]["daily"]["f1"] >= all_blocks["scores"]["daily"]["f1"] - .02)
        secondary = selections["secondary_developer"]
        conditions[f"r{repetition}_secondary_no_f1_loss"] = (
            secondary[name]["scores"]["daily"]["f1"] >= secondary["base18"]["scores"]["daily"]["f1"])
        conditions[f"r{repetition}_secondary_no_net_loss"] = (
            sum(row["net"] for row in corrections
                if row["target"] == "secondary_developer" and row["arm"] == name) >= 0)
        for control in ("visual_only", "numeric_only"):
            multimodal[f"r{repetition}_above_{control}"] = (
                main["scores"]["daily"]["f1"] > primary[f"{control}-r{repetition}_union"]["scores"]["daily"]["f1"])
    export(out / "decision.json", {
        "conditions": conditions, "multimodal_necessity": multimodal,
        "decision": "PROCEED_TO_NEW_CONFIRMATION" if all(conditions.values()) else "STOP_FIXED_GENERATED_POOLS",
        "new_api_calls": 0, "independent_confirmation": False})
    save_json(out / "state.json", {"status": "complete", "models": len(costs), "trials": len(trials),
                                   "new_api_calls": 0})
    export(out / "result-freeze.json", {path.relative_to(out).as_posix(): sha(path)
                                       for path in out.rglob("*") if path.is_file()
                                       and path.name not in {"result-freeze.json", "state.json"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "calibrate"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out.resolve())
    else:
        calibrate(args.out.resolve())
