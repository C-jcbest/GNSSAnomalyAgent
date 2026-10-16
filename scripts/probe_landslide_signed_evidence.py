"""Freeze, execute and score one axis-preserving evidence development experiment."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from analyze_landslide_dual_reference_confirmation import score_target
from analyze_landslide_review_utility_v2 import verify_receipts
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import save_json
from run_landslide_window_optimization import load_case_features, read_json
from write_landslide_dual_reference_report import full_table
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_priorities,
    review_prompt,
)
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_review_utility import correction_utility
from gnss_sim.landslide_signed_evidence import signed_block_facts, signed_review_prompt
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import parse_window_activity

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
PARENT = BASE / "sequential-followup-confirmation-comparison-v1/run"
ANALYSIS = BASE / "sequential-followup-confirmation-analysis-v1"
DATA = BASE / "sequential-followup-confirmation-data-v1"
PROCESS_CODES = BASE / "sequential-followup-process-predictions-v1/stage-codes.json"
SOURCES = [
    Path(__file__),
    ROOT / "src/gnss_sim/landslide_signed_evidence.py",
    ROOT / "src/gnss_sim/landslide_adaptive_review.py",
    ROOT / "src/gnss_sim/landslide_visual.py",
    ROOT / "scripts/probe_landslide_revision_scope.py",
    ROOT / "scripts/analyze_landslide_dual_reference_confirmation.py",
]


def prepare(out):
    verify_bindings(PARENT, "freeze.json")
    verify_bindings(PARENT, "prediction-freeze.json")
    verify_manifest(ANALYSIS)
    verify_receipts(PARENT)
    config = read_json(PARENT / "protocol.json")
    policy = ReviewPolicy(**config["policy"])
    out.mkdir(exist_ok=False)
    (out / "images").mkdir()
    (out / "source").mkdir()
    for path in SOURCES:
        (out / "source" / path.name).write_bytes(path.read_bytes())
    for name in ("stage-xgboost.json", "selections.json"):
        shutil.copyfile(PARENT / name, out / name)
    shutil.copyfile(ANALYSIS / "primary_ai/predictions.json", out / "prior-predictions.json")
    shutil.copyfile(ANALYSIS / "used-requests.json", out / "prior-used-requests.json")
    initial = read_json(out / "prior-predictions.json")
    selections = read_json(out / "selections.json")
    plan, facts = [], {}
    order = np.random.default_rng(20261011)
    for cid in config["case_ids"]:
        shutil.copytree(PARENT / "inputs" / cid, out / "inputs" / cid)
        raw_name = f"{cid}-raw.png"
        shutil.copyfile(PARENT / "images" / raw_name, out / "images" / raw_name)
        case, feature = load_case_features(out, cid)
        long = raw_motion_evidence(case, feature)
        for rep in (0, 1):
            choice = selections[f"{cid}-r{rep}"]
            windows = [(row["start"], row["stop"]) for row in choice["priorities"]]
            global_activity = np.array(initial[f"direct_visual_r{rep}"][cid]["activity"])
            priorities, selected = review_priorities(
                global_activity, feature, long, windows, policy
            )
            if priorities != choice["priorities"] or selected != choice["selected"]:
                raise ValueError("Original motion selection no longer replays exactly")
            adaptive = {row["index"] for row in selected}
            for index in sorted(adaptive | set(choice["fixed_indices"])):
                target = priorities[index]
                image_name = f"{cid}-window-{index:02d}.png"
                if not (out / "images" / image_name).exists():
                    shutil.copyfile(PARENT / "images" / image_name, out / "images" / image_name)
                arms = ["raw", "norm", "signed"] if index in adaptive else ["signed"]
                facts[f"{cid}-{index}"] = signed_block_facts(
                    case, feature.noise, target["start"], target["stop"]
                )
                for arm in order.permutation(arms):
                    prompt = (
                        signed_review_prompt(case, feature.noise, target["start"], target["stop"])
                        if arm == "signed"
                        else review_prompt(
                            feature, long, target["start"], target["stop"], arm == "norm"
                        )
                    )
                    plan.append(
                        {
                            "request_id": f"signed-probe-{cid}-window-{index:02d}-{arm}-r{rep}",
                            "case_id": cid,
                            "repeat": rep,
                            "arm": str(arm),
                            **target,
                            "prompt": prompt,
                            "images": [raw_name, image_name],
                        }
                    )
    write_json(out / "query-plan.json", plan)
    write_json(out / "raw-facts.json", facts)
    write_json(
        out / "protocol.json",
        {
            "date": "2026-10-11",
            "scope": "All12 exposed records0157-0168; development only. No new data or author labels.",
            "case_ids": config["case_ids"],
            "model": config["model"],
            "policy": config["policy"],
            "primary": "signed_adaptive_r{repeat}__conflict_scope",
            "mechanism": "Replace norm-only numeric feedback with fixed14-day signed NEU raw median facts. No predicted movement classes, thresholds, gap interpolation or initial-answer restoration.",
            "controls": "All46 original methods incl six target-matched numerical controls; contemporaneous raw/norm reviews at the same selected windows; signed fixed, same-count early/late, first-only, and whole-target controls.",
            "fixed": "Original initial answers,21/42 motion conflicts,14-day admission,two disjoint180-day windows,49-day authority,raw pixels,common target-matched stage head. No stage innovation claimed.",
            "continuation_gate": "Each repeat F1 strictly exceeds historical scoped main and fresh raw and norm scoped controls; mean F1 and summed net definite correction strictly exceed signed fixed whole-target and cheaper signed first-only; mean weak recall not below original. Any failure stops; process target cannot rescue primary failure. No seed,block,prompt,threshold,replicate or subset search.",
            "maximum_new_requests": len(plan),
            "repetitions": 2,
            "retries": 0,
            "hard_timeout_seconds": 1800,
            "new_training": False,
            "calibration": "Unchanged observations and targets; reuse frozen6/6 training-validation target-matched baseline calibrations, no tuning on these exposed records.",
            "request_policy": "temperature.1/top_p.3/max_tokens2048/no_thinking; arms randomized once seed20261011; all planned questions attempted once unless hard timeout.",
            "reference": "Author-AI visual/numeric reference plus separate latent-process sensitivity. Author has seen prior labels/results; neither independent expert nor field gold standard.",
        },
    )
    bindings = {str(path): sha(path) for path in SOURCES}
    for path in (
        PARENT / "freeze.json",
        PARENT / "prediction-freeze.json",
        ANALYSIS / "manifest.json",
        PROCESS_CODES,
        DATA / "codex-review/delivery-manifest.json",
    ):
        bindings[str(path)] = sha(path)
    write_json(out / "source-bindings.json", bindings)
    write_json(
        out / "freeze.json",
        {path.relative_to(out).as_posix(): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    save_json(out / "state.json", {"status": "prepared", "completed": 0, "total": len(plan)})
    print(
        {"status": "prepared", "new_requests": len(plan), "records": len(config["case_ids"])},
        flush=True,
    )


def verify(out):
    verify_bindings(out, "freeze.json")
    for name, expected in read_json(out / "source-bindings.json").items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Bound source changed: {name}")


def build_predictions(out, reviews):
    predictions = read_json(out / "prior-predictions.json")
    used = read_json(out / "prior-used-requests.json")
    choices = read_json(out / "selections.json")
    config = read_json(out / "protocol.json")
    head = XGBClassifier()
    head.load_model(out / "stage-xgboost.json")
    for cid in config["case_ids"]:
        case, feature = load_case_features(out, cid)
        codes = head.predict(feature.matrix).astype(int) + 1
        long = raw_motion_evidence(case, feature)
        for rep in (0, 1):
            initial = np.array(predictions[f"direct_visual_r{rep}"][cid]["activity"])
            choice = choices[f"{cid}-r{rep}"]
            selected = choice["selected"]
            fixed = [choice["priorities"][index] for index in choice["fixed_indices"]]
            count = len(selected)
            first = (
                [min(selected, key=lambda row: (-row["conflict_days"], row["start"]))]
                if selected
                else []
            )
            targets = {
                "fresh_raw": ("raw", selected),
                "fresh_norm": ("norm", selected),
                "signed_adaptive": ("signed", selected),
                "signed_fixed": ("signed", fixed),
                "signed_same_count_early": ("signed", fixed[:count]),
                "signed_same_count_late": ("signed", fixed[len(fixed) - count :] if count else []),
                "signed_first_only": ("signed", first),
            }
            authority = conflict_neighbourhood(
                initial, feature, long, ReviewPolicy(**config["policy"])
            )[1]
            for stem, (arm, windows) in targets.items():
                keys = [
                    f"signed-probe-{cid}-window-{target['index']:02d}-{arm}-r{rep}"
                    for target in windows
                ]
                local = [
                    dict(reviews[key], activity=np.array(reviews[key]["activity"])) for key in keys
                ]
                reviewed = combine_reviews(initial, local, feature.observed).activity
                scope = target_scopes(authority, feature.observed, windows)[0]
                variants = {
                    "conflict_scope": scoped_activity(initial, reviewed, scope, feature.observed)
                }
                if stem in ("fresh_raw", "fresh_norm", "signed_adaptive", "signed_fixed"):
                    variants["whole_target"] = reviewed
                for variant, activity in variants.items():
                    method = f"{stem}_r{rep}__{variant}"
                    predictions.setdefault(method, {})[cid] = learned_stages(
                        codes, activity, feature.supported
                    ).to_dict()
                    used.setdefault(method, {})[cid] = [f"{cid}-direct-r{rep}"] + keys
    return predictions, used


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Execute only a never-started frozen plan; no repeats")
    config = read_json(out / "protocol.json")
    plan = read_json(out / "query-plan.json")
    transport = VisualRequests(
        out / "requests", ROOT / ".env", config["model"], config["maximum_new_requests"]
    )
    reviews, failures = {}, {}
    started = time.monotonic()
    save_json(out / "state.json", {"status": "running", "completed": 0, "total": len(plan)})
    for index, row in enumerate(plan):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            save_json(
                out / "state.json", {"status": "timeout", "completed": index, "total": len(plan)}
            )
            raise RuntimeError("Registered wall-time limit exceeded; no automatic restart")
        _, feature = load_case_features(out, row["case_id"])
        activity = np.full(len(feature.observed), -1, dtype=int)
        status = "success"
        try:
            answer = transport.ask(
                row["request_id"], row["prompt"], [out / "images" / name for name in row["images"]]
            )
            activity = parse_window_activity(answer, feature.observed, row["start"], row["stop"])
        except (ValueError, RuntimeError) as error:
            failures[row["request_id"]] = str(error)
            status = "failed"
        reviews[row["request_id"]] = {
            "start": row["start"],
            "stop": row["stop"],
            "activity": activity.tolist(),
            "status": status,
        }
        save_json(out / "reviews.json", reviews)
        save_json(out / "failures.json", failures)
        save_json(
            out / "state.json",
            {
                "status": "running",
                "completed": index + 1,
                "total": len(plan),
                "seconds": time.monotonic() - started,
            },
        )
        print(f"Signed evidence {index + 1}/{len(plan)}: {row['request_id']} {status}", flush=True)
    predictions, used = build_predictions(out, reviews)
    write_json(out / "predictions.json", predictions)
    write_json(out / "used-requests.json", used)
    verify(out)
    write_json(
        out / "prediction-freeze.json",
        {
            name: sha(out / name)
            for name in ("reviews.json", "failures.json", "predictions.json", "used-requests.json")
        },
    )
    save_json(
        out / "state.json",
        {
            "status": "predictions_complete",
            "completed": len(plan),
            "total": len(plan),
            "seconds": time.monotonic() - started,
        },
    )


def score(out, destination):
    verify(out)
    verify_bindings(out, "prediction-freeze.json")
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("Complete the entire plan before scoring")
    destination.mkdir(exist_ok=False)
    receipts = verify_receipts(PARENT)
    new_receipts = verify_receipts(out)
    if set(receipts) & set(new_receipts):
        raise ValueError("Historical and contemporaneous receipt identities collide")
    receipts.update(new_receipts)
    replayed, used = build_predictions(out, read_json(out / "reviews.json"))
    if replayed != read_json(out / "predictions.json") or used != read_json(
        out / "used-requests.json"
    ):
        raise ValueError("All new predictions must replay exactly")
    authored = read_json(DATA / "codex-review/activity-decisions.json")[
        "weak_interiors_authored_before_predictions"
    ]
    features = {
        cid: load_case_features(out, cid)[1] for cid in read_json(out / "protocol.json")["case_ids"]
    }
    codes = read_json(PROCESS_CODES)
    secondary = read_json(ANALYSIS / "secondary_process/predictions.json")
    for method, records in replayed.items():
        if method not in secondary:
            secondary[method] = {
                cid: learned_stages(
                    np.array(codes[cid]), np.array(value["activity"]), features[cid].supported
                ).to_dict()
                for cid, value in records.items()
            }
    tables, decision, correction_rows = {}, {}, []
    for target, predictions in (("primary_ai", replayed), ("secondary_process", secondary)):
        labels = read_json(ANALYSIS / target / "reference-labels.json")
        observed = {cid: features[cid].observed for cid in labels}
        weak = {}
        for cid, rows in labels.items():
            if target == "primary_ai":
                mask = np.zeros(len(rows), dtype=bool)
                for start, stop in authored.get(cid, []):
                    mask[start:stop] = True
                weak[cid] = mask & (np.array([row["activity_label"] for row in rows]) == 1)
            else:
                weak[cid] = np.array(
                    [row["weak_signal"] == 1 and row["activity_label"] == 1 for row in rows]
                )
        results, summaries, records = score_target(
            labels, observed, weak, predictions, used, receipts
        )
        tables[target] = summaries
        folder = destination / target
        folder.mkdir()
        write_json(folder / "predictions.json", predictions)
        write_json(folder / "results.json", results)
        write_csv(folder / "summary.csv", summaries)
        write_csv(folder / "per-record.csv", records)
        values = {row["method"]: row for row in summaries}
        comparisons = []
        for stem, variant in (
            ("adaptive_no_feedback", "conflict_scope"),
            ("fresh_raw", "conflict_scope"),
            ("fresh_norm", "conflict_scope"),
            ("signed_fixed", "whole_target"),
            ("signed_first_only", "conflict_scope"),
        ):
            differences, weak_differences, net = [], [], 0
            for rep in (0, 1):
                main = f"signed_adaptive_r{rep}__conflict_scope"
                control = f"{stem}_r{rep}__{variant}"
                differences.append(values[main]["f1"] - values[control]["f1"])
                weak_differences.append(
                    values[main]["weak_recall"] - values[control]["weak_recall"]
                )
                for cid, rows in labels.items():
                    counts = correction_utility(
                        np.array([row["activity_label"] for row in rows]),
                        np.array(predictions[control][cid]["activity"]),
                        np.array(predictions[main][cid]["activity"]),
                        observed[cid],
                    )
                    net += counts["net_correct"]
                    correction_rows.append(
                        {
                            "target": target,
                            "case_id": cid,
                            "main": main,
                            "control": control,
                            **counts,
                        }
                    )
            if stem in ("adaptive_no_feedback", "fresh_raw", "fresh_norm"):
                passed = all(value > 0 for value in differences)
                if stem == "adaptive_no_feedback":
                    passed = passed and np.mean(weak_differences) >= 0
            else:
                passed = np.mean(differences) > 0 and net > 0
            comparisons.append(
                {
                    "control": stem,
                    "f1_difference_by_repeat": differences,
                    "mean_f1_difference": float(np.mean(differences)),
                    "net_correct_both_repeats": net,
                    "mean_weak_recall_difference": float(np.mean(weak_differences)),
                    "passed": bool(passed),
                }
            )
        decision[target] = {
            "passed": all(row["passed"] for row in comparisons),
            "comparisons": comparisons,
        }
    write_csv(destination / "corrections.csv", correction_rows)
    decision["candidate"] = (
        "CONTINUE_TO_NEW_CONFIRMATION" if decision["primary_ai"]["passed"] else "STOPPED"
    )
    decision["default_changed"] = False
    write_json(destination / "decision.json", decision)
    write_json(
        destination / "cost.json",
        {
            "new_launches": len(list((out / "requests").glob("*.started.json"))),
            "new_receipts": len(new_receipts),
            "reported_tokens": sum(
                row.get("usage", {}).get("total_tokens", 0) for row in new_receipts.values()
            ),
            "failures": len(read_json(out / "failures.json")),
            "seconds": read_json(out / "state.json")["seconds"],
            "historical_receipts_reused": len(receipts) - len(new_receipts),
            "new_records": 0,
            "new_training": False,
        },
    )
    text = "# 带符号原始块证据：完整开发比较\n\n## Material Passport\n\n类型：code experiment；参考：已暴露作者AI／潜在过程；状态：预测完成、独立逐日计数核验、确定性预测重放；不属于独立现场验证。\n\n"
    text += f"候选决策：**{decision['candidate']}**。原46方法与全部新消融保留，两个重复均报告。\n\n"
    for target, summaries in tables.items():
        text += f"## {target}\n\n" + full_table(summaries) + "\n\n"
    (destination / "report.md").write_text(text, encoding="utf-8")
    write_json(
        destination / "manifest.json",
        {
            "files": {
                path.relative_to(destination).as_posix(): sha(path)
                for path in destination.rglob("*")
                if path.is_file()
            }
        },
    )
    print(
        {
            "decision": decision["candidate"],
            "method_target_scores": sum(len(rows) for rows in tables.values()),
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out.resolve())
    elif args.action == "run":
        execute(args.out.resolve())
    else:
        if args.analysis is None:
            parser.error("score requires --analysis")
        score(args.out.resolve(), args.analysis.resolve())
