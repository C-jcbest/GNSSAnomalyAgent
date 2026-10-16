"""Pair alternative raw-evidence displays with unchanged bounded visual review."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from analyze_landslide_core_lockbox import developer_predictions
from analyze_landslide_review_utility_v2 import verify_receipts
from diagnose_landslide_feedback_contribution import compare_cases
from prepare_landslide_ai_reference_holdout import verify_manifest, write_csv
from probe_landslide_feedback_placement import fixed_predictions
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity, target_scopes
from report_landslide_core_lockbox import full_table
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_comparison import save_json
from run_landslide_window_optimization import load_case_features, read_json
from score_landslide_core_lockbox import score_target
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import ReviewPolicy, combine_reviews, raw_motion_evidence
from gnss_sim.landslide_displacement_comparison_view import render_comparison_supplement
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import VisualRequests
from gnss_sim.landslide_window_experiment import WINDOW_TASK, parse_window_activity

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/landslide-adaptive-review-2026-10-09"
DATA = BASE / "core-lockbox-data-v1"
PARENT = BASE / "core-lockbox-confirmation-v1"
PRIOR = BASE / "axis-persistence-report-v1"
NOTICE = """
The third image is a supplementary representation of the SAME local raw N/E/U.
Both available forms use disjoint14-day raw medians, only with at least eight
complete N/E/U observations. These are not independent sensors or activity labels.
A trace displays each median relative to the first valid median, divided by the
same full-record per-axis daily-difference MAD noise proxy. A comparison map shows
(column block median - row block median) / that noise proxy. Both map axes are
original days; map gray means an unavailable block, white means zero difference.
Map colors are display-clipped at +/-3; this is NOT a detection threshold or
calibrated uncertainty. Signed colors indicate displacement direction, not classes.
An endpoint difference across missing blocks does not prove motion during a gap.
Do not copy block edges as activity boundaries. Use the preserved original raw
curves to decide actual continuing displacement, platform, isolated step and
uncertainty. No rule prediction or hidden label is supplied by this supplement.
"""


def prepare(out):
    verify_manifest(PRIOR)
    config = read_json(PARENT / "protocol.json")
    selections = read_json(PARENT / "selections.json")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source").mkdir()
    sources = [
        Path(__file__),
        ROOT / "src/gnss_sim/landslide_displacement_comparison_view.py",
        ROOT / "tests/test_landslide_displacement_comparison_view.py",
        ROOT / "scripts/probe_landslide_revision_scope.py",
        ROOT / "src/gnss_sim/landslide_visual.py",
        ROOT / "src/gnss_sim/landslide_window_experiment.py",
        ROOT / "src/gnss_sim/landslide_adaptive_review.py",
        ROOT / "scripts/score_landslide_core_lockbox.py",
    ]
    for path in sources:
        shutil.copyfile(path, out / "source" / path.name)
    shutil.copytree(PARENT / "inputs", out / "inputs")
    (out / "images").mkdir()
    schedule = []
    for ordinal, cid in enumerate(config["case_ids"]):
        case, feature = load_case_features(DATA, cid)
        for rep in (0, 1):
            for target in selections[f"{cid}-r{rep}"]["selected"]:
                for name in (f"{cid}-raw.png", f"{cid}-window-{target['index']:02d}.png"):
                    destination = out / "images" / name
                    if not destination.exists():
                        shutil.copyfile(PARENT / "images" / name, destination)
                arms = ["comparison_map", "median_trace"]
                if (ordinal + rep + target["index"]) % 2:
                    arms.reverse()
                for arm in arms:
                    name = f"{cid}-window-{target['index']:02d}-{arm}.png"
                    image = out / "images" / name
                    if not image.exists():
                        facts = render_comparison_supplement(
                            case, feature.noise, target["start"], target["stop"], arm, image
                        )
                        write_json(
                            out / "images" / (image.stem + "-facts.json"),
                            {
                                key: np.where(np.isfinite(value), value, None).tolist()
                                for key, value in facts.items()
                            },
                        )
                    schedule.append(
                        {
                            "case_id": cid,
                            "repeat": rep,
                            "arm": arm,
                            **target,
                            "request_id": f"comparison-{cid}-window-{target['index']:02d}-{arm}-r{rep}",
                            "images": [
                                f"{cid}-raw.png",
                                f"{cid}-window-{target['index']:02d}.png",
                                name,
                            ],
                            "prompt": WINDOW_TASK.format(
                                start=target["start"], end=target["stop"] - 1
                            )
                            + NOTICE,
                        }
                    )
    write_json(out / "schedule.json", schedule)
    write_json(
        out / "protocol.json",
        {
            "registered_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Complete24 already exposed development records, fixed12 author subset; not independent confirmation or revival of stopped feedback/axis triggers",
            "change": "Only third-image representation: noise-scaled chronological pairwise raw block differences versus ordinary trace of EXACTLY the same block medians",
            "case_ids": config["case_ids"],
            "model": config["model"],
            "policy": config["policy"],
            "source_selection": str(PARENT / "selections.json"),
            "unchanged": "Original raw21/42 trigger,14 conflict days,at most2 disjoint180-day targets,49-day permission, initial answers and target-matched stage heads",
            "display": "Target-aligned14-day medians; >=8 completeNEU days; no interpolation, no labels; map fixed +/-3 display clipping, trace minimum6 units display range, all three axes kept",
            "paired_controls": "Both arms use same3 images by role, same third-image dimensions, exactly same prompt, alternating deterministic order, same2 repeats and request parameters",
            "maximum_requests": len(schedule),
            "retries": 0,
            "hard_timeout_seconds": 1800,
            "continuation_gate": "Both complete process repeats require map F1 and net corrections > trace, at least2 records benefit each repeat, mean weakRecall not lower; author complete12 mean F1 difference>0, total net corrections>0 and mean weakRecall not lower. Full fixed/direct/numerical effects reported but not required to beat all. Excluding initial failures is descriptive, not replacement gate",
            "failure": "Stop display candidate if the registered development gate fails; no prompt/scale/seed/repeat/subset tuning or best-control replacement",
            "next_if_pass": "Only then complete same-representation fixed-placement Agent controls and freeze truly unseen records; this probe cannot confirm the full method",
            "reference": "Unchanged controlled process truth and developer-exposed raw visual/numerical reference; no independent expert/field claim",
            "old_decisions_unchanged": [
                "NOT_CONFIRMED",
                "STOP_FEEDBACK_PROMOTION",
                "STOP_AXIS_WITNESS_BEFORE_CALLS",
            ],
        },
    )
    paths = sources + [
        PARENT / "protocol.json",
        PARENT / "selections.json",
        PRIOR / "manifest.json",
    ]
    paths.extend(path for path in PRIOR.rglob("*") if path.is_file())
    write_json(out / "source-bindings.json", {str(path): sha(path) for path in paths})
    write_json(
        out / "freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    write_json(out / "state.json", {"status": "prepared", "requests": len(schedule)})
    print("Prepared paired representations:", len(schedule), "queries", flush=True)


def verify(out):
    verify_bindings(out, "freeze.json")
    for filename, expected in read_json(out / "source-bindings.json").items():
        if sha(Path(filename)) != expected:
            raise ValueError("Frozen display source or old evidence changed: " + filename)


def execute(out):
    verify(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Execute only never-started plans; no restarts/retries")
    config = read_json(out / "protocol.json")
    schedule = read_json(out / "schedule.json")
    transport = VisualRequests(
        out / "requests", ROOT / ".env", config["model"], config["maximum_requests"]
    )
    features = {cid: load_case_features(out, cid)[1] for cid in config["case_ids"]}
    reviews, failures = {}, {}
    started = time.monotonic()
    save_json(out / "state.json", {"status": "running", "completed": 0, "total": len(schedule)})
    for ordinal, row in enumerate(schedule):
        if time.monotonic() - started > config["hard_timeout_seconds"]:
            save_json(out / "state.json", {"status": "timeout", "completed": ordinal})
            raise RuntimeError("Frozen timeout reached; keep partial answers, do not restart")
        feature = features[row["case_id"]]
        activity = np.full(len(feature.observed), -1, dtype=int)
        status = "success"
        try:
            answer = transport.ask(
                row["request_id"], row["prompt"], [out / "images" / name for name in row["images"]]
            )
            activity = parse_window_activity(answer, feature.observed, row["start"], row["stop"])
        except (ValueError, RuntimeError) as error:
            status = "failed"
            failures[row["request_id"]] = str(error)
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
            {"status": "running", "completed": ordinal + 1, "total": len(schedule)},
        )
        print(f"{ordinal + 1}/{len(schedule)} {row['request_id']} {status}", flush=True)
    verify(out)
    verify_receipts(out)
    write_json(
        out / "prediction-freeze.json",
        {name: sha(out / name) for name in ("reviews.json", "failures.json")},
    )
    save_json(
        out / "state.json",
        {"status": "predictions_complete", "seconds": time.monotonic() - started},
    )


def build_predictions(out):
    config = read_json(out / "protocol.json")
    choices = read_json(PARENT / "selections.json")
    previous = read_json(PRIOR / "primary_process/predictions.json")
    reviews = read_json(out / "reviews.json")
    lookup = {
        (row["case_id"], row["repeat"], row["index"], row["arm"]): row["request_id"]
        for row in read_json(out / "schedule.json")
    }
    head = XGBClassifier()
    head.load_model(DATA / "stage-xgboost.json")
    predictions, used = {}, {}
    pairs = {cid: load_case_features(DATA, cid) for cid in config["case_ids"]}
    for cid, (case, feature) in pairs.items():
        codes = head.predict(feature.matrix).astype(int) + 1
        long = raw_motion_evidence(case, feature)
        for rep in (0, 1):
            initial = np.array(previous[f"direct_visual_r{rep}"][cid]["activity"])
            selected = choices[f"{cid}-r{rep}"]["selected"]
            authority = conflict_neighbourhood(
                initial, feature, long, ReviewPolicy(**config["policy"])
            )[1]
            scope = target_scopes(authority, feature.observed, selected)[0]
            for arm in ("comparison_map", "median_trace"):
                keys = [lookup[(cid, rep, target["index"], arm)] for target in selected]
                reviewed = combine_reviews(
                    initial, [reviews[key] for key in keys], feature.observed
                ).activity
                activity = scoped_activity(initial, reviewed, scope, feature.observed)
                method = f"displacement_{arm}_r{rep}__conflict_scope"
                predictions.setdefault(method, {})[cid] = learned_stages(
                    codes, activity, feature.supported
                ).to_dict()
                used.setdefault(method, {})[cid] = [f"{cid}-direct-r{rep}"] + keys
    return predictions, used, pairs


def score(out, destination):
    verify(out)
    verify_bindings(out, "prediction-freeze.json")
    if read_json(out / "state.json")["status"] != "predictions_complete":
        raise ValueError("Score only terminal frozen answers")
    destination.mkdir(parents=True, exist_ok=False)
    additions, additional_used, pairs = build_predictions(out)
    receipts = verify_receipts(PARENT)
    receipts.update(verify_receipts(BASE / "feedback-placement-development-v1"))
    receipts.update(verify_receipts(out))
    used = read_json(BASE / "core-lockbox-analysis-v1/used-requests.json")
    _, feedback_used = fixed_predictions(BASE / "feedback-placement-development-v1", pairs)
    used.update(feedback_used)
    used.update(additional_used)
    comparisons, conditions = [], {}
    for target in ("primary_process", "secondary_developer"):
        previous = read_json(PRIOR / target / "predictions.json")
        labels = read_json(BASE / "core-lockbox-analysis-v1" / target / "reference-labels.json")
        if target == "primary_process":
            new = additions
            weak = {
                cid: np.array(
                    [
                        int(row["activity_label"]) == 1 and int(row["weak_signal"]) == 1
                        for row in rows
                    ]
                )
                for cid, rows in labels.items()
            }
        else:
            converted = developer_predictions(DATA, {cid: pairs[cid] for cid in labels}, additions)
            new = {name: converted[name] for name in additions}
            spans = read_json(DATA / "codex-review/activity-decisions.json")[
                "weak_interiors_authored_before_predictions"
            ]
            weak = {}
            for cid, rows in labels.items():
                mask = np.zeros(len(rows), dtype=bool)
                for start, stop in spans.get(cid, []):
                    mask[start:stop] = True
                weak[cid] = mask & (np.array([int(row["activity_label"]) for row in rows]) == 1)
        predictions = {**previous, **new}
        target_used = {
            method: {cid: keys for cid, keys in cases.items() if cid in labels}
            for method, cases in used.items()
        }
        observed = {cid: pairs[cid][1].observed for cid in labels}
        results, summaries, records = score_target(
            labels, observed, weak, predictions, target_used, receipts
        )
        old_results = read_json(PRIOR / target / "results.json")
        if any(results[name] != value for name, value in old_results.items()):
            raise ValueError("New display changed an original result dictionary")
        folder = destination / target
        folder.mkdir()
        write_json(folder / "predictions.json", predictions)
        write_json(folder / "results.json", results)
        write_csv(folder / "summary.csv", summaries)
        write_csv(folder / "per-record.csv", records)
        (folder / "all-methods.md").write_text(full_table(summaries), encoding="utf-8")
        record_table = {(row["method"], row["case_id"]): row for row in records}
        for rep in (0, 1):
            main = f"displacement_comparison_map_r{rep}__conflict_scope"
            controls = [
                f"displacement_median_trace_r{rep}__conflict_scope",
                f"adaptive_no_feedback_r{rep}__conflict_scope",
                f"direct_visual_r{rep}",
                f"fixed_image_review_r{rep}__whole_target",
                f"same_count_fixed_early_r{rep}__conflict_scope",
            ]
            controls.extend(name for name in previous if "_r0" not in name and "_r1" not in name)
            for population in ("complete", "valid_initial_repeat"):
                ids = [
                    cid
                    for cid in labels
                    if population == "complete"
                    or predictions[f"direct_visual_r{rep}"][cid]["status"] != "failed"
                ]
                for control in controls:
                    row = {
                        "target": target,
                        "repeat": rep,
                        "population": population,
                        "records": len(ids),
                        "control": control,
                        "treatment": main,
                        **compare_cases(
                            labels, predictions, results, record_table, ids, control, main
                        ),
                    }
                    comparisons.append(row)
    process_pairs = [
        row
        for row in comparisons
        if row["target"] == "primary_process"
        and row["population"] == "complete"
        and row["control"].startswith("displacement_median_trace")
    ]
    author_pairs = [
        row
        for row in comparisons
        if row["target"] == "secondary_developer"
        and row["population"] == "complete"
        and row["control"].startswith("displacement_median_trace")
    ]
    for row in process_pairs:
        conditions[f"process_r{row['repeat']}"] = (
            row["f1_difference"] > 0 and row["net_correct"] > 0 and row["benefited_records"] >= 2
        )
    conditions["process_mean_weak"] = np.mean(
        [row["treatment_weak_recall"] for row in process_pairs]
    ) >= np.mean([row["control_weak_recall"] for row in process_pairs])
    conditions["author_mean_f1"] = np.mean([row["f1_difference"] for row in author_pairs]) > 0
    conditions["author_net"] = sum(row["net_correct"] for row in author_pairs) > 0
    conditions["author_mean_weak"] = np.mean(
        [row["treatment_weak_recall"] for row in author_pairs]
    ) >= np.mean([row["control_weak_recall"] for row in author_pairs])
    write_csv(destination / "comparisons.csv", comparisons)
    write_json(
        destination / "decision.json",
        {
            "conditions": conditions,
            "decision": "COMPLETE_SAME_DISPLAY_AGENT_CONTROLS"
            if all(conditions.values())
            else "STOP_COMPARISON_DISPLAY",
            "old_decisions_unchanged": True,
            "independent_confirmation": False,
        },
    )
    new_receipts = verify_receipts(out)
    write_json(
        destination / "cost.json",
        {
            "new_requests": len(new_receipts),
            "responses": len(new_receipts),
            "reported_tokens": sum(
                row.get("usage", {}).get("total_tokens", 0) for row in new_receipts.values()
            ),
            "wall_seconds": read_json(out / "state.json")["seconds"],
            "failures": len(read_json(out / "failures.json")),
            "retries": 0,
        },
    )
    shutil.copyfile(Path(__file__), destination / "source.py")
    write_json(
        destination / "manifest.json",
        {
            "files": {
                str(path.relative_to(destination)): sha(path)
                for path in destination.rglob("*")
                if path.is_file()
            }
        },
    )
    print(read_json(destination / "decision.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out.resolve())
    elif args.mode == "run":
        execute(args.out.resolve())
    else:
        if args.analysis is None:
            parser.error("score requires --analysis")
        score(args.out.resolve(), args.analysis.resolve())
