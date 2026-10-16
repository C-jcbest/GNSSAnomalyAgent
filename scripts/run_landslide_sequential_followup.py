"""Run fresh censored-endpoint reviews before controls on a prepared comparison.

Inputs are supplied by the existing raw-reference comparison preparer. This
executor reads observations and frozen models, never reference-label files.
Every follow-up target is recorded after its first answer and before requesting
the second view. Identical questions within a record/repeat share one receipt.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from analyze_landslide_censored_followup_v2 import overlapping_scope
from probe_landslide_revision_scope import conflict_neighbourhood, scoped_activity
from run_landslide_adaptive_review import PLOT, ROOT, verify_freeze
from run_landslide_comparison import save_json
from run_landslide_window_optimization import load_case_features, read_json, render_window
from xgboost import XGBClassifier

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide_adaptive_review import (
    ReviewPolicy,
    combine_reviews,
    raw_motion_evidence,
    review_priorities,
    review_prompt,
)
from gnss_sim.landslide_censored_followup import continuation_choice, followup_targets
from gnss_sim.landslide_error_audit import regular_windows
from gnss_sim.landslide_evaluation import ActivityPrediction
from gnss_sim.landslide_local_experiment import learned_stages
from gnss_sim.landslide_visual import ACTIVITY_TASK, VisualRequests, parse_activity
from gnss_sim.landslide_window_experiment import parse_window_activity

ARMS = ("censored", "fixed_short", "wide", "no_numeric", "censored_latest", "first_only")


def execute(out):
    verify_freeze(out)
    if read_json(out / "state.json")["status"] != "prepared":
        raise ValueError("Run a never-started prepared comparison; no retries or overwrite")
    registration = read_json(out.parent / "source-bindings.json")
    data = Path(registration["data"])
    experiment = read_json(data / "protocol.json")
    expected = experiment["sequential_executor_sha256"]
    if sha(Path(__file__)) != expected:
        raise ValueError("Sequential executor differs from pre-generation registration")
    if sha(ROOT / "src/gnss_sim/landslide_censored_followup.py") != experiment["selector_sha256"]:
        raise ValueError("Follow-up selector differs from pre-generation registration")
    config = read_json(out / "protocol.json")
    policy = ReviewPolicy(**config["policy"])
    transport = VisualRequests(out / "requests", ROOT / ".env", config["model"],
                               experiment["cost"]["maximum_unique_remote_requests"])
    model = XGBClassifier()
    model.load_model(out / "stage-xgboost.json")
    predictions, followup, used, schedules, selections, failures = {}, {}, {}, {}, {}, {}
    save_json(out / "state.json", {"status": "running", "fresh_sequential": True})
    (out / "sequential-source.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    for rep in range(config["repetitions"]):
        for ordinal, cid in enumerate(config["case_ids"]):
            if time.monotonic() - started > config["hard_timeout_seconds"]:
                raise RuntimeError("Registered wall-time limit exceeded")
            case, feature = load_case_features(out, cid)
            observed = feature.observed
            raw = out / "images" / f"{cid}-raw.png"
            direct_key = f"{cid}-direct-r{rep}"
            prompt = ACTIVITY_TASK.format(last=len(case.dates) - 1) + (
                '\nReturn exactly JSON: {"activity":[[start,end]],"uncertain":[[start,end]]}. '
                "Adjacent intervals never share endpoints; use [100,149],[150,200].")
            try:
                payload = transport.ask(direct_key, prompt, [raw])
                initial = parse_activity(payload, observed)
                initial_status = "success"
            except (ValueError, RuntimeError) as error:
                failures[direct_key] = str(error)
                initial = np.full(len(observed), -1)
                initial_status = "failed"
            long = raw_motion_evidence(case, feature)
            windows = regular_windows(len(observed), config["window_days"], config["stride_days"])
            priorities, selected = review_priorities(initial, feature, long, windows, policy)
            centers = np.linspace(0, len(observed), policy.maximum_reviews + 2)[1:-1]
            fixed = sorted({min(range(len(windows)), key=lambda index:
                abs(sum(windows[index]) / 2 - center)) for center in centers})
            record_key = f"{cid}-r{rep}"
            selections[record_key] = {"priorities": priorities, "selected": selected,
                                      "fixed_indices": fixed}
            save_json(out / "selections.json", selections)
            conflicts, authority = conflict_neighbourhood(initial, feature, long, policy)
            cache = {}

            def observe(target, numeric=False):
                start, stop = target["start"], target["stop"]
                question = start, stop, numeric
                if question in cache:
                    return cache[question]
                original = next((row for row in priorities
                                 if (row["start"], row["stop"]) == (start, stop)), None)
                stem = f"window-{original['index']:02d}" if original else f"followup-{start}-{stop}"
                local = out / "images" / f"{cid}-{stem}.png"
                if not local.exists():
                    render_window(case, feature.noise, start, stop, local, PLOT)
                request_id = f"{cid}-{stem}-{'numeric' if numeric else 'image'}-r{rep}"
                try:
                    answer = transport.ask(request_id, review_prompt(feature, long, start, stop, numeric),
                                           [raw, local])
                    activity = parse_window_activity(answer, observed, start, stop)
                    status = "success"
                except (ValueError, RuntimeError) as error:
                    failures[request_id] = str(error)
                    activity = np.full(len(observed), -1)
                    status = "failed"
                result = {"start": start, "stop": stop, "activity": activity.tolist(),
                          "status": status, "request_id": request_id}
                cache[question] = result
                return result

            # The candidate acquires its first answer before any second or control view.
            first = None
            schedule = {"budget": len(selected), "targets": {}}
            if selected:
                first_target = min(selected, key=lambda row: (-row["conflict_days"], row["start"]))
                first = observe(first_target)
                remaining = next((row for row in selected if row["index"] != first_target["index"]), None)
                choice = continuation_choice(first_target, np.array(first["activity"]), conflicts,
                                             observed, policy.minimum_conflict_days)
                targets = followup_targets(first_target, remaining, choice, len(observed))
                schedule.update(first=first, continuation=choice, targets=targets)
            schedules[record_key] = schedule
            save_json(out / "followup-schedules.json", schedules)
            codes = model.predict(feature.matrix).astype(int) + 1
            # Main is asked first; subsequent identical control questions share this receipt.
            for arm in ARMS:
                reviews = [] if first is None else [first]
                second = None
                if first is not None and arm != "first_only":
                    target = schedule["targets"]["censored" if arm == "censored_latest" else arm]
                    if target is not None:
                        second = observe(target)
                        reviews.append(second)
                activity = initial.copy()
                if reviews:
                    reviewed = combine_reviews(initial, reviews, observed).activity
                    if arm == "censored_latest" and second is not None:
                        start, stop = second["start"], second["stop"]
                        reviewed[start:stop] = np.array(second["activity"])[start:stop]
                    scope = overlapping_scope(authority, observed, reviews)
                    activity = scoped_activity(initial, reviewed, scope, observed)
                method = f"first_only_r{rep}" if arm == "first_only" else f"followup_{arm}_r{rep}"
                followup.setdefault(method, {})[cid] = learned_stages(codes, activity, feature.supported).to_dict()
                used.setdefault(method, {})[cid] = [direct_key] + [row["request_id"] for row in reviews]
            # All existing pure-visual, fixed, numerical-feedback and scope controls remain.
            originals = {}
            adaptive = {row["index"] for row in selected}
            for index in sorted(set(fixed) | adaptive):
                conditions = [False, True] if index in adaptive else [False]
                if (ordinal + rep + index) % 2:
                    conditions.reverse()
                for numeric in conditions:
                    originals[index, numeric] = observe(priorities[index], numeric)
            gates = {
                "direct_visual": ActivityPrediction(initial, np.where(initial == 0, 0, -1), status=initial_status),
                "fixed_image_review": combine_reviews(initial, [originals[index, False] for index in fixed], observed),
                "adaptive_no_feedback": combine_reviews(initial, [originals[index, False]
                    for index in sorted(adaptive)], observed),
                "adaptive_numeric_feedback": combine_reviews(initial, [originals[index, True]
                    for index in sorted(adaptive)], observed),
            }
            for stem, gate in gates.items():
                result = gate if gate.status == "failed" else learned_stages(codes, gate.activity, feature.supported)
                predictions.setdefault(f"{stem}_r{rep}", {})[cid] = result.to_dict()
            for name, value in (("predictions.json", predictions), ("followup-predictions.json", followup),
                                ("followup-used-requests.json", used), ("failures.json", failures)):
                save_json(out / name, value)
            print(f"{cid} r{rep}: budget {len(selected)}, trigger "
                  f"{schedule.get('continuation', {}).get('numeric_admitted', False)}, "
                  f"fresh questions {len(cache) + 1}", flush=True)
    verify_freeze(out)
    write_json(out / "prediction-freeze.json", {name: sha(out / name) for name in (
        "predictions.json", "followup-predictions.json", "followup-used-requests.json",
        "followup-schedules.json", "selections.json", "failures.json", "sequential-source.py")})
    save_json(out / "state.json", {"status": "predictions_complete", "fresh_sequential": True,
                                    "seconds": time.monotonic() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    execute(parser.parse_args().run.resolve())
