"""Verify frozen local phase reviews and report corrections against strong stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from prepare_landslide_ai_reference_holdout import write_csv
from probe_landslide_endpoint_fit import export
from probe_landslide_phase_residual import ARMS, STAGE_CODES, numeric_table
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import read_json
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from gnss_sim.artifacts import sha
from gnss_sim.landslide_evaluation import ActivityPrediction
from gnss_sim.landslide_phase_residual import phase_evidence_window, phase_residual_evidence
from gnss_sim.landslide_visual import parse_stages


def score(records):
    actual = np.concatenate([row["actual"] for row in records])
    predicted = np.concatenate([row["predicted"] for row in records])
    baseline = np.concatenate([row["baseline"] for row in records])
    p, r, f1, support = precision_recall_fscore_support(actual, predicted, labels=[1, 2, 3], zero_division=0)
    return {
        "known_stage_days": len(actual), "correct": int((actual == predicted).sum()),
        "wrong_definite": int(((predicted > 0) & (predicted != actual)).sum()),
        "unknown": int((predicted == -1).sum()), "coverage": float((predicted > 0).mean()),
        "macro_f1": float(f1.mean()), "gained": int(((baseline != actual) & (predicted == actual)).sum()),
        "lost": int(((baseline == actual) & (predicted != actual)).sum()),
        "net": int((predicted == actual).sum() - (baseline == actual).sum()),
        "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(("A", "S", "D"))},
        "confusion_actual_ASD_pred_unknown_ASD": confusion_matrix(actual, predicted, labels=[-1, 1, 2, 3])[1:].tolist(),
    }


def report(run, out):
    verify_bindings(run, "pre-inference-freeze.json")
    verify_bindings(run, "inference-freeze.json")
    if read_json(run / "state.json")["status"] != "inference_complete":
        raise ValueError("Only terminal frozen inference may be scored")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    outputs = read_json(run / "outputs.json")
    packets = read_json(run / "packets.json")
    labels = read_json(run / "reference-labels.json")
    baselines = read_json(run / "baseline-predictions.json")
    selected = {row["probe_id"]: row for row in read_json(run / "diagnostic-selection.json")["selected"]}
    failures = read_json(run / "failures.json")
    records, contract_days, replayed = {}, 0, 0
    receipt_cost = []
    for item in read_json(run / "schedule.json"):
        request_id, key, arm = item["request_id"], item["probe_id"], item["arm"]
        packet = packets[key][arm]
        receipt = read_json(run / "requests" / (request_id + ".response.json"))
        started_receipt = read_json(run / "requests" / (request_id + ".started.json"))
        if receipt["prompt"] != packet["prompt"] or started_receipt["prompt"] != packet["prompt"]:
            raise ValueError("A registered input prompt changed")
        images = [run / name for name in packet["images"]]
        image_hashes = {path.name: sha(path) for path in images}
        if receipt["image_sha256"] != image_hashes or started_receipt["image_sha256"] != image_hashes:
            raise ValueError("A registered evidence image changed")
        baseline = baselines[key]
        activity, initial = np.array(baseline["activity"]), np.array(baseline["stage"])
        actual_prediction = outputs[request_id]
        days = len(activity)
        observed, supported = np.array(packet["observed"]), np.array(packet["stage_supported"])
        permitted = np.zeros(days, dtype=bool)
        permitted[slice(*packet["query"])] = True
        permitted &= observed & supported & (activity == 1)
        stages = initial.copy()
        if request_id not in failures:
            if receipt["status"] != "returned_json" or receipt["finish_reason"] != "stop":
                raise ValueError("Successful review lacks a complete frozen response")
            payload = json.loads(receipt["output"])
            for interval in payload["stages"]:
                if not packet["query"][0] <= interval[0] <= interval[1] < packet["query"][1]:
                    raise ValueError("Stage response escaped registered target")
            parsed = parse_stages(payload, activity, observed)
            stages[permitted] = parsed.stage[permitted]
        if stages.tolist() != actual_prediction["stage"]:
            raise ValueError("Response replay changed the saved review")
        if actual_prediction["activity"] != baseline["activity"]:
            raise ValueError("Phase review changed activity")
        if np.any(stages[~permitted] != initial[~permitted]):
            raise ValueError("Phase review changed dates outside permission")
        ActivityPrediction(activity, stages).validate(days)
        truth = np.array([STAGE_CODES.get(row["feature"], -1) if int(row["activity_label"]) == 1 else -1 for row in labels[key]])
        query = np.zeros(days, dtype=bool)
        query[slice(*packet["query"])] = True
        known = query & (truth > 0) & supported
        records.setdefault((arm, item["repetition"]), []).append({
            "probe_id": key, "target": packet["target"], "case_id": packet["case_id"],
            "selection_role": selected[key]["selection_role"],
            "actual": truth[known], "predicted": stages[known], "baseline": initial[known],
        })
        receipt_cost.append({**item, "reported_tokens": receipt.get("usage", {}).get("total_tokens", 0),
                             "seconds": receipt["seconds"], "status": actual_prediction["review_status"]})
        contract_days += days
        replayed += 1
    evidence_checks = []
    for key, arms in packets.items():
        packet = arms["raw_local"]
        case = read_json(run / (key + "-input.json"))
        values = np.array([row if row is not None else [np.nan] * 3 for row in case["displacement_mm"]])
        observed = np.isfinite(values).all(axis=1)
        window = phase_evidence_window(packet["activity"], observed, *packet["query"])
        if list(window) != packet["window"]:
            raise ValueError("Activity-contained horizon changed")
        old = read_json(run / (key + "-evidence.json"))
        evidence = phase_residual_evidence(values, old["noise_proxy_mm"], packet["activity"], *window)
        if evidence != old:
            raise ValueError("Motion-coordinate evidence does not match original observations")
        table = json.dumps(numeric_table(evidence), separators=(",", ":"))
        for arm in ("residual_feedback", "numeric_only"):
            if not arms[arm]["prompt"].endswith(table):
                raise ValueError("Numeric control saw different evidence")
        if arms["raw_local"]["prompt"] != arms["residual_view"]["prompt"]:
            raise ValueError("Representation-only contrast changed the prompt")
        evidence_checks.append({"probe_id": key, "original_values_dates_masks_match": True, "window": window})
    results, per_record, rows = {}, [], []
    for repetition in range(2):
        control = [{**row, "predicted": row["baseline"]} for row in records["raw_local", repetition]]
        for name in ("strong_stage", *ARMS):
            data = control if name == "strong_stage" else records[name, repetition]
            for target in ("all", "primary_process", "secondary_developer"):
                subset = data if target == "all" else [row for row in data if row["target"] == target]
                value = score(subset)
                results[f"{name}-r{repetition}-{target}"] = value
                rows.append({"method": name, "repetition": repetition, "target": target,
                             **{key: item for key, item in value.items() if key not in ("per_class", "confusion_actual_ASD_pred_unknown_ASD")}})
            for record in data:
                value = score([record])
                per_record.append({"method": name, "repetition": repetition, "probe_id": record["probe_id"],
                                   "target": record["target"], "case_id": record["case_id"],
                                   "selection_role": record["selection_role"], **{key: item for key, item in value.items()
                                                                               if key not in ("per_class", "confusion_actual_ASD_pred_unknown_ASD")}})
    gate = []
    for repetition in range(2):
        main = results[f"residual_feedback-r{repetition}-all"]
        for control in ("raw_local", "residual_view", "numeric_only"):
            gate.append({"repetition": repetition, "check": "main_macro_f1_strictly_above_" + control,
                         "passed": main["macro_f1"] > results[f"{control}-r{repetition}-all"]["macro_f1"]})
        for target in ("primary_process", "secondary_developer"):
            gate.append({"repetition": repetition, "check": "positive_net_vs_strong_" + target,
                         "passed": results[f"residual_feedback-r{repetition}-{target}"]["net"] > 0})
        changes = [row for row in per_record if row["method"] == "residual_feedback" and row["repetition"] == repetition]
        benefited = {(row["target"], row["case_id"]) for row in changes if row["net"] > 0 and row["selection_role"].startswith("largest")}
        control_losses = sum(row["lost"] for row in changes if row["selection_role"].startswith("correct"))
        gate.extend([{"repetition": repetition, "check": "two_distinct_error_records_benefit", "passed": len(benefited) >= 2},
                     {"repetition": repetition, "check": "correct_control_dates_preserved", "passed": control_losses == 0}])
    decision = "PROCEED_TO_ACTIVITY_CONTAINED_PHASE_CALIBRATION" if all(row["passed"] for row in gate) else "STOP_PHASE_RESIDUAL_PILOT"
    export(out / "decision.json", {"decision": decision, "gate": gate, "passed": sum(row["passed"] for row in gate), "total": len(gate)})
    export(out / "results.json", results)
    write_csv(out / "summary.csv", rows)
    write_csv(out / "per-target.csv", per_record)
    write_csv(out / "request-cost.csv", receipt_cost)
    export(out / "verification.json", {"replayed_saved_responses": replayed, "checked_output_calendar_days": contract_days,
                                       "activity_missing_support_scope_exact": True, "evidence_checks": evidence_checks,
                                       "no_reference_or_activity_revision": True, "remote_reinference": False,
                                       "failure_count": len(failures)})
    fig, panels = plt.subplots(1, 2, figsize=(11, 4.5))
    methods = ("strong_stage", *ARMS)
    positions = np.arange(len(methods))
    for repetition, color in ((0, "#147d72"), (1, "#cc6d4b")):
        f1 = [results[f"{name}-r{repetition}-all"]["macro_f1"] for name in methods]
        net = [results[f"{name}-r{repetition}-all"]["net"] for name in methods]
        panels[0].bar(positions + (repetition - .5) * .32, f1, .32, color=color, label=f"pass {repetition + 1}")
        panels[1].bar(positions + (repetition - .5) * .32, net, .32, color=color)
    for panel in panels:
        panel.set_xticks(positions, ["Strong\nstage", "Raw\nlocal", "Residual\nview", "Residual +\nnumbers", "Numbers\nonly"])
        panel.grid(axis="y", alpha=.2)
    panels[0].set_ylim(0, 1)
    panels[0].set_ylabel("Target A/S/D macro-F1")
    panels[0].legend()
    panels[1].axhline(0, color="black", linewidth=.7)
    panels[1].set_ylabel("Net exact stage corrections vs strong head (days)")
    fig.suptitle("Eight selected development targets; not full-record or independent effectiveness")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=170)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    costs = {"attempts": len(receipt_cost), "reported_tokens": sum(row["reported_tokens"] for row in receipt_cost),
             "reported_api_seconds": sum(row["seconds"] for row in receipt_cost), "failure_count": len(failures)}
    export(out / "cost-summary.json", costs)
    export(out / "manifest.json", {"files": {str(path.relative_to(out)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in out.rglob("*") if path.is_file()}})
    print(json.dumps({"decision": decision, "gate_passed": sum(row["passed"] for row in gate), "gate_total": len(gate), **costs}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.run.resolve(), arguments.out.resolve())
