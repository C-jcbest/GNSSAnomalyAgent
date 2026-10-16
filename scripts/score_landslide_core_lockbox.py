"""Common registered scoring with an explicit unavailable weak denominator."""

from __future__ import annotations

import numpy as np
from analyze_landslide_adaptive_review import boundary_errors
from analyze_landslide_codex_observation_reference import stage_reference_arrays
from sklearn.metrics import precision_recall_fscore_support

from gnss_sim.landslide_evaluation import ActivityPrediction, aggregate_scores, evaluate_case


def score_target(labels, observed, weak, predictions, used, receipts):
    """Apply the common task, checking day counts independently with sklearn."""
    results, summaries, records = {}, [], []
    for method, cases in predictions.items():
        if set(cases) != set(labels):
            raise ValueError("Every method must score every record")
        per_case, bounds03, bounds05, scores05 = {}, {}, {}, []
        actual_all, predicted_all, actual_stage, predicted_stage = [], [], [], []
        weak_total = weak_detected = unknown_positive = unknown_negative = unknown_abstained = 0
        for cid, value in cases.items():
            prediction = ActivityPrediction(
                np.array(value["activity"]),
                np.array(value["stage"]),
                status=value["status"],
                gated=value["gated"],
                stage_status=value["stage_status"],
            )
            if np.any(prediction.activity[~observed[cid]] != -1) or np.any(
                prediction.stage[~observed[cid]] != -1
            ):
                raise ValueError("Missing observation gained a decision")
            per_case[cid] = evaluate_case(labels[cid], prediction)
            scores05.append(evaluate_case(labels[cid], prediction, 0.5))
            bounds03[cid] = boundary_errors(labels[cid], prediction.activity, 0.3)
            bounds05[cid] = boundary_errors(labels[cid], prediction.activity, 0.5)
            actual = np.array([int(row["activity_label"]) for row in labels[cid]])
            known = actual >= 0
            actual_all.extend(actual[known].tolist())
            predicted_all.extend(prediction.activity[known].tolist())
            stages = stage_reference_arrays(labels[cid], prediction)
            interior = stages["stage_interior"]
            actual_stage.extend(stages["actual_stage"][interior].tolist())
            predicted_stage.extend(prediction.stage[interior].tolist())
            found = int((weak[cid] & (prediction.activity == 1)).sum())
            weak_detected += found
            weak_total += int(weak[cid].sum())
            unknown = observed[cid] & (actual == -1)
            unknown_positive += int((unknown & (prediction.activity == 1)).sum())
            unknown_negative += int((unknown & (prediction.activity == 0)).sum())
            unknown_abstained += int((unknown & (prediction.activity == -1)).sum())
            records.append(
                {
                    "method": method,
                    "case_id": cid,
                    **per_case[cid]["daily"],
                    "weak_days": int(weak[cid].sum()),
                    "weak_detected_days": found,
                }
            )
        aggregate = aggregate_scores(list(per_case.values()))
        p, r, f, _ = precision_recall_fscore_support(
            actual_all, predicted_all, labels=[1], zero_division=0
        )
        for metric, independent in (("precision", p[0]), ("recall", r[0]), ("f1", f[0])):
            if not np.isclose(aggregate["daily"][metric], independent):
                raise ValueError("Independent day counter disagrees")
        ip, ir, sf, support = precision_recall_fscore_support(
            actual_stage, predicted_stage, labels=[1, 2, 3], zero_division=0
        )
        interior_score = {
            "precision_ASD": ip.tolist(),
            "recall_ASD": ir.tolist(),
            "f1_ASD": sf.tolist(),
            "support_ASD": support.tolist(),
            "macro_f1": float(sf.mean()),
            "coverage": float(np.mean(np.array(predicted_stage) > 0)),
        }
        event05 = aggregate_scores(scores05)["events"]
        pairs = [pair for row in bounds03.values() for pair in row["matched_pairs"]]
        if (
            len(pairs) != aggregate["events"]["tp"]
            or sum(len(row["matched_pairs"]) for row in bounds05.values()) != event05["tp"]
        ):
            raise ValueError("Event and boundary counts differ")
        results[method] = {
            "aggregate": aggregate,
            "per_case": per_case,
            "events_iou05": event05,
            "internal_stages": interior_score,
            "boundaries_iou03": bounds03,
            "boundaries_iou05": bounds05,
        }
        keys = [key for values in used.get(method, {}).values() for key in values]
        summaries.append(
            {
                "method": method,
                **aggregate["daily"],
                "event_f1_iou03": aggregate["events"]["f1"],
                "event_f1_iou05": event05["f1"],
                "onset_core_mae_days": float(
                    np.mean([abs(pair["onset_error_days"]) for pair in pairs])
                )
                if pairs
                else None,
                "offset_core_mae_days": float(
                    np.mean([abs(pair["offset_error_days"]) for pair in pairs])
                )
                if pairs
                else None,
                "stage_end_to_end_macro_f1": aggregate["stage_macro_f1"],
                "stage_internal_macro_f1": interior_score["macro_f1"],
                "stage_internal_coverage": interior_score["coverage"],
                "coverage": aggregate["coverage"],
                "weak_days": weak_total,
                "weak_detected_days": weak_detected,
                "weak_recall": weak_detected / weak_total if weak_total else None,
                "unknown_reference_predicted_positive": unknown_positive,
                "unknown_reference_predicted_negative": unknown_negative,
                "unknown_reference_abstained": unknown_abstained,
                "requests": len(keys),
                "reported_tokens": sum(
                    receipts[key].get("usage", {}).get("total_tokens", 0) for key in keys
                ),
            }
        )
    return results, summaries, records
