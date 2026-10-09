"""Conditional agreement diagnostics against disclosed development stage reviews."""

from collections import Counter

from gnss_sim.landslide_evaluation import score_counts

STAGES = ("acceleration", "steady_motion", "deceleration")


def validate_aligned_rows(reference, prediction):
    """Keep the complete calendar and activity, before selecting evaluable dates."""
    if len(reference) != len(prediction):
        raise ValueError("Prediction must retain the complete reference calendar")
    seen = set()
    for actual, predicted in zip(reference, prediction, strict=True):
        identity = (actual["case_id"], actual["day_index"], actual["date"])
        predicted_identity = (predicted["case_id"], predicted["day_index"], predicted["date"])
        key = identity[:2]
        if identity != predicted_identity or key in seen:
            raise ValueError("Reference and prediction dates must align uniquely in order")
        seen.add(key)
        if actual["activity_label"] != predicted["activity_label"]:
            raise ValueError("Stage comparison cannot change frozen activity")
        if type(actual["stage_evaluable"]) is not bool:
            raise ValueError("Reference evaluability must be an explicit boolean")
        if actual["stage_evaluable"] and (
            actual["activity_label"] != 1 or actual["feature"] not in STAGES
        ):
            raise ValueError("Only definite observed activity stages are evaluable")
        if actual["activity_label"] == 1 and predicted["feature"] not in (*STAGES, "unknown"):
            raise ValueError("Active prediction must be a stage or an explicit abstention")
        if actual["activity_label"] != 1 and predicted["feature"] in STAGES:
            raise ValueError("Prediction stage is outside observed activity")


def stage_agreement(reference, prediction):
    """Count abstentions as misses on a common reference denominator.

    The arithmetic matches standard classwise precision/recall/F1, but this
    reference is a development review. These numbers do not certify accuracy.
    """
    validate_aligned_rows(reference, prediction)
    confusion = {stage: {feature: 0 for feature in (*STAGES, "unknown")} for stage in STAGES}
    active_predictions = Counter()
    unassessed_predictions = Counter()
    for actual, predicted in zip(reference, prediction, strict=True):
        if actual["activity_label"] != 1:
            continue
        active_predictions[predicted["feature"]] += 1
        if actual["stage_evaluable"]:
            confusion[actual["feature"]][predicted["feature"]] += 1
        else:
            unassessed_predictions[predicted["feature"]] += 1
    per_class = {}
    for stage in STAGES:
        tp = confusion[stage][stage]
        fp = sum(confusion[other][stage] for other in STAGES if other != stage)
        support = sum(confusion[stage].values())
        per_class[stage] = {"reference_support": support, **score_counts(tp, fp, support - tp)}
    evaluable = sum(sum(row.values()) for row in confusion.values())
    same = sum(confusion[stage][stage] for stage in STAGES)
    abstained = sum(row["unknown"] for row in confusion.values())
    all_classes_supported = all(per_class[stage]["reference_support"] > 0 for stage in STAGES)
    return {
        "reference_evaluable_days": evaluable,
        "observed_activity_days": sum(active_predictions.values()),
        "exact_agreement_days": same,
        "different_class_days": evaluable - same - abstained,
        "abstained_evaluable_days": abstained,
        "agreement_rate": same / evaluable if evaluable else None,
        "prediction_coverage_on_evaluable": (evaluable - abstained) / evaluable if evaluable else None,
        "active_prediction_features": dict(active_predictions),
        "reference_unknown_prediction_features": dict(unassessed_predictions),
        "confusion": confusion,
        "per_class": per_class,
        "development_agreement_macro_f1": (
            sum(per_class[stage]["f1"] for stage in STAGES) / len(STAGES)
            if all_classes_supported else None
        ),
        "all_three_reference_classes_supported": all_classes_supported,
        "meaning": "development agreement only; not independently verified accuracy",
    }


def support_filter_effect(reference, native, final):
    validate_aligned_rows(reference, native)
    validate_aligned_rows(reference, final)
    totals = Counter()
    for actual, before, after in zip(reference, native, final, strict=True):
        if before["feature"] == after["feature"]:
            continue
        if before["feature"] not in STAGES or after["feature"] != "unknown":
            raise ValueError("Support filtering may only suppress a definite stage")
        totals["suppressed_activity_days"] += 1
        if actual["stage_evaluable"]:
            totals["suppressed_evaluable_days"] += 1
            if before["feature"] == actual["feature"]:
                totals["native_agreement_to_abstention"] += 1
            else:
                totals["native_disagreement_to_abstention"] += 1
        else:
            totals["suppressed_reference_unknown_days"] += 1
    return {name: totals[name] for name in (
        "suppressed_activity_days", "suppressed_evaluable_days",
        "native_agreement_to_abstention", "native_disagreement_to_abstention",
        "suppressed_reference_unknown_days",
    )}


def disagreement_spans(reference, prediction):
    """Preserve gaps and label changes when listing reviewable discordant bands."""
    validate_aligned_rows(reference, prediction)
    spans = []
    for actual, predicted in zip(reference, prediction, strict=True):
        if not actual["stage_evaluable"] or actual["feature"] == predicted["feature"]:
            continue
        identity = (actual["case_id"], actual["feature"], predicted["feature"])
        previous = spans[-1] if spans else None
        if previous and (
            (previous["case_id"], previous["reference"], previous["prediction"]) == identity
            and previous["stop"] == actual["day_index"]
        ):
            previous["stop"] += 1
            previous["days"] += 1
        else:
            spans.append({"case_id": actual["case_id"], "start": actual["day_index"],
                          "stop": actual["day_index"] + 1, "days": 1,
                          "reference": actual["feature"], "prediction": predicted["feature"]})
    return spans
