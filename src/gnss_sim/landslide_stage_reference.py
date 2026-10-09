"""Human review contracts for conditional stage references, separate from predictions."""
from __future__ import annotations

import hashlib
from collections import Counter
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from gnss_sim.landslide_frozen_stage_heads import unclassified_rows, validate_activity
from gnss_sim.schemas import StrictModel

Feature = Literal["acceleration", "steady_motion", "deceleration", "unknown"]
UnknownReason = Literal["raw_signal_unclear", "insufficient_rate_support", "turning_unclear",
                        "window_conflict", "measurement_artifact", "activity_needs_review", "other"]


class ReviewerProfile(StrictModel):
    name: str
    background: str
    reviewer_type: Literal["human", "ai"] | None
    prior_predictions_seen: bool | None = Field(strict=True)
    prior_generation_labels_seen: bool | None = Field(strict=True)
    participated_in_activity_review: bool | None = Field(strict=True)


class ReferenceSegment(StrictModel):
    start: int = Field(strict=True, ge=0)
    stop: int = Field(strict=True, ge=1)
    feature: Feature
    raw_evidence: str
    rate_evidence: str
    nonzero_evidence: str
    steady_evidence: str
    unknown_reason: UnknownReason | None

    @model_validator(mode="after")
    def check_evidence(self):
        if self.start >= self.stop:
            raise ValueError("Reference segment must have positive half-open length")
        if not self.raw_evidence.strip():
            raise ValueError("Every decision needs raw evidence or an unresolved explanation")
        if self.feature == "unknown":
            if self.unknown_reason is None:
                raise ValueError("Unknown reference requires an explicit reason")
        else:
            if self.unknown_reason is not None or not self.rate_evidence.strip():
                raise ValueError("Definite reference requires rate evidence, not an unknown reason")
            if self.feature == "steady_motion":
                if not self.nonzero_evidence.strip() or not self.steady_evidence.strip():
                    raise ValueError("Steady reference requires both nonzero and steady evidence")
        return self


class InteriorReview(StrictModel):
    start: int = Field(strict=True, ge=0)
    stop: int = Field(strict=True, ge=1)
    review_status: Literal["unreviewed", "reviewed"]
    raw_activity_status: Literal["unreviewed", "confirmed", "needs_review"]
    raw_activity_evidence: str
    raw_views_seen: bool = Field(strict=True)
    auxiliary_views_seen: bool = Field(strict=True)
    reviewed_at: datetime | None
    segments: list[ReferenceSegment]


class CaseReference(StrictModel):
    case_id: str
    input_sha256: str
    activity_sha256: str
    interiors: list[InteriorReview]


class StageReferencePacket(StrictModel):
    schema_version: Literal["landslide-stage-reference-review-v1"]
    source_manifest_sha256: str
    reviewer: ReviewerProfile
    cases: list[CaseReference]


def cache_support_inventory(case, motions, start, stop):
    """Availability only. Counts never classify or mask the human reference."""
    observed = [i for i in range(start, stop) if case.displacement_mm[i] is not None]
    counts = {}
    for window, motion in motions.items():
        velocity = [i for i in observed if motion.velocity_mm_day[i] is not None
                    and np.all(np.isfinite(motion.velocity_mm_day[i]))]
        tangential = [i for i in velocity if motion.tangential_acceleration_mm_day2[i] is not None
                      and np.isfinite(motion.tangential_acceleration_mm_day2[i])]
        counts[str(window)] = {"velocity_days": len(velocity), "tangential_days": len(tangential)}
    return {"calendar_days": stop - start, "observed_days": len(observed), "windows": counts,
            "meaning": "Cached fit availability; not confidence or a reference label"}


def reference_template(contexts, source_manifest_sha256):
    cases = []
    for context in contexts:
        case, review = context["case"], context["review"]
        cases.append({
            "case_id": case.case_id, "input_sha256": review["input_sha256"],
            "activity_sha256": context["activity_sha256"],
            "interiors": [{"start": span["start"], "stop": span["stop"],
                           "review_status": "unreviewed", "raw_activity_status": "unreviewed",
                           "raw_activity_evidence": "", "raw_views_seen": False,
                           "auxiliary_views_seen": False, "reviewed_at": None, "segments": []}
                          for span in review["spans"] if span["state"] == "activity"],
        })
    return {
        "schema_version": "landslide-stage-reference-review-v1",
        "source_manifest_sha256": source_manifest_sha256,
        "reviewer": {"name": "", "background": "", "reviewer_type": None,
                     "prior_predictions_seen": None, "prior_generation_labels_seen": None,
                     "participated_in_activity_review": None},
        "cases": cases,
    }


def compile_reference(payload, contexts, source_manifest_sha256):
    """Reject incomplete reviews; preserve unknown and observation gaps explicitly."""
    packet = StageReferencePacket.model_validate(payload)
    if packet.source_manifest_sha256 != source_manifest_sha256:
        raise ValueError("Reference belongs to another source manifest")
    reviewer = packet.reviewer
    if not reviewer.name.strip() or not reviewer.background.strip():
        raise ValueError("Reviewer identity and background must be recorded")
    if any(value is None for value in (reviewer.reviewer_type, reviewer.prior_predictions_seen,
                                      reviewer.prior_generation_labels_seen,
                                      reviewer.participated_in_activity_review)):
        raise ValueError("Reviewer exposure and type must be explicitly declared")
    if [entry.case_id for entry in packet.cases] != [c["case"].case_id for c in contexts]:
        raise ValueError("Reference must retain all source cases in their original order")
    daily = []
    inventory = []
    for entry, context in zip(packet.cases, contexts, strict=True):
        case, review, rows = context["case"], context["review"], context["rows"]
        digest = context["activity_sha256"]
        validate_activity(case, review, rows, digest)
        if entry.input_sha256 != review["input_sha256"] or entry.activity_sha256 != digest:
            raise ValueError("Reference changed input or frozen activity provenance")
        spans = [span for span in review["spans"] if span["state"] == "activity"]
        if [(s.start, s.stop) for s in entry.interiors] != [(s["start"], s["stop"]) for s in spans]:
            raise ValueError("Reference changed or dropped a frozen activity interior")
        compiled = []
        for day, row in enumerate(unclassified_rows(rows)):
            reason = "outside_confirmed_activity"
            if case.displacement_mm[day] is None:
                reason = "missing_observation"
            compiled.append({**row, "stage_evaluable": False, "reference_reason": reason})
        for interior in entry.interiors:
            if interior.review_status != "reviewed" or not interior.raw_views_seen:
                raise ValueError("Unreviewed interior cannot become a reference")
            if interior.reviewed_at is None or interior.reviewed_at.tzinfo is None:
                raise ValueError("Reviewed interior needs a timestamp with timezone")
            if interior.raw_activity_status == "unreviewed" or not interior.raw_activity_evidence.strip():
                raise ValueError("Raw activity confirmation or challenge must precede stages")
            next_start = interior.start
            for segment in interior.segments:
                if segment.start != next_start or segment.stop > interior.stop:
                    raise ValueError("Reference must partition each interior without gaps or overlaps")
                if interior.raw_activity_status == "needs_review":
                    if segment.feature != "unknown" or segment.unknown_reason != "activity_needs_review":
                        raise ValueError("Challenged activity cannot receive definite reference stages")
                if segment.feature != "unknown" and not interior.auxiliary_views_seen:
                    raise ValueError("Definite stages need declared auxiliary review after raw review")
                for day in range(segment.start, segment.stop):
                    observed = case.displacement_mm[day] is not None
                    feature = segment.feature if observed else "unknown"
                    reason = segment.unknown_reason if segment.feature == "unknown" else "reviewer_definite"
                    compiled[day].update(feature=feature, stage_evaluable=observed and feature != "unknown",
                                         reference_reason=reason if observed else "missing_observation")
                next_start = segment.stop
            if next_start != interior.stop:
                raise ValueError("Unresolved dates must be explicit, not silently omitted")
            inventory.append({"case_id": case.case_id, "start": interior.start, "stop": interior.stop,
                              "raw_activity_status": interior.raw_activity_status,
                              **cache_support_inventory(case, context["motions"], interior.start, interior.stop)})
        daily.extend(compiled)
    features = Counter(row["feature"] for row in daily if row["activity_label"] == 1)
    evaluable = sum(row["stage_evaluable"] for row in daily)
    exposed = (reviewer.prior_predictions_seen or reviewer.prior_generation_labels_seen
               or reviewer.participated_in_activity_review)
    summary = {
        "status": "completed_declared_review", "cases": len(packet.cases), "interiors": len(inventory),
        "stage_evaluable_days": evaluable, "activity_features": dict(features),
        "stage_scoring_ready": evaluable > 0, "formal_reference_verified": False,
        "reviewer_declaration": reviewer.model_dump(),
        "reference_use": "development_only" if reviewer.reviewer_type == "ai" or exposed
        else "conditional_single_review_requires_provenance_verification",
        "activity_source": "frozen author review; not independently confirmed end-to-end truth",
        "mask": "observed days only; no automatic 61-day prediction support filter",
        "payload_sha256": hashlib.sha256(packet.model_dump_json().encode()).hexdigest(),
        "performance_scores": None,
    }
    return packet.model_dump(mode="json"), daily, summary, inventory
