import copy
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_raw_adjudication import compile_author_activity, raw_block_statistics
from gnss_sim.landslide_reference_interface import partition_sha256
from gnss_sim.landslide_stage_reference import compile_reference, reference_template


@pytest.fixture
def context():
    values = [(float(i), 0.0, 0.0) for i in range(730)]
    values[104] = None
    case = LandslideInput(case_id="case_0001", dates=[date(2023, 1, 1) + timedelta(days=i)
                                                    for i in range(730)], displacement_mm=values)
    record = {"input_sha256": raw_block_statistics(case)["input_sha256"],
              "views": [{"image": "raw.png", "image_sha256": "test"}]}
    review, rows = compile_author_activity({"spans": [
        {"start": 0, "stop": 100, "state": "stationary", "raw_evidence": "flat"},
        {"start": 100, "stop": 110, "state": "activity", "raw_evidence": "sustained raw movement"},
        {"start": 110, "stop": 730, "state": "unknown", "raw_evidence": "not reviewed"},
    ], "activity_notes": "author activity only"}, case, record)
    motion = SimpleNamespace(velocity_mm_day=[None] * 730,
                             tangential_acceleration_mm_day2=[None] * 730)
    return {"case": case, "review": review, "rows": rows, "activity_sha256": partition_sha256(review),
            "motions": {31: motion, 61: motion, 91: motion}}


def completed(context):
    packet = reference_template([context], "manifest-test")
    packet["reviewer"] = {"name": "Fixture reviewer; not actual research review",
                          "background": "software test fixture only", "reviewer_type": "human",
                          "prior_predictions_seen": False, "prior_generation_labels_seen": False,
                          "participated_in_activity_review": False}
    interior = packet["cases"][0]["interiors"][0]
    interior.update(review_status="reviewed", raw_activity_status="confirmed",
                    raw_activity_evidence="N raw displacement rises throughout the interior",
                    raw_views_seen=True, auxiliary_views_seen=True, reviewed_at="2026-10-08T10:00:00+08:00")
    interior["segments"] = [{"start": 100, "stop": 110, "feature": "steady_motion",
                             "raw_evidence": "Raw linear displacement", "rate_evidence": "Slope stays comparable",
                             "nonzero_evidence": "Raw cumulative change continues", "steady_evidence": "No sustained slope change",
                             "unknown_reason": None}]
    return packet


def test_template_cannot_become_reference(context):
    template = reference_template([context], "manifest-test")
    assert template["cases"][0]["interiors"][0]["segments"] == []
    with pytest.raises(ValueError, match="identity"):
        compile_reference(template, [context], "manifest-test")
    packet = completed(context)
    packet["cases"][0]["interiors"][0]["review_status"] = "unreviewed"
    with pytest.raises(ValueError, match="Unreviewed"):
        compile_reference(packet, [context], "manifest-test")


def test_missing_mask_is_shared_and_reference_does_not_copy_prediction_fit_mask(context):
    _, rows, summary, support = compile_reference(completed(context), [context], "manifest-test")
    assert rows[100]["feature"] == "steady_motion" and rows[100]["stage_evaluable"]
    assert rows[104]["feature"] == "unknown" and not rows[104]["stage_evaluable"]
    assert rows[104]["reference_reason"] == "missing_observation"
    assert rows[99]["feature"] == "none" and not rows[99]["stage_evaluable"]
    assert rows[110]["feature"] == "unknown" and not rows[110]["stage_evaluable"]
    assert summary["stage_evaluable_days"] == 9 and summary["formal_reference_verified"] is False
    assert support[0]["windows"]["61"]["velocity_days"] == 0
    assert summary["stage_scoring_ready"] is True


@pytest.mark.parametrize("mistake", ["gap", "overlap", "outside", "float", "boolean", "empty",
                                    "nonzero", "steady", "raw", "rate", "identity", "hash", "case_drop",
                                    "exposure", "timestamp", "raw_unseen", "aux_unseen", "extra"])
def test_invalid_review_cannot_silently_become_complete(context, mistake):
    packet = completed(context)
    interior = packet["cases"][0]["interiors"][0]
    segment = interior["segments"][0]
    if mistake == "gap":
        segment["start"] = 101
    elif mistake == "overlap":
        interior["segments"].append(copy.deepcopy(segment))
    elif mistake == "outside":
        segment["stop"] = 111
    elif mistake == "float":
        segment["start"] = 100.0
    elif mistake == "boolean":
        segment["start"] = True
    elif mistake == "empty":
        interior["segments"] = []
    elif mistake in {"nonzero", "steady", "raw", "rate"}:
        segment[mistake + "_evidence"] = " "
    elif mistake == "identity":
        packet["reviewer"]["name"] = " "
    elif mistake == "hash":
        packet["cases"][0]["activity_sha256"] = "wrong"
    elif mistake == "case_drop":
        packet["cases"] = []
    elif mistake == "exposure":
        packet["reviewer"]["prior_predictions_seen"] = None
    elif mistake == "timestamp":
        interior["reviewed_at"] = "2026-10-08T10:00:00"
    elif mistake == "raw_unseen":
        interior["raw_views_seen"] = False
    elif mistake == "aux_unseen":
        interior["auxiliary_views_seen"] = False
    elif mistake == "extra":
        packet["invented_score"] = 1.0
    with pytest.raises(ValueError):
        compile_reference(packet, [context], "manifest-test")


def test_challenged_activity_remains_explicitly_unclassified(context):
    packet = completed(context)
    interior = packet["cases"][0]["interiors"][0]
    interior["raw_activity_status"] = "needs_review"
    with pytest.raises(ValueError, match="Challenged"):
        compile_reference(packet, [context], "manifest-test")
    segment = interior["segments"][0]
    segment.update(feature="unknown", unknown_reason="activity_needs_review", rate_evidence="",
                   nonzero_evidence="", steady_evidence="")
    _, rows, summary, _ = compile_reference(packet, [context], "manifest-test")
    assert rows[100]["activity_label"] == 1 and rows[100]["feature"] == "unknown"
    assert summary["stage_evaluable_days"] == 0 and not summary["stage_scoring_ready"]
    assert summary["performance_scores"] is None


def test_exposure_is_disclosed_instead_of_certifying_independence(context):
    packet = completed(context)
    packet["reviewer"].update(reviewer_type="ai", prior_predictions_seen=True)
    _, _, summary, _ = compile_reference(packet, [context], "manifest-test")
    assert summary["reference_use"] == "development_only"
    assert not summary["formal_reference_verified"]
