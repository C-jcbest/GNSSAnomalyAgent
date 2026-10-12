import pytest

from gnss_research.vision import parse_prediction


def test_flat_visual_activity_does_not_invent_motion_phases():
    prediction = parse_prediction('{"events":[{"kind":"deformation","start":5,"end":20,'
                                  '"status":"confirmed","evidence":"sustained raw motion"}]}', 30)
    assert prediction.events[0].start == 5
    assert prediction.events[0].phases == []


def test_visual_decision_rejects_unrequested_phase_fields():
    with pytest.raises(ValueError):
        parse_prediction('{"events":[{"kind":"deformation","start":5,"end":20,'
                         '"status":"confirmed","evidence":"motion","phases":[]}]}', 30)
