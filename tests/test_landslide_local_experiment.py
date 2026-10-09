from __future__ import annotations

import numpy as np

from gnss_sim.landslide_local_experiment import (
    activity_prediction,
    execute_packet,
    learned_stages,
    request_schedule,
)


def test_schedule_alternates_arms_without_using_activity_labels():
    schedule = request_schedule(["case_0007", "case_0002", "case_0015"])
    assert [row["request_id"] for row in schedule[:6]] == [
        "case_0002-locate-global", "case_0002-locate-local", "case_0007-locate-local",
        "case_0007-locate-global", "case_0015-locate-global", "case_0015-locate-local",
    ]
    assert len({row["request_id"] for row in schedule}) == 12
    assert all(row["method"].startswith("stage_") for row in schedule[6:])


def test_activity_experiment_cannot_claim_definite_stages_and_preserves_gaps():
    prediction = activity_prediction({"activity": [[1, 3]], "uncertain": []}, np.array([True, True, False, True]))
    assert prediction.activity.tolist() == [0, 1, -1, 1]
    assert prediction.stage.tolist() == [0, -1, -1, -1]


def test_saved_stage_head_keeps_unsupported_activity_but_withholds_stage():
    gate = np.array([0, 1, 1, -1])
    codes = np.array([1, 2, 3, 1])
    prediction = learned_stages(codes, gate, np.array([True, True, False, True]))
    assert prediction.activity.tolist() == [0, 1, 1, -1]
    assert prediction.stage.tolist() == [0, 2, -1, -1]
    assert codes.tolist() == [1, 2, 3, 1]


def test_stage_failure_keeps_fixed_gate_and_does_not_retry():
    class InvalidResponse:
        calls = 0

        def ask(self, request_id, prompt, images):
            self.calls += 1
            return {"stages": [[0, 2, "A"]]}

    transport = InvalidResponse()
    packet = {"method": "stage_local", "request_id": "case-stage-local", "prompt": "frozen", "activity": [0, 1, 1]}
    failures = {}
    prediction = execute_packet(transport, packet, np.ones(3, dtype=bool), [], failures)
    assert transport.calls == 1
    assert prediction.activity.tolist() == [0, 1, 1]
    assert prediction.stage.tolist() == [-1, -1, -1]
    assert prediction.stage_status == "failed"
    assert "no automatic clipping" in failures[packet["request_id"]]
