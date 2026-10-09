from datetime import date, timedelta

import numpy as np

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_raw_adjudication import (
    compile_author_activity,
    raw_block_statistics,
    render_author_activity,
)


def observations():
    values = [(float(day), 2.0, -1.0) for day in range(730)]
    values[1] = None
    values[30:60] = [None] * 30
    return LandslideInput(case_id="case_0001", dates=[date(2023, 1, 1) + timedelta(days=day)
                                                    for day in range(730)], displacement_mm=values)


def test_blocks_keep_calendar_gaps_empty_blocks_and_short_tail():
    case = observations()
    result = raw_block_statistics(case)
    first, empty, tail = result["blocks"][0], result["blocks"][1], result["blocks"][-1]
    assert first["valid_days"] == 29
    assert first["median_NEU_mm"] == [15.0, 2.0, -1.0]
    assert empty == {"start": 30, "stop": 60, "valid_days": 0,
                     "median_NEU_mm": [None] * 3, "iqr_NEU_mm": [None] * 3}
    assert tail["start"] == 720 and tail["stop"] == 730 and tail["valid_days"] == 10
    assert tail["median_NEU_mm"] == [724.5, 2.0, -1.0]


def test_author_identity_and_stages_are_not_api_or_independent_claims():
    case = observations()
    record = {"input_sha256": raw_block_statistics(case)["input_sha256"],
              "views": [{"image": "raw.png", "image_sha256": "hash"}]}
    payload = {"spans": [{"start": 0, "stop": 10, "state": "activity", "raw_evidence": "raw change"},
                         {"start": 10, "stop": 730, "state": "unknown", "raw_evidence": "unresolved"}],
               "activity_notes": "author review"}
    review, rows = compile_author_activity(payload, case, record)
    assert "single-pass" not in review["reviewer"]
    assert review["prior_outputs_seen"] and not review["independent_expert"]
    assert review["stage_status"] == "not_started" and review["new_external_calls"] == 0
    assert rows[0]["activity_label"] == 1 and rows[1]["activity_label"] == -1
    assert rows[9]["activity_label"] == 1 and rows[10]["activity_label"] == -1
    assert all(row["feature"] == "unknown" for row in rows)


def test_local_strip_preserves_masks_and_original_axes(tmp_path):
    case = observations()
    rows = [{"activity_label": 1 if day < 20 else -1} for day in range(730)]
    view = {"view": [0, 59], "audit": {"panels": [
        {"xlim": [0, 59], "ylim": [-10, 100]},
        {"xlim": [0, 59], "ylim": [-20, 20]},
        {"xlim": [0, 59], "ylim": [-30, 30]},
    ], "tick_audit": {"ticks": [0, 30, 59]}}}
    audit = render_author_activity(case, view, rows, tmp_path / "review.png",
                                   {"figure_inches": [12, 7], "dpi": 70})
    expected = np.asarray([2 if day < 20 else 0 for day in range(730)])
    expected[1] = 3
    expected[30:60] = 3
    assert audit["label_colors"] == expected.tolist()
    assert audit["xlim"] == [0, 59]
    assert audit["ylim_NEU"] == [[-10, 100], [-20, 20], [-30, 30]]
