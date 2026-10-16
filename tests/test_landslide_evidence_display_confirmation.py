"""Scheduling and scope contracts for the unseen paired confirmation."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_landslide_evidence_display_confirmation import (  # noqa: E402
    fixed_targets,
    select_workflows,
    unique_local_queries,
)


def test_fixed_control_uses_calendar_thirds():
    windows = [(start, min(start + 180, 730)) for start in range(0, 730, 90)]
    assert fixed_targets(windows, 730) == [
        {"index": 2, "start": 180, "stop": 360},
        {"index": 4, "start": 360, "stop": 540},
    ]


def test_shared_views_deduplicate_without_confusing_displays():
    raw = {"index": 2, "start": 180, "stop": 360, "arm": "narrow_raw"}
    blocks = {**raw, "arm": "narrow_blocks"}
    assert unique_local_queries({"first": [raw], "second": [raw, blocks]}) == [blocks, raw]
    with pytest.raises(ValueError, match="different calendar"):
        unique_local_queries({"first": [raw], "second": [{**raw, "stop": 400}]})


def test_quiet_unknown_and_missing_dates_keep_bounded_authority():
    days = 730
    observed = np.ones(days, dtype=bool)
    observed[200:220] = False
    case = SimpleNamespace(displacement_mm=[None if not valid else [0., 0., 0.] for valid in observed])
    features = SimpleNamespace(observed=observed, progressive_rate=np.zeros(days),
                               signal_to_noise=np.zeros(days), noise=np.ones(3))
    initial = np.zeros(days, dtype=int)
    initial[~observed] = -1
    selected, scopes, _ = select_workflows(initial, case, features)
    assert len(selected) == 8
    assert len(selected["fixed_narrow_raw_budget2"]) == 2
    assert selected["adaptive84_evidence_display"] == []
    assert all(not scope.any() for scope in scopes.values())
    initial[300:320] = 1
    selected, scopes, _ = select_workflows(initial, case, features)
    assert all(len(targets) <= 2 for targets in selected.values())
    assert not any(scope[~observed].any() for scope in scopes.values())
    assert scopes["adaptive42_narrow_raw"][300:320].all()
    assert not scopes["adaptive42_narrow_raw"][0:100].any()
    assert len(unique_local_queries(selected)) <= 12
