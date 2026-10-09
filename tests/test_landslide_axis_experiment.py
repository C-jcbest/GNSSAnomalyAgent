from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gnss_sim.landslide_axis_experiment import (
    ARMS,
    axis_prompt,
    image_ticks,
    parse_axis_response,
    prompt_arm,
    render_tick_pair,
)
from gnss_sim.landslide_boundary_experiment import boundary_blocks, boundary_prompt


def test_tick_changes_do_not_change_prompt_or_parser_contract():
    assert axis_prompt("p0_t0", 450, 660) == axis_prompt("p0_t15", 450, 660)
    assert axis_prompt("p15_t0", 450, 660) == axis_prompt("p15_t15", 450, 660)
    assert axis_prompt("p0_t0", 450, 660) == boundary_prompt("b1_endpoints", 450, 660)
    assert axis_prompt("p15_t0", 450, 660) == boundary_prompt("b2_phase15", 450, 660)


@pytest.mark.parametrize("arm", ARMS)
def test_activity_inside_blocks_and_missing_dates(arm):
    payload = {
        "activity": [[520, 659]], "uncertain": [[500, 519]],
        "evidence": "Plateau then sustained movement", "first_activity_day": 520,
        "blocks": [
            {"start": left, "end": right - 1, "state": "mixed", "observation": "plateau then motion"}
            for left, right in boundary_blocks(450, 660, prompt_arm(arm))
        ],
    }
    observed = np.ones(700, dtype=bool)
    observed[550] = False
    result = parse_axis_response(payload, observed, 450, 660, arm)
    assert (result[450:500] == 0).all()
    assert (result[500:520] == -1).all()
    assert result[520] == result[659] == 1
    assert result[550] == result[660] == -1
    payload["blocks"][-1]["state"] = "stationary"
    with pytest.raises(ValueError, match="contradicts"):
        parse_axis_response(payload, observed, 450, 660, arm)
    payload["blocks"][-1]["state"] = "mixed"
    payload["activity"] = [[520, 660]]
    with pytest.raises(ValueError):
        parse_axis_response(payload, observed, 450, 660, arm)


def test_ticks_keep_original_calendar_and_last_day():
    assert image_ticks(405, 705, 0).tolist() == [*range(405, 705, 30), 704]
    assert image_ticks(405, 705, 15).tolist() == [*range(420, 705, 30), 704]
    assert image_ticks(0, 10, 15).tolist() == [9]
    with pytest.raises(ValueError):
        image_ticks(0, 100, 10)


def test_rendered_tick_pair_preserves_curve_geometry_and_missing_gaps(tmp_path):
    values = [[float(day), 2.0, -day / 5] for day in range(120)]
    values[55] = None
    case = SimpleNamespace(case_id="test", displacement_mm=values)
    paths = {phase: tmp_path / f"tick-{phase}.png" for phase in (0, 15)}
    audit = render_tick_pair(case, np.ones(3), 20, 100, paths,
                             {"single_plot_inches": [10.24, 6.3], "single_plot_dpi": 150})
    assert audit["geometry_unchanged"]
    assert audit["geometry"][0]["xlim"] == [20.0, 99.0]
    assert audit["geometry"][1]["ylim"] == [-5.5, 9.5]
    assert paths[0].read_bytes() != paths[15].read_bytes()
