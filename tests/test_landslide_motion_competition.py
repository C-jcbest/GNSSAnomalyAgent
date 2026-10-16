"""The tool must distinguish designed ramps from steps without producing labels."""
from types import SimpleNamespace

import numpy as np

from gnss_sim.landslide_motion_competition import motion_competition


def case_with_values(values):
    return SimpleNamespace(displacement_mm=[None if not np.isfinite(row).all() else row.tolist() for row in values])


def test_continuous_ramp_predicts_held_out_blocks_better_than_step():
    times = np.arange(182)
    values = times[:, None] * np.array([.07, -.04, -.03])
    result = motion_competition(case_with_values(values), np.ones(3), 0, 182)
    models = result["models"]
    assert result["status"] == "available"
    assert models["continuous_ramp"]["alternating_block_rmse_noise_units"] < 1e-10
    assert models["single_step"]["alternating_block_rmse_noise_units"] > .5
    assert "activity" not in result
    assert "stage" not in result


def test_single_jump_fits_step_but_hiding_blocks_can_hide_its_location():
    values = np.zeros((182, 3))
    values[84:] = [12, -8, 4]
    result = motion_competition(case_with_values(values), np.ones(3), 0, 182)
    models = result["models"]
    assert models["single_step"]["training_mse_noise_units"] < 1e-10
    assert models["continuous_ramp"]["training_mse_noise_units"] > .5
    assert models["single_step"]["alternating_block_rmse_noise_units"] > 0
    assert models["single_step"]["alternating_block_rmse_noise_units"] < models["stationary"]["alternating_block_rmse_noise_units"]
    assert "held-out error alone" in result["interpretation"]


def test_folds_partition_supported_blocks_and_do_not_fill_gaps():
    values = np.zeros((182, 3))
    values[28:42] = np.nan
    values[60:64] = np.nan
    result = motion_competition(case_with_values(values), np.ones(3), 0, 182)
    assert all(row["start"] != 28 for row in result["blocks"])
    for fitted in result["models"].values():
        for fold in fitted["folds"]:
            train, held_out = set(fold["training_block_indices"]), set(fold["held_out_block_indices"])
            assert not train & held_out
            assert train | held_out == set(range(len(result["blocks"])))
    short = motion_competition(case_with_values(values), np.ones(3), 0, 70)
    assert short["status"] == "unavailable"
    assert not short["models"]
