"""Guard calendar, component units and horizontal-reference semantics for real data."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_landslide_field_probe import (  # noqa: E402
    daily_coordinates,
    horizontal_reference_rate,
    load_ngl_records,
    relative_displacement,
)

HEADER = "site YYMMMDD __MJD reflon ____n0(m) _north(m) _e0(m) __east(m) u0(m) ____up(m) sig_n(m) sig_e(m) sig_u(m)"


def test_integer_reinitialization_preserves_displacement_and_axis_order(tmp_path):
    source = tmp_path / "OREO.tenv3"
    source.write_text(
        HEADER + "\nOREO 18MAR01 58178 -121.8 100 1.2 10 1.1 20 0.8 .001 .001 .004\n"
        "OREO 18MAR02 58179 -121.8 101 .3 11 .2 21 -.1 .001 .001 .004\n",
        encoding="utf-8",
    )
    records = load_ngl_records(source)
    calendar = [date(2018, 3, 1), date(2018, 3, 2), date(2018, 3, 3)]
    values = daily_coordinates(records, calendar)
    np.testing.assert_allclose(values[1] - values[0], [.1, .1, .1], atol=1e-12)
    assert np.isnan(values[2]).all()
    np.testing.assert_allclose(values[0], [101.2, 11.1, 20.8])


@pytest.mark.parametrize("second", [
    "OREO 18MAR01 58178 -121.8 100 1.2 10 1.1 20 .8 .001 .001 .004",
    "OREO 18MAR02 58178 -121.8 100 1.2 10 1.1 20 .8 .001 .001 .004",
    "OTHER 18MAR02 58179 -121.8 100 1.2 10 1.1 20 .8 .001 .001 .004",
])
def test_duplicate_dates_and_date_or_station_mismatch_are_rejected(tmp_path, second):
    source = tmp_path / "OREO.tenv3"
    source.write_text(HEADER + "\nOREO 18MAR01 58178 -121.8 100 1.2 10 1.1 20 .8 .001 .001 .004\n"
                      + second + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ngl_records(source)


def test_reference_removes_horizontal_rate_preserves_vertical_and_missing_days():
    day = np.arange(90, dtype=float)
    rates = np.array([.0002, -.0001, .0004])
    coordinates = np.array([1000, 2000, 3000]) + day[:, None] * rates
    coordinates[4] = np.nan
    estimated = horizontal_reference_rate(coordinates)
    np.testing.assert_allclose(estimated, [rates[0], rates[1], 0], atol=1e-12)
    values, _ = relative_displacement(coordinates, estimated)
    assert values[4] is None
    complete = np.array([row for row in values if row is not None])
    np.testing.assert_allclose(complete[:, :2], 0, atol=1e-8)
    np.testing.assert_allclose(np.array(values[-1]) - np.array(values[-2]), [0, 0, .4], atol=1e-8)
    with pytest.raises(ValueError, match="horizontal"):
        relative_displacement(coordinates, rates)
