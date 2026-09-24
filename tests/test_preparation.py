from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from gnss_anomaly.datasets import build_dataset, load_dataset
from gnss_anomaly.preparation import prepare_offline, trim_snapshot, verify_preparation
from gnss_anomaly.reporting import History
from gnss_anomaly.storage import digest, file_hash, read_json, write_json


def snapshot(root, station, values):
    alias = "S-" + digest(station["StationUUID"])[:8]
    folder = root / alias
    folder.mkdir(parents=True)
    times = pd.date_range("2026-01-01", periods=len(values), freq="h", tz="Asia/Shanghai")
    frame = pd.DataFrame(values, columns=["n_mm", "e_mm", "u_mm"])
    frame.insert(0, "timestamp", times)
    frame.to_csv(folder / "observations.csv", index=False)
    write_json(
        folder / "manifest.json",
        {
            "kind": "platform_snapshot",
            "station_alias": alias,
            "start": times[0].isoformat(),
            "end_exclusive": (times[-1].to_pydatetime() + timedelta(hours=1)).isoformat(),
            "created_at": times[0].isoformat(),
            "files": {"observations.csv": file_hash(folder / "observations.csv")},
            "units_verified": True,
        },
    )
    return folder


def test_trim_retains_partial_edges_internal_gaps_and_raw_values(tmp_path):
    values = np.array(
        [[np.nan] * 3, [0, np.nan, 1.1234567890123457], [np.nan] * 3, [2, 3, 4], [np.nan] * 3]
    )
    source = snapshot(tmp_path / "raw", {"StationUUID": "one"}, values)
    original = (source / "observations.csv").read_bytes()
    row, part = trim_snapshot(source, tmp_path / "trimmed", "group")
    assert row["leading_removed_hours"] == row["trailing_removed_hours"] == 1
    assert len(part) == 3 and row["audit"]["observed_timestamps"] == 2
    assert row["audit"]["longest_empty_gap_hours"] == 1
    assert row["end_exclusive"] == "2026-01-01T04:00:00+08:00"
    assert (source / "observations.csv").read_bytes() == original
    reread = pd.read_csv(tmp_path / "trimmed/observations.csv", float_precision="round_trip")
    np.testing.assert_array_equal(reread[["n_mm", "e_mm", "u_mm"]].to_numpy(), values[1:4])
    empty = snapshot(tmp_path / "empty", {"StationUUID": "empty"}, np.full((10, 3), np.nan))
    result, part = trim_snapshot(empty, tmp_path / "excluded", "group")
    assert result["status"] == "excluded_empty" and part is None
    assert not (tmp_path / "excluded").exists()
    (source / "observations.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        trim_snapshot(source, tmp_path / "bad", "group")


def test_preparation_source_splits_across_scales_and_future_injection(tmp_path):
    source = tmp_path / "data/snapshots/source"
    rows = [{"StationUUID": str(i), "StationGroupUUID": str(i // 2)} for i in range(8)]
    for r in rows:
        values = np.zeros((24 * 125, 3))
        values[:3] = np.nan
        values[-5:] = np.nan
        values[1500] = np.nan
        snapshot(source, r, values)
    write_json(tmp_path / "stations.json", rows)
    known = "S-" + digest("0")[:8]
    write_json(
        tmp_path / "references.json",
        {
            "cases": [
                {
                    "station_alias": known,
                    "start": "2026-01-02T00:00:00+08:00",
                    "end_exclusive": "2026-05-01T00:00:00+08:00",
                }
            ]
        },
    )
    out = tmp_path / "data/processed/offline-v1"
    report = prepare_offline(source, tmp_path / "stations.json", tmp_path / "references.json", out)
    assert verify_preparation(out) == report
    assert report["retained_stations"] == 8 and report["removed_hours"] == 64
    assert report["long_windows"] == 7 and report["auxiliary_cases"] == 1
    cat = read_json(out / "short/backgrounds.json")
    assert known not in {r["station_alias"] for r in cat["items"]}
    assert not any(r["approved"] for r in cat["items"])
    assert cat["group_splits"]["G-" + digest("0")[:8]] == "development"
    for r in cat["items"]:
        r["approved"] = True  # Fixture-only reviewed constant backgrounds.
    write_json(out / "short/backgrounds.json", cat)
    build_dataset(out / "short", tmp_path / "dataset", 5, seed=1)
    m = load_dataset(tmp_path / "dataset")
    assert all(r["split"] == cat["group_splits"][r["group"]] for r in m["records"])
    with pytest.raises(ValueError, match="checksum"):
        verify_preparation(out)
    history = History(tmp_path)
    index = history.data_index()
    assert len(index["snapshots"]) == 8
    assert all(s["trimmed"] for s in index["snapshots"])
    window, meta = history.snapshot(index["snapshots"][0]["key"])
    assert window.values[0] == (0, 0, 0) and len(window.values) == 2992
