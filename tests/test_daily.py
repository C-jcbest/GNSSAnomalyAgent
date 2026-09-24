import numpy as np
import pandas as pd
import pytest

from gnss_anomaly.contracts import Window
from gnss_anomaly.daily import daily_window, prepare_daily, select_daily, verify_daily
from gnss_anomaly.datasets import inject
from gnss_anomaly.figures import figure_bytes
from gnss_anomaly.reporting import History
from gnss_anomaly.storage import file_hash, read_json, write_json


def test_exact_local_hour_preserves_missing_partial_zero_and_edges():
    times = pd.date_range("2026-01-01", periods=6 * 24, freq="h", tz="Asia/Shanghai")
    frame = pd.DataFrame({"timestamp": times.tz_convert("UTC")})
    for c in ("n_mm", "e_mm", "u_mm"):
        frame[c] = np.arange(len(frame), dtype=float)
    frame.loc[[15, 15 + 5 * 24, 15 + 2 * 24], ["n_mm", "e_mm", "u_mm"]] = np.nan
    frame.loc[15 + 24, ["n_mm", "e_mm", "u_mm"]] = [0, np.nan, 1.23456789012345]
    part = select_daily(frame)
    w = daily_window(part, "daily")
    assert len(w.values) == 4 and w.sampling_hours == 24
    assert w.timestamps[0].hour == 7
    assert w.values[0] == (0, None, 1.23456789012345)
    assert w.values[1] == (None, None, None)  # Nearby valid 14/16h cannot substitute.
    payload = w.model_dump(mode="json")
    payload.pop("sampling_hours")
    with pytest.raises(ValueError, match="grid"):
        Window.model_validate(payload)
    with pytest.raises(ValueError, match="daily injection"):
        inject(w, "drift", np.random.default_rng(1))


def test_daily_bundle_lineage_splits_sparse_exclusion_history_and_figure(tmp_path):
    source = tmp_path / "hourly"
    groups = {"g0": "development", "g1": "validation", "g2": "test"}
    aliases = ["known", "dev", "val", "test", "short", "sparse", "no15"]
    for i, alias in enumerate(aliases):
        folder = source / "snapshots" / alias
        folder.mkdir(parents=True)
        n = 70 if alias == "short" else 200
        times = pd.date_range("2026-01-01", periods=n * 24, freq="h", tz="Asia/Shanghai")
        frame = pd.DataFrame({"timestamp": times.tz_convert("UTC")})
        for c in ("n_mm", "e_mm", "u_mm"):
            frame[c] = np.arange(len(frame), dtype=float) / 100
        if alias == "sparse":
            frame.loc[24 : len(frame) - 25, ["n_mm", "e_mm", "u_mm"]] = np.nan
        if alias == "no15":
            frame.loc[times.hour == 15, ["n_mm", "e_mm", "u_mm"]] = np.nan
        frame.to_csv(folder / "observations.csv", index=False)
        write_json(
            folder / "manifest.json",
            {
                "station_alias": alias,
                "site_alias": "g1" if alias == "val" else "g2" if alias == "test" else "g0",
                "files": {"observations.csv": file_hash(folder / "observations.csv")},
            },
        )
    write_json(source / "preparation-report.json", {"group_splits": groups})
    write_json(source / "real-cases/windows.json", {"items": [{"station_alias": "known"}]})
    files = {
        p.relative_to(source).as_posix(): file_hash(p) for p in source.rglob("*") if p.is_file()
    }
    write_json(source / "bundle.json", {"files": files})
    out = tmp_path / "data/processed/daily"
    r = prepare_daily(source, out)
    assert verify_daily(out) == r
    assert r["retained_stations"] == 6 and r["excluded_empty_stations"] == 1
    assert r["long_splits"] == {"development": 1, "validation": 1, "test": 1}
    assert {x["station_alias"]: x["status"] for x in r["stations"]}[
        "sparse"
    ] == "low_window_coverage"
    assert all(file_hash(source / p) == sha for p, sha in files.items())
    cases = read_json(out / "real-cases/windows.json")["items"]
    assert cases[0]["audit"]["expected_days"] == 200 and cases[0]["split"] == "auxiliary"
    for x in read_json(out / "long/windows.json")["items"]:
        assert not x["approved"] and x["audit"]["expected_days"] == 180
    history = History(tmp_path)
    index = history.data_index()
    assert len(index["snapshots"]) == 6
    w, _ = history.snapshot(index["snapshots"][0]["key"])
    assert w.sampling_hours == 24
    svg = figure_bytes(None, "snapshot", case=(w, None, None), trend=True).decode()
    assert "Beijing date (15:00)" in svg and "UTC daily median" not in svg
    with pytest.raises(FileExistsError):
        prepare_daily(source, out)
    (out / "snapshots/dev/observations.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_daily(out)
