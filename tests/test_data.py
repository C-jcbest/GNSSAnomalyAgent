from datetime import timedelta

import httpx
import numpy as np
import pytest

from gnss_anomaly.acquisition import Platform, local_time, normalize
from gnss_anomaly.contracts import Window
from gnss_anomaly.datasets import (
    build_dataset,
    build_pilot,
    demo_backgrounds,
    extract_window,
    load_dataset,
    prepare_backgrounds,
)
from gnss_anomaly.storage import child, file_hash, read_json, write_json


def test_normalize_units_boundary_missing_and_conflict():
    start, end = local_time("2025-01-01"), local_time("2025-01-01T03:00")
    row = {"DataTime": "2025-01-01 00:00:00", "N": 1.001, "E": None, "U": 3.002}
    frame, audit = normalize([row, row, dict(row, DataTime=end.isoformat())], [1, 2, 3], start, end)
    assert audit["duplicates"] == 1 and audit["outside_range"] == 1
    assert audit["observed_timestamps"] == 1 and audit["expected_hours"] == 3
    assert frame.n_mm[0] == pytest.approx(1) and frame.u_mm[0] == pytest.approx(2)
    assert np.isnan(frame.e_mm[0]) and np.isnan(frame.n_mm[1])
    assert str(frame.timestamp[0]) == "2024-12-31 16:00:00+00:00"
    with pytest.raises(ValueError, match="conflicting"):
        normalize([row, dict(row, N=2)], [1, 2, 3], start, end)
    with pytest.raises(ValueError, match="off-grid"):
        normalize([dict(row, DataTime="2025-01-01 00:15:00")], [1, 2, 3], start, end)


def test_platform_requires_success_and_does_not_expose_secret(monkeypatch):
    for name, value in {
        "BEIDOU_API_BASE_URL": "https://example.test",
        "BEIDOU_USERNAME": "user",
        "BEIDOU_PASSWORD": "private-pass",
    }.items():
        monkeypatch.setenv(name, value)
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"ResponseMsg": "private-pass"})
        )
    )
    platform = Platform(client)
    with pytest.raises(RuntimeError, match="ResponseCode missing") as exc:
        platform.login()
    assert "private-pass" not in str(exc.value)
    platform.close()


def test_build_preserves_groups_and_label_boundary(tmp_path):
    demo_backgrounds(tmp_path / "bg", 7)
    for name in ("a", "b"):
        assert build_dataset(tmp_path / "bg", tmp_path / name, 15, 7) == 360
    a, b = load_dataset(tmp_path / "a"), load_dataset(tmp_path / "b")
    assert a == b
    groups = {}
    for r in a["records"]:
        assert groups.setdefault(r["group"], r["split"]) == r["split"]
        data = read_json(child(tmp_path / "a", r["file"]))
        assert set(data) == {"case_id", "sampling_hours", "timestamps", "values"}
        Window.model_validate(data)
    assert set(groups.values()) == {"development", "validation", "test"}
    r = a["records"][0]
    child(tmp_path / "a", r["file"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(tmp_path / "a")


def test_unreviewed_sources_and_path_escape_rejected(tmp_path):
    demo_backgrounds(tmp_path / "bg")
    cat = read_json(tmp_path / "bg/backgrounds.json")
    for row in cat["items"]:
        row["approved"] = False
    write_json(tmp_path / "bg/backgrounds.json", cat)
    with pytest.raises(ValueError, match="source groups"):
        build_dataset(tmp_path / "bg", tmp_path / "dataset", 15)
    with pytest.raises(ValueError):
        child(tmp_path, "../outside.json")
    with pytest.raises(ValueError):
        prepare_backgrounds([], tmp_path / "empty", 0)


def test_window_timezone_and_grid(window):
    original = window()
    data = original.model_dump()
    data["timestamps"][1] += timedelta(minutes=1)
    with pytest.raises(ValueError):
        Window.model_validate(data)


def test_provisional_pilot_is_balanced_and_rejects_unreviewed_or_holdout_sources(tmp_path, window):
    source = tmp_path / "window.json"
    write_json(source, window(length=336).model_dump(mode="json"))
    review = tmp_path / "review.json"
    decision = {
        "scope": "pilot_only",
        "decision": "usable_for_pilot",
        "split": "test",
        "group": "g",
        "window_sha256": file_hash(source),
        "review_note": "fixture only",
    }
    write_json(review, decision)
    with pytest.raises(ValueError, match="development-only"):
        build_pilot(source, review, tmp_path / "rejected")
    decision["split"] = "development"
    write_json(review, decision)
    assert build_pilot(source, review, tmp_path / "pilot") == 24
    m = load_dataset(tmp_path / "pilot")
    assert m["kind"] == "platform_pilot_provisional"
    assert len({(r["kind"], r["state"]) for r in m["records"]}) == 24
    assert {r["split"] for r in m["records"]} == {"development"}
    source.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        build_pilot(source, review, tmp_path / "tampered")


def test_daily_collection_resume_and_offline_extraction(tmp_path):
    class FixturePlatform(Platform):
        def __init__(self):
            self.session = "fixture-session"
            self.queries = []
            self.interrupt = True

        def post(self, path, payload):
            self.queries.append(payload["BeginTime"])
            if len(self.queries) == 2 and self.interrupt:
                raise ValueError("simulated interruption")
            return {
                "ResponseCode": "200",
                "Data": [{"DataTime": payload["BeginTime"], "N": 1.001, "E": 2.001, "U": 3.001}],
            }

    platform = FixturePlatform()
    station = {"StationUUID": "fixture-station", "StationN0": 1, "StationE0": 2, "StationU0": 3}
    start, end = local_time("2025-01-01"), local_time("2025-01-03")
    with pytest.raises(ValueError, match="interruption"):
        platform.fetch(station, start, end, tmp_path / "snapshot", "S1")
    first = (tmp_path / "snapshot/raw/0000.json").read_bytes()
    platform.interrupt = False
    result = platform.fetch(station, start, end, tmp_path / "snapshot", "S1", resume=True)
    assert len(platform.queries) == 3 and result["observed_timestamps"] == 2
    assert (tmp_path / "snapshot/raw/0000.json").read_bytes() == first
    assert extract_window(tmp_path / "snapshot", tmp_path / "case", start, end) == 48
    extracted = Window.model_validate(read_json(tmp_path / "case/window.json"))
    assert extracted.values[1] == (None, None, None)
    assert not read_json(tmp_path / "case/provenance.json")["has_dense_ground_truth"]
    platform.fetch(station, start, end, tmp_path / "snapshot", "S1", resume=True)
    assert (
        len(platform.queries) == 3
    )  # A completed valid snapshot never contacts the platform again.
    with pytest.raises(ValueError, match="query changed"):
        platform.fetch(
            station, start, end + timedelta(days=1), tmp_path / "snapshot", "S1", resume=True
        )
