from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from gnss_anomaly.contracts import Prediction, Window
from gnss_anomaly.daily_dataset import build_daily_dataset
from gnss_anomaly.datasets import load_dataset
from gnss_anomaly.experiments import freeze_daily_validation, run_experiment
from gnss_anomaly.figures import figure_bytes
from gnss_anomaly.reporting import History
from gnss_anomaly.storage import file_hash, read_json, write_json


def daily_fixture(root: Path, monkeypatch):
    source = root / "source"
    catalog = []
    group_splits = {}
    for index, split in enumerate(("development", "validation", "test")):
        group = f"G{index}"
        group_splits[group] = split
        window = Window(
            case_id=f"source-{index}",
            sampling_hours=24,
            timestamps=[
                datetime(2026, 1, 1, 7, tzinfo=timezone.utc) + timedelta(days=day)
                for day in range(180)
            ],
            values=[(float(day + index), float(day * 2), 0.0) for day in range(180)],
        )
        relative = f"windows/{index}.json"
        path = source / "long" / relative
        write_json(path, window.model_dump(mode="json"))
        catalog.append(
            {
                "id": window.case_id,
                "group": group,
                "split": split,
                "file": relative,
                "sha256": file_hash(path),
            }
        )
    write_json(source / "long/windows.json", {"items": catalog})
    write_json(source / "bundle.json", {})
    monkeypatch.setattr(
        "gnss_anomaly.daily_dataset.verify_daily",
        lambda _: {
            "window_days": 180,
            "group_splits": group_splits,
        },
    )
    return source


def test_daily_dataset_is_source_isolated_and_unlabelled_to_methods(tmp_path, monkeypatch):
    source = daily_fixture(tmp_path, monkeypatch)
    dataset = tmp_path / "dataset"
    assert build_daily_dataset(source, dataset) == 72
    manifest = load_dataset(dataset)
    assert {
        split: sum(r["split"] == split for r in manifest["records"])
        for split in ("development", "validation", "test")
    } == {
        "development": 24,
        "validation": 24,
        "test": 24,
    }
    record = manifest["records"][0]
    input_window = read_json(dataset / record["file"])
    assert set(input_window) == {"case_id", "sampling_hours", "timestamps", "values"}
    assert not set(input_window) & {"kind", "split", "events", "injection"}
    assert read_json(dataset / "labels.json")[record["case_id"]]["injection"]
    with pytest.raises(FileExistsError):
        build_daily_dataset(source, dataset)


def test_native_only_keeps_original_cases_and_labels(tmp_path, monkeypatch):
    source = daily_fixture(tmp_path, monkeypatch)
    catalog = read_json(source / "long/windows.json")
    first = catalog["items"][0]
    source_window = read_json(source / "long" / first["file"])
    source_window["values"][150][2] = None
    write_json(source / "long" / first["file"], source_window)
    first["sha256"] = file_hash(source / "long" / first["file"])
    write_json(source / "long/windows.json", catalog)
    full = tmp_path / "full"
    native = tmp_path / "native"
    build_daily_dataset(source, full)
    assert build_daily_dataset(source, native, native_only=True) == 18
    full_manifest = load_dataset(full)
    native_manifest = load_dataset(native)
    assert native_manifest["protocol"] == "daily15-injection-native-v1"
    assert {record["state"] for record in native_manifest["records"]} == {"native"}
    original = {
        record["case_id"]: record
        for record in full_manifest["records"]
        if record["state"] == "native"
    }
    assert {record["case_id"] for record in native_manifest["records"]} == set(original)
    assert read_json(native / "labels.json") == {
        case_id: label
        for case_id, label in read_json(full / "labels.json").items()
        if case_id in original
    }
    for record in native_manifest["records"]:
        assert record == original[record["case_id"]]
        assert file_hash(native / record["file"]) == file_hash(full / record["file"])
        values = read_json(native / record["file"])["values"]
        assert sum(value is None for row in values for value in row) == (
            1 if record["source_group_id"] == first["id"] else 0
        )


def test_paired_union_reuses_one_visual_and_freeze_keeps_config(tmp_path, monkeypatch):
    source = daily_fixture(tmp_path, monkeypatch)
    dataset = tmp_path / "dataset"
    build_daily_dataset(source, dataset)
    config = read_json(Path(__file__).parents[1] / "configs/daily-comparison-v1.json")
    config["workers"] = 1
    config["detectors"]["iforest_trees"] = 5

    class ModelStub:
        requests = 0

        def __init__(self, _config):
            self.calls = []
            self.cost = None

        def reset(self):
            self.calls = []

        def close(self):
            pass

        def ask(self, prompt, schema, images):
            assert "injection" not in prompt and "source_group_id" not in prompt
            assert len(images) == 1
            ModelStub.requests += 1
            self.calls = [{"attempt": 1, "status": "ok", "prediction_status": "ok"}]
            return schema.model_validate({"events": [{"start": 54, "end": 54, "channels": ["N"]}]})

    monkeypatch.setattr("gnss_anomaly.experiments.ModelClient", ModelStub)
    methods = ["hampel", "cusum", "iforest", "visual", "daily_union"]
    run_experiment(dataset, tmp_path / "runs/dev", config, "development", methods, limit=1)
    assert ModelStub.requests == 1
    result_files = list((tmp_path / "runs/dev/results").glob("*.json"))
    rows = {r["method"]: r for r in map(read_json, result_files)}
    assert rows["daily_union"]["model_requests"] == 0
    assert rows["daily_union"]["prediction"]["status"] == "ok"
    assert np.array_equal(
        Prediction.model_validate(rows["daily_union"]["prediction"]).mask(180),
        Prediction.model_validate(rows["hampel"]["prediction"]).mask(180)
        | Prediction.model_validate(rows["visual"]["prediction"]).mask(180),
    )
    assert {tool["tool"] for tool in rows["daily_union"]["tools"]} == {"hampel", "visual"}
    history = History(tmp_path)
    detail = history.detail(next(r["key"] for r in history.index()["runs"] if r["name"] == "dev"))
    chart = figure_bytes(detail, "overview", "svg")
    assert b"Paired H+Visual union" in chart
    assert b"native background unverified" in chart
    with pytest.raises(ValueError, match="calibrate"):
        run_experiment(dataset, tmp_path / "test-denied", config, "test", methods)
    run_experiment(dataset, tmp_path / "validation", config, "validation", methods)
    frozen = freeze_daily_validation(dataset, tmp_path / "validation", tmp_path / "frozen.json")
    assert frozen["detectors"] == config["detectors"]
    with pytest.raises(ValueError, match="all five"):
        run_experiment(dataset, tmp_path / "test-subset", frozen, "test", ["visual"])
    frozen["model"]["max_tokens"] += 1
    with pytest.raises(ValueError, match="unchanged"):
        run_experiment(dataset, tmp_path / "test-changed", frozen, "test", methods)
