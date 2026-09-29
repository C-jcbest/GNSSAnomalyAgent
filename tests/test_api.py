import time
from pathlib import Path

from fastapi.testclient import TestClient

from gnss_sim.api import create_app
from gnss_sim.schemas import GenerationRequest
from gnss_sim.storage import DatasetStore


def test_generation_progress_persistence_and_truth_isolation(tmp_path):
    app = create_app(tmp_path / "generated", tmp_path / "no-built-web")
    with TestClient(app) as client:
        response = client.post("/api/datasets", json={"seed": 123, "count": 2, "case_type": "trend"})
        assert response.status_code == 202
        dataset_id = response.json()["dataset_id"]
        for _ in range(100):
            manifest = client.get(f"/api/datasets/{dataset_id}").json()
            if manifest["status"] in ("complete", "failed"):
                break
            time.sleep(0.1)
        assert manifest["status"] == "complete"
        assert manifest["generator_version"] == "synthetic-v1"
        assert manifest["generated_cases"] == 2
        assert manifest["type_counts"] == {"trend": 2}
        assert all(item["case_type"] == "trend" for item in manifest["cases"])
        assert all(item["event_count"] == 1 for item in manifest["cases"])
        assert client.get("/api/datasets").json()[0]["dataset_id"] == dataset_id

        path = tmp_path / "generated" / dataset_id / "cases" / "case_0001"
        case_input = client.get(f"/api/datasets/{dataset_id}/cases/case_0001").json()
        truth = client.get(f"/api/datasets/{dataset_id}/cases/case_0001/truth").json()
        assert "events" not in case_input
        assert "normal_background_mm" not in case_input
        assert "measurement_noise_mm" not in case_input
        assert "injected_deformation_mm" not in case_input
        assert "observation_artifact_mm" not in case_input
        assert len(truth["events"]) == 1
        assert truth["events"][0]["type"] == "trend"
        assert "measurement_noise_mm" in truth
        assert "anomaly_delta_mm" in truth
        assert "anomaly_delta_mm" not in case_input
        assert len(case_input["dates"]) == 365
        assert path.joinpath("input.json").is_file()
        assert path.joinpath("truth.json").is_file()
        assert client.get(f"/api/datasets/{dataset_id}/cases/../truth").status_code == 404
        assert client.post('/api/datasets', json={
            'seed': 123, 'count': 1, 'case_type': 'normal', 'preset': 'old'}).status_code == 422


def test_document_catalog_matches_current_files(tmp_path):
    root = Path(__file__).resolve().parents[1] / 'docs'
    with TestClient(create_app(tmp_path / 'data', tmp_path / 'no-web')) as client:
        documents = client.get('/api/docs').json()
        assert len({d['slug'] for d in documents}) == len(documents)
        assert {'data-generation', 'detection', 'development', 'project-status'} <= {
            d['slug'] for d in documents}
        for document in documents:
            response = client.get('/api/docs/' + document['slug'])
            assert response.status_code == 200 and document['group']
            assert response.text == (root / document['file']).read_text(encoding='utf-8')
        assert client.get('/api/docs/missing').status_code == 404


def test_manifest_read_retries_transient_windows_lock(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path / "generated")
    manifest = store.generate_sync(GenerationRequest(seed=42, count=1, case_type="normal"))
    original = Path.read_text
    attempts = 0

    def temporarily_locked(path, *args, **kwargs):
        nonlocal attempts
        if path.name == "manifest.json" and attempts == 0:
            attempts += 1
            raise PermissionError("temporary file lock")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", temporarily_locked)
    assert store.get_dataset(manifest.dataset_id) == manifest
    assert attempts == 1


def test_batches_reject_invalid_requests_and_refuse_directory_overwrite(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone
    from types import SimpleNamespace

    import pytest

    from gnss_sim import storage
    from gnss_sim.schemas import CASE_TYPES

    with TestClient(create_app(tmp_path, tmp_path / "no-web")) as client:
        for kind in ("acceleration", "all_scenarios", "multi_spike", "spike", "step", "slow_trend"):
            assert client.post("/api/datasets", json={
                "seed": 42, "count": 6, "case_type": kind}).status_code == 422
        assert client.post("/api/datasets", json={
            "seed": 42, "count": 5, "case_type": "all"}).status_code == 422
        invalid = tmp_path / "invalid-dataset"
        invalid.mkdir()
        (invalid / "manifest.json").write_text(json.dumps({"generator_version": "invalid"}))
        assert client.get("/api/datasets").json() == []
        assert client.get("/api/datasets/" + invalid.name).status_code == 404

    store = DatasetStore(tmp_path)
    request = GenerationRequest(seed=42, count=8, case_type="all")
    assigned = storage.allocate_case_types(request)
    assert assigned == storage.allocate_case_types(request)
    assert set(assigned) == set(CASE_TYPES) and len(assigned) == 8
    assert max(assigned.count(k) for k in CASE_TYPES) - min(assigned.count(k) for k in CASE_TYPES) == 0
    frozen_time = datetime(2026, 9, 28, tzinfo=timezone.utc)
    monkeypatch.setattr(storage, "datetime", SimpleNamespace(now=lambda tz: frozen_time))
    monkeypatch.setattr(storage, "uuid4", lambda: SimpleNamespace(hex="a" * 32))
    manifest = store.generate_sync(request)
    assert manifest.status == "complete" and manifest.generated_cases == 8
    manifest_path = tmp_path / manifest.dataset_id / "manifest.json"
    before = manifest_path.read_bytes()
    with pytest.raises(FileExistsError):
        store.generate_sync(request)
    assert manifest_path.read_bytes() == before
    store.executor.shutdown(wait=True)
