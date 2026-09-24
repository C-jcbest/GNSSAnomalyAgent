import time

from fastapi.testclient import TestClient

from gnss_sim.api import create_app
from gnss_sim.schemas import GenerationRequest
from gnss_sim.storage import DatasetStore


def test_generation_progress_persistence_and_truth_isolation(tmp_path):
    app = create_app(tmp_path / "generated", tmp_path / "no-built-web")
    with TestClient(app) as client:
        response = client.post("/api/datasets", json={"seed": 123, "count": 2, "case_type": "slow_trend"})
        assert response.status_code == 202
        dataset_id = response.json()["dataset_id"]
        for _ in range(100):
            manifest = client.get(f"/api/datasets/{dataset_id}").json()
            if manifest["status"] in ("complete", "failed"):
                break
            time.sleep(0.02)
        assert manifest["status"] == "complete"
        assert manifest["generator_version"] == "event-v3"
        assert manifest["generated_cases"] == 2
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
        assert truth["events"][0]["type"] == "slow_trend"
        assert "normal_background_mm" in truth
        assert "measurement_noise_mm" in truth
        assert "injected_deformation_mm" in truth
        assert "observation_artifact_mm" in truth
        assert "true_coordinate_mm" not in truth
        assert "ar_noise_mm" not in truth
        assert "active_motion" not in truth
        assert len(case_input["dates"]) == 365
        assert path.joinpath("input.json").is_file()
        assert path.joinpath("truth.json").is_file()
        assert client.get(f"/api/datasets/{dataset_id}/cases/../truth").status_code == 404


def test_same_request_reproduces_persisted_cases(tmp_path):
    store = DatasetStore(tmp_path / "generated")
    request = GenerationRequest(seed=20260923, count=2, case_type="normal")
    first = store.generate_sync(request)
    second = store.generate_sync(request)
    assert first.dataset_id != second.dataset_id
    assert first.generator_version == second.generator_version == "event-v3"
    for case_id in ("case_0001", "case_0002"):
        assert store.get_case_input(first.dataset_id, case_id) == store.get_case_input(
            second.dataset_id, case_id
        )
        assert store.get_case_truth(first.dataset_id, case_id) == store.get_case_truth(
            second.dataset_id, case_id
        )


def test_old_api_payload_and_dataset_are_not_supported(tmp_path):
    store = DatasetStore(tmp_path / "generated")
    legacy = store.root / "sim-20250101-000000-12345678"
    legacy.mkdir()
    legacy.joinpath("manifest.json").write_text('{"schema_version":"p1-dataset-v1"}')
    previous = store.root / "event-v1-20250101-000000-12345678"
    previous.mkdir()
    previous.joinpath("manifest.json").write_text('{"schema_version":"event-dataset-v1"}')
    former = store.root / "event-v2-20250101-000000-12345678"
    former.mkdir()
    former.joinpath("manifest.json").write_text('{"schema_version":"event-dataset-v2"}')
    with TestClient(create_app(store.root, tmp_path / "no-built-web")) as client:
        assert client.post("/api/datasets", json={"seed": 42, "count": 1}).status_code == 422
        assert client.post("/api/datasets", json={"seed": 42, "count": 1, "case_type": "normal", "preset": "normal-p1"}).status_code == 422
        assert client.get("/api/datasets").json() == []
        assert client.get(f"/api/datasets/{legacy.name}").status_code == 404
        assert client.get(f"/api/datasets/{previous.name}").status_code == 404
        assert client.get(f"/api/datasets/{former.name}").status_code == 404


def test_learning_documents_are_served_from_canonical_markdown(tmp_path):
    client = TestClient(create_app(tmp_path / "generated", tmp_path / "no-built-web"))
    index = client.get("/api/docs")
    assert index.status_code == 200
    assert [item["slug"] for item in index.json()] == [
        "research-design", "p1-normal-model", "data-and-reproducibility", "planned-methods"
    ]
    for item in index.json():
        document = client.get(f"/api/docs/{item['slug']}")
        assert document.status_code == 200
        assert document.text.startswith("# ")
    formula = client.get("/api/docs/p1-normal-model")
    assert formula.status_code == 200
    assert "text/plain" in formula.headers["content-type"]
    assert "B_{t,c}" in formula.text
    assert "\\sqrt{" in formula.text
    assert client.get("/api/docs/unknown").status_code == 404
