import io
import json
import threading
import zipfile
from http.server import ThreadingHTTPServer

import httpx
import pytest

from gnss_anomaly.contracts import Prediction
from gnss_anomaly.dashboard import handler_for
from gnss_anomaly.datasets import build_dataset, demo_backgrounds
from gnss_anomaly.evaluation import prf
from gnss_anomaly.experiments import run_experiment
from gnss_anomaly.figures import figure_bytes, paper_bundle
from gnss_anomaly.reporting import History
from gnss_anomaly.storage import read_json, write_json


@pytest.fixture
def history(tmp_path, config):
    dataset = tmp_path / "data/processed/demo"
    demo_backgrounds(tmp_path / "backgrounds")
    build_dataset(tmp_path / "backgrounds", dataset, 15)
    run_experiment(dataset, tmp_path / "runs/example", config, "development", ["hampel"], limit=4)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/metrics-criteria.md").write_text("# 指标标准", encoding="utf-8")
    return History(tmp_path)


def test_history_is_read_only_recovers_old_dataset_and_refuses_tampering(history):
    path = history.root / "runs/example/run.json"
    meta = read_json(path)
    meta.pop("dataset_path")  # Older runs still resolve by manifest hash.
    write_json(path, meta)
    paths = list(history.root.rglob("*.json"))
    before = {str(p): p.read_bytes() for p in paths}
    key = history.index()["runs"][0]["key"]
    detail = history.detail(key)
    assert detail["summary"]["all_records_present"]
    assert detail["summary"]["metric_version"] == "observed-cells-v1.1"
    result_key = detail["cases"][0]["result_key"]
    window, truth, prediction, _ = history.case(key, result_key)
    assert window.case_id == detail["cases"][0]["case_id"]
    assert len(truth.mask(len(window.values))) == 336
    assert before == {str(p): p.read_bytes() for p in paths}
    dataset, manifest = history.dataset(meta)
    (dataset / manifest["records"][0]["file"]).write_text("{}", encoding="utf-8")
    # Find a changed file actually used by this run and verify refusal.
    used = next(r for r in manifest["records"] if r["case_id"] == window.case_id)
    (dataset / used["file"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="哈希"):
        history.case(key, result_key)
    assert history.detail(key)["summary"]["records_present"] == 4


def test_partial_and_broken_runs_not_presented_as_complete(history):
    run = history.root / "runs/example"
    next((run / "results").glob("*.json")).unlink()
    index = history.index()
    assert not index["runs"][0]["complete"]
    assert any("不完整" in n for n in history.detail(index["runs"][0]["key"])["notices"])
    write_json(history.root / "runs/broken/run.json", {})
    assert len(history.index()["errors"]) == 1


def test_run_index_exposes_sampling_interval_for_daily_archive_filter(history):
    path = history.root / "runs/example/run.json"
    meta = read_json(path)
    assert history.index()["runs"][0]["sampling_hours"] == meta["sampling_hours"]
    meta["sampling_hours"] = 24
    write_json(path, meta)
    assert history.index()["runs"][0]["sampling_hours"] == 24
    meta.pop("sampling_hours")
    write_json(path, meta)
    assert history.index()["runs"][0]["sampling_hours"] == 1


def test_paper_exports_include_provenance_and_vector_formats(history):
    key = history.index()["runs"][0]["key"]
    detail = history.detail(key)
    png = figure_bytes(detail, "overview", "png")
    assert png.startswith(b"\x89PNG")
    assert figure_bytes(detail, "overview", "pdf").startswith(b"%PDF")
    svg = figure_bytes(detail, "heatmap", "svg")
    assert b"ENGINEERING DEMO" in svg and b"SMOKE RUN" in svg
    window, truth, prediction, _ = history.case(key, detail["cases"][0]["result_key"])
    assert b"Prediction (top band)" in figure_bytes(
        detail, "case", "svg", case=(window, truth, prediction), bounds=(10, 35)
    )
    label_chart = figure_bytes(detail, "labels", "svg", case=(window, truth, None))
    assert b"Injected target" in label_chart
    assert b"Prediction (top band)" not in label_chart
    normal_chart = figure_bytes(detail, "labels", "svg", case=(window, Prediction(), None))
    assert b"Injected target" not in normal_chart
    with pytest.raises(ValueError, match="不叠加检测预测"):
        figure_bytes(detail, "labels", case=(window, truth, prediction))
    with pytest.raises(ValueError):
        figure_bytes(detail, "case", case=(window, truth, prediction), bounds=(20, 10))
    with zipfile.ZipFile(io.BytesIO(paper_bundle(detail))) as bundle:
        assert "overview.pdf" in bundle.namelist() and "metrics.csv" in bundle.namelist()
        assert (
            json.loads(bundle.read("provenance.json"))["meta"]["run_id"] == detail["meta"]["run_id"]
        )


def test_http_routes_protect_local_files_and_serve_history(history):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(history))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{server.server_port}", trust_env=False
        ) as client:
            page = client.get("/")
            assert page.status_code == 200
            assert "每日 15:00" in page.text
            assert 'id="include-hourly"' in page.text
            assert client.get("/app.js").headers["Content-Type"].startswith("text/javascript")
            assert client.get("/.env").status_code == 404
            assert client.get("/api/run", params={"run": "../../.env"}).status_code == 400
            assert client.get("/api/runs", headers={"Host": "attacker.test"}).status_code == 403
            assert (
                client.get("/api/runs", headers={"Origin": "https://attacker.test"}).status_code
                == 403
            )
            assert client.post("/api/run").status_code == 501
            index = client.get("/api/runs").json()
            assert "injection_label_gallery_v1" in index["capabilities"]
            runs = index["runs"]
            detail = client.get("/api/run", params={"run": runs[0]["key"]}).json()
            result_key = detail["cases"][0]["result_key"]
            assert (
                client.get(
                    "/api/case", params={"run": runs[0]["key"], "result": result_key}
                ).status_code
                == 200
            )
            label = client.get("/api/label", params={"run": runs[0]["key"], "result": result_key})
            assert label.status_code == 200
            assert "truth" in label.json() and "prediction" not in label.json()
            chart = client.get(
                "/chart",
                params={"run": runs[0]["key"], "result": result_key, "chart": "labels"},
            )
            assert chart.status_code == 200 and b"Injected target" in chart.content
            export = client.get(
                "/export", params={"run": runs[0]["key"], "chart": "overview", "format": "pdf"}
            )
            assert (
                export.status_code == 200 and "attachment" in export.headers["Content-Disposition"]
            )
            assert client.get("/api/criteria").json()["text"] == "# 指标标准"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_normal_only_f1_is_not_a_quality_grade():
    assert prf(0, 5, 0)["f1"] is None
    assert prf(0, 0, 5)["f1"] == 0
    assert prf(2, 1, 2)["f1"] == pytest.approx(4 / 7)


def test_daily_trend_view_is_not_a_labelled_prediction(window):
    values = window(length=24 * 130)
    output = figure_bytes(None, "snapshot", "svg", case=(values, None, None), trend=True)
    assert b"daily median" in output and b"no imputation" in output
    assert b"Unlabelled platform observations" in output
    assert b"Truth (bottom band)" not in output
