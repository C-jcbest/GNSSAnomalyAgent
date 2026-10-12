import shutil

import pytest

from gnss_research.data import generate
from gnss_research.experiment import evaluate
from gnss_research.files import read_json, sha256, write_json


@pytest.mark.parametrize("method", ["isolation_forest", "visualtime_gnss"])
def test_failed_generic_method_has_no_deformation_score_and_dataset_is_portable(tmp_path, method):
    original = tmp_path / "original"
    generate(original, 7, 6, 365)
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    run = tmp_path / "run"
    manifest = read_json(original / "manifest.json")
    records = [row for row in manifest["records"] if row["split"] == "test"]
    write_json(run / "registration.json", {
        "dataset": str(tmp_path / "unavailable-original-location"),
        "manifest_sha256": sha256(original / "manifest.json"),
        "split": "test", "seed": 7, "methods": [method]})
    for record in records:
        write_json(run / "predictions" / method / record["station_id"] / "result.json",
                   {"status": "failed"})
    evaluate(run, run / "replay", relocated)
    summary = read_json(run / "replay/summary.json")["methods"][method]
    assert "types" not in summary
    assert summary["failed_records"] == 6
    assert summary["generic_daily_micro"]["tp"] == 0
    assert summary["generic_daily_micro"]["fn"] > 0
    (relocated / records[0]["observation"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset changed"):
        evaluate(run, run / "tampered", relocated)
