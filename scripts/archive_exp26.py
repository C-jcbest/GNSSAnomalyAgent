"""Verify saved EXP26 outputs and archive reproducible, credential-free evidence."""
from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

from experiment_exp26 import check_frozen
from report_exp25 import aggregate_cost, aggregate_rows

from gnss_research.files import read_json, sha256, write_json


def verify(visual, report):
    summary = read_json(report / "summary.json")
    rows = read_json(report / "per-station.json")
    for method, values in rows.items():
        assert aggregate_rows(values) == summary["methods"][method], method
    requests = []
    for phase in ("pilot", "remaining"):
        registration = read_json(visual / (phase + "-registration.json"))
        check_frozen(registration)
        for job in registration["jobs"]:
            work = visual / "predictions" / job["mode"] / job["station"]
            assert sha256(work / "request.json") == job["request_sha256"]
            response = read_json(work / "result.json")
            requests.append(response)
            if job["mode"].endswith("_image"):
                image = read_json(work / "request.json")
                noimage = read_json(visual / "predictions" / job["mode"].replace("_image", "_noimage")
                                    / job["station"] / "request.json")
                stripped = copy.deepcopy(image)
                for message in stripped["messages"]:
                    if isinstance(message["content"], list):
                        message["content"] = [item for item in message["content"] if item["type"] != "image_url"]
                assert stripped == noimage, "image ablation differs beyond image removal"
    receipts = read_json(report / "request-receipts.json")
    assert len(receipts) == len(requests)
    return aggregate_cost(receipts)


def main():
    costs = {}
    for cohort in ("development", "test"):
        costs[cohort] = verify(Path(f"runs/exp26-{cohort}-1517-visual-v1"),
                               Path(f"results/exp26-{cohort}-full-v1"))
    folders = [Path("data/exp26-development-1517"), Path("papers/exp26-counterfactual")]
    folders += [Path(f"runs/exp26-{cohort}-1517-{kind}-v1")
                for cohort in ("development", "test") for kind in ("baseline", "visual")]
    folders += sorted(Path("results").glob("exp26-*"))
    files = {file for folder in folders for file in folder.rglob("*") if file.is_file()}
    for folder, patterns in (("docs", ("exp26-*.md", "experiment-exp26*.md")),
                             ("scripts", ("*exp26*.py",)), ("src/gnss_research", ("*.py",)),
                             ("tests", ("test_exp*.py",))):
        for pattern in patterns:
            files.update(Path(folder).glob(pattern))
    files.update(Path("scripts").glob("*.py"))
    files.update(file for file in Path("vendor/affiliation-metrics-py").rglob("*")
                 if file.is_file() and ".git" not in file.parts and "__pycache__" not in file.parts)
    files.update(Path(name) for name in ("pyproject.toml", "uv.lock", "AGENTS.md", "README.md",
                                        "docs/project-status.md", "configs/upstream.json"))
    assert not any(file.name.startswith(".env") for file in files)
    target = Path("artifacts/backups/EXP26_CFV_DevelopmentAndTest_20261015.zip")
    manifest = {str(file).replace("\\", "/"): sha256(file) for file in sorted(files)}
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(files):
            archive.write(file, str(file).replace("\\", "/"))
        archive.writestr("EXP26-SHA256.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    receipt = {"experiment_id": "GNSS-VLM-EXP26", "archive": str(target), "bytes": target.stat().st_size,
               "files": len(files), "sha256": sha256(target), "costs": costs,
               "verified": ["frozen sources", "registered request SHA", "image-only deletion ablation",
                            "per-station metric reaggregation", "ZIP CRC"], "contains_credentials": False,
               "prior_backup": {"archive": "artifacts/backups/EXP25_ComprehensiveTest_20261015_4e5259c.zip",
                                "sha256": "88c4760e438613fdfc2d53d0e6c10a08970b60ac2135eefd650d380353c04e68"}}
    write_json(Path("docs/exp26-backup.json"), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
