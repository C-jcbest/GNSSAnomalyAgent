"""Preserve all large-cohort attempts, provider responses and frozen-source evidence."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from experiment_exp26 import check_frozen
from report_exp25 import aggregate_cost, request_receipts

from gnss_research.files import read_json, sha256, write_json

TEST_VISUAL = Path("D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp28-test-visual-v2")
TEST_ORIGINAL = Path("D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp28-test-visual-v1")
TEST_SUPPLEMENT = Path("D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/exp28-test-supplement-v1")


def archive_name(path):
    for folder, name in ((TEST_VISUAL, "exp28-test-visual-v2"), (TEST_ORIGINAL, "exp28-test-visual-v1"),
                         (TEST_SUPPLEMENT, "exp28-test-supplement-v1")):
        if path.is_absolute() and path.is_relative_to(folder):
            return Path("runs") / name / path.relative_to(folder)
    return path


def main():
    evidence = Path("results/exp28-evidence-v1")
    evidence.mkdir(exist_ok=False)
    costs = {}
    for split in ("validation", "test"):
        visual = Path("runs/exp28-validation-visual-v1") if split == "validation" else TEST_VISUAL
        report = Path(f"results/exp28-{split}-comprehensive-v1")
        for phase in ("pilot", "remaining"):
            registration = read_json(visual / f"{phase}-registration.json")
            check_frozen(registration)
            for job in registration["jobs"]:
                work = visual / "predictions" / job["mode"] / job["station"]
                if sha256(work / "request.json") != job["request_sha256"]:
                    raise ValueError("request bytes changed")
                target = evidence / "responses" / split / job["mode"] / job["station"]
                target.mkdir(parents=True)
                for name in ("response.json", "result.json"):
                    if (work / name).exists():
                        shutil.copyfile(work / name, target / name)
                if job["mode"] in ("ce_flash_image", "ce_vl_image"):
                    first = read_json(work / "request.json")
                    second = read_json(visual / "predictions" / job["mode"].replace("_image", "_noimage") / job["station"] / "request.json")
                    for message in first["messages"]:
                        if isinstance(message["content"], list):
                            message["content"] = [item for item in message["content"] if item["type"] != "image_url"]
                    if first != second:
                        raise ValueError("noimage ablation changed non-image inputs")
            shutil.copyfile(visual / f"{phase}-registration.json", evidence / f"{split}-{phase}-registration.json")
        supplement = Path("runs/exp28-validation-supplement-v1") if split == "validation" else TEST_SUPPLEMENT
        registration = read_json(supplement / "full-registration.json")
        check_frozen(registration)
        for job in registration["jobs"]:
            work = supplement / "predictions" / job["mode"] / job["station"]
            if sha256(work / "request.json") != job["request_sha256"]:
                raise ValueError("supplemental request bytes changed")
            target = evidence / "responses" / split / job["mode"] / job["station"]
            target.mkdir(parents=True)
            for name in ("response.json", "result.json"):
                if (work / name).exists():
                    shutil.copyfile(work / name, target / name)
            if job["mode"] == "cf_flash_noimage":
                original = read_json(visual / "predictions/cf_flash_image" / job["station"] / "request.json")
                for message in original["messages"]:
                    if isinstance(message["content"], list):
                        message["content"] = [item for item in message["content"] if item["type"] != "image_url"]
                if original != read_json(work / "request.json"):
                    raise ValueError("CFV noimage ablation changed non-image inputs")
        shutil.copyfile(supplement / "full-registration.json", evidence / f"{split}-supplement-registration.json")
        receipts = read_json(report / "request-receipts.json")
        costs[split] = aggregate_cost(receipts)
    write_json(evidence / "costs.json", costs)
    original_jobs = read_json(TEST_ORIGINAL / "pilot-registration.json")["jobs"]
    original_receipts = request_receipts([str(TEST_ORIGINAL / "predictions" / job["mode"] / job["station"])
                                         for job in original_jobs])
    probes = request_receipts(["runs/exp28-transport-check-v1"])
    write_json(evidence / "transport-attempts.json", {"original_failed_pilot": original_receipts, "health_probes": probes,
        "costs": {"original_failed_pilot": aggregate_cost(original_receipts), "health_probes": aggregate_cost(probes)}})
    roots = [Path(name) for name in ("src", "scripts", "tests", "configs", "docs", "vendor/affiliation-metrics-py",
        "papers/exp28-competing-review", "results/exp28-validation-pilot-v1", "results/exp28-validation-full-v1",
        "results/exp28-test-pilot-v1", "results/exp28-test-full-v1", "results/exp28-evidence-v1",
        "results/exp28-validation-comprehensive-v1", "results/exp28-test-comprehensive-v1",
        "data/exp28-expanded-1528-1530", "runs/exp28-shared-training-v1", "runs/exp28-validation-baseline-v1",
        "runs/exp28-test-baseline-v1", "runs/exp28-validation-visual-v1", "runs/exp28-validation-supplement-v1")]
    roots.append(TEST_VISUAL)
    roots.append(TEST_ORIGINAL)
    roots.append(TEST_SUPPLEMENT)
    roots.append(Path("runs/exp28-transport-check-v1"))
    files = {p for folder in roots for p in folder.rglob("*") if p.is_file()
             and not any(part in ("__pycache__", ".git", ".pytest_cache", ".ruff_cache") for part in p.parts)
             and not p.name.startswith(".env")}
    files.update(Path(name) for name in ("README.md", "AGENTS.md", "pyproject.toml", "uv.lock", ".gitignore", ".gitattributes"))
    files = sorted(p for p in files if p.exists())
    write_json(evidence / "archive-files.json", {str(archive_name(p)): {
        "original_path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files})
    files.append(evidence / "archive-files.json")
    archive = Path("D:/CodeWorkSpace/AgentWorkspace/GNSSAnomalyAgent-sim/artifacts/backups/EXP28_CEV_720Stations_20261015.zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
        for number, path in enumerate(files, 1):
            stream.write(path, archive_name(path))
            if number % 2000 == 0:
                print("archived", number, "/", len(files), flush=True)
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError("archive CRC failure")
    receipt = {"experiment_id": "GNSS-VLM-EXP28", "archive": str(archive), "files": len(files),
               "bytes": archive.stat().st_size, "sha256": sha256(archive), "costs": costs,
               "extra_transport_attempt_costs": {"original_failed_pilot": aggregate_cost(original_receipts),
                                                  "health_probes": aggregate_cost(probes)},
               "contains_credentials": False,
               "prior_backup": {key: read_json(Path("docs/exp27-backup.json"))[key] for key in ("archive", "sha256")}}
    write_json(Path("docs/exp28-backup.json"), receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
