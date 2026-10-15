"""Archive source, synthetic data and all EXP27 attempts without configuration secrets."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from experiment_exp26 import check_frozen
from report_exp25 import aggregate_cost

from gnss_research.files import read_json, sha256, write_json


def main():
    root = Path.cwd()
    visual = Path("runs/exp27-development-1518-visual-v2")
    report = Path("results/exp27-development-pilot-v1")
    registration = read_json(visual / "pilot-registration.json")
    check_frozen(registration)
    if not (visual / "pilot-completed.json").exists():
        raise ValueError("inference incomplete")
    receipts = read_json(report / "request-receipts.json")
    evidence = Path("results/exp27-evidence-v1")
    evidence.mkdir(exist_ok=False)
    for job in registration["jobs"]:
        work = visual / "predictions" / job["mode"] / job["station"]
        if sha256(work / "request.json") != job["request_sha256"]:
            raise ValueError("registered request changed")
        target = evidence / "responses" / job["mode"] / job["station"]
        target.mkdir(parents=True)
        for name in ("response.json", "result.json"):
            if (work / name).exists():
                shutil.copyfile(work / name, target / name)
    costs = aggregate_cost(receipts)
    write_json(evidence / "costs.json", costs)
    write_json(evidence / "registration.json", registration)
    write_json(evidence / "decision.json", {
        "experiment_id": "GNSS-VLM-EXP27", "decision": "cancel integration; no gain over frozen CFV",
        "expansion": False, "test_run": False, "new_requests": len(registration["jobs"]),
        "completed_validation_stations": 24, "triggered_stations": 2,
        "parent_commit": "6736b41", "stable_backup_tag": "backup/exp26-cfv-before-exp27-20261015"})
    sources = [Path(name) for name in ("src", "scripts", "tests", "configs", "docs", "vendor/affiliation-metrics-py",
        "papers/exp27-trend-cfv", "results/exp27-development-pilot-v1", "results/exp27-evidence-v1",
        "data/exp27-development-1518", "runs/exp27-development-1518-baseline-v1",
        "runs/exp27-development-1518-visual-v1", "runs/exp27-development-1518-visual-v2")]
    files = {path for folder in sources for path in folder.rglob("*") if path.is_file()
             and not any(part in ("__pycache__", ".git", ".pytest_cache", ".ruff_cache") for part in path.parts)
             and not path.name.startswith(".env")}
    files.update(Path(name) for name in ("README.md", "AGENTS.md", "pyproject.toml", "uv.lock", ".gitignore"))
    files = sorted(path for path in files if path.exists())
    manifest = {str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files}
    write_json(evidence / "archive-files.json", manifest)
    files.append(evidence / "archive-files.json")
    archive = Path("artifacts/backups/EXP27_ECV_TCV_DevelopmentPilot_NotAdopted_20261015.zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for path in files:
            stream.write(path, path.relative_to(root) if path.is_absolute() else path)
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError("archive CRC failure")
    receipt = {"experiment_id": "GNSS-VLM-EXP27", "archive": str(archive),
               "bytes": archive.stat().st_size, "files": len(files), "sha256": sha256(archive),
               "costs": costs, "contains_credentials": False,
               "prior_backup": read_json(Path("docs/exp26-backup.json")),
               "decision": "No improvement over CFV; stop expansion and test; keep negative results"}
    write_json(Path("docs/exp27-backup.json"), receipt)
    print(receipt)


if __name__ == "__main__":
    main()
