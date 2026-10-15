"""Preserve complete test inputs and responses and verify frozen evidence."""
from __future__ import annotations

import zipfile
from pathlib import Path

from report_exp25 import aggregate_rows

from gnss_research.files import read_json, sha256, write_json


def main():
    visual = Path("runs/exp25-test-1516-visual-v1")
    report = Path("results/exp25-test-1516-full-v1")
    verified_requests = 0
    for phase in ("pilot", "remaining"):
        if not (visual / f"{phase}-derived.json").exists():
            raise ValueError("all stations must finish")
        for stage in ("base", "recovery"):
            registration = read_json(visual / f"{phase}-{stage}-registration.json")
            for name, digest in registration["frozen_files"].items():
                if sha256(Path(name)) != digest:
                    raise ValueError(f"frozen experiment changed: {name}")
            for job in registration["jobs"]:
                work = visual / "predictions" / job["mode"] / job["station"]
                requests = job.get("requests", {"request.json": job.get("request_sha256")})
                for name, digest in requests.items():
                    if sha256(work / name) != digest:
                        raise ValueError(f"registered request changed: {work / name}")
                    verified_requests += 1
    summary = read_json(report / "summary.json")
    if summary["source"]["stations"] != 48:
        raise ValueError("full 48-station test required")
    rows = read_json(report / "per-station.json")
    station_sets = []
    for method, values in rows.items():
        station_sets.append({row["id"] for row in values})
        if len(values) != 48 or aggregate_rows(values) != summary["methods"][method]:
            raise ValueError(f"pooled metrics differ: {method}")
    if any(stations != station_sets[0] for stations in station_sets):
        raise ValueError("method station cohorts differ")
    if verified_requests != summary["request_total"]["requests"]:
        raise ValueError("request registration and receipt totals differ")
    roots = [visual, Path(summary["source"]["baseline"]), Path(summary["source"]["dataset"]),
             report, Path("results/exp25-historical-full-v3"),
             Path("results/exp25-test-1516-pelt-v1"), Path("results/exp25-historical-pelt-v1")]
    roots.extend(Path(folder) for folder in ("scripts", "docs", "configs", "tests", "src/gnss_research"))
    roots.extend(Path(folder) for folder in ("vendor/affiliation-metrics-py", "vendor/upstream/visualtime/src"))
    files = {path.resolve() for folder in roots for path in folder.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts}
    files.update(Path(name).resolve() for name in ("README.md", "pyproject.toml", "uv.lock", ".gitattributes"))
    files.update(path.resolve() for pattern in ("LICENSE*", "README*")
                 for path in Path("vendor/upstream/visualtime").glob(pattern) if path.is_file())
    root = Path.cwd().resolve()
    if not all(path.is_relative_to(root) and not path.name.startswith(".env") for path in files):
        raise ValueError("archive must contain workspace evidence without credentials")
    archive = Path("artifacts/backups/EXP25_ComprehensiveTest_20261015_4e5259c.zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(files):
            bundle.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("archive CRC verification failed")
    evidence = {"experiment_id": "GNSS-VLM-EXP25", "branch": "codex/exp25-comprehensive-evaluation",
        "parent_commit": "4e5259c", "backup_tag": "backup/pre-exp25-comprehensive-20261015-4e5259c",
        "archive": str(archive.resolve()), "archive_files": len(files), "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive), "verified_registered_requests": verified_requests,
        "verified_method_groups": len(rows), "same_stations": 48,
        "request_total": summary["request_total"], "previous_full_evidence": "docs/exp23-24-backup.json",
        "historical_full_evidence": "docs/exp17-19-evidence.json and prior linked archives",
        "note": "Full new test data, baselines, requests/images/responses and reports preserved. Historical predictions reference prior backups; no new historical requests."}
    write_json(Path("docs/exp25-backup.json"), evidence)
    print(evidence)


if __name__ == "__main__":
    main()
