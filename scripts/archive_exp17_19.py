"""Audit saved experiment evidence and preserve requests, data and plots locally."""
from __future__ import annotations

import zipfile
from pathlib import Path

from gnss_research.files import read_json, sha256, write_json
from gnss_research.metrics import aggregate

RUNS = {
    "exp17-factor-diagnostic-v1": "exp17-factor-diagnostic-v1",
    "exp17-development-1509-v1": "exp17-development-1509-visual-v1",
    "exp18-factor-diagnostic-v1": "exp18-factor-diagnostic-v1",
    "exp18-development-1512-v1": "exp18-development-1512-visual-v1",
    "exp19-thinking-diagnostic-v1": "exp19-thinking-diagnostic-v1",
}


def main():
    roots = set()
    ledger = []
    verified_methods = 0
    for result_name, run_name in RUNS.items():
        run = Path("runs") / run_name
        result = Path("results") / result_name
        frozen = read_json(run / "registration.json")
        for phase in ("pilot", "remaining"):
            if not (run / f"{phase}-completed.json").exists():
                raise ValueError(f"incomplete experiment: {run}")
        for path, digest in frozen.get("experiment_files", {}).items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"frozen source changed: {path}")
        if "source_sha256" in frozen:
            if sha256(Path("scripts/experiment_exp19.py")) != frozen["source_sha256"]:
                raise ValueError("native-thinking source changed")
        roots.update((run, Path(frozen["baseline_run"]), Path(frozen["dataset"]), result))
        if "parent" in frozen:
            roots.add(Path(frozen["parent"]))
        summary = read_json(result / "summary.json")
        rows = read_json(result / "per-station.json")
        for method, values in rows.items():
            if aggregate(values) != summary["methods"][method]:
                raise ValueError(f"pooled metrics differ: {result_name}/{method}")
            verified_methods += 1
        for job in frozen["pending"]:
            work = run / "predictions" / job["mode"] / job["station"]
            if sha256(work / "request.json") != job["request_sha256"]:
                raise ValueError("root request changed")
            request_paths = [work / "request.json"]
            if (work / "decision/request.json").exists():
                request_paths.append(work / "decision/request.json")
            for path in request_paths:
                receipt_path = path.with_name("result.json")
                if path.parent == work and (work / "readout-result.json").exists():
                    receipt_path = work / "readout-result.json"
                receipt = read_json(receipt_path)
                response = path.with_name("response.json")
                ledger.append({"experiment": frozen["experiment_id"], "run": run_name,
                    "mode": job["mode"], "station": job["station"],
                    "stage": "decision" if path.parent.name == "decision" else "initial",
                    "status": receipt["status"], "requested_model": receipt["requested_model"],
                    "served_model": receipt.get("served_model"), "usage": receipt.get("usage"),
                    "retry_count": receipt["retry_count"], "request_sha256": sha256(path),
                    "response_sha256": sha256(response) if response.exists() else None})
        for copied in frozen.get("original_requests", []):
            original = read_json(Path(copied["original_request_path"]))
            modified = read_json(run / "predictions" / copied["mode"] / copied["station"] / "request.json")
            if sha256(Path(copied["original_request_path"])) != copied["original_request_sha256"]:
                raise ValueError("nonthinking source request changed")
            original.pop("enable_thinking", None)
            modified.pop("enable_thinking", None)
            modified.pop("thinking_budget", None)
            if modified != original:
                raise ValueError("native mode changed more than its two registered parameters")
    root = Path.cwd().resolve()
    files = set()
    for directory in roots:
        files.update(path.resolve() for path in directory.rglob("*") if path.is_file())
    for folder in ("scripts", "docs", "configs", "src/gnss_research", "tests", "papers/exp17-visual-resolution"):
        files.update(path.resolve() for path in Path(folder).rglob("*") if path.is_file())
    files = {path for path in files if "__pycache__" not in path.parts}
    if not all(path.is_relative_to(root) and not path.name.startswith(".env") for path in files):
        raise ValueError("archive files must be workspace artifacts without credentials")
    archive = Path("artifacts/backups/EXP17-19_VisualInputStudy_20261015_6fc9f44.zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(files):
            bundle.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("archive CRC verification failed")
    evidence = {"branch": "codex/exp17-visual-resolution-workflow", "parent_commit": "6fc9f44",
        "backup_tag": "backup/pre-exp17-visual-evidence-20261015-6fc9f44",
        "prechange_tracked_archive": "artifacts/backups/EXP17_pre_6fc9f44/tracked.zip",
        "previous_full_evidence": "docs/exp10-16-evidence.json",
        "archive": str(archive.resolve()), "archive_files": len(files), "archive_sha256": sha256(archive),
        "requests": len(ledger), "failed_requests": sum(row["status"] != "success" for row in ledger),
        "retry_count": sum(row["retry_count"] for row in ledger), "ledger": ledger,
        "reaggregated_method_groups": verified_methods,
        "datasets": {str(path): sha256(path / "manifest.json") for path in roots
                     if (path / "manifest.json").exists()},
        "note": "Only new pending jobs are counted; copied controls are reused. "
                "EXP17 staged readout and decision are two actual requests. "
                "All data are development/diagnostic, no new formal test claims."}
    write_json(Path("docs/exp17-19-evidence.json"), evidence)
    print({key: evidence[key] for key in ("requests", "failed_requests", "retry_count",
                                        "reaggregated_method_groups", "archive_files", "archive_sha256")})


if __name__ == "__main__":
    main()
