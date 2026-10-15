"""Preserve paid responses and a local complete experiment archive without credentials."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from gnss_research.files import read_json, sha256, write_json

RUNS = {
    "exp10-development-v1": "exp10-development-visual-v1",
    "exp11-development-v1": "exp11-development-visual-v1",
    "exp12-development-v1": "exp12-development-qwen-vl-v1",
    "exp13-development-v1": "exp13-development-qwen-vl-v1",
    "exp14-development-v1": "exp14-development-visual-v1",
    "exp15-development-v1": "exp15-development-visual-v1",
    "exp15-test-1503-v1": "exp15-test-1503-visual-v1",
    "exp15-test-1504-v1": "exp15-test-1504-visual-v1",
    "exp16-development-v1": "exp16-development-visual-v1",
    "exp16-development-1508-v1": "exp16-development-1508-visual-v1",
}


def main():
    roots = set()
    ledger = []
    for result_name, run_name in RUNS.items():
        run = Path("runs") / run_name
        result = Path("results") / result_name
        frozen = read_json(run / "registration.json")
        assert (run / "pilot-completed.json").exists() and (run / "remaining-completed.json").exists()
        roots.update((run, Path(frozen["baseline_run"]), Path(frozen["dataset"]), result))
        for job in frozen["pending"]:
            work = run / "predictions" / job["mode"] / job["station"]
            receipt = read_json(work / "result.json")
            target = result / "responses" / job["mode"] / job["station"]
            target.mkdir(parents=True, exist_ok=False)
            for name in ("result.json", "response.json", "input-receipt.json", "candidates.json"):
                if (work / name).exists():
                    shutil.copyfile(work / name, target / name)
            ledger.append({"result": result_name, "mode": job["mode"], "station": job["station"],
                "request_sha256": sha256(work / "request.json"), "status": receipt["status"],
                "requested_model": receipt["requested_model"], "served_model": receipt.get("served_model"),
                "tokens": (receipt.get("usage") or {}).get("total_tokens", 0),
                "response_sha256": sha256(work / "response.json") if (work / "response.json").exists() else None})
    for grounding_name in ("visual-grounding-20261015-v1", "visual-grounding-pair-20261015-v1"):
        grounding = Path("runs") / grounding_name
        roots.add(grounding)
        for path in grounding.rglob("*"):
            if path.is_file() and path.name != "request.json":
                target = Path("results/exp10-16-diagnostics") / grounding_name / path.relative_to(grounding)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        for path in grounding.glob("*/*/result.json"):
            receipt = read_json(path)
            ledger.append({"result": grounding_name, "mode": path.parent.parent.name,
                           "station": path.parent.name, "status": receipt["status"],
                           "requested_model": receipt["requested_model"], "served_model": receipt.get("served_model"),
                           "request_sha256": sha256(path.with_name("request.json")),
                           "tokens": (receipt.get("usage") or {}).get("total_tokens", 0)})
    files = set()
    root = Path.cwd().resolve()
    for directory in roots:
        files.update(p.resolve() for p in directory.rglob("*") if p.is_file())
    for folder in ("scripts", "configs", "docs", "papers/exp10-visual-evidence", "src/gnss_research", "tests"):
        for path in Path(folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                files.add(path.resolve())
    files = {p for p in files if "__pycache__" not in p.parts}
    assert all(p.is_relative_to(root) and not p.name.startswith(".env") for p in files)
    archive = Path("artifacts/backups/EXP10-16_VisualEvidence_20261015_4495ff0.zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(files):
            bundle.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
    write_json(Path("docs/exp10-16-evidence.json"), {
        "branch": "codex/exp10-visual-evidence", "stable_commit": "4495ff0",
        "archive": str(archive.resolve()), "archive_sha256": sha256(archive),
        "archive_files": len(files), "requests": len(ledger),
        "failed_requests": sum(row["status"] != "success" for row in ledger),
        "retry_count": 0, "ledger": ledger,
        "datasets": {str(p): sha256(p / "manifest.json") for p in roots if (p / "manifest.json").exists()},
        "note": "Copied original controls in EXP14/15 are reused, counted only in their parent run. "
                "Simple image checks are input diagnostics, not GNSS accuracy samples. "
                "Full archive remains local; raw provider responses and numerical predictions are versioned."})
    print({"requests": len(ledger), "archive_files": len(files), "archive_sha256": sha256(archive)})


if __name__ == "__main__":
    main()
