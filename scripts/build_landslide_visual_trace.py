"""Build a local visual evidence browser without calling a model or changing predictions."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from gnss_sim.landslide_trace import build_trace, include_local_trace

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    comparison = ROOT / "artifacts/landslide-comparison-2026-10-07"
    parser.add_argument("--numeric", type=Path, default=comparison / "numeric-v1")
    parser.add_argument("--visual", type=Path, default=comparison / "visual-v3")
    parser.add_argument("--reference", type=Path,
                        default=ROOT / "artifacts/landslide-stage-diagnosis-2026-10-07/reference-v1")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts/landslide-visual-trace-2026-10-07")
    parser.add_argument("--local", type=Path, help="Append a completed frozen local-view experiment")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = build_trace(args.numeric, args.visual, args.reference, args.out)
    if args.local is not None:
        report = include_local_trace(report, args.local, args.out)
    (args.out / "trace-data.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    for path in (ROOT / "scripts/visual_trace").iterdir():
        if path.is_file():
            shutil.copyfile(path, args.out / path.name)
    shutil.copyfile(ROOT / "docs/landslide-stage-diagnosis.md", args.out / "diagnosis-notes.md")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(args.out / "index.html")


if __name__ == "__main__":
    main()
