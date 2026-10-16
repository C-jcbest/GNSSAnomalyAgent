"""Check live source bindings using either saved Windows or POSIX separators.

The frozen v2 runner verifies snapshot bytes but uses a POSIX prefix when
checking executing source paths. This separate guard closes that verification
gap without modifying any frozen inference dependency or outcome.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from run_landslide_comparison import ROOT, save_json, sha
from run_landslide_evidence_display_confirmation import verify
from run_landslide_window_optimization import read_json


def verify_live_sources(run, root=ROOT):
    sources = {}
    for name, expected in read_json(run / "pre-generation-freeze.json").items():
        normalized = name.replace("\\", "/")
        if not normalized.startswith("source/"):
            continue
        relative = normalized.removeprefix("source/")
        if sha(root / relative) != expected or sha(run / normalized) != expected:
            raise ValueError(f"Executing source differs from pre-generation snapshot: {relative}")
        sources[relative] = expected
    if not sources:
        raise ValueError("No executing source bindings were checked")
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run, generated=(args.run / "data-freeze.json").exists())
    bindings = verify_live_sources(args.run)
    if args.receipt.exists():
        raise ValueError("Do not overwrite an existing source-check receipt")
    save_json(args.receipt, {"live_sources": bindings, "live_source_count": len(bindings),
        "pre_generation_freeze_sha256": sha(args.run / "pre-generation-freeze.json"),
        "state_at_check": read_json(args.run / "state.json"),
        "scope": "Live and snapshot source bytes match; no inference or reference change"})
    print(f"Live pre-generation source bindings verified: {len(bindings)}", flush=True)


if __name__ == "__main__":
    main()
