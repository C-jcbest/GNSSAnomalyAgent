"""Prepare raw views and descriptive auxiliary evidence for direct Codex review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnss_sim.landslide import LandslideInput
from gnss_sim.landslide_diagnostics import derive_motion
from gnss_sim.landslide_raw_adjudication import raw_block_statistics


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(pack, out):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "source.py")
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if hashlib.sha256((pack / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Observation packet changed: " + name)
    for cid in manifest["case_ids"]:
        folder = out / cid
        folder.mkdir()
        source = pack / "observations" / cid / "input.json"
        shutil.copyfile(source, folder / "input.json")
        case = LandslideInput.model_validate_json(source.read_bytes())
        values = np.asarray([row if row is not None else [np.nan] * 3
                             for row in case.displacement_mm])
        days = np.arange(len(values))
        save(folder / "raw-blocks.json", raw_block_statistics(case))
        figure, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        for axis, name in enumerate(("N", "E", "U")):
            axes[axis].plot(days, values[:, axis], linewidth=.7)
            axes[axis].set_ylabel(name + " (mm)")
            axes[axis].grid(alpha=.2)
        axes[-1].set_xlabel("Original day index; gaps retained")
        axes[-1].set_xlim(0, len(values) - 1)
        figure.suptitle(cid + " | Raw observations only, no predictions or process labels")
        figure.tight_layout()
        figure.savefig(folder / "raw-global.png", dpi=120)
        plt.close(figure)
        figure, axes = plt.subplots(3, 4, figsize=(18, 7))
        for column, (start, stop) in enumerate(((0, 180), (180, 360), (360, 540),
                                               (540, len(values)))):
            for axis, name in enumerate(("N", "E", "U")):
                panel = axes[axis, column]
                panel.plot(days[start:stop], values[start:stop, axis], linewidth=.7)
                panel.set_ylabel(name + " (mm)")
                panel.set_xlim(start, stop - 1)
                panel.grid(alpha=.2)
                if axis == 0:
                    panel.set_title(f"[{start},{stop}) | local y-scale")
                if axis == 2:
                    panel.set_xlabel("Original day index")
        figure.suptitle(cid + " | Disjoint raw local views; no smoothing")
        figure.tight_layout()
        figure.savefig(folder / "raw-local.png", dpi=120)
        plt.close(figure)
        motions = {}
        for window in (31, 61, 91):
            motion = derive_motion(case, window)
            motions[window] = motion
            save(folder / f"motion-{window}.json", motion.model_dump(mode="json"))
        figure, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        for window, motion in motions.items():
            axes[0].plot(days, motion.speed_mm_day, label=f"{window} days", linewidth=.8)
            axes[1].plot(days, motion.tangential_acceleration_mm_day2,
                         label=f"{window} days", linewidth=.8)
            axes[2].plot(days, motion.valid_counts, label=f"{window} days", linewidth=.8)
        for panel in axes:
            panel.grid(alpha=.2)
            panel.legend()
        axes[0].set_ylabel("3D speed (mm/day)")
        axes[1].set_ylabel("Tangential acceleration")
        axes[2].set_ylabel("Valid support days")
        axes[2].set_xlabel("Original day index; centered observation-derived fits")
        figure.suptitle(cid + " | Auxiliary evidence, not labels or independent measurements")
        figure.tight_layout()
        figure.savefig(folder / "auxiliary.png", dpi=120)
        plt.close(figure)
    save(out / "provenance.json", {
        "reviewer": "Current Codex session, direct visual inspection plus observation statistics",
        "independent_expert": False, "prior_method_results_seen": True,
        "source": str(pack.resolve()), "case_ids": manifest["case_ids"],
        "new_remote_model_requests": 0, "process_reference_read_for_review": False,
        "activity_first": True, "stage_review_after_activity_freeze": True,
    })
    save(out / "preparation-manifest.json", {"files": {
        str(path.relative_to(out)).replace("\\", "/"):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*")) if path.is_file()}})
    print("Prepared", len(manifest["case_ids"]), "observation-only records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.pack, args.out)
