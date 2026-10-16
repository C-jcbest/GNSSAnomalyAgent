"""Generate an unfiltered 120-record extension with sealed process references."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from prepare_landslide_ai_reference_holdout import write_csv
from prepare_landslide_codex_observation_review import prepare as prepare_views
from prepare_landslide_core_lockbox import MODEL_FILES, MODEL_SOURCE
from prepare_landslide_motion_probe import FAMILIES, controlled_case
from probe_landslide_endpoint_fit import BASE, ROOT, export
from run_landslide_ai_reference_holdout import verify_bindings
from run_landslide_window_optimization import read_json

from gnss_sim.artifacts import sha
from gnss_sim.landslide import LandslideInput, LandslideRequest
from gnss_sim.landslide_diagnostics import derive_motion


def prepare(out: Path) -> None:
    calibration = BASE / "paired-persistence-calibration-v1"
    verify_bindings(calibration, "calibration-freeze.json")
    out.mkdir(parents=True, exist_ok=False)
    ids = [f"case_{index:04d}" for index in range(217, 337)]
    protocol = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "persistence-scale-v1",
        "method_name": "PP-VN / Persistence-Paired Visual-Numeric Learning",
        "primary_method": "visual_numeric_pair",
        "case_ids": ids,
        "generation": {
            "seed": 2026101141,
            "permutation_seed": 2026101142,
            "stream_indices": list(range(216, 336)),
            "days": 730,
            "families": list(FAMILIES),
            "records_per_family": 20,
        },
        "secondary_review_ids": ids[:24],
        "review_sampling": "First 24 opaque IDs; four of each family, shuffled before generation. Remaining 96 also shuffled. No prediction-dependent selection.",
        "selection": read_json(calibration / "selection.json"),
        "calibration": calibration.relative_to(ROOT).as_posix(),
        "refitting": False,
        "primary_reference": "Known synthetic process; full activity/A/S/D/episode/weak labels, raw gaps remain unknown.",
        "secondary_reference": "Codex raw visual and numerical review, activity frozen before stage review and inference. Prior historical design/results exposed; not independent expert truth.",
        "scope": "New complete same-generator scale extension, not external sites or field validation; report separately from the earlier 24-record confirmation, retain earlier negative results.",
        "sampling": "One fixed seed and all 120 records, no replacement, rejection, regeneration or best-seed substitution.",
        "analysis": "Frozen ten learning arms and nine existing numeric baselines; common scores, per-record and per-family results, whole-record paired bootstrap. No new remote VLM runs.",
        "training_data_expanded": False,
        "independent_of_method_selection_records": True,
        "field_accuracy_verified": False,
    }
    export(out / "protocol.json", protocol)
    bindings = {path.relative_to(ROOT).as_posix(): sha(path)
                for path in calibration.rglob("*") if path.is_file()}
    for name in MODEL_FILES:
        source = MODEL_SOURCE / name
        shutil.copyfile(source, out / name)
        bindings[source.relative_to(ROOT).as_posix()] = sha(source)
    export(out / "training-bindings.json", bindings)
    sources = [Path(__file__), ROOT / "scripts/prepare_landslide_motion_probe.py",
               ROOT / "scripts/prepare_landslide_codex_observation_review.py",
               ROOT / "src/gnss_sim/landslide.py",
               ROOT / "src/gnss_sim/landslide_diagnostics.py"]
    for source in sources:
        target = out / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    export(out / "pre-generation-freeze.json", {
        path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()
    })
    started = time.perf_counter()
    specification = protocol["generation"]
    request = LandslideRequest(seed=specification["seed"], count=120, days=730)
    rng = np.random.default_rng(specification["permutation_seed"])
    order = np.concatenate((rng.permutation(np.arange(216, 240)),
                            rng.permutation(np.arange(240, 336))))
    rows, ledger = [], []
    pack = out / "raw-review-pack"
    pack.mkdir()
    for cid, index in zip(ids, order, strict=True):
        original, labels, truth = controlled_case(request, int(index))
        case = LandslideInput.model_validate(dict(original.model_dump(), case_id=cid))
        folder = out / "inputs" / cid
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        for window in (31, 61, 91):
            motion = derive_motion(case, window)
            export(folder / f"motion-{window}.json", motion.model_dump(mode="json"))
        rows.extend(dict(row, case_id=cid) for row in labels)
        ledger.append({"case_id": cid, "stream_index": int(index), "truth": truth})
        if cid in protocol["secondary_review_ids"]:
            destination = pack / "observations" / cid
            destination.mkdir(parents=True)
            shutil.copyfile(folder / "input.json", destination / "input.json")
        print(f"Generated opaque observation {cid} ({len(ledger)}/120)", flush=True)
    sealed = out / "mechanism-sealed"
    sealed.mkdir()
    write_csv(sealed / "reference-labels.csv", rows)
    export(sealed / "generation-ledger.json", ledger)
    export(pack / "manifest.json", {
        "case_ids": protocol["secondary_review_ids"],
        "source": "Only raw observations, no process labels, family names or predictions",
        "files": {path.relative_to(pack).as_posix(): sha(path)
                  for path in pack.rglob("*") if path.is_file()},
    })
    prepare_views(pack, out / "codex-review")
    verify_bindings(out, "pre-generation-freeze.json")
    export(out / "data-freeze.json", {
        path.relative_to(out).as_posix(): sha(path)
        for path in out.rglob("*") if path.is_file()
    })
    export(out / "state.json", {
        "status": "raw_review_ready", "new_records": 120, "calendar_rows": len(rows),
        "review_records": 24, "seconds": time.perf_counter() - started,
        "new_api_calls": 0, "new_specific_reference_opened_by_reviewer": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out.resolve())
