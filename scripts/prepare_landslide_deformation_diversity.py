"""Freeze and generate diverse grouped data for deformation/quality supervision."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from probe_landslide_nuisance_evidence import ROOT, write_csv

from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide import LandslideRequest
from gnss_sim.landslide_deformation_diversity import FAMILIES, diverse_case
from gnss_sim.landslide_diagnostics import derive_motion


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    counts = {"train": 48, "validation": 24, "mechanism_test": 24}
    generator = np.random.default_rng(2026101248)
    split = {}
    assignments = []
    index = 337
    for group, count in counts.items():
        families = generator.permutation(np.tile(FAMILIES, count // len(FAMILIES)))
        ids = [f"case_{number:04d}" for number in range(index, index + count)]
        split[group] = ids
        for case_id, family in zip(ids, families, strict=True):
            assignments.append((group, case_id, str(family)))
        index += count
    write_json(
        out / "protocol.json",
        {
            "registered_utc": datetime.now(timezone.utc).isoformat(),
            "seed": 2026101247,
            "assignment_seed": 2026101248,
            "days": 730,
            "split": split,
            "families": list(FAMILIES),
            "records": 96,
            "stream_indices": list(range(1000, 1096)),
            "definition": "Target deformation separate from offset transitions/spikes; labels may coexist",
            "profiles": "constant/rising/falling/hump/irregular; durations35..259d, fast8..35d, recurrent35..179d with>=50d pause; log-uniform peak0.035..0.8mm/d or fast1..3.5; no fixed ASD sequence",
            "quality": "1..2 signed NEU offsets8..40mm or2..6 spikes6..20 times sampled white-noise sigma; first mixed artifact occurs inside an activity interval; do not resample if raw missing hides it",
            "noise": "Existing stable simulator measurement_noise only, independent streams; original missing calendar retained; old latent motion/artifacts excluded before explicit injections",
            "reference": "Known controlled activity, auxiliary process stages, original-day quality events; missing labels=-1; shifted platform is not repeated quality event",
            "auxiliary_query_target": "For a future local-window quality head, compute whether the observed central14-day target contains an injected transition/spike from TRAIN/VALIDATION labels only; exact event days remain canonical labels",
            "population": "Designed coverage, not field prevalence; each family6train/3validation/3test; no filtering, replacement or regeneration selection",
            "truth_access": "Train/validation references only for fitting/calibration; all test components/references in mechanism-sealed; do not read test labels before model/threshold freeze",
            "review": "Reserved first8 opaque test IDs for post-model-freeze Codex observation review; no independent expert reviewer; do not relabel process truth after predictions",
            "methods": "Next candidate is one shared visual/numeric trunk with independent deformation/quality heads; compare same-data single-head, dual numeric and visual single/dual, rules/PELT and refit strong raw84/chart-info XGBoost; no NCDA v1 threshold rescue",
            "baseline_fairness": "Refit/calibrate on these same complete train/validation records; old fitted models are historical comparisons only",
            "new_api_calls": 0,
            "independent_field_reference": False,
        },
    )
    sources = [
        Path(__file__),
        ROOT / "src/gnss_sim/landslide_deformation_diversity.py",
        ROOT / "src/gnss_sim/landslide.py",
        ROOT / "src/gnss_sim/landslide_diagnostics.py",
        ROOT / "tests/test_landslide_deformation_diversity.py",
    ]
    for source in sources:
        target = out / "source" / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
    write_json(
        out / "source-bindings.json", {str(path.relative_to(ROOT)): sha(path) for path in sources}
    )
    write_json(
        out / "pre-generation-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    request = LandslideRequest(seed=2026101247, count=96, days=730)
    training_rows = []
    evaluation_rows = []
    training_ledger = []
    evaluation_ledger = []
    started = time.perf_counter()
    for position, (group, case_id, family) in enumerate(assignments):
        stream_index = 1000 + position
        case, rows, truth = diverse_case(request, stream_index, family, case_id)
        folder = out / "inputs" / case_id
        folder.mkdir(parents=True)
        (folder / "input.json").write_text(case.model_dump_json(indent=2), encoding="utf-8")
        for window in (31, 61, 91):
            motion = derive_motion(case, window)
            write_json(folder / f"motion-{window}.json", motion.model_dump(mode="json"))
        ledger_row = {"case_id": case_id, "group": group, "truth": truth}
        if group == "mechanism_test":
            evaluation_rows.extend(rows)
            evaluation_ledger.append(ledger_row)
        else:
            training_rows.extend(rows)
            training_ledger.append(ledger_row)
        print(group, case_id, position + 1, "/ 96", flush=True)
    write_csv(out / "reference-labels.csv", training_rows)
    write_json(out / "training-generation-ledger.json", training_ledger)
    sealed = out / "mechanism-sealed"
    sealed.mkdir()
    write_csv(sealed / "reference-labels.csv", evaluation_rows)
    write_json(sealed / "generation-ledger.json", evaluation_ledger)
    pack = out / "raw-review-pack"
    pack.mkdir()
    review_ids = split["mechanism_test"][:8]
    for cid in review_ids:
        target = pack / "observations" / cid / "input.json"
        target.parent.mkdir(parents=True)
        shutil.copyfile(out / "inputs" / cid / "input.json", target)
    write_json(
        pack / "manifest.json",
        {
            "case_ids": review_ids,
            "scope": "Raw observations only; review after model freeze",
            "files": {
                str(path.relative_to(pack)): sha(path) for path in pack.rglob("*") if path.is_file()
            },
        },
    )
    write_json(
        out / "manifest.json",
        {
            "status": "generated_untrained_test_unopened",
            "split": split,
            "new_records": 96,
            "calendar_rows": 96 * 730,
            "training_validation_rows": len(training_rows),
            "sealed_test_rows": len(evaluation_rows),
            "generation_seconds": time.perf_counter() - started,
            "test_labels_used_for_model_selection": False,
            "new_api_calls": 0,
        },
    )
    for source in sources:
        if sha(source) != sha(out / "source" / source.name):
            raise ValueError("Generation source changed during preparation")
    write_json(
        out / "data-freeze.json",
        {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    )
    print("Generated96; train48/validation24/test24, test not scored", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out.resolve())
