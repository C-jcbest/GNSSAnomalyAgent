"""Check saved data/plot contracts without interpreting sealed test curves or scoring."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_landslide_background import historical_snapshot  # noqa: E402

from gnss_sim.artifacts import sha, write_json  # noqa: E402
from gnss_sim.landslide import LandslideInput  # noqa: E402
from gnss_sim.landslide_background import observation_noise  # noqa: E402
from gnss_sim.landslide_diagnostics import LandslideDiagnostics  # noqa: E402
from gnss_sim.landslide_observation_review import (  # noqa: E402
    ObservationReview,
    compile_observation_review,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path,
                        default=ROOT / "artifacts/landslide-background-2026-10-08/batch-v1")
    parser.add_argument("--url", default="http://127.0.0.1:18777/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    public = args.batch / "public"
    private = args.batch / "private"
    manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
    generation = json.loads((private / "generation-manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((public / "protocol.json").read_text(encoding="utf-8"))
    historical = json.loads((private / "historical-freeze-before.json").read_text(encoding="utf-8"))
    assert historical == historical_snapshot()
    for name, expected in generation["artifact_sha256"].items():
        assert sha(args.batch / name) == expected, name
    for name, expected in generation["source_sha256"].items():
        if sha(ROOT / name) == expected:
            continue
        assert name == "src/gnss_sim/landslide_background.py", name
        normalization = json.loads((private / "source-normalization-audit.json").read_text(encoding="utf-8"))
        assert normalization["source_sha256_before"] == expected
        assert normalization["source_sha256_after"] == sha(ROOT / name)
        assert normalization["verified_identical"] == {"inputs": 24, "truths": 24}
    assert len(manifest["cases"]) == 12
    assert manifest["new_model_calls"] == 0
    assert manifest["reference_status"] == "pending_observation_review"
    raw_count, diagnostic_count, pending_rejections, removed_ticks = 0, 0, 0, 0
    for record in manifest["cases"]:
        input_path = public / record["input"]
        case = LandslideInput.model_validate_json(input_path.read_bytes())
        assert sha(input_path) == record["input_sha256"]
        with (public / record["csv"]).open(encoding="utf-8-sig", newline="") as source:
            csv_rows = list(csv.DictReader(source))
        assert len(csv_rows) == len(case.dates) == 1095
        for day, row in enumerate(csv_rows):
            assert int(row["day_index"]) == day and row["date"] == str(case.dates[day])
            observed = case.displacement_mm[day]
            assert int(row["observed"]) == int(observed is not None)
            if observed is None:
                assert all(row[axis + "_mm"] == "" for axis in ("N", "E", "U"))
            else:
                assert [float(row[axis + "_mm"]) for axis in ("N", "E", "U")] == list(observed)
        values = np.array([row if row is not None else [np.nan] * 3 for row in case.displacement_mm])
        noise = observation_noise(values)
        for view in record["views"]:
            assert sha(public / view["image"]) == view["image_sha256"]
            start, end = view["view"]
            audit = view["audit"]
            ticks = audit["tick_audit"]
            assert ticks["ticks"][0] == start and ticks["ticks"][-1] == end
            bounds = ticks["label_bounds_pixels"]
            assert all(right[0] - left[1] >= 8 for left, right in zip(bounds, bounds[1:]))
            removed_ticks += len(ticks["removed_ticks"])
            for axis, panel in enumerate(audit["panels"]):
                series = values[start:end + 1, axis]
                coordinates = np.column_stack([np.arange(start, end + 1), series])
                assert panel["data_sha256"] == hashlib.sha256(coordinates.tobytes()).hexdigest()
                assert panel["missing_days"] == np.flatnonzero(~np.isfinite(series)).tolist()
                assert panel["xlim"] == [start, end]
                finite = series[np.isfinite(series)]
                plot = protocol["plot"]
                span = max(float(finite.max() - finite.min()),
                           plot["noise_span_multiplier"] * noise[axis], plot["minimum_span_mm"])
                assert np.isclose(panel["ylim"][1] - panel["ylim"][0], span * 1.5)
            raw_count += 1
        for window, files in record["diagnostics"].items():
            diagnostics = LandslideDiagnostics.model_validate_json((public / files["data"]).read_bytes())
            assert diagnostics.input_sha256 == record["input_sha256"]
            assert diagnostics.window_days == int(window)
            assert sha(public / files["image"]) == files["image_sha256"]
            for day, observation in enumerate(case.displacement_mm):
                if observation is None:
                    assert diagnostics.velocity_mm_day[day] is None
            diagnostic_count += 1
        review = ObservationReview.model_validate_json((public / record["review"]).read_bytes())
        assert review.activity_status == "pending" and not review.stages and not review.episodes
        motion = LandslideDiagnostics.model_validate_json(
            (public / record["diagnostics"]["61"]["data"]).read_bytes(),
        )
        try:
            compile_observation_review(case, review, motion, record["views"][0]["image_sha256"])
        except ValueError as error:
            assert "Pending" in str(error)
            pending_rejections += 1
        else:
            raise AssertionError("A pending template was exported as reference labels")
    assert len(list((private / "sealed_test").glob("*/input.json"))) == 12
    assert not list((private / "sealed_test").rglob("*.png"))
    assert not list((private / "sealed_test").rglob("*diagnostics*"))
    assert not any(path.is_symlink() for path in public.rglob("*"))
    http = {}
    for path in public.rglob("*"):
        if not path.is_file():
            continue
        name = path.relative_to(public).as_posix()
        with urlopen(args.url + name, timeout=10) as response:
            http[name] = response.status
            assert response.read() == path.read_bytes()
    blocked = {}
    for name in ("private/protocol.json", "../private/sealed_test/case_0001/input.json",
                 "%2e%2e/private/sealed_test/case_0001/input.json", "truth.json"):
        try:
            urlopen(args.url + name, timeout=10)
        except HTTPError as error:
            assert error.code == 404
            blocked[name] = error.code
        else:
            raise AssertionError("Private material is accessible from the public review root")
    write_json(args.output, {
        "status": "VERIFIED saved data/plot contracts; not model performance or independent labels",
        "development_records": 12, "sealed_test_records": 12,
        "raw_views": raw_count, "auxiliary_views": diagnostic_count,
        "pending_templates_rejected": pending_rejections, "interior_ticks_removed": removed_ticks,
        "historical_files_unchanged": len(historical), "http_resources": http,
        "private_http_paths_rejected": blocked, "new_model_calls": 0,
    })
    print(f"VERIFIED: {raw_count} raw plots, {diagnostic_count} auxiliary plots, {len(http)} HTTP resources")


if __name__ == "__main__":
    main()
