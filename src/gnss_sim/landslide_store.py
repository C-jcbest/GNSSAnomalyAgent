"""Persistent batches for the multi-year simulation domain."""
from __future__ import annotations

import csv
import io
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from gnss_sim import landslide
from gnss_sim.artifacts import sha, write_json
from gnss_sim.landslide import (
    SCENARIOS,
    LandslideInput,
    LandslideManifest,
    LandslideRequest,
    LandslideSummary,
    LandslideTruth,
    allocate_scenarios,
    simulate_case,
)
from gnss_sim.storage import _read_json, _write_json


class LandslideStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="landslide-generate")

    def _dataset_dir(self, dataset_id: str) -> Path:
        if not re.fullmatch(r"landslide-v1-\d{8}-\d{6}-[a-f0-9]{8}", dataset_id):
            raise FileNotFoundError(dataset_id)
        return self.root / dataset_id

    def _case_file(self, dataset_id: str, case_id: str, filename: str) -> Path:
        if not re.fullmatch(r"case_\d{4}", case_id):
            raise FileNotFoundError(case_id)
        return self._dataset_dir(dataset_id) / "cases" / case_id / filename

    def list_datasets(self) -> list[LandslideManifest]:
        manifests = []
        for path in self.root.glob("landslide-v1-*/manifest.json"):
            try:
                manifests.append(LandslideManifest.model_validate(_read_json(path)))
            except (OSError, ValueError):
                continue
        return sorted(manifests, key=lambda item: item.created_at, reverse=True)

    def get_dataset(self, dataset_id: str) -> LandslideManifest:
        return LandslideManifest.model_validate(
            _read_json(self._dataset_dir(dataset_id) / "manifest.json"))

    def get_case(self, dataset_id: str, case_id: str) -> LandslideInput:
        return LandslideInput.model_validate(
            _read_json(self._case_file(dataset_id, case_id, "input.json")))

    def get_truth(self, dataset_id: str, case_id: str) -> LandslideTruth:
        return LandslideTruth.model_validate(
            _read_json(self._case_file(dataset_id, case_id, "truth.json")))

    def observations_csv(self, dataset_id: str, case_id: str) -> str:
        case = self.get_case(dataset_id, case_id)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["date", "N_mm", "E_mm", "U_mm"])
        for day, observation in zip(case.dates, case.displacement_mm):
            writer.writerow([day.isoformat(), *(observation if observation is not None else ("", "", ""))])
        return output.getvalue()

    def _new_manifest(self, request: LandslideRequest) -> LandslideManifest:
        created_at = datetime.now(timezone.utc)
        dataset_id = f"landslide-v1-{created_at:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        scenarios = allocate_scenarios(request)
        manifest = LandslideManifest(
            dataset_id=dataset_id, created_at=created_at, status="queued", request=request,
            generator_sha256=sha(Path(landslide.__file__)),
            type_counts={kind: scenarios.count(kind) for kind in SCENARIOS if kind in scenarios},
        )
        directory = self._dataset_dir(dataset_id)
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "manifest.json", manifest.model_dump(mode="json"))
        # Keep the exact generating source with each batch, not only a hash of mutable code.
        (directory / "generator.py").write_bytes(Path(landslide.__file__).read_bytes())
        return manifest

    def start(self, request: LandslideRequest) -> LandslideManifest:
        manifest = self._new_manifest(request)
        self.executor.submit(self._run, manifest.model_copy(deep=True))
        return manifest

    def generate_sync(self, request: LandslideRequest) -> LandslideManifest:
        return self._run(self._new_manifest(request))

    def _run(self, manifest: LandslideManifest) -> LandslideManifest:
        path = self._dataset_dir(manifest.dataset_id) / "manifest.json"
        manifest.status = "running"
        _write_json(path, manifest.model_dump(mode="json"))
        try:
            for index, scenario in enumerate(allocate_scenarios(manifest.request)):
                case, truth = simulate_case(manifest.request, index, scenario)
                case_path = self._case_file(manifest.dataset_id, case.case_id, "input.json")
                write_json(case_path, case.model_dump(mode="json"))
                write_json(case_path.with_name("truth.json"), truth.model_dump(mode="json"))
                missing_days = len(truth.missing_indices)
                manifest.cases.append(LandslideSummary(
                    case_id=case.case_id, scenario=scenario, missing_days=missing_days,
                    observed_days=manifest.request.days - missing_days,
                ))
                values = [value for row in case.displacement_mm if row is not None for value in row]
                low, high = manifest.observed_extent_mm
                manifest.observed_extent_mm = (min(low, min(values)), max(high, max(values)))
                manifest.generated_cases += 1
                _write_json(path, manifest.model_dump(mode="json"))
            manifest.status = "complete"
        except Exception as exc:
            manifest.status = "failed"
            manifest.error = f"{type(exc).__name__}: {exc}"
        _write_json(path, manifest.model_dump(mode="json"))
        return manifest
