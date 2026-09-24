from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from gnss_sim.generator import GENERATOR_VERSION, generate_case
from gnss_sim.schemas import CaseInput, CaseSummary, CaseTruth, DatasetManifest, GenerationRequest

DATASET_ID_PATTERN = re.compile(r"^event-v4-\d{8}-\d{6}-[a-f0-9]{8}$")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class DatasetStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gnss-generate")

    def _dataset_dir(self, dataset_id: str) -> Path:
        if not DATASET_ID_PATTERN.fullmatch(dataset_id):
            raise FileNotFoundError(dataset_id)
        return self.root / dataset_id

    def _case_dir(self, dataset_id: str, case_id: str) -> Path:
        if not re.fullmatch(r"case_\d{4}", case_id):
            raise FileNotFoundError(case_id)
        return self._dataset_dir(dataset_id) / "cases" / case_id

    def list_datasets(self) -> list[DatasetManifest]:
        manifests = []
        for path in self.root.glob("*/manifest.json"):
            try:
                manifests.append(DatasetManifest.model_validate(_read_json(path)))
            except (OSError, ValueError):
                continue
        return sorted(manifests, key=lambda item: item.created_at, reverse=True)

    def get_dataset(self, dataset_id: str) -> DatasetManifest:
        path = self._dataset_dir(dataset_id) / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(dataset_id)
        try:
            return DatasetManifest.model_validate(_read_json(path))
        except ValueError as exc:
            raise FileNotFoundError(dataset_id) from exc

    def get_case_input(self, dataset_id: str, case_id: str) -> CaseInput:
        path = self._case_dir(dataset_id, case_id) / "input.json"
        if not path.is_file():
            raise FileNotFoundError(case_id)
        try:
            return CaseInput.model_validate(_read_json(path))
        except ValueError as exc:
            raise FileNotFoundError(case_id) from exc

    def get_case_truth(self, dataset_id: str, case_id: str) -> CaseTruth:
        path = self._case_dir(dataset_id, case_id) / "truth.json"
        if not path.is_file():
            raise FileNotFoundError(case_id)
        try:
            return CaseTruth.model_validate(_read_json(path))
        except ValueError as exc:
            raise FileNotFoundError(case_id) from exc

    def _new_manifest(self, request: GenerationRequest) -> DatasetManifest:
        created_at = datetime.now(timezone.utc)
        dataset_id = f"{GENERATOR_VERSION}-{created_at:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            created_at=created_at,
            status="queued",
            request=request,
        )
        _write_json(self._dataset_dir(dataset_id) / "manifest.json", manifest.model_dump(mode="json"))
        return manifest

    def start(self, request: GenerationRequest) -> DatasetManifest:
        manifest = self._new_manifest(request)
        self.executor.submit(self._run, manifest)
        return manifest

    def generate_sync(self, request: GenerationRequest) -> DatasetManifest:
        return self._run(self._new_manifest(request))

    def _run(self, manifest: DatasetManifest) -> DatasetManifest:
        path = self._dataset_dir(manifest.dataset_id) / "manifest.json"
        manifest.status = "running"
        _write_json(path, manifest.model_dump(mode="json"))
        try:
            for index in range(manifest.request.count):
                case_id = f"case_{index + 1:04d}"
                case_seed = int(
                    np.random.SeedSequence([manifest.request.seed, index]).generate_state(
                        1, dtype=np.uint32
                    )[0]
                )
                case_input, truth = generate_case(case_id, case_seed, manifest.request.case_type)
                case_dir = self._case_dir(manifest.dataset_id, case_id)
                _write_json(case_dir / "input.json", case_input.model_dump(mode="json"))
                _write_json(case_dir / "truth.json", truth.model_dump(mode="json"))
                manifest.cases.append(
                    CaseSummary(case_id=case_id, case_seed=case_seed, event_count=len(truth.events))
                )
                manifest.generated_cases += 1
                _write_json(path, manifest.model_dump(mode="json"))
            manifest.status = "complete"
        except Exception as exc:
            manifest.status = "failed"
            manifest.error = f"{type(exc).__name__}: {exc}"
        _write_json(path, manifest.model_dump(mode="json"))
        return manifest
