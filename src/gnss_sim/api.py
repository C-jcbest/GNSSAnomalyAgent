from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from gnss_sim.landslide import LandslideInput, LandslideManifest, LandslideRequest, LandslideTruth
from gnss_sim.landslide_diagnostics import (
    LandslideDiagnostics,
    derive_motion,
    render_diagnostics,
)
from gnss_sim.landslide_store import LandslideStore
from gnss_sim.schemas import CaseInput, CaseTruth, DatasetManifest, GenerationRequest
from gnss_sim.storage import DatasetStore

# The API owns document order and grouping; the reader uses this same catalog.
DOCUMENT_GROUPS = {
    "项目文档": (
        ("data-generation", "数据生成", "data-generation.md"),
        ("landslide-design", "长期滑坡数据设计", "landslide-design.md"),
        ("landslide-source-notes", "滑坡文献核查", "landslide-source-notes.md"),
        ("detection", "检测与评价", "detection.md"),
        ("development", "开发与运行", "development.md"),
        ("project-status", "项目状态", "project-status.md"),
    ),
}
DOCUMENTS = tuple(
    {"slug": slug, "title": title, "file": file, "group": group}
    for group, entries in DOCUMENT_GROUPS.items()
    for slug, title, file in entries
)


def create_app(data_root: Path | None = None, web_root: Path | None = None) -> FastAPI:
    root = data_root or Path(os.environ.get("GNSS_SIM_DATA_DIR", "data/generated"))
    store = DatasetStore(root)
    landslide_store = LandslideStore(root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            store.executor.shutdown(wait=True)
            landslide_store.executor.shutdown(wait=True)

    app = FastAPI(title="GNSS Simulation Lab", version="0.2.0", lifespan=lifespan)
    app.state.store = store
    app.state.landslide_store = landslide_store

    @app.post("/api/landslide-datasets", response_model=LandslideManifest, status_code=202)
    def create_landslide_dataset(request: LandslideRequest) -> LandslideManifest:
        return landslide_store.start(request)

    @app.get("/api/landslide-datasets", response_model=list[LandslideManifest])
    def list_landslide_datasets() -> list[LandslideManifest]:
        return landslide_store.list_datasets()

    @app.get("/api/landslide-datasets/{dataset_id}", response_model=LandslideManifest)
    def get_landslide_dataset(dataset_id: str) -> LandslideManifest:
        try:
            return landslide_store.get_dataset(dataset_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc

    @app.get("/api/landslide-datasets/{dataset_id}/cases/{case_id}", response_model=LandslideInput)
    def get_landslide_case(dataset_id: str, case_id: str) -> LandslideInput:
        try:
            return landslide_store.get_case(dataset_id, case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Case not found") from exc

    @app.get("/api/landslide-datasets/{dataset_id}/cases/{case_id}/truth",
             response_model=LandslideTruth)
    def get_landslide_truth(dataset_id: str, case_id: str) -> LandslideTruth:
        try:
            return landslide_store.get_truth(dataset_id, case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Case truth not found") from exc

    @app.get("/api/landslide-datasets/{dataset_id}/cases/{case_id}/observations.csv")
    def export_landslide_observations(dataset_id: str, case_id: str) -> Response:
        try:
            content = landslide_store.observations_csv(dataset_id, case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Case not found") from exc
        return Response(content=content, media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{case_id}.csv"'})

    @app.get("/api/landslide-datasets/{dataset_id}/cases/{case_id}/diagnostics",
             response_model=LandslideDiagnostics)
    def get_landslide_diagnostics(
        dataset_id: str, case_id: str,
        window_days: Annotated[int, Query(ge=31, le=91)] = 61,
    ) -> LandslideDiagnostics:
        if window_days not in (31, 61, 91):
            raise HTTPException(status_code=422, detail="窗口只能为 31、61 或 91 日")
        try:
            case = landslide_store.get_case(dataset_id, case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Case not found") from exc
        return derive_motion(case, window_days)

    @app.get("/api/landslide-datasets/{dataset_id}/cases/{case_id}/diagnostics.png")
    def export_landslide_diagnostics(
        dataset_id: str, case_id: str,
        window_days: Annotated[int, Query(ge=31, le=91)] = 61,
    ) -> Response:
        if window_days not in (31, 61, 91):
            raise HTTPException(status_code=422, detail="窗口只能为 31、61 或 91 日")
        try:
            case = landslide_store.get_case(dataset_id, case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Case not found") from exc
        diagnostics = derive_motion(case, window_days)
        return Response(content=render_diagnostics(case, diagnostics), media_type="image/png",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{case_id}-{window_days}d.png"'})

    @app.get("/api/docs")
    def list_documents() -> list[dict[str, str]]:
        return [dict(item) for item in DOCUMENTS]

    @app.get("/api/docs/{slug}", response_class=PlainTextResponse)
    def get_document(slug: str) -> str:
        document = next((item for item in DOCUMENTS if item["slug"] == slug), None)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        path = Path(__file__).resolve().parents[2] / "docs" / document["file"]
        if not path.is_file():
            raise HTTPException(status_code=503, detail="Document unavailable")
        return path.read_text(encoding="utf-8")

    @app.post("/api/datasets", response_model=DatasetManifest, status_code=202)
    def create_dataset(request: GenerationRequest) -> DatasetManifest:
        return store.start(request)

    @app.get("/api/datasets", response_model=list[DatasetManifest])
    def list_datasets() -> list[DatasetManifest]:
        return store.list_datasets()

    @app.get("/api/datasets/{dataset_id}", response_model=DatasetManifest)
    def get_dataset(dataset_id: str) -> DatasetManifest:
        try:
            return store.get_dataset(dataset_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Dataset not found") from exc

    @app.get("/api/datasets/{dataset_id}/cases/{case_id}", response_model=CaseInput)
    def get_case_input(dataset_id: str, case_id: str) -> CaseInput:
        try:
            return store.get_case_input(dataset_id, case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Case not found") from exc

    @app.get("/api/datasets/{dataset_id}/cases/{case_id}/truth", response_model=CaseTruth)
    def get_case_truth(dataset_id: str, case_id: str) -> CaseTruth:
        try:
            return store.get_case_truth(dataset_id, case_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Case truth not found") from exc

    dist = web_root or Path(__file__).resolve().parents[2] / "web" / "dist"
    if dist.is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(dist / "index.html")

    return app
