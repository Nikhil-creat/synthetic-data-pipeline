"""
main.py - FastAPI service for the Synthetic Data Pipeline.

Endpoints:
    GET  /health     liveness + RAG backend
    POST /ingest     upload a CSV (tabular) or a ZIP of class-named folders (images)
    POST /resample   agentic routing + resampling of the TRAIN partition
    POST /train      train a classifier on the resampled data (self-healing)
    POST /evaluate   Precision / Recall / F1 / PR-AUC on the untouched test set

Sessions live in memory (one pipeline per dataset_id). For multi-replica deployments,
move them to shared storage.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import secrets
import shutil
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from pipeline import (
    CNNFeatureExtractor,
    DataValidationError,
    PipelineConfig,
    PipelineError,
    RAGStrategyEngine,
    StageOrderError,
    SUPPORTED_STRATEGIES,
    SyntheticDataPipeline,
    make_demo_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sdp.api")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "50"))
API_KEY = os.getenv("API_KEY", "")
OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
Strategy = Literal["auto", "smote", "adasyn", "borderline_smote", "smote_tomek", "random_over"]


class Session:
    """A pipeline plus a lock so stages for one dataset never run concurrently."""

    def __init__(self, pipeline: SyntheticDataPipeline) -> None:
        self.pipeline = pipeline
        self.lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = PipelineConfig()
    app.state.cfg = cfg
    app.state.rag = await run_in_threadpool(RAGStrategyEngine, cfg)  # connect + seed Qdrant once
    app.state.extractor = None                                         # built lazily on first image upload
    app.state.sessions = {}
    logger.info("Ready. RAG backend: %s", app.state.rag.backend)
    yield


async def require_key(request: Request) -> None:
    """Optional shared-secret auth: enforced only when API_KEY is set."""
    if not API_KEY or request.url.path in OPEN_PATHS:
        return
    if not secrets.compare_digest(request.headers.get("x-api-key", ""), API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


app = FastAPI(title="Synthetic Data Pipeline", version="1.1.0", lifespan=lifespan, dependencies=[Depends(require_key)])
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StageOrderError)
async def _stage_error(_: Request, exc: StageOrderError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(DataValidationError)
async def _data_error(_: Request, exc: DataValidationError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(PipelineError)
async def _pipeline_error(_: Request, exc: PipelineError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# ------------------------------- schemas ---------------------------------- #
class ResampleRequest(BaseModel):
    dataset_id: str
    strategy: Strategy = "auto"
    k_neighbors: Optional[int] = Field(None, ge=1, le=50)
    sampling_ratio: Optional[float] = Field(None, gt=0, le=1)


class TrainRequest(BaseModel):
    dataset_id: str
    classifier: Literal["logreg", "rf"] = "logreg"


class EvaluateRequest(BaseModel):
    dataset_id: str
    threshold_source: Literal["validation", "test"] = "validation"


# ------------------------------- helpers ---------------------------------- #
def _get_session(dataset_id: str) -> Session:
    session = app.state.sessions.get(dataset_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset_id '{dataset_id}'.")
    return session


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds MAX_UPLOAD_MB.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    return data


def _extract_image_zip(raw: bytes, dest: Path) -> tuple[List[str], List[str]]:
    """Safely unzip (blocks path traversal) and return (paths, labels) from class folders."""
    paths: List[str] = []
    labels: List[str] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise DataValidationError("Upload is not a valid ZIP file.") from exc
    with zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise DataValidationError("Unsafe path in ZIP archive.")
            if info.is_dir() or Path(info.filename).suffix.lower() not in IMAGE_EXTS:
                continue
            parts = Path(info.filename).parts
            if len(parts) < 2:
                continue  # need class_name/file.jpg
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            paths.append(str(target))
            labels.append(parts[-2])
    if not paths:
        raise DataValidationError("No images found. Use class_name/image.jpg inside the ZIP.")
    return paths, labels


def _ingest_images_sync(pipe: SyntheticDataPipeline, raw: bytes, extractor):
    tmp = Path(tempfile.mkdtemp(prefix="sdp_imgs_"))
    try:
        paths, labels = _extract_image_zip(raw, tmp)
        return pipe.ingest_images(paths, labels)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # features are extracted; raw images are not kept


# ------------------------------- endpoints -------------------------------- #
@app.get("/health")
async def health():
    return {"status": "ok", "rag_backend": app.state.rag.backend, "sessions": len(app.state.sessions)}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    kind: Literal["tabular", "image"] = Form("tabular"),
    target: str = Form("target"),
    classifier: Literal["logreg", "rf"] = Form("logreg"),
    random_state: int = Form(42),
):
    """Upload a dataset. Splits stratified FIRST, then fits preprocessing on train only."""
    raw = await _read_upload(file)
    cfg = PipelineConfig(classifier=classifier, random_state=random_state)
    pipe = SyntheticDataPipeline(cfg, app.state.rag, app.state.extractor)
    try:
        if kind == "tabular":
            try:
                df = pd.read_csv(io.BytesIO(raw))
            except Exception as exc:
                raise DataValidationError(f"Could not parse CSV: {exc}") from exc
            summary = await run_in_threadpool(pipe.ingest_tabular, df, target)
        else:
            summary = await run_in_threadpool(_ingest_images_sync, pipe, raw, app.state.extractor)
            app.state.extractor = pipe._extractor  # reuse the loaded backbone next time
    except PipelineError:
        raise
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc
    dataset_id = uuid.uuid4().hex[:12]
    app.state.sessions[dataset_id] = Session(pipe)
    while len(app.state.sessions) > MAX_SESSIONS:  # evict oldest to bound memory
        app.state.sessions.pop(next(iter(app.state.sessions)))
    return {"dataset_id": dataset_id, "kind": kind, **summary}


@app.post("/resample")
async def resample(req: ResampleRequest):
    """Agent profiles the data, consults the RAG store, routes, resamples (with self-healing)."""
    session = _get_session(req.dataset_id)
    async with session.lock:
        return await run_in_threadpool(
            session.pipeline.resample, req.strategy, req.k_neighbors, req.sampling_ratio
        )


@app.post("/train")
async def train(req: TrainRequest):
    session = _get_session(req.dataset_id)
    async with session.lock:
        return await run_in_threadpool(session.pipeline.train, req.classifier)


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    session = _get_session(req.dataset_id)
    async with session.lock:
        return await run_in_threadpool(session.pipeline.evaluate, req.threshold_source)


class CompareRequest(BaseModel):
    dataset_id: str
    classifier: Literal["logreg", "rf"] = "logreg"


@app.get("/strategies")
async def strategies():
    """Resamplers the agent can choose from."""
    return {"strategies": list(SUPPORTED_STRATEGIES), "auto": "agent routes on imbalance, noise and dimension"}


@app.get("/datasets")
async def datasets():
    return {"datasets": [
        {"dataset_id": k, "modality": v.pipeline.data.modality, "rows_fit": len(v.pipeline.data.y_fit)}
        for k, v in app.state.sessions.items() if v.pipeline.data is not None
    ]}


@app.post("/compare")
async def compare(req: CompareRequest):
    """Benchmark every resampler on validation data (run /resample first for k and ratio)."""
    session = _get_session(req.dataset_id)
    async with session.lock:
        return await run_in_threadpool(session.pipeline.compare_strategies, req.classifier)


@app.get("/report/{dataset_id}")
async def report(dataset_id: str):
    """Markdown model card plus metrics for the last evaluation."""
    session = _get_session(dataset_id)
    async with session.lock:
        return await run_in_threadpool(session.pipeline.report)


@app.get("/demo-data")
async def demo_data(n: int = 4000, minority_fraction: float = 0.01, seed: int = 42):
    """Download a synthetic imbalanced CSV (target column 'target') to try /ingest."""
    if not (500 <= n <= 100000) or not (0.002 <= minority_fraction <= 0.3):
        raise HTTPException(status_code=400, detail="n in [500, 100000], minority_fraction in [0.002, 0.3].")
    df = await run_in_threadpool(make_demo_dataset, n, minority_fraction, 8, seed)
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=demo_imbalanced.csv"})
