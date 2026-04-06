"""
FastAPI application — entry point for the serving layer.

Architecture: Lifespan-managed singleton services + thin route handlers.
  - Lifespan: loads models into ModelRegistry, constructs all services once,
    attaches them to app.state for DI retrieval.
  - Routes: delegate entirely to services; no business logic here.
  - Exception handlers: map domain exceptions to HTTP codes per the design
    error table. All error bodies are structured {"error": str, "type": str}
    — raw tracebacks never reach the client.

Technical Debt: GraderService.predict() runs TF inference synchronously
inside an async handler. This blocks the FastAPI event loop for the duration
of inference. Fix: wrap the inference call in asyncio.to_thread() or migrate
to TF Serving / Triton over gRPC so the event loop is never CPU-blocked.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse

from pregrader.config import PregraderSettings, load_settings
from pregrader.enums import CardType
from pregrader.exceptions import (
    BatchSizeError,
    ImageIngestionError,
    InferenceError,
    InvalidImageFormatError,
    ImageResolutionError,
    ModelNotFoundError,
    PreprocessingError,
)
from pregrader.logging_config import get_logger
from pregrader.registry import ModelRegistry
from pregrader.schemas import GradeResult
from pregrader.services.grader import GraderService
from pregrader.services.ingestion import ImageIngestionService
from pregrader.services.preprocessing import PreprocessingService

from pregrader.api.dependencies import (
    GraderDep,
    IngestionDep,
    PreprocessingDep,
    RegistryDep,
)

logger = get_logger(service="api")

# ---------------------------------------------------------------------------
# Lifespan: model loading and service wiring
# ---------------------------------------------------------------------------

# Artifact path map — driven by settings so no card type is hard-coded here.
_ARTIFACT_PATH_MAP = {
    CardType.pokemon: "pokemon_model_artifact_path",
    CardType.one_piece: "one_piece_model_artifact_path",
    CardType.sports: "sports_model_artifact_path",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models and wire services before the first request is served.

    Why asynccontextmanager instead of on_event("startup")?
    - The lifespan pattern is the FastAPI-recommended approach since v0.93.
    - It keeps startup and shutdown logic co-located and avoids the deprecated
      on_event API.
    - Errors during startup propagate cleanly and prevent the server from
      accepting requests in a broken state.
    """
    settings: PregraderSettings = load_settings()
    registry = ModelRegistry()

    # Load each enabled card type — fail fast if an artifact path is missing.
    for card_type in settings.enabled_card_types:
        attr = _ARTIFACT_PATH_MAP[card_type]
        artifact_path = getattr(settings, attr, None)
        if artifact_path is None:
            logger.warning(
                "artifact_path_not_configured",
                card_type=card_type.value,
                setting=attr,
            )
            continue
        registry.load(card_type, artifact_path)

    # Construct services once — they are stateless beyond their injected deps.
    ingestion_service = ImageIngestionService(settings)
    preprocessing_service = PreprocessingService()
    grader_service = GraderService(registry, settings)

    # Attach to app.state so DI providers can retrieve them per-request.
    app.state.settings = settings
    app.state.registry = registry
    app.state.ingestion_service = ingestion_service
    app.state.preprocessing_service = preprocessing_service
    app.state.grader_service = grader_service

    logger.info("startup_complete", registry_ready=registry.is_ready)

    yield  # Server is live — handle requests

    # Shutdown: nothing to clean up for TF SavedModels (GC handles it).
    logger.info("shutdown_complete")


# ---------------------------------------------------------------------------
# App instantiation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TCG Pre-Grader API",
    version="0.1.0",
    description="CNN-based TCG card pre-grading (Pokemon MVP)",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Exception handlers — domain errors → structured HTTP responses
# ---------------------------------------------------------------------------

def _error_body(exc: Exception) -> dict:
    return {"error": str(exc), "type": type(exc).__name__}


@app.exception_handler(BatchSizeError)
async def handle_batch_size_error(request, exc: BatchSizeError) -> JSONResponse:
    return JSONResponse(status_code=422, content=_error_body(exc))


@app.exception_handler(InvalidImageFormatError)
async def handle_invalid_format(request, exc: InvalidImageFormatError) -> JSONResponse:
    return JSONResponse(status_code=422, content=_error_body(exc))


@app.exception_handler(ImageResolutionError)
async def handle_resolution_error(request, exc: ImageResolutionError) -> JSONResponse:
    return JSONResponse(status_code=422, content=_error_body(exc))


@app.exception_handler(ImageIngestionError)
async def handle_ingestion_error(request, exc: ImageIngestionError) -> JSONResponse:
    # Catch-all for any ImageIngestionError subclass not handled above.
    return JSONResponse(status_code=422, content=_error_body(exc))


@app.exception_handler(ModelNotFoundError)
async def handle_model_not_found(request, exc: ModelNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content=_error_body(exc))


@app.exception_handler(PreprocessingError)
async def handle_preprocessing_error(request, exc: PreprocessingError) -> JSONResponse:
    return JSONResponse(status_code=422, content=_error_body(exc))


@app.exception_handler(InferenceError)
async def handle_inference_error(request, exc: InferenceError) -> JSONResponse:
    return JSONResponse(status_code=500, content=_error_body(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Always returns 200 — confirms the process is alive."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
async def ready(registry: RegistryDep) -> JSONResponse:
    """Returns 200 when the model registry is ready, 503 otherwise.

    Why a separate /ready from /health?
    - /health = process is alive (used by process supervisors / load balancers
      to decide whether to restart the container).
    - /ready = process can serve traffic (used by orchestrators like Kubernetes
      to decide whether to route requests to this instance).
    Conflating them means a starting-up instance gets traffic before models load.
    """
    if registry.is_ready:
        return JSONResponse(status_code=200, content={"status": "ready"})
    return JSONResponse(status_code=503, content={"status": "not_ready"})


@app.post("/predict", response_model=list[GradeResult], tags=["grading"])
async def predict(
    ingestion: IngestionDep,
    preprocessing: PreprocessingDep,
    grader: GraderDep,
    files: list[UploadFile],
    card_type: CardType = CardType.pokemon,
) -> list[GradeResult]:
    """Grade a batch of card images.

    Pipeline: ImageIngestionService → PreprocessingService → GraderService.
    Each service raises a domain exception on failure; the exception handlers
    above map those to the appropriate HTTP status codes.

    Technical Debt: The preprocessing loop is synchronous and runs in the
    async handler. For large batches, move to asyncio.gather() with
    asyncio.to_thread() wrapping the CPU-bound preprocessing calls.

    Args:
        files: Multipart-uploaded image files (JPEG or PNG, ≥300×420px).
        card_type: Which model to use. Defaults to pokemon.

    Returns:
        List of GradeResult objects — may be shorter than input if any
        cards raised InferenceError (partial-failure resilience).
    """
    # Step 1: Validate format, resolution, and batch size.
    validated = await ingestion.validate_and_load(files)

    # Step 2: Preprocess each validated image into a PreprocessedCard.
    preprocessed = [
        preprocessing.preprocess(raw_bytes)
        for _image_id, raw_bytes in validated
    ]

    # Step 3: Run inference and decode ordinal outputs to GradeResult objects.
    results = await grader.predict(preprocessed, card_type)

    logger.info(
        "predict_complete",
        card_type=card_type.value,
        input_count=len(files),
        output_count=len(results),
    )

    return results
