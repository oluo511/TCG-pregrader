"""
FastAPI dependency injection providers.

Design pattern: Dependency Injection via FastAPI's Depends() system.
Each provider is a callable that FastAPI resolves per-request (or at startup
for app-state-backed singletons). Services are constructed once and stored on
app.state during the lifespan; providers here just retrieve them so route
handlers stay thin and testable.

Why app.state instead of module-level globals?
- Module globals are process-wide singletons that survive test isolation
  boundaries, causing cross-test contamination.
- app.state is scoped to the FastAPI instance, so each TestClient gets a
  clean slate when the lifespan is re-run.
"""

from typing import Annotated

from fastapi import Depends, Request

from pregrader.config import PregraderSettings
from pregrader.registry import ModelRegistry
from pregrader.services.grader import GraderService
from pregrader.services.ingestion import ImageIngestionService
from pregrader.services.preprocessing import PreprocessingService


# ---------------------------------------------------------------------------
# App-state retrievers — pull singletons set during lifespan
# ---------------------------------------------------------------------------


def get_settings(request: Request) -> PregraderSettings:
    return request.app.state.settings


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_ingestion_service(request: Request) -> ImageIngestionService:
    return request.app.state.ingestion_service


def get_preprocessing_service(request: Request) -> PreprocessingService:
    return request.app.state.preprocessing_service


def get_grader_service(request: Request) -> GraderService:
    return request.app.state.grader_service


# ---------------------------------------------------------------------------
# Annotated type aliases — import these in route handlers for clean signatures
# ---------------------------------------------------------------------------

SettingsDep = Annotated[PregraderSettings, Depends(get_settings)]
RegistryDep = Annotated[ModelRegistry, Depends(get_registry)]
IngestionDep = Annotated[ImageIngestionService, Depends(get_ingestion_service)]
PreprocessingDep = Annotated[PreprocessingService, Depends(get_preprocessing_service)]
GraderDep = Annotated[GraderService, Depends(get_grader_service)]
