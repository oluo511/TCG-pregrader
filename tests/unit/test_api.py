"""
Unit tests for FastAPI routes.

Strategy: Use FastAPI's TestClient with a fully mocked app.state so no real
model artifacts or TF are needed. Each test overrides the DI providers via
app.dependency_overrides, which is the idiomatic FastAPI testing pattern.

Why mock at the service layer rather than the route layer?
- Mocking services tests the full request → DI → service → response path.
- Mocking at the route level would skip exception handler wiring, which is
  exactly what we need to verify (HTTP 422, 404, 500, 503 mappings).

Why not use pytest-asyncio here?
- TestClient runs the ASGI app synchronously via httpx under the hood.
  No async test functions needed for route-level tests.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pregrader.api.app import app
from pregrader.api.dependencies import (
    get_grader_service,
    get_ingestion_service,
    get_preprocessing_service,
    get_registry,
)
from pregrader.enums import CardType
from pregrader.exceptions import (
    BatchSizeError,
    InferenceError,
    InvalidImageFormatError,
    ImageResolutionError,
    ModelNotFoundError,
)
from pregrader.schemas import GradeResult, Subgrades

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_VALID_GRADE_RESULT = GradeResult(
    image_id="pikachu.jpg",
    card_type=CardType.pokemon,
    overall_grade=8,
    subgrades=Subgrades(centering=8.0, corners=7.5, edges=8.0, surface=8.5),
    confidence=0.82,
)

# Minimal valid JPEG magic bytes — enough to pass format validation.
_JPEG_BYTES = b"\xff\xd8\xff" + b"\x00" * 10


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ingestion():
    """ImageIngestionService mock — returns one validated (id, bytes) tuple."""
    svc = MagicMock()
    svc.validate_and_load = AsyncMock(return_value=[("pikachu.jpg", _JPEG_BYTES)])
    return svc


@pytest.fixture
def mock_preprocessing():
    """PreprocessingService mock — returns a minimal PreprocessedCard."""
    from pregrader.schemas import CardRegion, PreprocessedCard

    card = PreprocessedCard(
        image_id="pikachu.jpg",
        full_tensor=[[[0.0, 0.0, 0.0]]],
        regions=[
            CardRegion(name=n, tensor=[[[0.0, 0.0, 0.0]]])
            for n in ("centering", "corners", "edges", "surface")
        ],
    )
    svc = MagicMock()
    svc.preprocess.return_value = card
    return svc


@pytest.fixture
def mock_grader():
    """GraderService mock — returns one valid GradeResult."""
    svc = MagicMock()
    svc.predict = AsyncMock(return_value=[_VALID_GRADE_RESULT])
    return svc


@pytest.fixture
def mock_registry():
    """ModelRegistry mock — is_ready=True by default."""
    registry = MagicMock()
    registry.is_ready = True
    return registry


@pytest.fixture
def client(mock_ingestion, mock_preprocessing, mock_grader, mock_registry):
    """TestClient with all services mocked via dependency_overrides."""
    app.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion
    app.dependency_overrides[get_preprocessing_service] = lambda: mock_preprocessing
    app.dependency_overrides[get_grader_service] = lambda: mock_grader
    app.dependency_overrides[get_registry] = lambda: mock_registry
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthRoute:
    def test_health_always_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /ready
# ---------------------------------------------------------------------------

class TestReadyRoute:
    def test_ready_returns_200_when_registry_ready(self, client: TestClient) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_ready_returns_503_when_registry_not_ready(
        self, mock_ingestion, mock_preprocessing, mock_grader
    ) -> None:
        not_ready_registry = MagicMock()
        not_ready_registry.is_ready = False

        app.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion
        app.dependency_overrides[get_preprocessing_service] = lambda: mock_preprocessing
        app.dependency_overrides[get_grader_service] = lambda: mock_grader
        app.dependency_overrides[get_registry] = lambda: not_ready_registry

        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.get("/ready")

        app.dependency_overrides.clear()
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# POST /predict — happy path
# ---------------------------------------------------------------------------

class TestPredictHappyPath:
    def test_valid_request_returns_200_with_grade_results(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/predict",
            files=[("files", ("pikachu.jpg", io.BytesIO(_JPEG_BYTES), "image/jpeg"))],
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 1
        assert body[0]["overall_grade"] == 8
        assert body[0]["card_type"] == "pokemon"
        assert "subgrades" in body[0]

    def test_response_body_matches_grade_result_schema(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/predict",
            files=[("files", ("pikachu.jpg", io.BytesIO(_JPEG_BYTES), "image/jpeg"))],
        )
        body = response.json()[0]
        # Verify all GradeResult fields are present and in range.
        assert 1 <= body["overall_grade"] <= 10
        assert 0.0 <= body["confidence"] <= 1.0
        for dim in ("centering", "corners", "edges", "surface"):
            assert 1.0 <= body["subgrades"][dim] <= 10.0


# ---------------------------------------------------------------------------
# POST /predict — ingestion errors → HTTP 422
# ---------------------------------------------------------------------------

class TestPredictIngestionErrors:
    def test_invalid_image_format_returns_422(
        self, mock_preprocessing, mock_grader, mock_registry
    ) -> None:
        ingestion = MagicMock()
        ingestion.validate_and_load = AsyncMock(
            side_effect=InvalidImageFormatError("Not a JPEG or PNG.")
        )
        app.dependency_overrides[get_ingestion_service] = lambda: ingestion
        app.dependency_overrides[get_preprocessing_service] = lambda: mock_preprocessing
        app.dependency_overrides[get_grader_service] = lambda: mock_grader
        app.dependency_overrides[get_registry] = lambda: mock_registry

        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post(
                "/predict",
                files=[("files", ("bad.gif", io.BytesIO(b"GIF89a"), "image/gif"))],
            )

        app.dependency_overrides.clear()
        assert response.status_code == 422
        body = response.json()
        assert "error" in body
        assert body["type"] == "InvalidImageFormatError"

    def test_image_resolution_too_low_returns_422(
        self, mock_preprocessing, mock_grader, mock_registry
    ) -> None:
        ingestion = MagicMock()
        ingestion.validate_and_load = AsyncMock(
            side_effect=ImageResolutionError("Resolution 100x100 below minimum.")
        )
        app.dependency_overrides[get_ingestion_service] = lambda: ingestion
        app.dependency_overrides[get_preprocessing_service] = lambda: mock_preprocessing
        app.dependency_overrides[get_grader_service] = lambda: mock_grader
        app.dependency_overrides[get_registry] = lambda: mock_registry

        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post(
                "/predict",
                files=[("files", ("tiny.jpg", io.BytesIO(_JPEG_BYTES), "image/jpeg"))],
            )

        app.dependency_overrides.clear()
        assert response.status_code == 422
        assert response.json()["type"] == "ImageResolutionError"

    def test_batch_size_exceeded_returns_422(
        self, mock_preprocessing, mock_grader, mock_registry
    ) -> None:
        ingestion = MagicMock()
        ingestion.validate_and_load = AsyncMock(
            side_effect=BatchSizeError("Batch size 51 exceeds maximum of 50.")
        )
        app.dependency_overrides[get_ingestion_service] = lambda: ingestion
        app.dependency_overrides[get_preprocessing_service] = lambda: mock_preprocessing
        app.dependency_overrides[get_grader_service] = lambda: mock_grader
        app.dependency_overrides[get_registry] = lambda: mock_registry

        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post(
                "/predict",
                files=[("files", (f"card_{i}.jpg", io.BytesIO(_JPEG_BYTES), "image/jpeg")) for i in range(51)],
            )

        app.dependency_overrides.clear()
        assert response.status_code == 422
        assert response.json()["type"] == "BatchSizeError"


# ---------------------------------------------------------------------------
# POST /predict — model not found → HTTP 404
# ---------------------------------------------------------------------------

class TestPredictModelNotFound:
    def test_unknown_card_type_returns_404_with_structured_body(
        self, mock_ingestion, mock_preprocessing, mock_registry
    ) -> None:
        grader = MagicMock()
        grader.predict = AsyncMock(
            side_effect=ModelNotFoundError(
                "No model loaded for card_type='one_piece'. Loaded types: ['pokemon']"
            )
        )
        app.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion
        app.dependency_overrides[get_preprocessing_service] = lambda: mock_preprocessing
        app.dependency_overrides[get_grader_service] = lambda: grader
        app.dependency_overrides[get_registry] = lambda: mock_registry

        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post(
                "/predict",
                files=[("files", ("card.jpg", io.BytesIO(_JPEG_BYTES), "image/jpeg"))],
                data={"card_type": "one_piece"},
            )

        app.dependency_overrides.clear()
        assert response.status_code == 404
        body = response.json()
        assert body["type"] == "ModelNotFoundError"
        assert "one_piece" in body["error"]

    def test_error_body_never_contains_raw_traceback(
        self, mock_ingestion, mock_preprocessing, mock_registry
    ) -> None:
        grader = MagicMock()
        grader.predict = AsyncMock(
            side_effect=ModelNotFoundError("No model for 'sports'.")
        )
        app.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion
        app.dependency_overrides[get_preprocessing_service] = lambda: mock_preprocessing
        app.dependency_overrides[get_grader_service] = lambda: grader
        app.dependency_overrides[get_registry] = lambda: mock_registry

        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post(
                "/predict",
                files=[("files", ("card.jpg", io.BytesIO(_JPEG_BYTES), "image/jpeg"))],
                data={"card_type": "sports"},
            )

        app.dependency_overrides.clear()
        body_str = response.text
        # Raw tracebacks contain "Traceback" — must never appear in response.
        assert "Traceback" not in body_str
        assert "traceback" not in body_str


# ---------------------------------------------------------------------------
# POST /predict — inference error → HTTP 500
# ---------------------------------------------------------------------------

class TestPredictInferenceError:
    def test_inference_error_returns_500_with_structured_body(
        self, mock_ingestion, mock_preprocessing, mock_registry
    ) -> None:
        grader = MagicMock()
        grader.predict = AsyncMock(
            side_effect=InferenceError("GPU out of memory.")
        )
        app.dependency_overrides[get_ingestion_service] = lambda: mock_ingestion
        app.dependency_overrides[get_preprocessing_service] = lambda: mock_preprocessing
        app.dependency_overrides[get_grader_service] = lambda: grader
        app.dependency_overrides[get_registry] = lambda: mock_registry

        with TestClient(app, raise_server_exceptions=False) as c:
            response = c.post(
                "/predict",
                files=[("files", ("card.jpg", io.BytesIO(_JPEG_BYTES), "image/jpeg"))],
            )

        app.dependency_overrides.clear()
        assert response.status_code == 500
        body = response.json()
        assert body["type"] == "InferenceError"
        assert "error" in body
