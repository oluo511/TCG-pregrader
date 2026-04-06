"""
Unit tests for ImageIngestionService.

Covers the four concrete scenarios from Task 4.4:
  1. BatchSizeError raised before any file is opened when batch > 50
  2. InvalidImageFormatError with a GIF magic byte payload
  3. ImageResolutionError with a 100×100 JPEG
  4. Successful load returns (image_id, bytes) tuples

Uses AsyncMock stubs for UploadFile — no FastAPI runtime required.
"""

import io
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from pregrader.config import PregraderSettings
from pregrader.exceptions import (
    BatchSizeError,
    ImageResolutionError,
    InvalidImageFormatError,
)
from pregrader.services.ingestion import ImageIngestionService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings() -> PregraderSettings:
    return PregraderSettings(
        pokemon_model_artifact_path="/tmp/fake_model",
        max_batch_size=50,
    )


@pytest.fixture
def service(settings: PregraderSettings) -> ImageIngestionService:
    return ImageIngestionService(settings)


def _make_upload_file(data: bytes, filename: str = "card.jpg") -> AsyncMock:
    """Minimal UploadFile stub."""
    mock = AsyncMock()
    mock.filename = filename
    mock.read = AsyncMock(return_value=data)
    return mock


def _jpeg_bytes(width: int = 300, height: int = 420) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(100, 100, 100)).save(buf, format="JPEG")
    return buf.getvalue()


def _png_bytes(width: int = 300, height: int = 420) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(100, 100, 100)).save(buf, format="PNG")
    return buf.getvalue()


# GIF magic bytes: 47 49 46 38 (GIF8)
_GIF_PAYLOAD = b"GIF89a" + b"\x00" * 20


# ---------------------------------------------------------------------------
# Test 1: BatchSizeError raised before any file I/O
# Requirements: 1.5, 1.6
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_size_error_raised_before_file_read(
    service: ImageIngestionService,
) -> None:
    """BatchSizeError must be raised when batch > 50, and no file must be read."""
    files = [_make_upload_file(b"placeholder", f"img_{i}.jpg") for i in range(51)]

    with pytest.raises(BatchSizeError) as exc_info:
        await service.validate_and_load(files)

    assert "51" in str(exc_info.value)
    assert "50" in str(exc_info.value)

    # Critical: no file I/O should have occurred
    for f in files:
        f.read.assert_not_called()


@pytest.mark.asyncio
async def test_batch_size_error_message_contains_counts(
    service: ImageIngestionService,
) -> None:
    """Error message must include both the submitted count and the limit."""
    files = [_make_upload_file(b"x", f"img_{i}.jpg") for i in range(75)]

    with pytest.raises(BatchSizeError) as exc_info:
        await service.validate_and_load(files)

    msg = str(exc_info.value)
    assert "75" in msg
    assert "50" in msg


# ---------------------------------------------------------------------------
# Test 2: InvalidImageFormatError with GIF magic bytes
# Requirements: 1.1, 1.4
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_format_gif_rejected(service: ImageIngestionService) -> None:
    """A GIF payload must raise InvalidImageFormatError."""
    files = [_make_upload_file(_GIF_PAYLOAD, "card.gif")]

    with pytest.raises(InvalidImageFormatError) as exc_info:
        await service.validate_and_load(files)

    assert "card.gif" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invalid_format_random_bytes_rejected(
    service: ImageIngestionService,
) -> None:
    """Arbitrary bytes with no valid magic prefix must raise InvalidImageFormatError."""
    payload = b"\x00\x01\x02\x03" + b"\xff" * 100
    files = [_make_upload_file(payload, "corrupt.bin")]

    with pytest.raises(InvalidImageFormatError):
        await service.validate_and_load(files)


# ---------------------------------------------------------------------------
# Test 3: ImageResolutionError with a 100×100 JPEG
# Requirements: 1.2, 1.3
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolution_error_small_jpeg(service: ImageIngestionService) -> None:
    """A 100×100 JPEG must raise ImageResolutionError."""
    files = [_make_upload_file(_jpeg_bytes(100, 100), "small.jpg")]

    with pytest.raises(ImageResolutionError) as exc_info:
        await service.validate_and_load(files)

    msg = str(exc_info.value)
    assert "small.jpg" in msg
    assert "100" in msg


@pytest.mark.asyncio
async def test_resolution_error_wide_but_short(service: ImageIngestionService) -> None:
    """An image that meets width but not height must still raise ImageResolutionError."""
    files = [_make_upload_file(_jpeg_bytes(400, 100), "wide_short.jpg")]

    with pytest.raises(ImageResolutionError):
        await service.validate_and_load(files)


@pytest.mark.asyncio
async def test_resolution_error_tall_but_narrow(service: ImageIngestionService) -> None:
    """An image that meets height but not width must still raise ImageResolutionError."""
    files = [_make_upload_file(_jpeg_bytes(100, 500), "narrow_tall.jpg")]

    with pytest.raises(ImageResolutionError):
        await service.validate_and_load(files)


# ---------------------------------------------------------------------------
# Test 4: Successful load returns (image_id, bytes) tuples
# Requirements: 1.1, 1.2, 1.5
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_jpeg_returns_tuple(service: ImageIngestionService) -> None:
    """A valid JPEG at minimum resolution must return (filename, bytes)."""
    data = _jpeg_bytes(300, 420)
    files = [_make_upload_file(data, "pikachu.jpg")]

    results = await service.validate_and_load(files)

    assert len(results) == 1
    image_id, raw_bytes = results[0]
    assert image_id == "pikachu.jpg"
    assert raw_bytes == data


@pytest.mark.asyncio
async def test_successful_png_returns_tuple(service: ImageIngestionService) -> None:
    """A valid PNG at minimum resolution must return (filename, bytes)."""
    data = _png_bytes(300, 420)
    files = [_make_upload_file(data, "charizard.png")]

    results = await service.validate_and_load(files)

    assert len(results) == 1
    image_id, raw_bytes = results[0]
    assert image_id == "charizard.png"
    assert raw_bytes == data


@pytest.mark.asyncio
async def test_successful_batch_preserves_order(service: ImageIngestionService) -> None:
    """Results must be returned in the same order as the input files."""
    filenames = [f"card_{i}.jpg" for i in range(5)]
    files = [_make_upload_file(_jpeg_bytes(300, 420), name) for name in filenames]

    results = await service.validate_and_load(files)

    assert [r[0] for r in results] == filenames


@pytest.mark.asyncio
async def test_successful_load_at_exact_boundary_resolution(
    service: ImageIngestionService,
) -> None:
    """Images at exactly 300×420 must be accepted (boundary is inclusive)."""
    data = _jpeg_bytes(300, 420)
    files = [_make_upload_file(data, "boundary.jpg")]

    results = await service.validate_and_load(files)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_successful_load_at_max_batch_size(
    service: ImageIngestionService,
) -> None:
    """A batch of exactly 50 images must be accepted."""
    data = _jpeg_bytes(300, 420)
    files = [_make_upload_file(data, f"card_{i}.jpg") for i in range(50)]

    results = await service.validate_and_load(files)
    assert len(results) == 50


@pytest.mark.asyncio
async def test_image_id_from_filename(service: ImageIngestionService) -> None:
    """image_id must be derived from UploadFile.filename."""
    data = _jpeg_bytes(300, 420)
    files = [_make_upload_file(data, "my_special_card.jpg")]

    results = await service.validate_and_load(files)
    assert results[0][0] == "my_special_card.jpg"
