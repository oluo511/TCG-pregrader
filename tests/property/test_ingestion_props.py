# Feature: pokemon-card-pregrader, Property 1: Image format acceptance
# Feature: pokemon-card-pregrader, Property 2: Resolution threshold enforcement
# Feature: pokemon-card-pregrader, Property 3: Batch size boundary

"""
Property-based tests for ImageIngestionService.

Strategy: Use in-memory PIL images rather than real files to keep tests
hermetic and fast. UploadFile is stubbed with a minimal async-compatible
class — no FastAPI/Starlette runtime needed.
"""

import io
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from pregrader.config import PregraderSettings
from pregrader.exceptions import (
    BatchSizeError,
    ImageResolutionError,
    InvalidImageFormatError,
)
from pregrader.services.ingestion import ImageIngestionService, _JPEG_MAGIC, _PNG_MAGIC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_SETTINGS = PregraderSettings(
    pokemon_model_artifact_path="/tmp/fake_model",
    max_batch_size=50,
)


def _make_settings(max_batch_size: int = 50) -> PregraderSettings:
    return PregraderSettings(
        pokemon_model_artifact_path="/tmp/fake_model",
        max_batch_size=max_batch_size,
    )


def _pil_image_bytes(width: int, height: int, fmt: str = "JPEG") -> bytes:
    """Create a minimal in-memory image of the given dimensions."""
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_upload_file(data: bytes, filename: str = "test.jpg") -> AsyncMock:
    """Minimal UploadFile stub — read() returns the provided bytes."""
    mock = AsyncMock()
    mock.filename = filename
    mock.read = AsyncMock(return_value=data)
    return mock


# ---------------------------------------------------------------------------
# Property 1: Image format acceptance
# Validates: Requirements 1.1, 1.4
# ---------------------------------------------------------------------------

# Strategy: generate random bytes, then prepend a magic-byte prefix drawn
# from {valid JPEG, valid PNG, random 4 bytes}. The service must accept iff
# the prefix matches JPEG or PNG.
_VALID_MAGIC = st.sampled_from([_JPEG_MAGIC, _PNG_MAGIC])
_INVALID_MAGIC = st.binary(min_size=4, max_size=4).filter(
    lambda b: not b[:3] == _JPEG_MAGIC and not b[:4] == _PNG_MAGIC
)


@given(suffix=st.binary(min_size=0, max_size=64))
@settings(max_examples=100)
def test_property1_valid_magic_bytes_accepted(suffix: bytes) -> None:
    """Any byte sequence starting with a valid JPEG or PNG magic prefix
    must pass the format gate (magic bytes check only — not a full decode).

    **Validates: Requirements 1.1, 1.4**
    """
    # We test the private helper directly to isolate the magic-bytes gate
    # from the resolution gate (which requires a decodable image).
    service = ImageIngestionService(_DEFAULT_SETTINGS)

    for magic in (_JPEG_MAGIC, _PNG_MAGIC):
        payload = magic + suffix
        # Should not raise — magic bytes are valid
        service._validate_magic_bytes(payload, image_id="test.img")


@given(prefix=_INVALID_MAGIC, suffix=st.binary(min_size=0, max_size=64))
@settings(max_examples=100)
def test_property1_invalid_magic_bytes_rejected(prefix: bytes, suffix: bytes) -> None:
    """Any byte sequence whose first 4 bytes don't match JPEG or PNG must
    raise InvalidImageFormatError.

    **Validates: Requirements 1.1, 1.4**
    """
    service = ImageIngestionService(_DEFAULT_SETTINGS)
    payload = prefix + suffix

    with pytest.raises(InvalidImageFormatError):
        service._validate_magic_bytes(payload, image_id="bad.gif")


# ---------------------------------------------------------------------------
# Property 2: Resolution threshold enforcement
# Validates: Requirements 1.2, 1.3
# ---------------------------------------------------------------------------

# Dimensions strategy: sample widths and heights independently around the
# threshold so we get good coverage of the boundary region.
_WIDTH_STRATEGY = st.integers(min_value=1, max_value=600)
_HEIGHT_STRATEGY = st.integers(min_value=1, max_value=840)


@given(width=_WIDTH_STRATEGY, height=_HEIGHT_STRATEGY)
@settings(max_examples=100)
def test_property2_resolution_threshold(width: int, height: int) -> None:
    """Images below 300×420 must raise ImageResolutionError; images at or
    above the threshold must pass.

    **Validates: Requirements 1.2, 1.3**
    """
    service = ImageIngestionService(_DEFAULT_SETTINGS)
    raw_bytes = _pil_image_bytes(width, height, fmt="JPEG")

    below_threshold = width < 300 or height < 420

    if below_threshold:
        with pytest.raises(ImageResolutionError) as exc_info:
            service._validate_resolution(raw_bytes, image_id=f"{width}x{height}.jpg")
        # Error message must identify the image and the requirement
        assert str(width) in str(exc_info.value)
        assert str(height) in str(exc_info.value)
    else:
        # Must not raise
        service._validate_resolution(raw_bytes, image_id=f"{width}x{height}.jpg")


# ---------------------------------------------------------------------------
# Property 3: Batch size boundary
# Validates: Requirements 1.5, 1.6
# ---------------------------------------------------------------------------

@given(batch_size=st.integers(min_value=1, max_value=100))
@settings(max_examples=100)
@pytest.mark.asyncio
async def test_property3_batch_size_boundary(batch_size: int) -> None:
    """Batches of ≤ 50 images must be accepted; batches of > 50 must raise
    BatchSizeError before any file is read (i.e., read() is never called).

    **Validates: Requirements 1.5, 1.6**
    """
    service = ImageIngestionService(_DEFAULT_SETTINGS)

    # Build stub UploadFile list — read() should never be called for oversized batches
    files = [_make_upload_file(b"placeholder", f"img_{i}.jpg") for i in range(batch_size)]

    if batch_size > 50:
        with pytest.raises(BatchSizeError):
            await service.validate_and_load(files)
        # Confirm no file was opened — read() must not have been called
        for f in files:
            f.read.assert_not_called()
    else:
        # For valid batch sizes, we need real image bytes to pass subsequent gates.
        # Replace stubs with actual valid JPEG bytes at minimum resolution.
        valid_bytes = _pil_image_bytes(300, 420, fmt="JPEG")
        valid_files = [
            _make_upload_file(valid_bytes, f"img_{i}.jpg") for i in range(batch_size)
        ]
        results = await service.validate_and_load(valid_files)
        assert len(results) == batch_size
