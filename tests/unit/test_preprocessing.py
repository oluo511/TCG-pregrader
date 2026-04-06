"""
Unit tests for PreprocessingService.

Covers Task 5.3:
  1. Output tensor shape (312, 224, 3) and pixel range [0.0, 1.0]
  2. Perspective correction failure path — logs WARNING, does not raise
  3. All four region names present in output

Uses synthetic PIL images — no real card photos required.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pregrader.schemas import PreprocessedCard
from pregrader.services.preprocessing import PreprocessingService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(width: int = 400, height: int = 500) -> bytes:
    """Create a minimal in-memory JPEG of the given dimensions."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 80, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width: int = 400, height: int = 500) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(50, 150, 250)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def service() -> PreprocessingService:
    return PreprocessingService()


# ---------------------------------------------------------------------------
# Test 1: Output tensor shape and pixel range
# Requirements: 2.1, 2.2
# ---------------------------------------------------------------------------

def test_full_tensor_shape_is_312_224_3(service: PreprocessingService) -> None:
    """full_tensor must have shape (312, 224, 3) regardless of input size."""
    result = service.preprocess(_make_jpeg_bytes(400, 500), image_id="test.jpg")

    arr = np.array(result.full_tensor)
    assert arr.shape == (312, 224, 3), f"Expected (312, 224, 3), got {arr.shape}"


def test_full_tensor_pixel_range_is_0_to_1(service: PreprocessingService) -> None:
    """All pixel values in full_tensor must be in [0.0, 1.0]."""
    result = service.preprocess(_make_jpeg_bytes(300, 420), image_id="test.jpg")

    arr = np.array(result.full_tensor)
    assert arr.min() >= 0.0, f"Min pixel value {arr.min()} is below 0.0"
    assert arr.max() <= 1.0, f"Max pixel value {arr.max()} is above 1.0"


def test_full_tensor_dtype_is_float(service: PreprocessingService) -> None:
    """full_tensor values must be floats, not integers."""
    result = service.preprocess(_make_jpeg_bytes(), image_id="test.jpg")
    arr = np.array(result.full_tensor)
    assert np.issubdtype(arr.dtype, np.floating)


def test_output_is_preprocessed_card_instance(service: PreprocessingService) -> None:
    """preprocess() must return a PreprocessedCard."""
    result = service.preprocess(_make_jpeg_bytes(), image_id="card_001.jpg")
    assert isinstance(result, PreprocessedCard)
    assert result.image_id == "card_001.jpg"


def test_png_input_produces_correct_shape(service: PreprocessingService) -> None:
    """PNG input must produce the same (312, 224, 3) output shape as JPEG."""
    result = service.preprocess(_make_png_bytes(600, 800), image_id="card.png")
    arr = np.array(result.full_tensor)
    assert arr.shape == (312, 224, 3)


def test_small_input_image_still_produces_correct_shape(
    service: PreprocessingService,
) -> None:
    """Even a 300×420 minimum-resolution image must resize to (312, 224, 3)."""
    result = service.preprocess(_make_jpeg_bytes(300, 420), image_id="small.jpg")
    arr = np.array(result.full_tensor)
    assert arr.shape == (312, 224, 3)


# ---------------------------------------------------------------------------
# Test 2: Perspective correction failure path
# Requirements: 2.3, 2.4
# ---------------------------------------------------------------------------

def test_perspective_correction_failure_does_not_raise(
    service: PreprocessingService,
) -> None:
    """If perspective correction raises internally, preprocess() must not raise.
    It must return a valid PreprocessedCard with the uncorrected image.
    """
    # Patch cv2.findContours to simulate a detection failure (no contours found)
    with patch("pregrader.services.preprocessing.cv2.findContours", return_value=([], None)):
        result = service.preprocess(_make_jpeg_bytes(), image_id="no_contour.jpg")

    assert isinstance(result, PreprocessedCard)
    arr = np.array(result.full_tensor)
    assert arr.shape == (312, 224, 3)


def test_perspective_correction_exception_does_not_raise(
    service: PreprocessingService,
) -> None:
    """If cv2 raises an unexpected exception, preprocess() must still succeed."""
    with patch(
        "pregrader.services.preprocessing.cv2.cvtColor",
        side_effect=RuntimeError("simulated cv2 crash"),
    ):
        result = service.preprocess(_make_jpeg_bytes(), image_id="cv2_crash.jpg")

    assert isinstance(result, PreprocessedCard)


def test_perspective_correction_failure_logs_warning(
    service: PreprocessingService,
    capsys: pytest.CaptureFixture,
) -> None:
    """A perspective correction failure must emit a WARNING log entry to stdout."""
    with patch("pregrader.services.preprocessing.cv2.findContours", return_value=([], None)):
        service.preprocess(_make_jpeg_bytes(), image_id="warn_test.jpg")

    # structlog emits JSON to stdout — check that a warning was written
    captured = capsys.readouterr()
    assert "warning" in captured.out.lower() or "perspective_correction" in captured.out


# ---------------------------------------------------------------------------
# Test 3: All four region names present in output
# Requirements: 2.5
# ---------------------------------------------------------------------------

def test_four_regions_returned(service: PreprocessingService) -> None:
    """preprocess() must return exactly four CardRegion objects."""
    result = service.preprocess(_make_jpeg_bytes(), image_id="regions.jpg")
    assert len(result.regions) == 4


def test_all_region_names_present(service: PreprocessingService) -> None:
    """The four regions must be named centering, corners, edges, surface."""
    result = service.preprocess(_make_jpeg_bytes(), image_id="regions.jpg")
    names = {r.name for r in result.regions}
    assert names == {"centering", "corners", "edges", "surface"}


def test_region_tensors_have_float_values(service: PreprocessingService) -> None:
    """All region tensor values must be floats in [0.0, 1.0]."""
    result = service.preprocess(_make_jpeg_bytes(), image_id="regions.jpg")
    for region in result.regions:
        arr = np.array(region.tensor)
        assert arr.min() >= 0.0, f"Region '{region.name}' has min {arr.min()} < 0.0"
        assert arr.max() <= 1.0, f"Region '{region.name}' has max {arr.max()} > 1.0"


def test_centering_region_shape(service: PreprocessingService) -> None:
    """centering region must be 80% of 312×224 = (249, 179, 3)."""
    result = service.preprocess(_make_jpeg_bytes(), image_id="centering.jpg")
    centering = next(r for r in result.regions if r.name == "centering")
    arr = np.array(centering.tensor)
    expected_h = int(312 * 0.80)  # 249
    expected_w = int(224 * 0.80)  # 179
    assert arr.shape == (expected_h, expected_w, 3), (
        f"centering shape {arr.shape} != ({expected_h}, {expected_w}, 3)"
    )


def test_surface_region_shape(service: PreprocessingService) -> None:
    """surface region must be 60% of 312×224 = (187, 134, 3)."""
    result = service.preprocess(_make_jpeg_bytes(), image_id="surface.jpg")
    surface = next(r for r in result.regions if r.name == "surface")
    arr = np.array(surface.tensor)
    expected_h = int(312 * 0.60)  # 187
    expected_w = int(224 * 0.60)  # 134
    assert arr.shape == (expected_h, expected_w, 3), (
        f"surface shape {arr.shape} != ({expected_h}, {expected_w}, 3)"
    )


def test_corners_region_has_four_patches(service: PreprocessingService) -> None:
    """corners region must concatenate 4 patches horizontally → shape (40, 160, 3)."""
    result = service.preprocess(_make_jpeg_bytes(), image_id="corners.jpg")
    corners = next(r for r in result.regions if r.name == "corners")
    arr = np.array(corners.tensor)
    # Shape: (40, 40*4, 3) = (40, 160, 3)
    assert arr.shape[0] == 40, f"Expected height 40, got {arr.shape[0]}"
    assert arr.shape[1] == 160, f"Expected width 160 (4×40), got {arr.shape[1]}"
    assert arr.shape[2] == 3


def test_edges_region_has_four_strips(service: PreprocessingService) -> None:
    """edges region must concatenate 4 strips horizontally → 3D tensor."""
    result = service.preprocess(_make_jpeg_bytes(), image_id="edges.jpg")
    edges = next(r for r in result.regions if r.name == "edges")
    arr = np.array(edges.tensor)
    # Must be 3D (H, W, 3) — not 4D
    assert arr.ndim == 3, f"Expected 3D tensor, got {arr.ndim}D"
    assert arr.shape[2] == 3
