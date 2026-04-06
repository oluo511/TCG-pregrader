# Feature: pokemon-card-pregrader, Property 4: Preprocessor output shape invariant
# Feature: pokemon-card-pregrader, Property 5: Card region extraction completeness

"""
Property-based tests for PreprocessingService.

Strategy: Generate random valid images of varying sizes using PIL, then
assert universal invariants on the PreprocessingService output. No mocking
of the service itself — we test the real pipeline end-to-end.

Why vary image dimensions? The resize step must produce (312, 224, 3)
regardless of input size. Testing across many random dimensions gives
confidence that the resize logic is not accidentally correct only for
specific inputs.
"""

import io

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from pregrader.services.preprocessing import PreprocessingService

# ---------------------------------------------------------------------------
# Shared strategy: generate valid card images as JPEG bytes
# ---------------------------------------------------------------------------

# Minimum dimensions per Requirement 1.2 (ingestion gate).
# We generate images above this threshold so they would pass ingestion.
# Upper bound is kept modest to keep test runtime reasonable.
_image_strategy = st.builds(
    lambda w, h, r, g, b: _make_jpeg_bytes(w, h, r, g, b),
    w=st.integers(min_value=300, max_value=800),
    h=st.integers(min_value=420, max_value=1000),
    r=st.integers(min_value=0, max_value=255),
    g=st.integers(min_value=0, max_value=255),
    b=st.integers(min_value=0, max_value=255),
)


def _make_jpeg_bytes(
    width: int, height: int, r: int = 128, g: int = 128, b: int = 128
) -> bytes:
    """Create a solid-color JPEG of the given dimensions."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(r, g, b)).save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Property 4: Preprocessor output shape invariant
# Validates: Requirements 2.1, 2.2
# ---------------------------------------------------------------------------

@given(image_bytes=_image_strategy)
@settings(max_examples=100)
def test_property4_full_tensor_shape_invariant(image_bytes: bytes) -> None:
    """For any valid input image (regardless of original dimensions), the
    preprocessed full_tensor must have shape (312, 224, 3) and all pixel
    values must lie in [0.0, 1.0].

    **Validates: Requirements 2.1, 2.2**
    """
    service = PreprocessingService()
    result = service.preprocess(image_bytes, image_id="prop4_test.jpg")

    arr = np.array(result.full_tensor)

    # Shape invariant: always (H=312, W=224, C=3)
    assert arr.shape == (312, 224, 3), (
        f"full_tensor shape {arr.shape} != (312, 224, 3)"
    )

    # Pixel range invariant: all values in [0.0, 1.0]
    assert arr.min() >= 0.0, f"Pixel min {arr.min()} < 0.0"
    assert arr.max() <= 1.0, f"Pixel max {arr.max()} > 1.0"


# ---------------------------------------------------------------------------
# Property 5: Card region extraction completeness
# Validates: Requirements 2.5
# ---------------------------------------------------------------------------

@given(image_bytes=_image_strategy)
@settings(max_examples=100)
def test_property5_region_extraction_completeness(image_bytes: bytes) -> None:
    """For any valid input image, the result must contain exactly four
    CardRegion objects with names {"centering", "corners", "edges", "surface"}.

    **Validates: Requirements 2.5**
    """
    service = PreprocessingService()
    result = service.preprocess(image_bytes, image_id="prop5_test.jpg")

    # Cardinality invariant
    assert len(result.regions) == 4, (
        f"Expected 4 regions, got {len(result.regions)}"
    )

    # Name completeness invariant
    names = {r.name for r in result.regions}
    assert names == {"centering", "corners", "edges", "surface"}, (
        f"Region names {names} != {{'centering', 'corners', 'edges', 'surface'}}"
    )

    # All region tensors must also have values in [0.0, 1.0]
    for region in result.regions:
        arr = np.array(region.tensor)
        assert arr.min() >= 0.0, (
            f"Region '{region.name}' has pixel min {arr.min()} < 0.0"
        )
        assert arr.max() <= 1.0, (
            f"Region '{region.name}' has pixel max {arr.max()} > 1.0"
        )
