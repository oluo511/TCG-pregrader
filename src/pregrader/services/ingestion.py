"""
ImageIngestionService — validates and loads raw uploaded image files.

Design pattern: Fail-fast validation pipeline.
Each gate is ordered by cost: batch size (O(1) count) → magic bytes
(read first 4 bytes) → resolution (decode image header). We never do
expensive I/O on a batch that will be rejected for a cheap reason.

Technical Debt: UploadFile.read() loads the entire file into memory.
At scale (large images, large batches), stream to a temp file and use
memory-mapped I/O instead. For MVP with max_batch_size=50 and typical
card photos (~2–5 MB each), in-memory is acceptable.
"""

import io
from typing import Final

from fastapi import UploadFile
from PIL import Image

from pregrader.config import PregraderSettings
from pregrader.exceptions import (
    BatchSizeError,
    ImageResolutionError,
    InvalidImageFormatError,
)
from pregrader.logging_config import get_logger

# Magic byte signatures for supported formats.
# JPEG: FF D8 FF (first 3 bytes)
# PNG:  89 50 4E 47 (first 4 bytes — "\x89PNG")
_JPEG_MAGIC: Final[bytes] = b"\xff\xd8\xff"
_PNG_MAGIC: Final[bytes] = b"\x89PNG"

# Minimum card image dimensions per Requirements 1.2
_MIN_WIDTH: Final[int] = 300
_MIN_HEIGHT: Final[int] = 420


class ImageIngestionService:
    """Validates and loads raw uploaded image files before preprocessing.

    Validation order (fail-fast, cheapest first):
      1. Batch size ≤ max_batch_size  → BatchSizeError
      2. Magic bytes (JPEG / PNG)     → InvalidImageFormatError
      3. Decoded resolution ≥ 300×420 → ImageResolutionError

    Returns a list of (image_id, raw_bytes) tuples for all files that
    pass all three gates. Any single failure raises immediately — the
    entire batch is rejected, consistent with the design's fail-fast
    strategy.
    """

    def __init__(self, settings: PregraderSettings) -> None:
        self._max_batch_size = settings.max_batch_size
        self._logger = get_logger(service="ingestion")

    async def validate_and_load(
        self, files: list[UploadFile]
    ) -> list[tuple[str, bytes]]:
        """Validate all files and return (image_id, raw_bytes) pairs.

        Args:
            files: Uploaded files from the multipart request.

        Returns:
            List of (image_id, raw_bytes) in the same order as input.

        Raises:
            BatchSizeError: Batch exceeds max_batch_size (checked first,
                before any file I/O).
            InvalidImageFormatError: A file's magic bytes are not JPEG/PNG.
            ImageResolutionError: A file's decoded dimensions are below
                the 300×420 minimum.
        """
        # --- Gate 1: Batch size (O(1), no I/O) ---
        # Checked before opening any file so oversized batches never
        # trigger unnecessary disk/network reads.
        if len(files) > self._max_batch_size:
            self._logger.error(
                "batch_size_exceeded",
                batch_size=len(files),
                max_batch_size=self._max_batch_size,
            )
            raise BatchSizeError(
                f"Batch size {len(files)} exceeds the maximum of "
                f"{self._max_batch_size} images per request."
            )

        results: list[tuple[str, bytes]] = []

        for upload_file in files:
            image_id = upload_file.filename or "unknown"
            raw_bytes = await upload_file.read()

            # --- Gate 2: Magic bytes format check ---
            self._validate_magic_bytes(raw_bytes, image_id)

            # --- Gate 3: Resolution check (requires PIL decode) ---
            self._validate_resolution(raw_bytes, image_id)

            results.append((image_id, raw_bytes))

        return results

    def _validate_magic_bytes(self, raw_bytes: bytes, image_id: str) -> None:
        """Reject files whose leading bytes don't match JPEG or PNG signatures.

        Why magic bytes over MIME type? MIME is client-supplied and trivially
        spoofed. Magic bytes are read from the file content itself.

        Raises:
            InvalidImageFormatError: If neither JPEG nor PNG signature matches.
        """
        if raw_bytes[:3] == _JPEG_MAGIC or raw_bytes[:4] == _PNG_MAGIC:
            return

        self._logger.warning(
            "invalid_image_format",
            image_id=image_id,
            leading_bytes=raw_bytes[:4].hex(),
        )
        raise InvalidImageFormatError(
            f"Image '{image_id}' is not a valid JPEG or PNG. "
            f"Expected magic bytes FF D8 FF (JPEG) or 89 50 4E 47 (PNG), "
            f"got: {raw_bytes[:4].hex().upper()!r}."
        )

    def _validate_resolution(self, raw_bytes: bytes, image_id: str) -> None:
        """Reject images below the minimum 300×420 pixel threshold.

        Uses PIL to decode only the image header (not full pixel data) via
        Image.open() — PIL is lazy and won't decompress pixels until
        getdata() or similar is called.

        Raises:
            ImageResolutionError: If width < 300 or height < 420.
        """
        with Image.open(io.BytesIO(raw_bytes)) as img:
            width, height = img.size

        if width < _MIN_WIDTH or height < _MIN_HEIGHT:
            self._logger.warning(
                "image_resolution_too_low",
                image_id=image_id,
                width=width,
                height=height,
                min_width=_MIN_WIDTH,
                min_height=_MIN_HEIGHT,
            )
            raise ImageResolutionError(
                f"Image '{image_id}' resolution {width}×{height} is below "
                f"the minimum required {_MIN_WIDTH}×{_MIN_HEIGHT} pixels."
            )
