"""
Exception hierarchy for the TCG Pre-Grader.

Why a custom hierarchy instead of plain ValueError/RuntimeError?
- Allows FastAPI exception handlers to map domain errors to HTTP codes
  without catching overly broad built-in exceptions.
- Enables structured logging with exception type as a field.
- Makes the error contract explicit in function signatures.
"""


class PregraderError(Exception):
    """Base class for all domain exceptions. Catch this at the API boundary
    to guarantee no raw tracebacks leak to clients."""


# ---------------------------------------------------------------------------
# Image Ingestion Errors
# ---------------------------------------------------------------------------


class ImageIngestionError(PregraderError):
    """Raised when an uploaded image fails any pre-processing validation gate."""


class InvalidImageFormatError(ImageIngestionError):
    """Magic bytes do not match JPEG (FF D8 FF) or PNG (89 50 4E 47).

    Why magic bytes instead of MIME type? MIME types are client-supplied and
    trivially spoofed. Magic bytes are read from the file itself.
    """


class ImageResolutionError(ImageIngestionError):
    """Image dimensions are below the minimum required (300×420 px).

    Raised after format validation passes — we only decode the image header
    to check dimensions, not the full pixel data.
    """


class BatchSizeError(ImageIngestionError):
    """Batch exceeds max_batch_size (default 50).

    Checked *before* opening any file so we never do unnecessary I/O on an
    oversized batch.
    """


# ---------------------------------------------------------------------------
# Preprocessing Errors
# ---------------------------------------------------------------------------


class PreprocessingError(PregraderError):
    """Raised when the preprocessing pipeline cannot produce a valid tensor.

    Note: perspective correction failure is NOT raised as an error — it logs
    a WARNING and continues with the uncorrected image. Only unrecoverable
    failures (e.g., corrupt pixel data) raise this.
    """


# ---------------------------------------------------------------------------
# Inference Errors
# ---------------------------------------------------------------------------


class InferenceError(PregraderError):
    """Raised when the model fails to produce a prediction for an image.

    Per-image InferenceErrors are caught at the batch level, logged with
    image_id, and the remaining images continue processing.
    """


class ModelNotFoundError(InferenceError):
    """Requested card_type has no loaded model in the ModelRegistry.

    Maps to HTTP 404. The message always includes the requested card_type
    value so the client knows exactly what was missing.

    Example message:
        "No model loaded for card_type='one_piece'. Loaded types: ['pokemon']"
    """


# ---------------------------------------------------------------------------
# Configuration Errors
# ---------------------------------------------------------------------------


class ConfigurationError(PregraderError):
    """Raised at startup when a required config value is missing or invalid.

    Wraps Pydantic ValidationError so callers don't need to import Pydantic
    just to catch config failures. The message always includes the field name.
    """
