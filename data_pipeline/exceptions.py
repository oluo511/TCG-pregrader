"""
Pipeline exception hierarchy.

All pipeline errors inherit from PipelineError so callers can catch the base
class for broad handling or specific subclasses for targeted recovery.
No silent failures — every error carries enough context to log and act on.
"""


class PipelineError(Exception):
    """Base class for all training data pipeline errors."""


class QuotaExhaustedError(PipelineError):
    """
    Raised by PSAClient when the daily API call quota has been exhausted.
    The orchestrator catches this to halt further PSA calls for the run
    without stopping image downloads already in flight.
    """


class CertLookupError(PipelineError):
    """
    Raised by PSAClient on a non-retryable 4xx response (excluding 429).
    Carries the cert number and HTTP status code so the operator can
    investigate bad cert numbers vs. API permission issues.
    """

    def __init__(self, cert_number: str, status_code: int) -> None:
        self.cert_number = cert_number
        self.status_code = status_code
        super().__init__(
            f"PSA cert lookup failed for {cert_number!r}: HTTP {status_code}"
        )


class InvalidImageError(PipelineError):
    """
    Raised when downloaded bytes fail magic-byte validation (not JPEG/PNG)
    or when an image is too small for label-region masking (height < 100px).
    """


class DownloadError(PipelineError):
    """
    Raised by ImageDownloader after exhausting all retry attempts.
    Wraps the underlying cause so the orchestrator can log the root error.
    """


class ConfigurationError(PipelineError):
    """
    Raised at startup when required environment variables (e.g. PSA_API_TOKEN)
    are absent or invalid. The CLI catches this and exits with code 1.
    """
