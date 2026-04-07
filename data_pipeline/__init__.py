"""
Training Data Pipeline — standalone async module for collecting and labeling
PSA slab photos for TCG Pre-Grader CNN training.

Data flow:
  eBay / CardLadder scrapers → Deduplicator → PSA Client → Manifest Builder
"""

from data_pipeline.config import PipelineSettings
from data_pipeline.exceptions import (
    CertLookupError,
    ConfigurationError,
    DownloadError,
    InvalidImageError,
    PipelineError,
    QuotaExhaustedError,
)
from data_pipeline.models import CertRecord, RawListing, ScrapedRecord
from data_pipeline.psa_client import PSAClient, QuotaState

__all__ = [
    # Config
    "PipelineSettings",
    # Exceptions
    "PipelineError",
    "QuotaExhaustedError",
    "CertLookupError",
    "InvalidImageError",
    "DownloadError",
    "ConfigurationError",
    # Models
    "CertRecord",
    "RawListing",
    "ScrapedRecord",
    # PSA Client
    "PSAClient",
    "QuotaState",
]
