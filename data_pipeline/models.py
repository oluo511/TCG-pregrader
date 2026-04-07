"""
Data models for the training data pipeline.

CertRecord is the authoritative label record sourced from the PSA Public API.
RawListing is the intermediate scraper output before cert lookup.
ScrapedRecord is the post-dedup, post-PSA-lookup record ready for download.

All models are frozen (immutable) where appropriate to prevent accidental
mutation after validation — a common source of subtle bugs in async pipelines.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CertRecord(BaseModel):
    """
    Validated label data from the PSA Public API.

    frozen=True: once constructed and validated, a CertRecord is immutable.
    This is safe to share across concurrent tasks without defensive copying.
    """

    model_config = ConfigDict(frozen=True)

    cert_number: str
    overall_grade: int = Field(ge=1, le=10)
    centering: float = Field(ge=1.0, le=10.0)
    corners: float = Field(ge=1.0, le=10.0)
    edges: float = Field(ge=1.0, le=10.0)
    surface: float = Field(ge=1.0, le=10.0)
    # False when grade is sourced from listing metadata, not PSA API (Req 3.5)
    verified: bool = True


class RawListing(BaseModel):
    """
    Intermediate scraper output before cert lookup.

    Produced by EbayScraper / CardLadderScraper before the PSA API call.
    cert_number is None when the listing title/metadata contains no PSA cert.
    """

    source: Literal["ebay", "cardladder"]
    listing_url: str
    image_url: str
    title: str
    raw_grade: int | None = None  # grade from listing metadata, pre-PSA-verification
    cert_number: str | None = None


class ScrapedRecord(BaseModel):
    """
    Post-dedup, post-PSA-lookup record ready for image download and manifest write.

    cert_record carries the authoritative PSA label; image_url is the slab photo
    to download. source is preserved for logging and dedup tracking.
    """

    cert_record: CertRecord
    image_url: str
    source: Literal["ebay", "cardladder"]
