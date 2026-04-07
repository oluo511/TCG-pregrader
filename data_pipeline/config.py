"""
PipelineSettings — single source of truth for all pipeline configuration.

Uses Pydantic BaseSettings so every field can be overridden via environment
variable or .env file without code changes. SecretStr on psa_api_token ensures
the token is never exposed in logs, repr, or model_dump() output (Req 9.4).

Field naming convention: snake_case matches the env var name exactly
(pydantic-settings maps PSA_API_TOKEN → psa_api_token automatically).
"""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # silently drop unknown env vars — avoids startup noise
    )

    # -------------------------------------------------------------------------
    # Secrets — SecretStr redacts value in str/repr/model_dump (Req 9.4)
    # -------------------------------------------------------------------------
    psa_api_token: SecretStr = Field(
        ..., description="PSA Public API bearer token (required)"
    )

    # -------------------------------------------------------------------------
    # PSA Client
    # -------------------------------------------------------------------------
    psa_daily_quota: int = Field(
        default=500,
        ge=1,
        description="Max PSA API calls per 24-hour window",
    )
    psa_quota_state_path: Path = Field(
        default=Path(".quota_state.json"),
        description="Path to JSON file persisting daily quota counter",
    )
    psa_base_url: str = Field(
        default="https://api.psacard.com/publicapi/cert/GetByCertNumber/",
        description="Base URL for PSA cert lookup endpoint",
    )

    # -------------------------------------------------------------------------
    # Deduplicator
    # -------------------------------------------------------------------------
    seen_certs_path: Path = Field(
        default=Path(".seen_certs.json"),
        description="Path to JSON file persisting seen cert numbers across runs",
    )

    # -------------------------------------------------------------------------
    # Scrapers — crawl delays and per-grade limits
    # -------------------------------------------------------------------------
    ebay_crawl_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Minimum seconds between consecutive requests to eBay",
    )
    cardladder_crawl_delay: float = Field(
        default=3.0,
        ge=0.0,
        description="Minimum seconds between consecutive requests to cardladder.com",
    )
    max_listings_per_grade: int = Field(
        default=500,
        ge=1,
        description="Max eBay listings to process per PSA grade per run",
    )
    max_records_per_grade: int = Field(
        default=500,
        ge=1,
        description="Max Card Ladder records to process per PSA grade per run",
    )
    max_concurrent_requests: int = Field(
        default=5,
        ge=1,
        description="Semaphore size for concurrent HTTP requests within a scraper",
    )

    # -------------------------------------------------------------------------
    # Image quality thresholds (Req 13.1–13.5)
    # -------------------------------------------------------------------------
    min_sharpness: float = Field(
        default=100.0,
        ge=0.0,
        description="Laplacian variance threshold; images below this are rejected",
    )
    min_luminance: float = Field(
        default=30.0,
        ge=0.0,
        le=255.0,
        description="Mean luminance floor; images below this are underexposed",
    )
    max_luminance: float = Field(
        default=230.0,
        ge=0.0,
        le=255.0,
        description="Mean luminance ceiling; images above this are overexposed/glare",
    )
    max_skew_angle: float = Field(
        default=5.0,
        ge=0.0,
        le=90.0,
        description="Max slab tilt in degrees before perspective correction is attempted",
    )

    # -------------------------------------------------------------------------
    # Augmentation probabilities (Req 12.1–12.4)
    # -------------------------------------------------------------------------
    glare_probability: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Probability of applying glare simulation per training image",
    )
    label_occlusion_probability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of occluding the label region per training image",
    )

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------
    output_dir: Path = Field(
        default=Path("data/raw_slabs/"),
        description="Directory where downloaded slab images are saved",
    )
    manifest_path: Path = Field(
        default=Path("data/manifest.csv"),
        description="Path to the manifest CSV consumed by ManifestLoader",
    )

    # -------------------------------------------------------------------------
    # CNN input dimensions — must match the model's expected input shape
    # Dimensions: (input_height, input_width, 3) — HWC convention
    # -------------------------------------------------------------------------
    input_width: int = Field(
        default=224,
        ge=1,
        description="Target image width in pixels (standard CNN input)",
    )
    input_height: int = Field(
        default=224,
        ge=1,
        description="Target image height in pixels (standard CNN input)",
    )

    # -------------------------------------------------------------------------
    # Label region — bottom fraction of slab photo containing the PSA grade label
    # -------------------------------------------------------------------------
    label_region_fraction: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description=(
            "Fraction of image height occupied by the PSA label region "
            "(bottom 15% of a standard slab photo)"
        ),
    )
