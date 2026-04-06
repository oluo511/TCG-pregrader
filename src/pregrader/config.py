"""
Runtime configuration for the TCG Pre-Grader.

Why pydantic-settings over os.environ / configparser?
- Type coercion and validation happen at import time — the app fails fast
  with a clear error rather than crashing mid-request on a bad cast.
- .env file support is built-in; no extra dotenv loading boilerplate.
- Settings are a typed object, not a stringly-typed dict, so IDEs and type
  checkers can catch misconfigured field access.

Technical Debt: TrainingConfig is a separate BaseSettings subclass in
schemas.py. If the number of config classes grows, consolidate into a
single settings module with nested models.
"""

from pathlib import Path
from typing import Any, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import EnvSettingsSource

from pregrader.enums import CardType
from pregrader.exceptions import ConfigurationError

# Fields whose env values are comma-separated strings rather than JSON arrays.
# pydantic-settings' EnvSettingsSource calls json.loads() on list fields before
# validators run, which breaks "pokemon,one_piece" style values. We subclass
# EnvSettingsSource to intercept those fields and convert them to JSON arrays
# before the base class attempts to decode them.
_CSV_LIST_FIELDS = {"enabled_card_types"}


class _CsvAwareEnvSource(EnvSettingsSource):
    """Custom env source that converts comma-separated strings to JSON arrays
    for fields listed in _CSV_LIST_FIELDS.

    Why subclass instead of a field_validator?
    pydantic-settings calls json.loads() inside prepare_field_value() *before*
    Pydantic's own validators fire. A mode="before" field_validator never sees
    the raw string — the SettingsError is raised first. Subclassing lets us
    normalise the value at the source level, before any JSON parsing occurs.
    """

    def prepare_field_value(
        self,
        field_name: str,
        field: Any,
        value: Any,
        value_is_complex: bool,
    ) -> Any:
        # Convert "pokemon,one_piece" → '["pokemon","one_piece"]' so the base
        # class json.loads() call succeeds and Pydantic coerces each element.
        if (
            field_name in _CSV_LIST_FIELDS
            and isinstance(value, str)
            and not value.strip().startswith("[")
        ):
            import json

            items = [item.strip() for item in value.split(",") if item.strip()]
            value = json.dumps(items)
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class PregraderSettings(BaseSettings):
    """All runtime parameters for the serving layer.

    Loaded from environment variables and/or a `.env` file at startup.
    Pydantic raises ValidationError on missing required fields or type
    mismatches; the application catches this and re-raises as ConfigurationError
    so the startup log always names the offending field.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra env vars without raising — avoids breakage when the
        # environment has unrelated variables set (e.g., CI secrets).
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls: type, **kwargs: Any) -> tuple:  # type: ignore[override]
        """Replace the default EnvSettingsSource with our CSV-aware subclass."""
        init_settings = kwargs.get("init_settings")
        env_settings = _CsvAwareEnvSource(settings_cls)
        dotenv_settings = kwargs.get("dotenv_settings")
        secrets_settings = kwargs.get("secrets_settings")
        # Return only non-None sources to avoid TypeError from None entries.
        sources = [s for s in [init_settings, env_settings, dotenv_settings, secrets_settings] if s is not None]
        return tuple(sources)

    # --- Model artifact paths ---
    # Only pokemon is required for MVP; others are optional so the app can
    # start without them and return HTTP 404 for those card types.
    pokemon_model_artifact_path: Path = Field(
        ...,
        description="Filesystem path to the TF SavedModel directory for Pokemon cards.",
    )
    one_piece_model_artifact_path: Optional[Path] = Field(
        default=None,
        description="Path to One Piece model artifact. None = not loaded at startup.",
    )
    sports_model_artifact_path: Optional[Path] = Field(
        default=None,
        description="Path to sports card model artifact. None = not loaded at startup.",
    )

    # --- Registry bootstrap ---
    # Drives which models ModelRegistry.load() is called for at startup.
    # Accepts either a JSON array or a comma-separated string in .env:
    #   ENABLED_CARD_TYPES=pokemon,one_piece   ← human-friendly
    #   ENABLED_CARD_TYPES=["pokemon"]         ← JSON array also works
    enabled_card_types: list[CardType] = Field(
        default=[CardType.pokemon],
        description="Card types to load at startup. Must have a corresponding artifact path.",
    )

    # --- Preprocessing dimensions ---
    # These must match the input shape the model was trained on.
    # Changing them without retraining will silently degrade accuracy.
    input_width: int = Field(default=224, ge=1)
    input_height: int = Field(default=312, ge=1)

    # --- Ingestion limits ---
    max_batch_size: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum images per /predict request. Checked before any I/O.",
    )

    # --- API server ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)

    # --- Observability ---
    log_level: str = Field(
        default="INFO",
        description="structlog log level. One of DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Reject unknown log levels early so misconfigured deployments fail at
        startup rather than silently emitting no logs."""
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'")
        return upper


def load_settings() -> PregraderSettings:
    """Load and validate settings, re-raising Pydantic errors as ConfigurationError.

    Why wrap ValidationError? The rest of the codebase only needs to catch
    PregraderError subclasses at the boundary — it shouldn't need to import
    Pydantic just to handle a startup failure.
    """
    from pydantic import ValidationError

    try:
        return PregraderSettings()
    except ValidationError as exc:
        # Extract the first missing/invalid field name for the error message.
        missing_fields = [str(e["loc"]) for e in exc.errors()]
        raise ConfigurationError(
            f"Invalid configuration. Problem fields: {missing_fields}. "
            f"Details: {exc}"
        ) from exc
