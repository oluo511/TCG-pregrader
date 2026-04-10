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
from pydantic_settings.sources import DotEnvSettingsSource, EnvSettingsSource

from pregrader.enums import CardType
from pregrader.exceptions import ConfigurationError

# Fields whose env values are comma-separated strings rather than JSON arrays.
# pydantic-settings calls json.loads() inside prepare_field_value() *before*
# Pydantic's own validators fire, so a mode="before" field_validator never sees
# the raw string — the SettingsError is raised first. We must intercept at the
# source level for BOTH the OS-env source AND the dotenv source.
_CSV_LIST_FIELDS = {"enabled_card_types"}


class _CsvNormMixin:
    """Mixin that normalises comma-separated env values to JSON arrays.

    Applied to both EnvSettingsSource (OS env vars) and DotEnvSettingsSource
    (.env file) so that ENABLED_CARD_TYPES=pokemon,one_piece works in either
    context without requiring JSON array syntax.
    """

    def prepare_field_value(
        self,
        field_name: str,
        field: Any,
        value: Any,
        value_is_complex: bool,
    ) -> Any:
        import json

        # Convert "pokemon,one_piece" → '["pokemon","one_piece"]' so the base
        # class json.loads() call succeeds and Pydantic coerces each element.
        if (
            field_name in _CSV_LIST_FIELDS
            and isinstance(value, str)
            and not value.strip().startswith("[")
        ):
            items = [item.strip() for item in value.split(",") if item.strip()]
            value = json.dumps(items)
        return super().prepare_field_value(field_name, field, value, value_is_complex)  # type: ignore[misc]


class _CsvAwareEnvSource(_CsvNormMixin, EnvSettingsSource):
    """OS environment variable source with CSV normalisation."""


class _CsvAwareDotEnvSource(_CsvNormMixin, DotEnvSettingsSource):
    """.env file source with CSV normalisation."""


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
        """Replace both env sources with CSV-aware subclasses.

        Why replace dotenv_settings too?
        DotEnvSettingsSource is a separate class from EnvSettingsSource.
        Both call prepare_field_value() independently, so the CSV mixin must
        be applied to both or .env file values still hit json.loads() raw.

        Why forward kwargs to _CsvAwareDotEnvSource instead of constructing
        it fresh?
        pydantic-settings passes a pre-configured DotEnvSettingsSource via
        kwargs["dotenv_settings"] that already reflects any _env_file=None
        override passed at instantiation time. Constructing a new instance
        from settings_cls alone would always read model_config's env_file,
        ignoring runtime overrides. We instead subclass on-the-fly using the
        same init args so the override is preserved.
        """
        init_settings = kwargs.get("init_settings")
        env_settings = _CsvAwareEnvSource(settings_cls)

        # Preserve the dotenv source only if pydantic-settings provided one.
        # When _env_file=None is passed at instantiation, pydantic-settings
        # sets dotenv_settings=None — we must honour that to allow tests to
        # suppress .env loading for env-isolation scenarios.
        raw_dotenv = kwargs.get("dotenv_settings")
        if raw_dotenv is not None:
            # Re-wrap with our mixin so CSV normalisation applies to .env values.
            dotenv_settings = _CsvAwareDotEnvSource(
                settings_cls,
                env_file=raw_dotenv.env_file,
                env_file_encoding=raw_dotenv.env_file_encoding,
            )
        else:
            dotenv_settings = None

        secrets_settings = kwargs.get("secrets_settings")
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
