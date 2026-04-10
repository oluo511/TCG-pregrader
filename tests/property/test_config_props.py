"""
# Feature: pokemon-card-pregrader, Property 16: Configuration validation completeness

Property 16: For any PregraderSettings instantiation with a missing required
field or an out-of-range value, Pydantic must raise a ValidationError before
the application accepts requests; for any fully valid configuration,
instantiation must succeed.

Validates: Requirements 10.2, 10.3
"""

import os
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A minimal valid config dict — all required fields present, all values in range.
# We build invalid variants by removing or corrupting individual fields.
_VALID_BASE: dict[str, Any] = {
    "POKEMON_MODEL_ARTIFACT_PATH": "/tmp/pokemon_model",
    "ENABLED_CARD_TYPES": "pokemon",
    "INPUT_WIDTH": "224",
    "INPUT_HEIGHT": "312",
    "MAX_BATCH_SIZE": "50",
    "API_HOST": "0.0.0.0",
    "API_PORT": "8000",
    "LOG_LEVEL": "INFO",
}

# Fields that are required (no default) — removing any one must cause ValidationError.
_REQUIRED_FIELDS = ["POKEMON_MODEL_ARTIFACT_PATH"]

# Strategies for out-of-range integer values.
_invalid_port = st.one_of(st.integers(max_value=0), st.integers(min_value=65536))
_invalid_batch = st.one_of(st.integers(max_value=0), st.integers(min_value=201))
_invalid_log_level = (
    st.text(min_size=1, alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"))
    .filter(lambda s: s.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
)


def _make_env(overrides: dict[str, Any]) -> dict[str, str]:
    """Merge overrides into the valid base config, converting all values to str."""
    env = {**_VALID_BASE, **{k: str(v) for k, v in overrides.items()}}
    return env


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@given(st.sampled_from(_REQUIRED_FIELDS))
@settings(max_examples=10)  # Only one required field in MVP; keep suite fast.
def test_missing_required_field_raises_validation_error(missing_field: str) -> None:
    """
    **Validates: Requirements 10.2, 10.3**

    Removing any required field from the environment must cause Pydantic to
    raise ValidationError at settings construction time — never silently default.
    """
    # Build env without the required field and inject into os.environ so
    # PregraderSettings (which reads from env) sees the incomplete config.
    env = {k: v for k, v in _VALID_BASE.items() if k != missing_field}

    _assert_invalid_env(env)


@given(_invalid_port)
def test_out_of_range_api_port_raises_validation_error(port: int) -> None:
    """
    **Validates: Requirements 10.2, 10.3**

    API_PORT must be in [1, 65535]. Values outside this range must be rejected
    at startup, not at the first incoming connection.
    """
    env = _make_env({"API_PORT": port})
    _assert_invalid_env(env)


@given(_invalid_batch)
def test_out_of_range_batch_size_raises_validation_error(batch_size: int) -> None:
    """
    **Validates: Requirements 10.2, 10.3**

    MAX_BATCH_SIZE must be in [1, 200]. Zero or negative values would allow
    empty batches; values > 200 would bypass the ingestion guard.
    """
    env = _make_env({"MAX_BATCH_SIZE": batch_size})
    _assert_invalid_env(env)


@given(_invalid_log_level)
def test_invalid_log_level_raises_validation_error(level: str) -> None:
    """
    **Validates: Requirements 10.2, 10.3**

    LOG_LEVEL must be one of the five stdlib levels. An unknown level would
    silently fall back to WARNING in some logging frameworks — we reject it
    explicitly so misconfigured deployments fail fast.
    """
    env = _make_env({"LOG_LEVEL": level})
    _assert_invalid_env(env)


@given(
    st.fixed_dictionaries(
        {
            "POKEMON_MODEL_ARTIFACT_PATH": st.just("/tmp/pokemon_model"),
            "INPUT_WIDTH": st.integers(min_value=1, max_value=4096),
            "INPUT_HEIGHT": st.integers(min_value=1, max_value=4096),
            "MAX_BATCH_SIZE": st.integers(min_value=1, max_value=200),
            "API_PORT": st.integers(min_value=1, max_value=65535),
            "LOG_LEVEL": st.sampled_from(
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            ),
        }
    )
)
def test_valid_config_instantiates_successfully(config: dict[str, Any]) -> None:
    """
    **Validates: Requirements 10.2, 10.3**

    Any fully valid configuration dict must produce a PregraderSettings
    instance without raising. This guards against overly strict validators
    that reject legitimate values.
    """
    env = _make_env(config)
    _assert_valid_env(env)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_invalid_env(env: dict[str, str]) -> None:
    """Inject env vars, attempt to construct PregraderSettings, assert ValidationError.

    We pass _env_file=None to suppress .env file loading so the test controls
    the full configuration surface via os.environ only. Without this, a valid
    .env on disk would mask missing required fields and the ValidationError
    would never fire.
    """
    import sys

    # Isolate env manipulation — restore original env after the assertion.
    original = {k: os.environ.get(k) for k in env}
    try:
        # Clear any existing values that might mask the test scenario.
        for k in list(os.environ.keys()):
            if k in _VALID_BASE:
                del os.environ[k]
        os.environ.update(env)

        # Import here to avoid module-level side effects from settings loading.
        from pregrader.config import PregraderSettings

        with pytest.raises(ValidationError):
            # _env_file=None disables .env file loading so only os.environ is
            # consulted — gives the test full control over the config surface.
            PregraderSettings(_env_file=None)
    finally:
        _restore_env(original)


def _assert_valid_env(env: dict[str, str]) -> None:
    """Inject env vars, construct PregraderSettings, assert no exception raised."""
    original = {k: os.environ.get(k) for k in env}
    try:
        for k in list(os.environ.keys()):
            if k in _VALID_BASE:
                del os.environ[k]
        os.environ.update(env)

        from pregrader.config import PregraderSettings

        # _env_file=None: same isolation rationale as _assert_invalid_env.
        instance = PregraderSettings(_env_file=None)
        # Spot-check that the instance actually holds the injected values.
        assert instance.api_port == int(env["API_PORT"])
        assert instance.log_level == env["LOG_LEVEL"].upper()
    finally:
        _restore_env(original)


def _restore_env(snapshot: dict[str, str | None]) -> None:
    """Restore os.environ to its pre-test state."""
    for k, v in snapshot.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
