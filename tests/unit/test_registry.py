"""
Unit tests for ModelRegistry.

Covers Task 6.3:
  1. ModelNotFoundError raised with card_type in message for unloaded type
  2. is_ready is False before any load, True after successful load
  3. get() returns the correct model object after load()

Uses unittest.mock.patch to mock tf.saved_model.load — no real TF artifact
required. Path.exists() is also patched to control the filesystem check.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pregrader.enums import CardType
from pregrader.exceptions import ModelNotFoundError
from pregrader.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> ModelRegistry:
    """Return a fresh, empty ModelRegistry."""
    return ModelRegistry()


def _fake_model() -> MagicMock:
    """Return a mock object standing in for a TF SavedModel."""
    return MagicMock(name="FakeSavedModel")


# ---------------------------------------------------------------------------
# Test 1: ModelNotFoundError raised with card_type in message
# Requirements: 3.5, 5.7
# ---------------------------------------------------------------------------

def test_get_unloaded_type_raises_model_not_found_error() -> None:
    """get() on an empty registry must raise ModelNotFoundError."""
    registry = _make_registry()

    with pytest.raises(ModelNotFoundError):
        registry.get(CardType.pokemon)


def test_get_unloaded_type_message_contains_card_type_value() -> None:
    """ModelNotFoundError message must include the requested card_type value."""
    registry = _make_registry()

    with pytest.raises(ModelNotFoundError) as exc_info:
        registry.get(CardType.one_piece)

    assert "one_piece" in str(exc_info.value)


def test_get_unloaded_type_message_contains_loaded_types() -> None:
    """ModelNotFoundError message must list currently loaded types."""
    registry = _make_registry()

    # Load pokemon so it appears in the 'Loaded types' list
    fake_path = Path("/tmp/fake_pokemon")
    with (
        patch("pregrader.registry.tf.saved_model.load", return_value=_fake_model()),
        patch.object(Path, "exists", return_value=True),
    ):
        registry.load(CardType.pokemon, fake_path)

    # Now request one_piece — message must mention pokemon as a loaded type
    with pytest.raises(ModelNotFoundError) as exc_info:
        registry.get(CardType.one_piece)

    msg = str(exc_info.value)
    assert "one_piece" in msg
    assert "pokemon" in msg


def test_get_unloaded_type_message_format() -> None:
    """ModelNotFoundError message must match the exact format from the design doc."""
    registry = _make_registry()

    with pytest.raises(ModelNotFoundError) as exc_info:
        registry.get(CardType.sports)

    msg = str(exc_info.value)
    # Design doc format: "No model loaded for card_type='sports'. Loaded types: []"
    assert "No model loaded for card_type='sports'" in msg
    assert "Loaded types:" in msg


# ---------------------------------------------------------------------------
# Test 2: is_ready flag behaviour
# Requirements: 5.5, 5.6
# ---------------------------------------------------------------------------

def test_is_ready_false_on_empty_registry() -> None:
    """is_ready must be False before any model is loaded."""
    registry = _make_registry()
    assert registry.is_ready is False


def test_is_ready_true_after_successful_load() -> None:
    """is_ready must be True after at least one successful load()."""
    registry = _make_registry()
    fake_path = Path("/tmp/fake_pokemon")

    with (
        patch("pregrader.registry.tf.saved_model.load", return_value=_fake_model()),
        patch.object(Path, "exists", return_value=True),
    ):
        registry.load(CardType.pokemon, fake_path)

    assert registry.is_ready is True


def test_is_ready_false_after_load_error() -> None:
    """is_ready must be False if any load() raised ModelNotFoundError."""
    registry = _make_registry()
    missing_path = Path("/nonexistent/path")

    # Path.exists() returns False → load() raises ModelNotFoundError
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(ModelNotFoundError):
            registry.load(CardType.pokemon, missing_path)

    assert registry.is_ready is False


def test_is_ready_false_when_no_models_loaded_despite_no_error() -> None:
    """is_ready requires at least one loaded model — empty registry is not ready."""
    registry = _make_registry()
    # _load_error is False by default, but no models loaded → not ready
    assert registry.is_ready is False


# ---------------------------------------------------------------------------
# Test 3: get() returns the correct model after load()
# Requirements: 5.6
# ---------------------------------------------------------------------------

def test_get_returns_loaded_model() -> None:
    """get() must return the exact model object registered by load()."""
    registry = _make_registry()
    fake_path = Path("/tmp/fake_pokemon")
    expected_model = _fake_model()

    with (
        patch("pregrader.registry.tf.saved_model.load", return_value=expected_model),
        patch.object(Path, "exists", return_value=True),
    ):
        registry.load(CardType.pokemon, fake_path)

    result = registry.get(CardType.pokemon)
    assert result is expected_model


def test_get_returns_correct_model_for_each_type() -> None:
    """get() must route to the correct model when multiple types are loaded."""
    registry = _make_registry()
    pokemon_model = _fake_model()
    sports_model = _fake_model()

    with patch.object(Path, "exists", return_value=True):
        with patch("pregrader.registry.tf.saved_model.load", return_value=pokemon_model):
            registry.load(CardType.pokemon, Path("/tmp/pokemon"))
        with patch("pregrader.registry.tf.saved_model.load", return_value=sports_model):
            registry.load(CardType.sports, Path("/tmp/sports"))

    assert registry.get(CardType.pokemon) is pokemon_model
    assert registry.get(CardType.sports) is sports_model


def test_load_missing_artifact_raises_model_not_found_error() -> None:
    """load() must raise ModelNotFoundError when artifact_path does not exist."""
    registry = _make_registry()

    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(ModelNotFoundError):
            registry.load(CardType.pokemon, Path("/nonexistent/model"))


def test_load_missing_artifact_message_contains_card_type() -> None:
    """ModelNotFoundError from load() must include the card_type value."""
    registry = _make_registry()

    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.load(CardType.one_piece, Path("/nonexistent/model"))

    assert "one_piece" in str(exc_info.value)
