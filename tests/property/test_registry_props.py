# Feature: pokemon-card-pregrader, Property 9: Registry load-once invariant per card type
# Feature: pokemon-card-pregrader, Property 17: Unknown card type returns ModelNotFoundError / HTTP 404

"""
Property-based tests for ModelRegistry.

Strategy: Mock tf.saved_model.load and Path.exists so tests are hermetic —
no real TF artifacts or filesystem access required. Hypothesis drives the
card type selection and request counts.
"""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pregrader.enums import CardType
from pregrader.exceptions import ModelNotFoundError
from pregrader.registry import ModelRegistry

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# All valid CardType values as a Hypothesis strategy
_card_type_strategy = st.sampled_from(list(CardType))

# Strategy for a non-empty subset of CardType values to load into the registry
_loaded_types_strategy = st.lists(
    _card_type_strategy,
    min_size=1,
    max_size=len(CardType),
    unique=True,
)

# Strategy for a CardType that is NOT in a given loaded set — used in Property 17
# We generate the full set and filter; Hypothesis handles the assume() contract.
_all_card_types = set(CardType)


# ---------------------------------------------------------------------------
# Property 9: Registry load-once invariant per card type
# Validates: Requirements 5.6
# ---------------------------------------------------------------------------

@given(
    card_type=_card_type_strategy,
    n_requests=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=100)
def test_property9_load_called_exactly_once_per_card_type(
    card_type: CardType,
    n_requests: int,
) -> None:
    """Regardless of how many get() calls are made, load() must be invoked
    exactly once per card type.

    This property validates that the registry does not reload on every request
    (which would be catastrophically expensive for large TF SavedModels).

    **Validates: Requirements 5.6**
    """
    registry = ModelRegistry()
    fake_model = MagicMock(name="FakeSavedModel")

    with (
        patch("pregrader.registry.tf.saved_model.load", return_value=fake_model) as mock_load,
        patch.object(Path, "exists", return_value=True),
    ):
        # Load the model once — simulating startup
        registry.load(card_type, Path(f"/tmp/{card_type.value}"))

        # Simulate N prediction requests — each calls get(), never load() again
        for _ in range(n_requests):
            registry.get(card_type)

        # tf.saved_model.load must have been called exactly once, not N times
        assert mock_load.call_count == 1, (
            f"Expected tf.saved_model.load called 1 time, got {mock_load.call_count} "
            f"after {n_requests} get() calls for card_type={card_type.value}"
        )


@given(
    card_types=_loaded_types_strategy,
    n_requests=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_property9_load_once_per_type_multi_type_registry(
    card_types: list[CardType],
    n_requests: int,
) -> None:
    """With multiple card types loaded, each type's model must be loaded exactly
    once regardless of how many get() calls are made across all types.

    **Validates: Requirements 5.6**
    """
    registry = ModelRegistry()
    fake_model = MagicMock(name="FakeSavedModel")

    with (
        patch("pregrader.registry.tf.saved_model.load", return_value=fake_model) as mock_load,
        patch.object(Path, "exists", return_value=True),
    ):
        # Load each type once at startup
        for ct in card_types:
            registry.load(ct, Path(f"/tmp/{ct.value}"))

        load_count_after_startup = mock_load.call_count
        assert load_count_after_startup == len(card_types), (
            f"Expected {len(card_types)} load calls at startup, got {load_count_after_startup}"
        )

        # Simulate N requests per type — no additional load() calls should occur
        for _ in range(n_requests):
            for ct in card_types:
                registry.get(ct)

        # Total load count must not have increased after startup
        assert mock_load.call_count == len(card_types), (
            f"load() was called {mock_load.call_count} times total; "
            f"expected exactly {len(card_types)} (once per type)"
        )


# ---------------------------------------------------------------------------
# Property 17: Unknown card type returns ModelNotFoundError
# Validates: Requirements 3.5, 5.7
# ---------------------------------------------------------------------------

@given(
    loaded_types=_loaded_types_strategy,
)
@settings(max_examples=100)
def test_property17_unloaded_type_raises_model_not_found_error(
    loaded_types: list[CardType],
) -> None:
    """For any card_type not present in the registry, get() must raise
    ModelNotFoundError. The error message must include the requested
    card_type value.

    **Validates: Requirements 3.5, 5.7**
    """
    # Determine which types are NOT loaded — skip if all types are loaded
    unloaded_types = [ct for ct in CardType if ct not in loaded_types]
    if not unloaded_types:
        # All card types are loaded — no unloaded type to test; skip this example
        return

    registry = ModelRegistry()
    fake_model = MagicMock(name="FakeSavedModel")

    with (
        patch("pregrader.registry.tf.saved_model.load", return_value=fake_model),
        patch.object(Path, "exists", return_value=True),
    ):
        for ct in loaded_types:
            registry.load(ct, Path(f"/tmp/{ct.value}"))

    # Each unloaded type must raise ModelNotFoundError with the type in the message
    for unloaded in unloaded_types:
        with pytest.raises(ModelNotFoundError) as exc_info:
            registry.get(unloaded)

        msg = str(exc_info.value)
        assert unloaded.value in msg, (
            f"ModelNotFoundError message must contain '{unloaded.value}', got: {msg!r}"
        )


@given(card_type=_card_type_strategy)
@settings(max_examples=100)
def test_property17_empty_registry_always_raises(card_type: CardType) -> None:
    """An empty registry must raise ModelNotFoundError for every CardType value,
    and the message must always include the requested card_type.

    **Validates: Requirements 3.5, 5.7**
    """
    registry = ModelRegistry()

    with pytest.raises(ModelNotFoundError) as exc_info:
        registry.get(card_type)

    msg = str(exc_info.value)
    assert card_type.value in msg, (
        f"Expected '{card_type.value}' in error message, got: {msg!r}"
    )
    # Loaded types list must be empty
    assert "[]" in msg or "Loaded types: []" in msg, (
        f"Expected empty loaded types list in message, got: {msg!r}"
    )


@given(
    loaded_types=_loaded_types_strategy,
)
@settings(max_examples=100)
def test_property17_error_message_lists_loaded_types(
    loaded_types: list[CardType],
) -> None:
    """ModelNotFoundError message must include the list of currently loaded types,
    so operators can diagnose which types are available.

    **Validates: Requirements 3.5, 5.7**
    """
    unloaded_types = [ct for ct in CardType if ct not in loaded_types]
    if not unloaded_types:
        return

    registry = ModelRegistry()
    fake_model = MagicMock(name="FakeSavedModel")

    with (
        patch("pregrader.registry.tf.saved_model.load", return_value=fake_model),
        patch.object(Path, "exists", return_value=True),
    ):
        for ct in loaded_types:
            registry.load(ct, Path(f"/tmp/{ct.value}"))

    unloaded = unloaded_types[0]
    with pytest.raises(ModelNotFoundError) as exc_info:
        registry.get(unloaded)

    msg = str(exc_info.value)
    # Every loaded type's value must appear in the message
    for loaded in loaded_types:
        assert loaded.value in msg, (
            f"Expected loaded type '{loaded.value}' in error message, got: {msg!r}"
        )
