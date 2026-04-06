# Feature: pokemon-card-pregrader, Property 6: GradeResult output validity
# Feature: pokemon-card-pregrader, Property 8: Response cardinality
# Feature: pokemon-card-pregrader, Property 10: Batch partial-failure resilience

"""
Property-based tests for GraderService.

Strategy: Mock the TF model as a plain callable returning a dict of numpy
arrays. Hypothesis drives card counts, cumulative probability distributions,
and failure injection. No real TF artifacts or filesystem access required.

All three properties are tested with @settings(max_examples=100) per the
design doc testing strategy.
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pregrader.config import PregraderSettings
from pregrader.enums import CardType
from pregrader.exceptions import InferenceError
from pregrader.registry import ModelRegistry
from pregrader.schemas import CardRegion, GradeResult, PreprocessedCard
from pregrader.services.grader import GraderService

# ---------------------------------------------------------------------------
# Shared test settings
# ---------------------------------------------------------------------------

_SETTINGS = PregraderSettings(pokemon_model_artifact_path="/tmp/fake_model")

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: 9 monotonically non-decreasing floats in [0, 1] — valid cumulative probs.
# We generate 9 independent uniforms and sort them to enforce monotonicity,
# which mirrors the sigmoid outputs of a well-trained ordinal regression head.
_cumprob_strategy = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=9,
    max_size=9,
).map(sorted).map(np.array)

# Strategy: a full model output dict with all 5 heads
def _model_output_strategy() -> st.SearchStrategy[dict[str, np.ndarray]]:
    return st.fixed_dictionaries({
        "overall": _cumprob_strategy,
        "centering": _cumprob_strategy,
        "corners": _cumprob_strategy,
        "edges": _cumprob_strategy,
        "surface": _cumprob_strategy,
    })

# Strategy: a single PreprocessedCard with a minimal valid tensor
def _card_strategy(image_id: str = "card.jpg") -> st.SearchStrategy[PreprocessedCard]:
    return st.just(
        PreprocessedCard(
            image_id=image_id,
            # Minimal 1x1x3 tensor — shape doesn't matter for mocked inference
            full_tensor=[[[0.5, 0.5, 0.5]]],
            regions=[
                CardRegion(name=name, tensor=[[[0.5, 0.5, 0.5]]])
                for name in ("centering", "corners", "edges", "surface")
            ],
        )
    )

# Strategy: a batch of N cards with unique image_ids
def _batch_strategy(
    min_size: int = 1,
    max_size: int = 10,
) -> st.SearchStrategy[list[PreprocessedCard]]:
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: st.just([
            PreprocessedCard(
                image_id=f"card_{i}.jpg",
                full_tensor=[[[0.5, 0.5, 0.5]]],
                regions=[
                    CardRegion(name=name, tensor=[[[0.5, 0.5, 0.5]]])
                    for name in ("centering", "corners", "edges", "surface")
                ],
            )
            for i in range(n)
        ])
    )


def _make_mock_model(outputs: dict[str, np.ndarray]) -> MagicMock:
    """Return a mock TF model callable that returns the given outputs dict."""
    model = MagicMock()
    model.return_value = outputs
    return model


def _make_registry(model: Any) -> ModelRegistry:
    """Return a mock registry that always returns the given model."""
    registry = MagicMock(spec=ModelRegistry)
    registry.get.return_value = model
    return registry


# ---------------------------------------------------------------------------
# Property 6: GradeResult output validity
# Validates: Requirements 3.1, 3.2, 3.3, 4.1, 4.4
# ---------------------------------------------------------------------------

@given(outputs=_model_output_strategy())
@settings(max_examples=100)
def test_property6_grade_result_output_validity(
    outputs: dict[str, np.ndarray],
) -> None:
    """For any valid cumulative probability outputs from the model, the returned
    GradeResult must satisfy all field range constraints:
      - overall_grade ∈ {1..10} (integer)
      - all four subgrades ∈ [1.0, 10.0] (float)
      - confidence ∈ [0.0, 1.0] (float)

    **Validates: Requirements 3.1, 3.2, 3.3, 4.1, 4.4**
    """
    model = _make_mock_model(outputs)
    registry = _make_registry(model)
    service = GraderService(registry, _SETTINGS)

    card = PreprocessedCard(
        image_id="test.jpg",
        full_tensor=[[[0.5, 0.5, 0.5]]],
        regions=[
            CardRegion(name=name, tensor=[[[0.5, 0.5, 0.5]]])
            for name in ("centering", "corners", "edges", "surface")
        ],
    )

    results = asyncio.run(service.predict([card], CardType.pokemon))

    assert len(results) == 1
    r: GradeResult = results[0]

    # overall_grade must be an integer in {1..10}
    assert isinstance(r.overall_grade, int), (
        f"overall_grade must be int, got {type(r.overall_grade)}"
    )
    assert 1 <= r.overall_grade <= 10, (
        f"overall_grade={r.overall_grade} out of range [1, 10]"
    )

    # confidence must be a float in [0.0, 1.0]
    assert 0.0 <= r.confidence <= 1.0, (
        f"confidence={r.confidence} out of range [0.0, 1.0]"
    )

    # all four subgrades must be floats in [1.0, 10.0]
    for name, value in [
        ("centering", r.subgrades.centering),
        ("corners", r.subgrades.corners),
        ("edges", r.subgrades.edges),
        ("surface", r.subgrades.surface),
    ]:
        assert 1.0 <= value <= 10.0, (
            f"subgrade '{name}'={value} out of range [1.0, 10.0]"
        )


# ---------------------------------------------------------------------------
# Property 8: Response cardinality
# Validates: Requirements 5.2, 6.2
# ---------------------------------------------------------------------------

@given(
    batch=_batch_strategy(min_size=1, max_size=15),
    outputs=_model_output_strategy(),
)
@settings(max_examples=100)
def test_property8_response_cardinality(
    batch: list[PreprocessedCard],
    outputs: dict[str, np.ndarray],
) -> None:
    """For any batch of N valid cards, the service must return exactly N
    GradeResult objects in the same order as the inputs.

    **Validates: Requirements 5.2, 6.2**
    """
    model = _make_mock_model(outputs)
    registry = _make_registry(model)
    service = GraderService(registry, _SETTINGS)

    results = asyncio.run(service.predict(batch, CardType.pokemon))

    n = len(batch)
    assert len(results) == n, (
        f"Expected {n} results for batch of {n} cards, got {len(results)}"
    )

    # Order must be preserved — image_ids must match input order
    for i, (card, result) in enumerate(zip(batch, results)):
        assert result.image_id == card.image_id, (
            f"Position {i}: expected image_id='{card.image_id}', "
            f"got '{result.image_id}'"
        )


# ---------------------------------------------------------------------------
# Property 10: Batch partial-failure resilience
# Validates: Requirements 6.5
# ---------------------------------------------------------------------------

@given(
    batch_size=st.integers(min_value=2, max_value=15),
    fail_index=st.integers(min_value=0, max_value=14),
    outputs=_model_output_strategy(),
)
@settings(max_examples=100)
def test_property10_partial_failure_resilience(
    batch_size: int,
    fail_index: int,
    outputs: dict[str, np.ndarray],
) -> None:
    """For any batch of N cards where exactly one card causes an InferenceError,
    the service must return exactly N-1 GradeResult objects for the remaining
    cards, and the failed card's image_id must not appear in the results.

    **Validates: Requirements 6.5**
    """
    # Clamp fail_index to a valid position within this batch
    fail_index = fail_index % batch_size

    batch = [
        PreprocessedCard(
            image_id=f"card_{i}.jpg",
            full_tensor=[[[0.5, 0.5, 0.5]]],
            regions=[
                CardRegion(name=name, tensor=[[[0.5, 0.5, 0.5]]])
                for name in ("centering", "corners", "edges", "surface")
            ],
        )
        for i in range(batch_size)
    ]

    # Track call count to inject failure at the right position
    call_count = 0

    def model_side_effect(tensor: np.ndarray) -> dict[str, np.ndarray]:
        nonlocal call_count
        idx = call_count
        call_count += 1
        if idx == fail_index:
            raise RuntimeError(f"Simulated failure at index {idx}")
        return outputs

    model = MagicMock(side_effect=model_side_effect)
    registry = _make_registry(model)
    service = GraderService(registry, _SETTINGS)

    results = asyncio.run(service.predict(batch, CardType.pokemon))

    # Must return exactly N-1 results
    assert len(results) == batch_size - 1, (
        f"Expected {batch_size - 1} results with one failure, got {len(results)}"
    )

    # The failed card must not appear in results
    failed_id = f"card_{fail_index}.jpg"
    result_ids = {r.image_id for r in results}
    assert failed_id not in result_ids, (
        f"Failed card '{failed_id}' must not appear in results"
    )

    # All returned results must have valid field ranges
    for r in results:
        assert 1 <= r.overall_grade <= 10
        assert 0.0 <= r.confidence <= 1.0
        assert 1.0 <= r.subgrades.centering <= 10.0
        assert 1.0 <= r.subgrades.corners <= 10.0
        assert 1.0 <= r.subgrades.edges <= 10.0
        assert 1.0 <= r.subgrades.surface <= 10.0
