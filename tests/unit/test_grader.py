"""
Unit tests for GraderService.

Covers:
  1. Ordinal decoding correctness (_decode_ordinal, _decode_subgrade)
  2. Per-image InferenceError handling — errored card is skipped, batch continues
  3. ModelNotFoundError propagation — registry.get() failure is NOT swallowed

The TF model is mocked as a plain callable returning a dict of numpy arrays,
matching the contract described in the design doc.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pregrader.config import PregraderSettings
from pregrader.enums import CardType
from pregrader.exceptions import InferenceError, ModelNotFoundError
from pregrader.registry import ModelRegistry
from pregrader.schemas import GradeResult, PreprocessedCard, CardRegion
from pregrader.services.grader import GraderService, _decode_ordinal, _decode_subgrade

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings() -> PregraderSettings:
    return PregraderSettings(pokemon_model_artifact_path="/tmp/fake_model")


def _make_registry(model: MagicMock | None = None) -> ModelRegistry:
    """Return a registry with a mocked pokemon model pre-loaded."""
    registry = MagicMock(spec=ModelRegistry)
    if model is not None:
        registry.get.return_value = model
    else:
        registry.get.return_value = _make_model()
    return registry


def _make_model(grade_index: int = 7) -> MagicMock:
    """Return a mock TF model callable.

    Produces cumulative probabilities that concentrate mass at `grade_index`
    (0-indexed, so grade_index=7 → PSA grade 8).

    The cumulative probs ramp from 0 to 1 with a sharp step at grade_index:
      P(Y≤k) ≈ 0  for k < grade_index
      P(Y≤k) ≈ 1  for k ≥ grade_index
    """
    # Build 9 cumulative probs with a step at grade_index
    cumprobs = np.zeros(9, dtype=np.float32)
    for i in range(9):
        cumprobs[i] = 0.0 if i < grade_index else 1.0

    outputs = {
        "overall": cumprobs.copy(),
        "centering": cumprobs.copy(),
        "corners": cumprobs.copy(),
        "edges": cumprobs.copy(),
        "surface": cumprobs.copy(),
    }
    model = MagicMock()
    model.return_value = outputs
    return model


def _make_card(image_id: str = "test_card.jpg") -> PreprocessedCard:
    """Return a minimal PreprocessedCard with correct tensor shapes."""
    # 312 x 224 x 3 full tensor, all zeros
    full_tensor = [[[0.0, 0.0, 0.0] for _ in range(224)] for _ in range(312)]
    regions = [
        CardRegion(name=name, tensor=[[[0.0, 0.0, 0.0]]])
        for name in ("centering", "corners", "edges", "surface")
    ]
    return PreprocessedCard(
        image_id=image_id,
        full_tensor=full_tensor,
        regions=regions,
    )


# ---------------------------------------------------------------------------
# Section 1: Ordinal decoding unit tests
# ---------------------------------------------------------------------------

class TestDecodeOrdinal:
    """Tests for the _decode_ordinal helper function."""

    def test_step_at_grade_1_returns_grade_1(self) -> None:
        """All mass at grade 1: cumprobs all 1.0 → grade 1."""
        # P(Y≤k) = 1.0 for all k → P(Y=1) = 1.0, all others 0.0
        cumprobs = np.ones(9, dtype=np.float64)
        grade, confidence = _decode_ordinal(cumprobs)
        assert grade == 1
        assert confidence == pytest.approx(1.0, abs=1e-6)

    def test_step_at_grade_10_returns_grade_10(self) -> None:
        """All mass at grade 10: cumprobs all 0.0 → grade 10."""
        # P(Y≤k) = 0.0 for all k → P(Y=10) = 1.0
        cumprobs = np.zeros(9, dtype=np.float64)
        grade, confidence = _decode_ordinal(cumprobs)
        assert grade == 10
        assert confidence == pytest.approx(1.0, abs=1e-6)

    def test_step_at_grade_5_returns_grade_5(self) -> None:
        """Sharp step at index 4 (0-indexed) → grade 5."""
        # P(Y≤k) = 0 for k<4, 1 for k≥4 → P(Y=5) = 1.0
        cumprobs = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
        grade, confidence = _decode_ordinal(cumprobs)
        assert grade == 5

    def test_confidence_is_max_prob_mass(self) -> None:
        """Confidence must equal the maximum P(Y=k) value."""
        # Uniform distribution: each P(Y=k) = 0.1
        # cumprobs: P(Y≤k) = k/10 for k=1..9
        cumprobs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        grade, confidence = _decode_ordinal(cumprobs)
        # All prob masses are 0.1 — confidence should be 0.1
        assert confidence == pytest.approx(0.1, abs=1e-6)

    def test_grade_in_valid_range(self) -> None:
        """Decoded grade must always be in {1..10}."""
        for i in range(9):
            cumprobs = np.zeros(9, dtype=np.float64)
            cumprobs[i:] = 1.0
            grade, _ = _decode_ordinal(cumprobs)
            assert 1 <= grade <= 10

    def test_confidence_in_valid_range(self) -> None:
        """Confidence must always be in [0.0, 1.0]."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            # Random monotonically non-decreasing cumprobs in [0, 1]
            raw = np.sort(rng.uniform(0, 1, 9))
            _, confidence = _decode_ordinal(raw)
            assert 0.0 <= confidence <= 1.0


class TestDecodeSubgrade:
    """Tests for the _decode_subgrade helper function."""

    def test_all_mass_at_grade_1_returns_1(self) -> None:
        """P(Y≤k)=1 for all k → expected value = 1.0."""
        cumprobs = np.ones(9, dtype=np.float64)
        result = _decode_subgrade(cumprobs)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_all_mass_at_grade_10_returns_10(self) -> None:
        """P(Y≤k)=0 for all k → expected value = 10.0."""
        cumprobs = np.zeros(9, dtype=np.float64)
        result = _decode_subgrade(cumprobs)
        assert result == pytest.approx(10.0, abs=1e-6)

    def test_uniform_distribution_returns_midpoint(self) -> None:
        """Uniform distribution → expected value = 5.5."""
        # P(Y=k) = 0.1 for k=1..10 → E[Y] = 5.5
        cumprobs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        result = _decode_subgrade(cumprobs)
        assert result == pytest.approx(5.5, abs=1e-6)

    def test_result_in_valid_range(self) -> None:
        """Subgrade must always be in [1.0, 10.0]."""
        rng = np.random.default_rng(99)
        for _ in range(50):
            raw = np.sort(rng.uniform(0, 1, 9))
            result = _decode_subgrade(raw)
            assert 1.0 <= result <= 10.0


# ---------------------------------------------------------------------------
# Section 2: Per-image InferenceError handling
# ---------------------------------------------------------------------------

class TestPerImageErrorHandling:
    """GraderService must skip errored cards and continue the batch."""

    @pytest.mark.asyncio
    async def test_single_card_inference_error_returns_empty_list(
        self, settings: PregraderSettings
    ) -> None:
        """If the only card raises InferenceError, result list is empty."""
        model = MagicMock(side_effect=RuntimeError("GPU OOM"))
        registry = _make_registry(model)
        service = GraderService(registry, settings)

        results = await service.predict([_make_card("bad.jpg")], CardType.pokemon)
        assert results == []

    @pytest.mark.asyncio
    async def test_one_bad_card_in_batch_skipped_rest_returned(
        self, settings: PregraderSettings
    ) -> None:
        """One failing card must be skipped; the other two must be returned."""
        good_model = _make_model(grade_index=7)  # returns grade 8
        bad_call_count = 0

        def model_side_effect(tensor: np.ndarray) -> dict:
            nonlocal bad_call_count
            # Fail on the second call (index 1)
            bad_call_count += 1
            if bad_call_count == 2:
                raise RuntimeError("simulated failure")
            return good_model(tensor)

        model = MagicMock(side_effect=model_side_effect)
        registry = _make_registry(model)
        service = GraderService(registry, settings)

        cards = [_make_card("card_0.jpg"), _make_card("card_1.jpg"), _make_card("card_2.jpg")]
        results = await service.predict(cards, CardType.pokemon)

        # card_1 failed — only card_0 and card_2 should be in results
        assert len(results) == 2
        result_ids = {r.image_id for r in results}
        assert "card_0.jpg" in result_ids
        assert "card_2.jpg" in result_ids
        assert "card_1.jpg" not in result_ids

    @pytest.mark.asyncio
    async def test_all_cards_fail_returns_empty_list(
        self, settings: PregraderSettings
    ) -> None:
        """If every card fails, an empty list is returned (not an exception)."""
        model = MagicMock(side_effect=RuntimeError("all fail"))
        registry = _make_registry(model)
        service = GraderService(registry, settings)

        cards = [_make_card(f"card_{i}.jpg") for i in range(5)]
        results = await service.predict(cards, CardType.pokemon)
        assert results == []

    @pytest.mark.asyncio
    async def test_successful_card_has_valid_grade_result(
        self, settings: PregraderSettings
    ) -> None:
        """A successful card must produce a GradeResult with valid field ranges."""
        registry = _make_registry(_make_model(grade_index=7))
        service = GraderService(registry, settings)

        results = await service.predict([_make_card("pikachu.jpg")], CardType.pokemon)

        assert len(results) == 1
        r = results[0]
        assert isinstance(r, GradeResult)
        assert r.image_id == "pikachu.jpg"
        assert r.card_type == CardType.pokemon
        assert 1 <= r.overall_grade <= 10
        assert 0.0 <= r.confidence <= 1.0
        assert 1.0 <= r.subgrades.centering <= 10.0
        assert 1.0 <= r.subgrades.corners <= 10.0
        assert 1.0 <= r.subgrades.edges <= 10.0
        assert 1.0 <= r.subgrades.surface <= 10.0


# ---------------------------------------------------------------------------
# Section 3: ModelNotFoundError propagation
# ---------------------------------------------------------------------------

class TestModelNotFoundPropagation:
    """ModelNotFoundError from registry.get() must NOT be caught by GraderService."""

    @pytest.mark.asyncio
    async def test_model_not_found_propagates(
        self, settings: PregraderSettings
    ) -> None:
        """ModelNotFoundError must propagate out of predict() unchanged."""
        registry = MagicMock(spec=ModelRegistry)
        registry.get.side_effect = ModelNotFoundError(
            "No model loaded for card_type='one_piece'. Loaded types: ['pokemon']"
        )
        service = GraderService(registry, settings)

        with pytest.raises(ModelNotFoundError) as exc_info:
            await service.predict([_make_card()], CardType.one_piece)

        assert "one_piece" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_model_not_found_is_not_wrapped_as_inference_error(
        self, settings: PregraderSettings
    ) -> None:
        """ModelNotFoundError must NOT be caught and re-raised as InferenceError."""
        registry = MagicMock(spec=ModelRegistry)
        registry.get.side_effect = ModelNotFoundError("No model for 'sports'.")
        service = GraderService(registry, settings)

        # Must raise ModelNotFoundError, not InferenceError
        with pytest.raises(ModelNotFoundError):
            await service.predict([_make_card()], CardType.sports)

    @pytest.mark.asyncio
    async def test_model_not_found_message_preserved(
        self, settings: PregraderSettings
    ) -> None:
        """The original ModelNotFoundError message must be preserved."""
        expected_msg = "No model loaded for card_type='one_piece'. Loaded types: ['pokemon']"
        registry = MagicMock(spec=ModelRegistry)
        registry.get.side_effect = ModelNotFoundError(expected_msg)
        service = GraderService(registry, settings)

        with pytest.raises(ModelNotFoundError) as exc_info:
            await service.predict([], CardType.one_piece)

        assert str(exc_info.value) == expected_msg
