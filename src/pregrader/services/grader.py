"""
GraderService — runs CNN inference and decodes ordinal outputs to PSA grades.

Design pattern: Service Layer with injected ModelRegistry.
GraderService is stateless beyond its constructor dependencies; the registry
and settings are injected so the FastAPI lifespan, CLI, and tests all share
the same construction contract.

Ordinal regression decoding (why not argmax on raw logits?):
The model is trained with a cumulative-link / CORN loss, so its output head
emits P(Y ≤ k) for k=1..9 as 9 sigmoid values — NOT class probabilities.
We must convert to P(Y=k) = P(Y≤k) - P(Y≤k-1) before taking argmax.
Skipping this step and argmax-ing the raw cumulative values would always
predict grade 9 (the last sigmoid is always the largest).

Technical Debt: Inference runs synchronously inside an async method via
asyncio's default thread pool (implicit blocking). For production, wrap
the numpy/TF call in asyncio.to_thread() or move to a dedicated model
server (TF Serving / Triton) over gRPC so the FastAPI event loop is never
blocked by CPU-bound inference.
"""

from typing import Any

import numpy as np

from pregrader.config import PregraderSettings
from pregrader.enums import CardType
from pregrader.exceptions import InferenceError
from pregrader.logging_config import get_logger
from pregrader.registry import ModelRegistry
from pregrader.schemas import GradeResult, PreprocessedCard, Subgrades

logger = get_logger(service="grader")

# The model outputs 9 cumulative probabilities P(Y≤1) … P(Y≤9).
# Grade 10 is implied: P(Y=10) = 1 - P(Y≤9).
# We represent all 10 probability masses in a length-10 array.
_NUM_GRADES: int = 10
_NUM_CUMULATIVE: int = 9  # model output length

# Canonical subgrade region names — must match PreprocessedCard.regions order.
_REGION_NAMES: tuple[str, ...] = ("centering", "corners", "edges", "surface")


def _decode_ordinal(cumulative_probs: np.ndarray) -> tuple[int, float]:
    """Convert cumulative probabilities to a PSA grade and confidence.

    The model outputs P(Y≤k) for k=1..9 as a 1-D array of 9 sigmoid values.
    We compute the probability mass for each grade:

        P(Y=k) = P(Y≤k) - P(Y≤k-1),  with P(Y≤0) = 0.0

    Grade 10 gets the remaining mass: P(Y=10) = 1 - P(Y≤9).

    Args:
        cumulative_probs: 1-D numpy array of shape (9,), values in [0, 1].

    Returns:
        (grade, confidence) where grade ∈ {1..10} and confidence ∈ [0.0, 1.0].
    """
    # Prepend P(Y≤0) = 0.0 and append P(Y≤10) = 1.0 so we can diff uniformly.
    # Shape becomes (11,): [0.0, P(Y≤1), …, P(Y≤9), 1.0]
    padded = np.concatenate([[0.0], cumulative_probs, [1.0]])

    # P(Y=k) for k=1..10 — diff of consecutive cumulative values.
    # Clamp to [0, 1] to guard against floating-point violations from the model.
    prob_mass = np.diff(padded).clip(0.0, 1.0)  # shape (10,)

    # argmax gives 0-indexed position; +1 maps to PSA scale {1..10}.
    grade: int = int(np.argmax(prob_mass)) + 1
    confidence: float = float(np.max(prob_mass))

    return grade, confidence


def _decode_subgrade(cumulative_probs: np.ndarray) -> float:
    """Decode a subgrade head output to a float on the PSA 1.0–10.0 scale.

    Uses the same ordinal decoding as _decode_ordinal but returns a float
    (the expected value E[Y] = Σ k·P(Y=k)) rather than the mode, which
    gives smoother subgrade values that better reflect the probability
    distribution across adjacent grades.

    Why expected value for subgrades but mode for overall_grade?
    - overall_grade must be an integer (PSA scale) → mode is appropriate.
    - Subgrades are floats (1.0–10.0) → expected value captures nuance
      between adjacent grades (e.g., 7.4 vs 7.0 for a near-8 card).

    Args:
        cumulative_probs: 1-D numpy array of shape (9,), values in [0, 1].

    Returns:
        Float subgrade in [1.0, 10.0].
    """
    padded = np.concatenate([[0.0], cumulative_probs, [1.0]])
    prob_mass = np.diff(padded).clip(0.0, 1.0)  # shape (10,)

    # Grade values 1..10 as a float array for the dot product.
    grades = np.arange(1, _NUM_GRADES + 1, dtype=np.float64)
    expected: float = float(np.dot(prob_mass, grades))

    # Clamp to valid range — floating-point edge cases can push slightly outside.
    return float(np.clip(expected, 1.0, 10.0))


class GraderService:
    """Wraps the loaded TF SavedModel and decodes ordinal outputs to GradeResult.

    Lifecycle:
      1. Instantiated once at startup with a populated ModelRegistry.
      2. predict() is called per request — it fetches the model from the
         registry, runs inference, and decodes outputs.
      3. ModelNotFoundError from registry.get() propagates to the caller
         (maps to HTTP 404 at the API boundary — do NOT catch it here).
      4. Per-image InferenceError is caught, logged, and the card is skipped
         so the rest of the batch continues (Requirement 6.5).
    """

    def __init__(self, registry: ModelRegistry, settings: PregraderSettings) -> None:
        """
        Args:
            registry: Populated ModelRegistry — must have the target card_type
                      loaded before predict() is called.
            settings: Runtime settings (currently used for future config hooks;
                      stored for forward-compatibility with batch size limits).
        """
        self._registry = registry
        self._settings = settings
        self._logger = get_logger(service="grader")

    async def predict(
        self,
        cards: list[PreprocessedCard],
        card_type: CardType,
    ) -> list[GradeResult]:
        """Run inference on a batch of preprocessed cards.

        Fetches the model for card_type from the registry, runs each card
        through the model, and decodes ordinal outputs to GradeResult objects.

        ModelNotFoundError from registry.get() is intentionally NOT caught —
        it propagates to the API/CLI layer and maps to HTTP 404 / stderr exit.

        Per-image InferenceError IS caught: the card is skipped, the error is
        logged with image_id, and the remaining cards continue processing.
        This satisfies Requirement 6.5 (partial-failure resilience).

        Args:
            cards: List of PreprocessedCard objects from PreprocessingService.
            card_type: Which model to use for inference.

        Returns:
            List of GradeResult objects — may be shorter than `cards` if any
            cards raised InferenceError during inference.

        Raises:
            ModelNotFoundError: If card_type is not loaded in the registry.
        """
        # Fetch model — ModelNotFoundError propagates (maps to HTTP 404).
        # This is the single registry lookup for the entire batch; the model
        # object is reused for every card in the loop below.
        model: Any = self._registry.get(card_type)

        results: list[GradeResult] = []

        for card in cards:
            try:
                grade_result = self._infer_single(card, card_type, model)
                results.append(grade_result)
            except InferenceError:
                # Per-image failure: log and skip — do not abort the batch.
                # The error is already logged inside _infer_single with image_id.
                continue

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_single(
        self,
        card: PreprocessedCard,
        card_type: CardType,
        model: Any,
    ) -> GradeResult:
        """Run inference on one card and decode outputs to a GradeResult.

        Args:
            card: PreprocessedCard with full_tensor and four region crops.
            card_type: Used to populate GradeResult.card_type.
            model: Loaded TF SavedModel callable.

        Returns:
            GradeResult with overall_grade, subgrades, and confidence.

        Raises:
            InferenceError: On any model call or decoding failure.
                Logged with image_id before raising so the batch loop can
                skip this card without losing the diagnostic context.
        """
        try:
            # Convert the nested Python list to a numpy array.
            # The model expects a batch dimension, so we add axis 0:
            # (H, W, C) → (1, H, W, C).
            full_np = np.array(card.full_tensor, dtype=np.float32)[np.newaxis, ...]

            # Call the model — returns a dict with keys:
            #   "overall":    shape (9,)  — cumulative probs for overall grade
            #   "centering":  shape (9,)  — cumulative probs for centering subgrade
            #   "corners":    shape (9,)  — cumulative probs for corners subgrade
            #   "edges":      shape (9,)  — cumulative probs for edges subgrade
            #   "surface":    shape (9,)  — cumulative probs for surface subgrade
            # Each value is a 1-D tensor of 9 sigmoid outputs.
            outputs: dict[str, Any] = model(full_np)

            # Extract and convert each output to a 1-D numpy float64 array.
            # We work in numpy (not TF ops) after inference so this module
            # has no hard TF dependency beyond the model call itself.
            overall_cumprobs = np.array(outputs["overall"], dtype=np.float64).flatten()

            # Decode overall grade and confidence from cumulative probabilities.
            overall_grade, confidence = _decode_ordinal(overall_cumprobs)

            # Decode each subgrade head using the expected-value decoder.
            subgrades = Subgrades(
                centering=_decode_subgrade(
                    np.array(outputs["centering"], dtype=np.float64).flatten()
                ),
                corners=_decode_subgrade(
                    np.array(outputs["corners"], dtype=np.float64).flatten()
                ),
                edges=_decode_subgrade(
                    np.array(outputs["edges"], dtype=np.float64).flatten()
                ),
                surface=_decode_subgrade(
                    np.array(outputs["surface"], dtype=np.float64).flatten()
                ),
            )

            self._logger.info(
                "inference_complete",
                image_id=card.image_id,
                card_type=card_type.value,
                overall_grade=overall_grade,
                confidence=round(confidence, 4),
            )

            return GradeResult(
                image_id=card.image_id,
                card_type=card_type,
                overall_grade=overall_grade,
                subgrades=subgrades,
                confidence=confidence,
            )

        except Exception as exc:
            self._logger.error(
                "inference_failed",
                image_id=card.image_id,
                card_type=card_type.value,
                error=str(exc),
            )
            raise InferenceError(
                f"Inference failed for image '{card.image_id}': {exc}"
            ) from exc
