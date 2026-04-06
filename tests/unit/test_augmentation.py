"""
Unit tests for AugmentationPipeline.

Strategy: Feed known synthetic tensors and assert invariants on the output.
We can't assert exact pixel values (transforms are random), but we can
assert shape preservation and value range clamping — these must hold
regardless of the random state.
"""

import numpy as np
import pytest
import tensorflow as tf

from pregrader.training.augmentation import AugmentationPipeline

# Standard card tensor shape after preprocessing.
_H, _W, _C = 312, 224, 3


def _make_tensor(value: float = 0.5) -> tf.Tensor:
    """Return a solid-color float32 tensor of card shape."""
    return tf.constant(
        np.full((_H, _W, _C), value, dtype=np.float32)
    )


class TestAugmentationOutputShape:
    def test_output_shape_matches_input(self) -> None:
        """Shape must be preserved — (312, 224, 3) in, (312, 224, 3) out."""
        pipeline = AugmentationPipeline()
        output = pipeline.apply(_make_tensor())
        assert output.shape == (_H, _W, _C)

    def test_output_dtype_is_float32(self) -> None:
        """Output must remain float32 — TF model expects float32 input."""
        pipeline = AugmentationPipeline()
        output = pipeline.apply(_make_tensor())
        assert output.dtype == tf.float32


class TestAugmentationValueRange:
    def test_output_values_in_0_1(self) -> None:
        """All pixel values must be clamped to [0.0, 1.0] after augmentation."""
        pipeline = AugmentationPipeline()
        output = pipeline.apply(_make_tensor(0.5))
        arr = output.numpy()
        assert arr.min() >= 0.0, f"Min value {arr.min()} below 0.0"
        assert arr.max() <= 1.0, f"Max value {arr.max()} above 1.0"

    def test_bright_image_stays_clamped(self) -> None:
        """Brightness jitter on a near-white image must not exceed 1.0."""
        pipeline = AugmentationPipeline()
        output = pipeline.apply(_make_tensor(0.95))
        assert output.numpy().max() <= 1.0

    def test_dark_image_stays_clamped(self) -> None:
        """Brightness jitter on a near-black image must not go below 0.0."""
        pipeline = AugmentationPipeline()
        output = pipeline.apply(_make_tensor(0.05))
        assert output.numpy().min() >= 0.0


class TestAugmentationNonDeterminism:
    def test_repeated_calls_produce_different_outputs(self) -> None:
        """Augmentation must be non-deterministic — 100 calls should not all match."""
        pipeline = AugmentationPipeline()
        input_tensor = _make_tensor(0.5)
        outputs = [pipeline.apply(input_tensor).numpy() for _ in range(20)]

        # At least some outputs must differ from the first.
        first = outputs[0]
        num_different = sum(
            1 for o in outputs[1:] if not np.allclose(o, first, atol=1e-4)
        )
        assert num_different >= 5, (
            f"Expected at least 5/19 augmented outputs to differ, got {num_different}"
        )
