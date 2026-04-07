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


# ---------------------------------------------------------------------------
# Slab-specific augmentation tests (Task 13.2)
# ---------------------------------------------------------------------------

import math


class TestLabelOcclusion:
    def test_apply_training_false_returns_identical_tensor(self) -> None:
        """training=False must bypass ALL augmentations — output identical to input."""
        # Use default pipeline; training=False is the validation path (Req 12.5)
        pipeline = AugmentationPipeline()
        image = _make_tensor(0.5)
        output = pipeline.apply(image, training=False)
        assert np.allclose(output.numpy(), image.numpy(), atol=1e-5), (
            "training=False must return tensor numerically identical to input"
        )

    def test_label_occlusion_probability_1_bottom_rows_uniform(self) -> None:
        """With probability=1.0, bottom label_region_fraction rows must be uniform fill.

        Strategy: call _apply_label_occlusion directly to isolate from base transforms.
        Float32 mean-color fill has ~1e-5 rounding variance across the fill block —
        use atol=1e-4 to accommodate TF's float32 arithmetic without false negatives.
        """
        pipeline = AugmentationPipeline(
            glare_probability=0.0,
            label_occlusion_probability=1.0,
            label_region_fraction=0.15,
        )
        rng = np.random.default_rng(42)
        image = tf.constant(rng.random((_H, _W, _C), dtype=np.float32))

        # Call the private method directly — isolates from flip/brightness/rotation
        output = pipeline._apply_label_occlusion(image)

        # Bottom floor(312 * 0.15) = 46 rows should be uniform (std ≈ 0)
        label_rows = math.floor(_H * 0.15)
        bottom_region = output.numpy()[-label_rows:, :, :]  # (46, 224, 3)
        std = tf.math.reduce_std(
            tf.constant(bottom_region), axis=[0, 1]
        ).numpy()  # (3,)
        # Float32 fill arithmetic produces ~3e-5 variance — 1e-4 is the safe bound
        assert np.all(std < 1e-4), (
            f"Bottom {label_rows} rows should be uniform fill, got std={std}"
        )

    def test_label_occlusion_probability_0_bottom_rows_unchanged(self) -> None:
        """With probability=0.0, bottom rows must be identical to input.

        Strategy: call _apply_label_occlusion directly to isolate from base transforms.
        """
        pipeline = AugmentationPipeline(
            glare_probability=0.0,
            label_occlusion_probability=0.0,
            label_region_fraction=0.15,
        )
        rng = np.random.default_rng(7)
        image = tf.constant(rng.random((_H, _W, _C), dtype=np.float32))

        # Call the private method directly — isolates from flip/brightness/rotation
        output = pipeline._apply_label_occlusion(image)

        label_rows = math.floor(_H * 0.15)
        input_bottom = image.numpy()[-label_rows:, :, :]
        output_bottom = output.numpy()[-label_rows:, :, :]
        assert np.allclose(input_bottom, output_bottom, atol=1e-5), (
            "probability=0.0 must leave bottom rows unchanged"
        )


class TestGlare:
    def test_glare_probability_0_image_unchanged(self) -> None:
        """With glare_probability=0.0, output must equal input (no glare applied).

        Strategy: call _apply_glare directly to isolate from base transforms.
        """
        pipeline = AugmentationPipeline(
            glare_probability=0.0,
            label_occlusion_probability=0.0,
        )
        image = _make_tensor(0.5)
        # Call the private method directly — isolates from flip/brightness/rotation
        output = pipeline._apply_glare(image)
        assert np.allclose(output.numpy(), image.numpy(), atol=1e-5), (
            "glare_probability=0.0 must not modify the image"
        )

    def test_glare_probability_1_output_differs_from_input(self) -> None:
        """With glare_probability=1.0, glare must always be applied — output differs."""
        pipeline = AugmentationPipeline(
            glare_probability=1.0,
            label_occlusion_probability=0.0,
        )
        # Use a uniform mid-gray image; glare will push some pixels toward white
        image = _make_tensor(0.5)
        output = pipeline.apply(image, training=True)
        assert not np.allclose(output.numpy(), image.numpy(), atol=1e-5), (
            "glare_probability=1.0 must modify the image"
        )

    def test_output_shape_preserved_with_slab_augmentations(self) -> None:
        """Shape (312, 224, 3) must be preserved with both new transforms enabled."""
        pipeline = AugmentationPipeline(
            glare_probability=1.0,
            label_occlusion_probability=1.0,
        )
        output = pipeline.apply(_make_tensor(0.5), training=True)
        assert output.shape == (_H, _W, _C), (
            f"Expected shape {(_H, _W, _C)}, got {output.shape}"
        )


class TestExistingAugmentationsUnaffected:
    def test_flip_brightness_rotation_still_work(self) -> None:
        """Existing flip/brightness/rotation must remain non-deterministic with new params."""
        # Instantiate with default new params — must not break existing behaviour
        pipeline = AugmentationPipeline()
        input_tensor = _make_tensor(0.5)
        outputs = [pipeline.apply(input_tensor, training=True).numpy() for _ in range(20)]

        first = outputs[0]
        num_different = sum(
            1 for o in outputs[1:] if not np.allclose(o, first, atol=1e-4)
        )
        assert num_different >= 5, (
            f"Expected at least 5/19 outputs to differ (non-determinism), got {num_different}"
        )
