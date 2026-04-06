"""
AugmentationPipeline — randomized image transforms for training data.

Design pattern: Stateless transform pipeline using TF ops exclusively.
Each transform is applied independently so they compose without coupling —
adding or removing a transform doesn't affect the others.

Why TF ops instead of PIL/numpy?
- This runs inside a tf.data.Dataset.map() call in the training loop.
  TF ops execute in graph mode (compiled), which is significantly faster
  than Python-level PIL transforms when processing thousands of images.
- PIL/numpy would require py_function() wrapping, which breaks graph
  compilation and kills parallelism.

Why these three transforms specifically?
- Horizontal flip: cards can be photographed from either orientation;
  the model should be invariant to left/right mirroring.
- Brightness jitter ±20%: simulates variable lighting conditions in
  card photos (phone flash, natural light, scanner lamp).
- Rotation ±5°: simulates slight tilt in handheld photography.
  Kept small (±5°) because large rotations would crop card edges and
  destroy the corner/edge region crops that subgrade heads rely on.

Technical Debt: tfa.image.rotate (TensorFlow Addons) is the clean way
to do arbitrary rotation in TF, but TFA is deprecated for TF 2.16+.
We use tf.keras.layers.RandomRotation as a workaround, which requires
wrapping the tensor in a fake batch. At scale, migrate to
keras_cv.layers.RandomRotation or implement a custom tfa-free rotation
using tf.raw_ops.ImageProjectiveTransformV3.
"""

from typing import Final

import tensorflow as tf

# Rotation factor is expressed as a fraction of 2π in Keras convention.
# ±5° = ±5/360 ≈ ±0.0139 of a full rotation.
_ROTATION_FACTOR: Final[float] = 5.0 / 360.0

# Brightness delta: max absolute change in pixel intensity (0.0–1.0 scale).
_BRIGHTNESS_DELTA: Final[float] = 0.2


class AugmentationPipeline:
    """Applies randomized transforms to a single image tensor.

    Designed to be called inside tf.data.Dataset.map() during training.
    Each call produces a different random augmentation — this is intentional
    and is what gives the model exposure to varied inputs per epoch.

    NOT applied during validation or inference — only training data should
    be augmented. The DatasetBuilder / TrainingLoop are responsible for
    calling apply() only on the training split.

    Why pre-instantiate RandomRotation in __init__?
    tf.keras.layers.RandomRotation creates a tf.Variable (its seed state)
    at construction time. If instantiated inside a tf.data.map() lambda,
    TF's graph tracing creates a new Variable on every trace — which raises
    ValueError. Pre-instantiating in __init__ creates the Variable once,
    outside any tf.function scope.
    """

    def __init__(self) -> None:
        # Pre-instantiate outside any tf.function scope — see class docstring.
        self._rotation_layer = tf.keras.layers.RandomRotation(
            factor=_ROTATION_FACTOR,
            fill_mode="reflect",
        )

    def apply(self, image_tensor: tf.Tensor) -> tf.Tensor:
        """Apply random horizontal flip, brightness jitter, and rotation.

        Each transform is applied independently with its own random state.
        The order is: flip → brightness → rotation, which matches standard
        augmentation pipeline conventions (geometric transforms last).

        Args:
            image_tensor: Float32 tensor of shape (H, W, 3), values in [0.0, 1.0].

        Returns:
            Augmented float32 tensor of the same shape, values clamped to [0.0, 1.0].
        """
        image = image_tensor

        # Random horizontal flip — 50% probability.
        image = tf.image.random_flip_left_right(image)

        # Random brightness jitter — delta drawn uniformly from [-0.2, +0.2].
        # clip_by_value ensures we stay in [0.0, 1.0] after the shift.
        image = tf.image.random_brightness(image, max_delta=_BRIGHTNESS_DELTA)
        image = tf.clip_by_value(image, 0.0, 1.0)

        # Random rotation ±5° using tfa-free approach.
        # We use tf.keras.layers.RandomRotation but it must be pre-instantiated
        # (see __init__) to avoid creating tf.Variables inside tf.function traces.
        image = tf.expand_dims(image, axis=0)
        image = self._rotation_layer(image, training=True)
        image = tf.squeeze(image, axis=0)

        # Final clamp — rotation interpolation can push values slightly outside [0,1].
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image
