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

Why these three base transforms?
- Horizontal flip: cards can be photographed from either orientation;
  the model should be invariant to left/right mirroring.
- Brightness jitter ±20%: simulates variable lighting conditions in
  card photos (phone flash, natural light, scanner lamp).
- Rotation ±5°: simulates slight tilt in handheld photography.
  Kept small (±5°) because large rotations would crop card edges and
  destroy the corner/edge region crops that subgrade heads rely on.

Why two slab-specific augmentations?
- Glare simulation: PSA slab photos frequently contain specular highlights
  from overhead lighting. Training on synthetic glare makes the model
  robust to this common real-world artifact (Req 12.1).
- Label occlusion: The PSA grade label at the bottom of the slab is
  sometimes partially obscured by fingers, stickers, or shadows. Replacing
  the bottom 15% with a mean-color fill teaches the model to grade from
  card surface features, not the label (Req 12.2).

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

    def __init__(
        self,
        glare_probability: float = 0.3,
        label_occlusion_probability: float = 0.5,
        label_region_fraction: float = 0.15,
    ) -> None:
        # Pre-instantiate outside any tf.function scope — see class docstring.
        self._rotation_layer = tf.keras.layers.RandomRotation(
            factor=_ROTATION_FACTOR,
            fill_mode="reflect",
        )
        # Slab-specific augmentation probabilities (Req 12.1, 12.2, 12.4)
        self._glare_probability = glare_probability
        self._label_occlusion_probability = label_occlusion_probability
        # Fraction of image height occupied by the PSA label region (bottom 15%)
        self._label_region_fraction = label_region_fraction

    def apply(self, image_tensor: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Apply random horizontal flip, brightness jitter, rotation, and slab transforms.

        Order: flip → brightness → rotation → glare → label occlusion.
        ALL transforms are guarded by the `training` flag — when False, the tensor
        is returned unchanged. This satisfies Req 12.5 (validation images never
        augmented) and mirrors the standard Keras layer convention where
        training=False is a strict passthrough.

        Why guard ALL transforms, not just the slab-specific ones?
        Validation metrics must reflect real-world slab photo conditions without
        any synthetic modification. Applying flip/brightness/rotation to validation
        images would distort the loss signal and make val metrics non-comparable
        across epochs. The `training` flag is the single authoritative gate.

        Args:
            image_tensor: Float32 tensor of shape (H, W, 3), values in [0.0, 1.0].
            training: When False, returns image_tensor unchanged. Existing callers
                      pass no argument → default True → unaffected.

        Returns:
            Augmented float32 tensor of the same shape, values clamped to [0.0, 1.0].
            When training=False, returns the input tensor unmodified.
        """
        if not training:
            return image_tensor

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

        # Slab-specific augmentations — applied after geometric transforms so
        # artifact simulation occurs in pixel space (Req 12.3).
        image = self._apply_glare(image)
        image = self._apply_label_occlusion(image)

        return image

    def _apply_glare(self, image: tf.Tensor) -> tf.Tensor:
        """Simulate specular glare by overlaying a semi-transparent ellipse.

        # (H, W, 3) — float32, values in [0.0, 1.0]

        Why upper 85% only? The bottom 15% is the PSA label region. Glare on
        the label would be redundant with label occlusion and could confuse the
        model about label presence. Real-world glare from overhead lighting
        predominantly hits the card surface, not the label (Req 12.1).

        Why tf.cond instead of Python if? This method may be traced by
        tf.function (e.g., inside tf.data.Dataset.map). Python if on a tensor
        value is not graph-safe — tf.cond is the correct graph-mode branch.

        Args:
            image: Float32 tensor of shape (H, W, 3), values in [0.0, 1.0].

        Returns:
            Float32 tensor of shape (H, W, 3) with glare applied or unchanged.
        """
        # Probability gate — sample once per image
        apply_gate = tf.random.uniform([], 0.0, 1.0)

        def glare_fn() -> tf.Tensor:
            # (H, W, 3) — read dynamic shape for graph-mode compatibility
            H = tf.shape(image)[0]
            W = tf.shape(image)[1]

            # Sample ellipse center — constrained to upper 85% to avoid label region
            cx = tf.random.uniform([], 0, W, dtype=tf.int32)
            cy = tf.random.uniform(
                [],
                0,
                tf.cast(tf.cast(H, tf.float32) * 0.85, tf.int32),
                dtype=tf.int32,
            )

            # Sample ellipse axes as fractions of image dimensions
            rx = tf.random.uniform([], 0.05, 0.25) * tf.cast(W, tf.float32)
            ry = tf.random.uniform([], 0.05, 0.25) * tf.cast(H, tf.float32)

            # Sample glare intensity — moderate range avoids total washout
            intensity = tf.random.uniform([], 0.3, 0.7)

            # Build coordinate grids — (W,) and (H,) ranges cast to float
            xs = tf.cast(tf.range(W), tf.float32)  # (W,)
            ys = tf.cast(tf.range(H), tf.float32)  # (H,)

            # meshgrid produces X shape (H, W) and Y shape (H, W)
            X, Y = tf.meshgrid(xs, ys)

            # Ellipse mask: 1.0 inside ellipse, 0.0 outside — shape (H, W)
            mask = tf.cast(
                ((X - tf.cast(cx, tf.float32)) / rx) ** 2
                + ((Y - tf.cast(cy, tf.float32)) / ry) ** 2
                <= 1.0,
                tf.float32,
            )

            # Expand to (H, W, 1) for broadcast against (H, W, 3)
            mask = tf.expand_dims(mask, axis=-1)

            # Blend: inside ellipse → push toward white (1.0) by intensity
            blended = image * (1.0 - intensity * mask) + intensity * mask

            # Clamp to valid pixel range
            return tf.clip_by_value(blended, 0.0, 1.0)

        # Graph-safe conditional — only compute glare_fn() when gate passes
        return tf.cond(
            apply_gate < self._glare_probability,
            true_fn=glare_fn,
            false_fn=lambda: image,
        )

    def _apply_label_occlusion(self, image: tf.Tensor) -> tf.Tensor:
        """Replace the bottom label_region_fraction of rows with mean-color fill.

        # (H, W, 3) — float32, values in [0.0, 1.0]

        Why mean-color fill instead of black or white? A solid black or white
        fill is an obvious synthetic artifact that the model could learn to
        detect. Mean-color fill is visually neutral and harder to distinguish
        from a naturally obscured label region (Req 12.2).

        Why tf.cond? Same graph-mode safety reason as _apply_glare — this
        method may be traced inside tf.data.Dataset.map.

        Args:
            image: Float32 tensor of shape (H, W, 3), values in [0.0, 1.0].

        Returns:
            Float32 tensor of shape (H, W, 3) with label region occluded or unchanged.
        """
        # Probability gate — sample once per image
        apply_gate = tf.random.uniform([], 0.0, 1.0)

        def occlusion_fn() -> tf.Tensor:
            # (H, W, 3) — read dynamic height for graph-mode compatibility
            H = tf.shape(image)[0]

            # Number of rows to replace at the bottom of the image
            label_rows = tf.cast(
                tf.cast(H, tf.float32) * self._label_region_fraction, tf.int32
            )
            keep_rows = H - label_rows

            # Compute mean color across all spatial positions — shape (3,)
            # This gives a neutral fill that blends with the card's color palette
            mean_color = tf.reduce_mean(image, axis=[0, 1])

            # Build fill block: (label_rows, W, 3) filled with mean_color
            fill = (
                tf.ones([label_rows, tf.shape(image)[1], 3], dtype=tf.float32)
                * mean_color
            )

            # Splice: keep top rows unchanged, replace bottom rows with fill
            return tf.concat([image[:keep_rows, :, :], fill], axis=0)

        # Graph-safe conditional
        return tf.cond(
            apply_gate < self._label_occlusion_probability,
            true_fn=occlusion_fn,
            false_fn=lambda: image,
        )
