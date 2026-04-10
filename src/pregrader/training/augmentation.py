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

    def apply_batch(self, images: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Apply augmentation to a batch of images (rank-4 tensor).

        This is the preferred entry point for tf.data.map() — operating on
        batches avoids the expand_dims/squeeze workaround needed for rank-3
        single images, and RandomRotation handles rank-4 natively.

        Args:
            images: Float32 tensor of shape (B, H, W, 3), values in [0.0, 1.0].
            training: When False, returns images unchanged.

        Returns:
            Augmented float32 tensor of shape (B, H, W, 3).
        """
        if not training:
            return images

        # Apply per-image ops that work on rank-4 batches natively
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_brightness(images, max_delta=_BRIGHTNESS_DELTA)
        images = tf.clip_by_value(images, 0.0, 1.0)

        # RandomRotation accepts rank-4 (B, H, W, C) natively
        images = self._rotation_layer(images, training=True)
        images = tf.clip_by_value(images, 0.0, 1.0)

        # Apply slab-specific augmentations per image using vectorized_map
        # to avoid the dynamic shape issues of map() on rank-3 tensors.
        images = tf.vectorized_map(self._apply_glare, images)
        images = tf.vectorized_map(self._apply_label_occlusion, images)

        return images

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

        # Random rotation ±5°.
        # expand_dims/squeeze wraps the single image as a fake batch so
        # RandomRotation receives rank-4 input as it expects.
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

        Implementation note: uses static shape (_HEIGHT, _WIDTH) constants
        rather than dynamic tf.shape() to avoid shape inference issues when
        traced inside tf.data.map(). This means the augmentation is tied to
        the fixed input dimensions — acceptable since the pipeline always
        produces (312, 224, 3) images after preprocessing.
        """
        apply_gate = tf.random.uniform([], 0.0, 1.0)

        def glare_fn() -> tf.Tensor:
            # Use static dimensions to avoid dynamic shape issues in graph mode.
            # These must match the actual image dimensions from the pipeline.
            H_static = 312
            W_static = 224

            cx = tf.random.uniform([], 0, W_static, dtype=tf.int32)
            cy = tf.random.uniform([], 0, int(H_static * 0.85), dtype=tf.int32)
            rx = tf.random.uniform([], 0.05, 0.25) * float(W_static)
            ry = tf.random.uniform([], 0.05, 0.25) * float(H_static)
            intensity = tf.random.uniform([], 0.3, 0.7)

            # Build coordinate grids with static shapes — no dynamic tile needed.
            # X[i,j] = j (column), Y[i,j] = i (row)
            col_idx = tf.cast(tf.range(W_static), tf.float32)          # (W,)
            row_idx = tf.cast(tf.range(H_static), tf.float32)          # (H,)

            # Static tile: shapes are known at trace time
            X = tf.tile(tf.reshape(col_idx, [1, W_static]), [H_static, 1])  # (H, W)
            Y = tf.tile(tf.reshape(row_idx, [H_static, 1]), [1, W_static])  # (H, W)

            mask = tf.cast(
                ((X - tf.cast(cx, tf.float32)) / rx) ** 2
                + ((Y - tf.cast(cy, tf.float32)) / ry) ** 2
                <= 1.0,
                tf.float32,
            )  # (H, W)

            mask = tf.expand_dims(mask, axis=-1)  # (H, W, 1)
            blended = image * (1.0 - intensity * mask) + intensity * mask
            return tf.clip_by_value(blended, 0.0, 1.0)

        return tf.cond(
            apply_gate < self._glare_probability,
            true_fn=glare_fn,
            false_fn=lambda: image,
        )

    def _apply_label_occlusion(self, image: tf.Tensor) -> tf.Tensor:
        """Replace the bottom label_region_fraction of rows with mean-color fill.

        # (H, W, 3) — float32, values in [0.0, 1.0]

        Uses static width constant (224) to avoid dynamic shape issues in
        tf.data.map() graph tracing — same constraint as _apply_glare.
        """
        apply_gate = tf.random.uniform([], 0.0, 1.0)

        def occlusion_fn() -> tf.Tensor:
            H = tf.shape(image)[0]

            label_rows = tf.cast(
                tf.cast(H, tf.float32) * self._label_region_fraction, tf.int32
            )
            keep_rows = H - label_rows

            mean_color = tf.reduce_mean(image, axis=[0, 1])  # (3,)

            # Build a mask: 1.0 for rows to keep, 0.0 for label region rows.
            # This avoids concat with dynamic shapes entirely.
            # keep_mask shape: (H, 1, 1) — broadcasts against (H, W, 3)
            keep_mask = tf.cast(
                tf.range(H) < keep_rows, tf.float32
            )  # (H,)
            keep_mask = tf.reshape(keep_mask, [H, 1, 1])  # (H, 1, 1)

            # Fill mask: 1.0 for label region rows, 0.0 for card body rows
            fill_mask = 1.0 - keep_mask  # (H, 1, 1)

            # Blend: keep original pixels in card body, replace with mean in label
            return image * keep_mask + mean_color * fill_mask

        return tf.cond(
            apply_gate < self._label_occlusion_probability,
            true_fn=occlusion_fn,
            false_fn=lambda: image,
        )
