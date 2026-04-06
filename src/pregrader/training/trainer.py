"""
TrainingLoop — builds, trains, and saves the ordinal regression CNN.

Architecture: EfficientNetB0 backbone + multi-head ordinal regression.
  - Backbone: EfficientNetB0 pretrained on ImageNet (transfer learning).
    Frozen for the first N epochs, then unfrozen for fine-tuning.
  - Head: 5 parallel Dense(9, sigmoid) outputs — one for overall grade,
    one per subgrade dimension (centering, corners, edges, surface).
    Each head outputs 9 cumulative probabilities P(Y≤k) for k=1..9.
  - Loss: binary cross-entropy per threshold, summed across heads.
    This approximates the CORN (Conditional Ordinal Regression for Neural
    networks) loss without requiring TensorFlow Addons (deprecated in TF 2.16+).

Why EfficientNetB0 over ResNet or VGG?
- EfficientNetB0 achieves comparable accuracy to ResNet50 at ~5x fewer
  parameters, which matters for inference latency on CPU (no GPU at serving).
- ImageNet pretraining gives strong low-level feature detectors (edges,
  textures) that transfer well to card surface defect detection.

Technical Debt:
- The backbone is frozen for all epochs in this MVP implementation.
  Production approach: freeze for epoch 1–5, unfreeze top N layers for
  fine-tuning in epochs 6+. This requires a LearningRateScheduler and
  a second compile() call — deferred to post-MVP.
- Training runs synchronously on CPU by default. For real training,
  use tf.distribute.MirroredStrategy for multi-GPU or submit to a
  managed training service (Vertex AI, SageMaker).
- The 300-line rule: this file is approaching the limit. If the model
  architecture grows (more heads, attention layers), split into
  trainer.py (training loop) and model.py (architecture definition).
"""

import json
from pathlib import Path
from typing import Any

import tensorflow as tf

from pregrader.logging_config import get_logger
from pregrader.schemas import TrainingConfig
from pregrader.training.augmentation import AugmentationPipeline

logger = get_logger(service="trainer")

# Number of ordinal thresholds — model outputs P(Y≤k) for k=1..9.
_NUM_THRESHOLDS: int = 9

# Input shape must match PreprocessingService output and DatasetBuilder resize.
_INPUT_SHAPE: tuple[int, int, int] = (312, 224, 3)


def _build_model(config: TrainingConfig) -> tf.keras.Model:
    """Construct the EfficientNetB0 + ordinal regression head model.

    Architecture:
      Input (312, 224, 3)
        → EfficientNetB0 backbone (ImageNet weights, frozen)
        → GlobalAveragePooling2D
        → Dense(256, relu) — shared representation layer
        → 5x Dense(9, sigmoid) — one head per grade dimension

    The 5 output heads share the same backbone and dense layer, which
    forces the model to learn a shared card quality representation before
    specializing per dimension. This is more parameter-efficient than
    5 independent models and regularizes the subgrade heads via the
    shared gradient signal from the overall grade head.

    Args:
        config: TrainingConfig with backbone and pretrained_weights settings.

    Returns:
        Compiled tf.keras.Model with 5 sigmoid output heads.
    """
    inputs = tf.keras.Input(shape=_INPUT_SHAPE, name="card_image")

    # Backbone — EfficientNetB0 with ImageNet weights.
    # include_top=False removes the ImageNet classification head.
    # We freeze the backbone so ImageNet features are preserved during
    # the initial training phase on our small dataset.
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=config.pretrained_weights if config.pretrained_weights != "none" else None,
        input_shape=_INPUT_SHAPE,
    )
    backbone.trainable = False  # Frozen — see Technical Debt note above.

    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    # Shared dense layer — learns a card-quality embedding before the
    # heads specialize. Dropout regularizes against overfitting on small datasets.
    x = tf.keras.layers.Dense(256, activation="relu", name="shared_dense")(x)
    x = tf.keras.layers.Dropout(0.3, name="shared_dropout")(x)

    # Five parallel ordinal regression heads — each outputs 9 sigmoid values
    # representing P(Y≤k) for k=1..9 on the PSA 1–10 scale.
    overall = tf.keras.layers.Dense(_NUM_THRESHOLDS, activation="sigmoid", name="overall")(x)
    centering = tf.keras.layers.Dense(_NUM_THRESHOLDS, activation="sigmoid", name="centering")(x)
    corners = tf.keras.layers.Dense(_NUM_THRESHOLDS, activation="sigmoid", name="corners")(x)
    edges = tf.keras.layers.Dense(_NUM_THRESHOLDS, activation="sigmoid", name="edges")(x)
    surface = tf.keras.layers.Dense(_NUM_THRESHOLDS, activation="sigmoid", name="surface")(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "overall": overall,
            "centering": centering,
            "corners": corners,
            "edges": edges,
            "surface": surface,
        },
        name="tcg_pregrader",
    )

    return model


def _make_ordinal_targets(
    labels: tf.Tensor,
    num_thresholds: int = _NUM_THRESHOLDS,
) -> tf.Tensor:
    """Convert integer grade labels to ordinal threshold targets.

    For a grade label y (0-indexed, 0–9), the ordinal target vector is:
        t[k] = 1  if k < y   (P(Y≤k) should be low — grade is above threshold k)
        t[k] = 0  if k >= y  (P(Y≤k) should be high — grade is at or below threshold k)

    Wait — this is inverted from the standard CORN formulation. Let me be precise:
    We want P(Y≤k) to be HIGH when the true grade is LOW (easy to beat threshold k).
    For grade y (0-indexed):
        t[k] = 1  if y <= k  (true grade is at or below threshold k+1)
        t[k] = 0  if y > k   (true grade exceeds threshold k+1)

    Args:
        labels: Integer tensor of shape (batch,), values in [0, 9] (0-indexed grades).
        num_thresholds: Number of ordinal thresholds (9 for PSA 1–10 scale).

    Returns:
        Float32 tensor of shape (batch, num_thresholds) with binary targets.
    """
    # thresholds shape: (1, num_thresholds) — broadcast against labels (batch, 1)
    thresholds = tf.cast(tf.range(num_thresholds), tf.int32)[tf.newaxis, :]
    labels_expanded = tf.cast(labels, tf.int32)[:, tf.newaxis]
    # t[i, k] = 1 if label[i] <= k, else 0
    targets = tf.cast(labels_expanded <= thresholds, tf.float32)
    return targets


class TrainingLoop:
    """Orchestrates model construction, training, and artifact saving.

    Lifecycle:
      1. train() builds the model, prepares datasets, runs the training loop.
      2. The trained model is saved as a TF SavedModel to config.output_dir.
      3. Returns the artifact path for use by ModelRegistry.load().
    """

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        config: TrainingConfig,
    ) -> Path:
        """Build, train, and save the model.

        Args:
            train_ds: Unbatched training dataset of (image, label) pairs.
            val_ds: Unbatched validation dataset of (image, label) pairs.
            config: TrainingConfig with hyperparameters and output paths.

        Returns:
            Path to the saved TF SavedModel directory.
        """
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)

        augmentation = AugmentationPipeline()

        # Apply augmentation only to training data, then batch and prefetch.
        # Prefetch overlaps data loading with model execution — critical for
        # GPU utilization. AUTOTUNE lets TF tune the buffer size dynamically.
        train_ds_prepared = (
            train_ds
            .map(
                lambda img, lbl: (augmentation.apply(img), lbl),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds_prepared = (
            val_ds
            .batch(config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        model = _build_model(config)

        # Compile with Adam and binary cross-entropy per head.
        # Each head gets equal loss weight — adjust if subgrade heads
        # underfit relative to the overall grade head.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
            loss={
                head: "binary_crossentropy"
                for head in ("overall", "centering", "corners", "edges", "surface")
            },
            loss_weights={head: 1.0 for head in ("overall", "centering", "corners", "edges", "surface")},
        )

        logger.info(
            "training_start",
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            output_dir=str(config.output_dir),
        )

        # TensorBoard callback for loss/metric visualization — only added when
        # TensorBoard is installed. In test environments it may not be present.
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(config.output_dir / "best_checkpoint.keras"),
                save_best_only=True,
                monitor="val_loss",
                verbose=0,
            ),
        ]

        try:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir=str(config.log_dir))
            )
        except Exception:
            # TensorBoard not installed — skip silently, training continues.
            logger.warning("tensorboard_unavailable", log_dir=str(config.log_dir))

        # The dataset yields (image, label) but the model expects
        # (image, {head: ordinal_targets}) — we need a wrapper dataset.
        def _add_ordinal_targets(image: tf.Tensor, label: tf.Tensor):
            targets = _make_ordinal_targets(label)
            return image, {
                "overall": targets,
                "centering": targets,
                "corners": targets,
                "edges": targets,
                "surface": targets,
            }

        train_ds_final = train_ds_prepared.map(_add_ordinal_targets)
        val_ds_final = val_ds_prepared.map(_add_ordinal_targets)

        model.fit(
            train_ds_final,
            validation_data=val_ds_final,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # Export as TF SavedModel — the format ModelRegistry.load() expects.
        # Keras 3 uses model.export() for SavedModel format (TFLite/TFServing compatible).
        # model.save() now defaults to the .keras format.
        artifact_path = config.output_dir / "saved_model"
        model.export(str(artifact_path))

        logger.info(
            "training_complete",
            artifact_path=str(artifact_path),
        )

        return artifact_path
