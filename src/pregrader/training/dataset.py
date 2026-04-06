"""
DatasetBuilder — builds train/val/test tf.data.Dataset splits from manifest rows.

Design pattern: Deterministic pipeline with fixed-seed shuffle.
The split strategy is: shuffle once with a fixed seed → slice by index.
This guarantees:
  1. Reproducibility — same seed always produces the same splits.
  2. Partition disjointness — index slicing on a shuffled list is mutually
     exclusive by construction; no sample can appear in two splits.
  3. Union completeness — the three slices cover every index exactly once.

Why not sklearn train_test_split?
- We're already in the TF ecosystem; tf.data pipelines compose naturally
  with the training loop (batching, prefetch, augmentation).
- sklearn would require converting back to tf.data anyway, adding a
  round-trip through numpy that's unnecessary overhead.

Technical Debt: Dataset is built from in-memory lists of image paths.
At scale (>100k images), replace with tf.data.Dataset.from_generator()
backed by a lazy file iterator, or pre-convert to TFRecord shards and use
tf.data.TFRecordDataset for O(1) memory overhead during dataset construction.
"""

import random
from pathlib import Path
from typing import Final

import numpy as np
import tensorflow as tf

from pregrader.logging_config import get_logger
from pregrader.schemas import ManifestRow, TrainingConfig

logger = get_logger(service="dataset_builder")

# Fixed seed for reproducible shuffles — changing this invalidates all
# previously generated splits, so treat it as a versioned constant.
_SHUFFLE_SEED: Final[int] = 42

# Target image dimensions — must match PreprocessingService output.
_IMG_HEIGHT: Final[int] = 312
_IMG_WIDTH: Final[int] = 224
_IMG_CHANNELS: Final[int] = 3


def _load_and_preprocess_image(image_path: str, label: int) -> tuple[tf.Tensor, int]:
    """Load a JPEG/PNG from disk and normalize to [0.0, 1.0].

    This function runs inside a tf.data pipeline (via map()), so it must
    use TF ops only — no Python I/O or numpy calls.

    Args:
        image_path: String tensor — path to the image file.
        label: Integer overall_grade label (1–10).

    Returns:
        (image_tensor, label) where image_tensor has shape (312, 224, 3)
        and values in [0.0, 1.0].
    """
    raw = tf.io.read_file(image_path)
    # decode_image handles JPEG, PNG, and BMP; expand_animations=False
    # prevents GIF frames from adding a batch dimension.
    image = tf.image.decode_image(raw, channels=_IMG_CHANNELS, expand_animations=False)
    image = tf.image.resize(image, [_IMG_HEIGHT, _IMG_WIDTH])
    # Cast to float32 and normalize — uint8 [0,255] → float32 [0.0, 1.0].
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


class DatasetBuilder:
    """Converts a list of ManifestRow objects into train/val/test tf.data splits.

    Lifecycle:
      1. Shuffle rows with a fixed seed for reproducibility.
      2. Slice into train / val / test by index using configured ratios.
      3. Build a tf.data.Dataset for each split with image loading + preprocessing.
      4. Log grade class distribution and split sizes before returning.

    The returned datasets are unbatched — callers (TrainingLoop) apply
    .batch() and .prefetch() according to their own TrainingConfig.
    """

    def build(
        self,
        rows: list[ManifestRow],
        config: TrainingConfig,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Build and return (train, val, test) tf.data.Dataset splits.

        Args:
            rows: Validated ManifestRow objects from ManifestLoader.
            config: TrainingConfig with train_ratio, val_ratio, batch_size.

        Returns:
            (train_ds, val_ds, test_ds) — unbatched, preprocessed datasets.

        Raises:
            ValueError: If rows is empty (no data to split).
        """
        if not rows:
            raise ValueError("Cannot build dataset from empty manifest.")

        # Deterministic shuffle — same seed always produces the same order.
        # We shuffle the list in-place on a copy to avoid mutating the caller's data.
        shuffled = rows[:]
        random.seed(_SHUFFLE_SEED)
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * config.train_ratio)
        val_end = train_end + int(n * config.val_ratio)
        # Test split gets the remainder — guaranteed non-empty by TrainingConfig validator.

        train_rows = shuffled[:train_end]
        val_rows = shuffled[train_end:val_end]
        test_rows = shuffled[val_end:]

        # Log grade distribution before building datasets so operators can
        # spot class imbalance before committing to a training run.
        self._log_statistics(train_rows, val_rows, test_rows, n)

        train_ds = self._build_split(train_rows, config)
        val_ds = self._build_split(val_rows, config)
        test_ds = self._build_split(test_rows, config)

        return train_ds, val_ds, test_ds

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_split(
        self,
        rows: list[ManifestRow],
        config: TrainingConfig,
    ) -> tf.data.Dataset:
        """Build a tf.data.Dataset for a single split.

        Args:
            rows: ManifestRow objects for this split.
            config: TrainingConfig (batch_size used by callers, not here).

        Returns:
            Unbatched tf.data.Dataset of (image_tensor, label) pairs.
        """
        # Extract parallel lists — tf.data.Dataset.from_tensor_slices
        # requires homogeneous arrays, not a list of objects.
        paths = [str(row.image_path) for row in rows]
        # Labels are 0-indexed for cross-entropy loss: grade 1 → label 0.
        labels = [row.overall_grade - 1 for row in rows]

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        # map() applies the image loading function to each (path, label) pair.
        # num_parallel_calls=AUTOTUNE lets TF tune parallelism to available CPUs.
        ds = ds.map(
            _load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds

    def _log_statistics(
        self,
        train_rows: list[ManifestRow],
        val_rows: list[ManifestRow],
        test_rows: list[ManifestRow],
        total: int,
    ) -> None:
        """Log split sizes and grade class distribution before training.

        Why log before returning (not after)?
        Operators need to see the distribution before committing GPU time.
        If the distribution is badly skewed, they should abort and re-balance
        the manifest rather than discover it from a poor eval report.
        """
        all_grades = [r.overall_grade for r in train_rows + val_rows + test_rows]
        grade_counts = {
            grade: all_grades.count(grade)
            for grade in range(1, 11)
            if all_grades.count(grade) > 0
        }

        logger.info(
            "dataset_statistics",
            total_samples=total,
            train_size=len(train_rows),
            val_size=len(val_rows),
            test_size=len(test_rows),
            grade_distribution=grade_counts,
        )
