"""
Full training run on the cleaned manifest.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 20 --batch-size 16 --output artifacts/v1

What this does:
  1. Loads manifest_clean.csv (deduped + slab-filtered)
  2. Computes class weights to compensate for grade imbalance
  3. Trains EfficientNetB0 with ordinal regression heads
  4. Saves the best checkpoint + final SavedModel artifact

Class weighting strategy:
  Inverse-frequency weighting — grades with fewer samples get higher loss weight.
  This prevents the model from ignoring rare grades (1-5) in favour of
  the more common high grades (9-10).
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tensorflow as tf

from pregrader.training.manifest import ManifestLoader
from pregrader.training.dataset import DatasetBuilder
from pregrader.training.trainer import TrainingLoop, _make_ordinal_targets
from pregrader.schemas import TrainingConfig
from pregrader.logging_config import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger(service="train_script")

MANIFEST_PATH = Path("data/manifest_clean.csv")
OUTPUT_DIR = Path("artifacts/v1/")


def compute_class_weights(rows) -> dict[int, float]:
    """Inverse-frequency class weights for the 10 PSA grades (0-indexed).

    Weight for class c = total_samples / (n_classes * count_c)
    This is sklearn's 'balanced' strategy — keeps the effective learning
    rate equal across all grades regardless of sample count.
    """
    grades = [r.overall_grade - 1 for r in rows]  # 0-indexed
    counts = Counter(grades)
    n_total = len(grades)
    n_classes = 10
    weights = {
        g: n_total / (n_classes * counts[g])
        for g in range(n_classes)
        if counts[g] > 0
    }
    # Fill missing grades with max weight
    max_w = max(weights.values())
    for g in range(n_classes):
        if g not in weights:
            weights[g] = max_w
    return weights


def main(epochs: int = 15, batch_size: int = 16, output_dir: Path = OUTPUT_DIR) -> None:
    print("=== TCG Pregrader — Full Training Run ===\n")

    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found.")
        print("Run first: python scripts/clean_dataset.py")
        sys.exit(1)

    # Load manifest
    rows = ManifestLoader().load(MANIFEST_PATH)
    grade_counts = Counter(r.overall_grade for r in rows)
    print(f"Loaded {len(rows)} samples across grades:")
    for g in range(1, 11):
        print(f"  Grade {g:>2}: {grade_counts[g]:>4}")
    print()

    # Class weights
    class_weights = compute_class_weights(rows)
    print("Class weights (inverse frequency):")
    for g in range(10):
        print(f"  Grade {g+1:>2} (idx {g}): {class_weights[g]:.3f}")
    print()

    # Build datasets
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        train_ratio=0.75,
        val_ratio=0.15,
        output_dir=output_dir,
        log_dir=output_dir / "logs",
    )

    builder = DatasetBuilder()
    train_ds, val_ds, _ = builder.build(rows, config)

    train_count = sum(1 for _ in train_ds)
    val_count = sum(1 for _ in val_ds)
    print(f"Train samples: {train_count}, Val samples: {val_count}\n")

    # Train
    print(f"Training for {epochs} epochs, batch_size={batch_size}...")
    print(f"Output: {output_dir}\n")

    loop = TrainingLoop()
    artifact_path = loop.train(
        train_ds,
        val_ds,
        config,
        class_weight=class_weights,
    )

    print(f"\nTraining complete. SavedModel: {artifact_path}")
    print("\nUpdate .env to point at this artifact:")
    print(f"  POKEMON_MODEL_ARTIFACT_PATH={artifact_path.resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, output_dir=args.output)
