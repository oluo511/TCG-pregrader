"""
Smoke test: verify the full training pipeline runs end-to-end on synthetic data.

Usage (from TCG-pregrader/):
    python scripts/smoke_test_training.py

What this validates:
  ManifestLoader → DatasetBuilder → AugmentationPipeline → TrainingLoop → SavedModel

If this completes without error, the training pipeline is correctly wired.
The model won't learn anything meaningful from synthetic data — that's fine.
We're testing the plumbing, not the model quality.
"""

import sys
from pathlib import Path

# Add src to path so pregrader imports work
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pregrader.training.manifest import ManifestLoader
from pregrader.training.dataset import DatasetBuilder
from pregrader.training.trainer import TrainingLoop
from pregrader.schemas import TrainingConfig

MANIFEST_PATH = Path("data/manifest.csv")
OUTPUT_DIR = Path("artifacts/smoke_test/")

def main() -> None:
    print("=== Training Smoke Test ===\n")

    # Step 1: Load manifest
    print(f"Loading manifest from {MANIFEST_PATH}...")
    if not MANIFEST_PATH.exists():
        print("ERROR: manifest not found. Run first:")
        print("  python -m data_pipeline.generate_synthetic_data --images-per-grade 50")
        sys.exit(1)

    rows = ManifestLoader().load(MANIFEST_PATH)
    print(f"  Loaded {len(rows)} rows across grades {sorted(set(r.overall_grade for r in rows))}\n")

    # Step 2: Build datasets — use minimal config for speed
    config = TrainingConfig(
        epochs=2,
        batch_size=8,
        train_ratio=0.70,
        val_ratio=0.15,
        output_dir=OUTPUT_DIR,
        log_dir=OUTPUT_DIR / "logs",
    )

    print("Building train/val/test splits...")
    builder = DatasetBuilder()
    train_ds, val_ds, test_ds = builder.build(rows, config)

    # DatasetBuilder returns unbatched datasets — TrainingLoop handles
    # batching internally. Count elements before batching for the log.
    train_count = sum(1 for _ in train_ds)
    val_count   = sum(1 for _ in val_ds)
    print(f"  Train samples: {train_count}")
    print(f"  Val samples:   {val_count}\n")

    # Step 3: Train for 2 epochs
    print(f"Training for {config.epochs} epochs (smoke test — expect ~random accuracy)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    artifact_path = TrainingLoop().train(train_ds, val_ds, config)

    print(f"\nTraining complete. SavedModel written to: {artifact_path}")
    print("\nSmoke test PASSED — pipeline is correctly wired.")

if __name__ == "__main__":
    main()
