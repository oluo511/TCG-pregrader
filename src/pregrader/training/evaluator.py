"""
Evaluator — computes post-training metrics and writes an evaluation report.

Design pattern: Pure functions for metric computation + a thin orchestrator.
_compute_mae() and _compute_within_one() are module-level functions (not
methods) so they can be tested independently without constructing an Evaluator
or loading a model. The Evaluator.evaluate() method orchestrates inference,
metric computation, and report writing.

Metrics computed:
  - MAE (Mean Absolute Error): average |predicted_grade - true_grade|.
    Target: MAE < 0.5 grades for a production-quality model.
  - ±1 Accuracy: fraction of predictions within 1 grade of true grade.
    Target: >85% for a production-quality model.
  - 10×10 Confusion Matrix: rows = true grade, cols = predicted grade.
    Useful for identifying systematic bias (e.g., always predicting 7–8).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from pregrader.logging_config import get_logger
from pregrader.services.grader import _decode_ordinal

logger = get_logger(service="evaluator")


# ---------------------------------------------------------------------------
# Pure metric functions — testable without a model or dataset
# ---------------------------------------------------------------------------

def _compute_mae(pairs: list[tuple[int, int]]) -> float:
    """Compute Mean Absolute Error over (predicted, actual) grade pairs.

    Args:
        pairs: List of (predicted_grade, actual_grade) tuples.
            Grades are integers on the PSA 1–10 scale.

    Returns:
        Mean absolute error as a float. Returns 0.0 for empty input.
    """
    if not pairs:
        return 0.0
    return float(np.mean([abs(pred - actual) for pred, actual in pairs]))


def _compute_within_one(pairs: list[tuple[int, int]]) -> float:
    """Compute the fraction of predictions within ±1 grade of the true grade.

    Args:
        pairs: List of (predicted_grade, actual_grade) tuples.

    Returns:
        Fraction in [0.0, 1.0]. Returns 0.0 for empty input.
    """
    if not pairs:
        return 0.0
    within_one = sum(1 for pred, actual in pairs if abs(pred - actual) <= 1)
    return within_one / len(pairs)


def _compute_confusion_matrix(
    pairs: list[tuple[int, int]],
) -> list[list[int]]:
    """Build a 10×10 confusion matrix for PSA grades 1–10.

    Args:
        pairs: List of (predicted_grade, actual_grade) tuples.
            Grades are integers on the PSA 1–10 scale (1-indexed).

    Returns:
        10×10 list of lists where matrix[true-1][pred-1] = count.
    """
    matrix = [[0] * 10 for _ in range(10)]
    for pred, actual in pairs:
        # Clamp to valid range defensively — model output should already be 1–10.
        true_idx = max(0, min(9, actual - 1))
        pred_idx = max(0, min(9, pred - 1))
        matrix[true_idx][pred_idx] += 1
    return matrix


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """Runs inference on a test dataset and produces an evaluation report.

    Lifecycle:
      1. evaluate() iterates the test dataset, runs model inference per batch.
      2. Decodes ordinal outputs to integer grades using _decode_ordinal.
      3. Computes MAE, ±1 accuracy, and confusion matrix.
      4. Writes a JSON report to output_dir/eval_report.json.
      5. Returns the metrics dict for programmatic use (e.g., CI gates).
    """

    def evaluate(
        self,
        model: Any,
        test_ds: tf.data.Dataset,
        output_dir: Path,
    ) -> dict:
        """Run evaluation and write the report.

        Args:
            model: Loaded TF SavedModel (or Keras model) with the 5-head
                   ordinal regression architecture.
            test_ds: Unbatched test dataset of (image, label) pairs.
            output_dir: Directory to write eval_report.json.

        Returns:
            Dict with keys: mae, within_one_accuracy, confusion_matrix,
            total_samples.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs: list[tuple[int, int]] = []

        # Batch for efficient inference — batch size 32 is safe for CPU eval.
        batched_ds = test_ds.batch(32)

        for images, labels in batched_ds:
            outputs = model(images, training=False)
            overall_cumprobs = outputs["overall"].numpy()  # shape (batch, 9)

            for i, label in enumerate(labels.numpy()):
                cumprobs = overall_cumprobs[i].astype(np.float64)
                predicted_grade, _ = _decode_ordinal(cumprobs)
                # Labels are 0-indexed in the dataset; convert back to 1-indexed PSA scale.
                true_grade = int(label) + 1
                pairs.append((predicted_grade, true_grade))

        mae = _compute_mae(pairs)
        within_one = _compute_within_one(pairs)
        confusion = _compute_confusion_matrix(pairs)

        metrics = {
            "mae": round(mae, 4),
            "within_one_accuracy": round(within_one, 4),
            "confusion_matrix": confusion,
            "total_samples": len(pairs),
        }

        report_path = output_dir / "eval_report.json"
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        logger.info(
            "evaluation_complete",
            mae=metrics["mae"],
            within_one_accuracy=metrics["within_one_accuracy"],
            total_samples=metrics["total_samples"],
            report_path=str(report_path),
        )

        return metrics
