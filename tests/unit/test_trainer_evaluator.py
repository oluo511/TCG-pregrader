"""
Unit tests for TrainingLoop and Evaluator.

Strategy:
  - Metric functions (_compute_mae, _compute_within_one): pure functions,
    tested with known inputs and expected outputs.
  - Evaluator.evaluate(): mocked model returning fixed cumulative probs,
    verifies JSON report is written and parseable with correct structure.
  - TrainingLoop.train(): runs 1 epoch on a tiny synthetic dataset (5 images)
    to verify the SavedModel artifact is written without crashing.
    We use pretrained_weights="none" to skip the ImageNet download.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from pregrader.schemas import TrainingConfig
from pregrader.training.evaluator import (
    Evaluator,
    _compute_confusion_matrix,
    _compute_mae,
    _compute_within_one,
)
from pregrader.training.trainer import TrainingLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_png(path: Path, size: tuple[int, int] = (224, 312)) -> Path:
    img = Image.new("RGB", size, color=(100, 150, 200))
    img.save(path, format="PNG")
    return path


def _make_tiny_dataset(tmp_path: Path, n: int = 6) -> tf.data.Dataset:
    """Build a minimal tf.data.Dataset of (image_tensor, label) pairs."""
    images = []
    labels = []
    for i in range(n):
        img_path = _write_png(tmp_path / f"card_{i}.png")
        raw = tf.io.read_file(str(img_path))
        img = tf.image.decode_png(raw, channels=3)
        img = tf.image.resize(img, [312, 224])
        img = tf.cast(img, tf.float32) / 255.0
        images.append(img.numpy())
        labels.append(i % 9)  # 0-indexed grades 0–8

    ds = tf.data.Dataset.from_tensor_slices(
        (np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32))
    )
    return ds


# ---------------------------------------------------------------------------
# _compute_mae
# ---------------------------------------------------------------------------

class TestComputeMAE:
    def test_perfect_predictions_mae_is_zero(self) -> None:
        pairs = [(5, 5), (8, 8), (3, 3)]
        assert _compute_mae(pairs) == pytest.approx(0.0)

    def test_known_mae(self) -> None:
        # |5-4| + |7-9| + |3-3| = 1 + 2 + 0 = 3, mean = 1.0
        pairs = [(5, 4), (7, 9), (3, 3)]
        assert _compute_mae(pairs) == pytest.approx(1.0)

    def test_empty_pairs_returns_zero(self) -> None:
        assert _compute_mae([]) == pytest.approx(0.0)

    def test_single_pair(self) -> None:
        assert _compute_mae([(8, 5)]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# _compute_within_one
# ---------------------------------------------------------------------------

class TestComputeWithinOne:
    def test_all_within_one(self) -> None:
        pairs = [(5, 5), (5, 6), (5, 4)]
        assert _compute_within_one(pairs) == pytest.approx(1.0)

    def test_none_within_one(self) -> None:
        pairs = [(1, 5), (2, 8), (3, 9)]
        assert _compute_within_one(pairs) == pytest.approx(0.0)

    def test_half_within_one(self) -> None:
        pairs = [(5, 5), (1, 9)]  # first within, second not
        assert _compute_within_one(pairs) == pytest.approx(0.5)

    def test_empty_pairs_returns_zero(self) -> None:
        assert _compute_within_one([]) == pytest.approx(0.0)

    def test_boundary_exactly_one_apart(self) -> None:
        assert _compute_within_one([(7, 8)]) == pytest.approx(1.0)
        assert _compute_within_one([(7, 9)]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _compute_confusion_matrix
# ---------------------------------------------------------------------------

class TestComputeConfusionMatrix:
    def test_matrix_is_10x10(self) -> None:
        matrix = _compute_confusion_matrix([(5, 5)])
        assert len(matrix) == 10
        assert all(len(row) == 10 for row in matrix)

    def test_perfect_prediction_on_diagonal(self) -> None:
        pairs = [(grade, grade) for grade in range(1, 11)]
        matrix = _compute_confusion_matrix(pairs)
        for i in range(10):
            assert matrix[i][i] == 1

    def test_off_diagonal_prediction(self) -> None:
        # True grade 8, predicted grade 6 → matrix[7][5] = 1
        matrix = _compute_confusion_matrix([(6, 8)])
        assert matrix[7][5] == 1


# ---------------------------------------------------------------------------
# Evaluator.evaluate()
# ---------------------------------------------------------------------------

class TestEvaluatorReport:
    def test_eval_report_written_and_parseable(self, tmp_path: Path) -> None:
        """evaluate() must write a valid JSON report to output_dir."""
        # Mock model: always predicts grade 8 (cumprobs step at index 7).
        # Must return batch-shaped output matching the input batch size.
        def mock_call(images, training=False):
            batch_size = images.shape[0] or tf.shape(images)[0]
            cumprobs = np.zeros((batch_size, 9), dtype=np.float32)
            cumprobs[:, 7:] = 1.0
            return {"overall": tf.constant(cumprobs)}

        mock_model = MagicMock(side_effect=mock_call)

        # Tiny dataset: 3 samples with label 7 (0-indexed → grade 8).
        images = np.zeros((3, 312, 224, 3), dtype=np.float32)
        labels = np.array([7, 7, 7], dtype=np.int32)
        test_ds = tf.data.Dataset.from_tensor_slices((images, labels))

        evaluator = Evaluator()
        metrics = evaluator.evaluate(mock_model, test_ds, tmp_path)

        report_path = tmp_path / "eval_report.json"
        assert report_path.exists()
        parsed = json.loads(report_path.read_text())
        assert "mae" in parsed
        assert "within_one_accuracy" in parsed
        assert "confusion_matrix" in parsed
        assert "total_samples" in parsed

    def test_confusion_matrix_is_10x10(self, tmp_path: Path) -> None:
        def mock_call(images, training=False):
            batch_size = images.shape[0] or tf.shape(images)[0]
            cumprobs = np.zeros((batch_size, 9), dtype=np.float32)
            cumprobs[:, 7:] = 1.0
            return {"overall": tf.constant(cumprobs)}

        mock_model = MagicMock(side_effect=mock_call)

        images = np.zeros((2, 312, 224, 3), dtype=np.float32)
        labels = np.array([7, 7], dtype=np.int32)
        test_ds = tf.data.Dataset.from_tensor_slices((images, labels))

        metrics = Evaluator().evaluate(mock_model, test_ds, tmp_path)
        assert len(metrics["confusion_matrix"]) == 10
        assert all(len(row) == 10 for row in metrics["confusion_matrix"])

    def test_perfect_model_mae_is_zero(self, tmp_path: Path) -> None:
        """A model that always predicts the correct grade should have MAE=0."""
        def mock_call(images, training=False):
            batch_size = images.shape[0] or tf.shape(images)[0]
            cumprobs = np.zeros((batch_size, 9), dtype=np.float32)
            cumprobs[:, 7:] = 1.0
            return {"overall": tf.constant(cumprobs)}

        mock_model = MagicMock(side_effect=mock_call)

        images = np.zeros((3, 312, 224, 3), dtype=np.float32)
        labels = np.array([7, 7, 7], dtype=np.int32)
        test_ds = tf.data.Dataset.from_tensor_slices((images, labels))

        metrics = Evaluator().evaluate(mock_model, test_ds, tmp_path)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["within_one_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TrainingLoop.train() — smoke test
# ---------------------------------------------------------------------------

class TestTrainingLoopSmoke:
    def test_saved_model_artifact_exists_after_training(
        self, tmp_path: Path
    ) -> None:
        """train() must produce a SavedModel artifact at config.output_dir/saved_model."""
        ds = _make_tiny_dataset(tmp_path)

        config = TrainingConfig(
            pretrained_weights="none",  # Skip ImageNet download in tests.
            epochs=1,
            batch_size=2,
            output_dir=tmp_path / "artifacts",
            log_dir=tmp_path / "logs",
        )

        artifact_path = TrainingLoop().train(ds, ds, config)

        assert artifact_path.exists(), f"SavedModel not found at {artifact_path}"

    def test_train_returns_path_object(self, tmp_path: Path) -> None:
        ds = _make_tiny_dataset(tmp_path)
        config = TrainingConfig(
            pretrained_weights="none",
            epochs=1,
            batch_size=2,
            output_dir=tmp_path / "artifacts",
            log_dir=tmp_path / "logs",
        )
        result = TrainingLoop().train(ds, ds, config)
        assert isinstance(result, Path)
