"""
Unit tests for the Typer CLI (pregrader predict).

Strategy: Use Typer's CliRunner (wraps Click's runner) to invoke the CLI
in-process without spawning a subprocess. Services are mocked via
unittest.mock.patch so no real model artifacts or TF are needed.

Why patch at the module level rather than injecting via constructor?
The CLI constructs services internally (no DI container like FastAPI has),
so patching the class constructors is the cleanest isolation point.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from pregrader.cli import app
from pregrader.enums import CardType
from pregrader.exceptions import ModelNotFoundError
from pregrader.schemas import GradeResult, Subgrades

runner = CliRunner()

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_VALID_RESULT = GradeResult(
    image_id="pikachu.jpg",
    card_type=CardType.pokemon,
    overall_grade=8,
    subgrades=Subgrades(centering=8.0, corners=7.5, edges=8.0, surface=8.5),
    confidence=0.82,
)

# Minimal valid JPEG bytes — passes magic byte check.
_JPEG_BYTES = b"\xff\xd8\xff" + b"\x00" * 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fake_image(tmp_path: Path, name: str = "pikachu.jpg") -> Path:
    """Write minimal JPEG bytes to a temp file and return its path."""
    p = tmp_path / name
    p.write_bytes(_JPEG_BYTES)
    return p


def _mock_services(grader_results: list[GradeResult] | None = None):
    """Context manager stack that mocks all three service constructors."""
    import contextlib

    results = grader_results if grader_results is not None else [_VALID_RESULT]

    mock_settings = MagicMock()
    mock_settings.pokemon_model_artifact_path = Path("/fake/model")
    mock_settings.one_piece_model_artifact_path = None
    mock_settings.sports_model_artifact_path = None

    mock_registry = MagicMock()
    mock_registry.is_ready = True

    mock_ingestion = MagicMock()
    mock_ingestion._validate_magic_bytes = MagicMock()
    mock_ingestion._validate_resolution = MagicMock()

    mock_preprocessing = MagicMock()
    mock_preprocessing.preprocess.return_value = MagicMock(image_id="pikachu.jpg")

    mock_grader = MagicMock()
    mock_grader.predict = AsyncMock(return_value=results)

    return (
        mock_settings,
        mock_registry,
        mock_ingestion,
        mock_preprocessing,
        mock_grader,
    )


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

class TestPathValidation:
    def test_nonexistent_file_exits_with_code_1(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [str(tmp_path / "missing.jpg")])
        assert result.exit_code == 1

    def test_nonexistent_file_prints_error_to_stderr(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [str(tmp_path / "missing.jpg")])
        # CliRunner merges stdout/stderr by default — check combined output.
        assert "Error" in result.output
        assert "missing.jpg" in result.output

    def test_multiple_files_one_missing_exits_with_code_1(
        self, tmp_path: Path
    ) -> None:
        real = _write_fake_image(tmp_path, "real.jpg")
        result = runner.invoke(
            app, [str(real), str(tmp_path / "ghost.jpg")]
        )
        assert result.exit_code == 1
        assert "ghost.jpg" in result.output


# ---------------------------------------------------------------------------
# Card type not in registry
# ---------------------------------------------------------------------------

class TestCardTypeNotInRegistry:
    def test_unconfigured_card_type_exits_with_code_1(
        self, tmp_path: Path
    ) -> None:
        image = _write_fake_image(tmp_path)

        with patch("pregrader.cli.load_settings") as mock_load:
            settings = MagicMock()
            settings.one_piece_model_artifact_path = None
            mock_load.return_value = settings

            result = runner.invoke(
                app, [str(image), "--card-type", "one_piece"]
            )

        assert result.exit_code == 1
        assert "one_piece" in result.output

    def test_model_not_found_error_exits_with_code_1(
        self, tmp_path: Path
    ) -> None:
        image = _write_fake_image(tmp_path)

        with (
            patch("pregrader.cli.load_settings") as mock_load,
            patch("pregrader.cli.ModelRegistry") as MockRegistry,
            patch("pregrader.cli.ImageIngestionService") as MockIngestion,
            patch("pregrader.cli.PreprocessingService") as MockPreprocessing,
            patch("pregrader.cli.GraderService") as MockGrader,
        ):
            settings = MagicMock()
            settings.pokemon_model_artifact_path = Path("/fake/model")
            mock_load.return_value = settings

            registry = MagicMock()
            registry.load.side_effect = ModelNotFoundError(
                "No model loaded for card_type='pokemon'."
            )
            MockRegistry.return_value = registry

            result = runner.invoke(app, [str(image)])

        assert result.exit_code == 1
        assert "Error" in result.output


# ---------------------------------------------------------------------------
# Stdout output
# ---------------------------------------------------------------------------

class TestStdoutOutput:
    def test_valid_prediction_prints_json_to_stdout(
        self, tmp_path: Path
    ) -> None:
        image = _write_fake_image(tmp_path)

        with (
            patch("pregrader.cli.load_settings") as mock_load,
            patch("pregrader.cli.ModelRegistry") as MockRegistry,
            patch("pregrader.cli.ImageIngestionService") as MockIngestion,
            patch("pregrader.cli.PreprocessingService") as MockPreprocessing,
            patch("pregrader.cli.GraderService") as MockGrader,
        ):
            settings = MagicMock()
            settings.pokemon_model_artifact_path = Path("/fake/model")
            mock_load.return_value = settings

            MockRegistry.return_value = MagicMock()

            ingestion = MagicMock()
            ingestion._validate_magic_bytes = MagicMock()
            ingestion._validate_resolution = MagicMock()
            MockIngestion.return_value = ingestion

            preprocessing = MagicMock()
            preprocessing.preprocess.return_value = MagicMock(image_id="pikachu.jpg")
            MockPreprocessing.return_value = preprocessing

            grader = MagicMock()
            grader.predict = AsyncMock(return_value=[_VALID_RESULT])
            MockGrader.return_value = grader

            result = runner.invoke(app, [str(image)])

        assert result.exit_code == 0
        # Output must be valid JSON.
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert parsed[0]["overall_grade"] == 8
        assert parsed[0]["card_type"] == "pokemon"


# ---------------------------------------------------------------------------
# --output flag
# ---------------------------------------------------------------------------

class TestOutputFlag:
    def test_output_flag_writes_json_to_file(self, tmp_path: Path) -> None:
        image = _write_fake_image(tmp_path)
        output_file = tmp_path / "results.json"

        with (
            patch("pregrader.cli.load_settings") as mock_load,
            patch("pregrader.cli.ModelRegistry") as MockRegistry,
            patch("pregrader.cli.ImageIngestionService") as MockIngestion,
            patch("pregrader.cli.PreprocessingService") as MockPreprocessing,
            patch("pregrader.cli.GraderService") as MockGrader,
        ):
            settings = MagicMock()
            settings.pokemon_model_artifact_path = Path("/fake/model")
            mock_load.return_value = settings

            MockRegistry.return_value = MagicMock()

            ingestion = MagicMock()
            ingestion._validate_magic_bytes = MagicMock()
            ingestion._validate_resolution = MagicMock()
            MockIngestion.return_value = ingestion

            preprocessing = MagicMock()
            preprocessing.preprocess.return_value = MagicMock(image_id="pikachu.jpg")
            MockPreprocessing.return_value = preprocessing

            grader = MagicMock()
            grader.predict = AsyncMock(return_value=[_VALID_RESULT])
            MockGrader.return_value = grader

            result = runner.invoke(
                app, [str(image), "--output", str(output_file)]
            )

        assert result.exit_code == 0
        assert output_file.exists()
        parsed = json.loads(output_file.read_text())
        assert isinstance(parsed, list)
        assert parsed[0]["overall_grade"] == 8

    def test_output_flag_confirms_write_in_stdout(self, tmp_path: Path) -> None:
        image = _write_fake_image(tmp_path)
        output_file = tmp_path / "results.json"

        with (
            patch("pregrader.cli.load_settings") as mock_load,
            patch("pregrader.cli.ModelRegistry") as MockRegistry,
            patch("pregrader.cli.ImageIngestionService") as MockIngestion,
            patch("pregrader.cli.PreprocessingService") as MockPreprocessing,
            patch("pregrader.cli.GraderService") as MockGrader,
        ):
            settings = MagicMock()
            settings.pokemon_model_artifact_path = Path("/fake/model")
            mock_load.return_value = settings

            MockRegistry.return_value = MagicMock()

            ingestion = MagicMock()
            ingestion._validate_magic_bytes = MagicMock()
            ingestion._validate_resolution = MagicMock()
            MockIngestion.return_value = ingestion

            preprocessing = MagicMock()
            preprocessing.preprocess.return_value = MagicMock(image_id="pikachu.jpg")
            MockPreprocessing.return_value = preprocessing

            grader = MagicMock()
            grader.predict = AsyncMock(return_value=[_VALID_RESULT])
            MockGrader.return_value = grader

            result = runner.invoke(
                app, [str(image), "--output", str(output_file)]
            )

        assert "Results written to" in result.output
