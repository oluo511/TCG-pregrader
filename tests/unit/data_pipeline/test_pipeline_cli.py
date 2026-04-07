"""
Unit tests for data_pipeline.cli (the `data-pipeline` entry point).

Uses typer.testing.CliRunner — no real HTTP, no real filesystem I/O.
All external dependencies (PipelineSettings, Orchestrator) are patched
so tests are fast, deterministic, and environment-independent.

Requirements covered: 10.1, 10.2, 10.3, 10.4
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from data_pipeline.cli import app
from data_pipeline.reporter import GradeReport

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grade_report() -> GradeReport:
    """Minimal valid GradeReport for patching Orchestrator.run()."""
    return GradeReport(
        counts_per_grade={},
        rejection_counts={},
        grades_below_warning=[],
        grades_at_target=[],
        total_images=0,
    )


# ---------------------------------------------------------------------------
# 10.1 — Default grades 1–10 when --grades not provided
# ---------------------------------------------------------------------------

def test_run_defaults_grades_1_to_10() -> None:
    """
    Req 10.1: Omitting --grades must collect all PSA grades 1–10.
    We capture the `grades` kwarg passed to Orchestrator.run() and assert
    it equals list(range(1, 11)).
    """
    captured: dict = {}

    async def fake_run(grades: list[int], max_per_grade: int) -> GradeReport:
        captured["grades"] = grades
        return _make_grade_report()

    with patch("data_pipeline.cli.PipelineSettings"), \
         patch("data_pipeline.cli.Orchestrator") as mock_orch:
        mock_orch.return_value.run = fake_run
        # Single-command Typer apps: CliRunner invokes the command directly.
        # Passing ["run"] would treat "run" as an extra positional arg — wrong.
        result = runner.invoke(app, [], env={"PSA_API_TOKEN": "test-token"})

    assert result.exit_code == 0, result.output
    assert captured["grades"] == list(range(1, 11))


# ---------------------------------------------------------------------------
# 10.2 — Default max-per-grade is 500
# ---------------------------------------------------------------------------

def test_run_defaults_max_per_grade_500() -> None:
    """
    Req 10.2: Omitting --max-per-grade must default to 500.
    """
    captured: dict = {}

    async def fake_run(grades: list[int], max_per_grade: int) -> GradeReport:
        captured["max_per_grade"] = max_per_grade
        return _make_grade_report()

    with patch("data_pipeline.cli.PipelineSettings"), \
         patch("data_pipeline.cli.Orchestrator") as mock_orch:
        mock_orch.return_value.run = fake_run
        result = runner.invoke(app, [], env={"PSA_API_TOKEN": "test-token"})

    assert result.exit_code == 0, result.output
    assert captured["max_per_grade"] == 500


# ---------------------------------------------------------------------------
# 10.3 — Missing PSA_API_TOKEN → exit code 1 + error message
# ---------------------------------------------------------------------------

def test_run_exits_1_on_missing_psa_token() -> None:
    """
    Req 10.3: When PipelineSettings raises (missing PSA_API_TOKEN), the CLI
    must exit with code 1 and print a configuration error message.

    We patch PipelineSettings directly to raise so the test is independent
    of the actual environment and pydantic-settings internals.
    """
    with patch(
        "data_pipeline.cli.PipelineSettings",
        side_effect=Exception("PSA_API_TOKEN required"),
    ):
        result = runner.invoke(app, [])

    assert result.exit_code == 1
    # CliRunner merges stdout/stderr into result.output
    assert "Configuration error" in result.output or "PSA_API_TOKEN" in result.output


def test_run_exits_1_on_configuration_error() -> None:
    """
    Req 10.3: ConfigurationError (our own exception subclass) must also
    produce exit code 1 — covers the explicit except ConfigurationError branch.
    """
    from data_pipeline.exceptions import ConfigurationError

    with patch(
        "data_pipeline.cli.PipelineSettings",
        side_effect=ConfigurationError("bad config"),
    ):
        result = runner.invoke(app, [])

    assert result.exit_code == 1
    assert "bad config" in result.output


# ---------------------------------------------------------------------------
# 10.4 — --output-dir and --manifest-path override PipelineSettings values
# ---------------------------------------------------------------------------

def test_run_overrides_output_dir_and_manifest_path(tmp_path: Path) -> None:
    """
    Req 10.4: CLI flags --output-dir and --manifest-path must override the
    corresponding PipelineSettings fields before the Orchestrator is created.

    We use a real PipelineSettings instance (token injected via env) so that
    model_copy(update=…) works correctly — mocking PipelineSettings would make
    model_copy() return another MagicMock, defeating the assertion.
    """
    import os

    custom_output = tmp_path / "my_images"
    custom_manifest = tmp_path / "my_manifest.csv"

    captured_settings: dict = {}

    async def fake_run(grades: list[int], max_per_grade: int) -> GradeReport:
        return _make_grade_report()

    mock_orch_instance = MagicMock()
    mock_orch_instance.run = fake_run

    def fake_orch_init(settings):
        captured_settings["settings"] = settings
        return mock_orch_instance

    # Inject a real token so PipelineSettings validates successfully.
    # We still mock Orchestrator to avoid real HTTP calls.
    with patch.dict(os.environ, {"PSA_API_TOKEN": "test-token"}, clear=False), \
         patch("data_pipeline.cli.Orchestrator", side_effect=fake_orch_init):
        result = runner.invoke(
            app,
            [
                "--output-dir", str(custom_output),
                "--manifest-path", str(custom_manifest),
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured_settings["settings"].output_dir == custom_output
    assert captured_settings["settings"].manifest_path == custom_manifest
