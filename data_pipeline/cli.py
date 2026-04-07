"""
CLI entry point for the TCG training data pipeline.

Design pattern: Thin adapter — the CLI owns only argument parsing and
settings override. All business logic lives in Orchestrator. This keeps
the CLI testable (CliRunner) without spinning up real HTTP connections.

Data flow:
  CLI args → PipelineSettings (with overrides) → Orchestrator.run() → GradeReport → stdout

Why asyncio.run() here instead of an async command?
  Typer does not natively support async commands. asyncio.run() is the
  idiomatic bridge: it creates a fresh event loop, runs the coroutine to
  completion, and tears the loop down cleanly. One call per CLI invocation
  is fine — this is not a long-lived server process.

Why catch broad Exception after ConfigurationError?
  Pydantic-settings raises pydantic.ValidationError (not ConfigurationError)
  when a required field like PSA_API_TOKEN is missing from the environment.
  We catch both so the operator always sees a clean "Configuration error: …"
  message instead of a raw Pydantic traceback.
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import typer

from data_pipeline.config import PipelineSettings
from data_pipeline.exceptions import ConfigurationError
from data_pipeline.orchestrator import Orchestrator

app = typer.Typer(
    help="TCG training data pipeline — scrape, download, and filter PSA slab photos."
)


@app.command()
def run(
    grades: Annotated[
        list[int] | None,
        typer.Option("--grades", help="PSA grades to collect (default: 1–10)"),
    ] = None,
    max_per_grade: Annotated[
        int,
        typer.Option("--max-per-grade", help="Max records per grade per source"),
    ] = 500,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Directory for downloaded images"),
    ] = None,
    manifest_path: Annotated[
        Path | None,
        typer.Option("--manifest-path", help="Path to manifest CSV"),
    ] = None,
) -> None:
    """Scrape, download, and filter PSA slab photos for CNN training."""

    # Default to all PSA grades (1–10) when the caller omits --grades.
    # This is the standard full-dataset collection mode.
    if grades is None:
        grades = list(range(1, 11))

    # Load settings from environment / .env file.
    # ConfigurationError is raised by our own validation logic (e.g., missing
    # PSA_API_TOKEN detected at PSAClient init). The broad Exception catch
    # handles pydantic.ValidationError for missing required BaseSettings fields.
    try:
        settings = PipelineSettings()
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=1)

    # Override PipelineSettings paths when the operator provides CLI flags.
    # model_copy(update=…) returns a new immutable instance — we never mutate
    # the original settings object, which keeps the pattern safe for testing.
    if output_dir is not None:
        settings = settings.model_copy(update={"output_dir": output_dir})
    if manifest_path is not None:
        settings = settings.model_copy(update={"manifest_path": manifest_path})

    # Run the full pipeline synchronously from the CLI's perspective.
    # Orchestrator.run() is async internally (concurrent scrapers, async HTTP),
    # but the CLI is a one-shot process so asyncio.run() is the right bridge.
    report = asyncio.run(
        Orchestrator(settings).run(grades=grades, max_per_grade=max_per_grade)
    )

    # GradeReporter already printed the grade distribution table to stdout
    # inside report(). We just confirm completion with the aggregate count.
    typer.echo(f"\nPipeline complete. Total images: {report.total_images}")
