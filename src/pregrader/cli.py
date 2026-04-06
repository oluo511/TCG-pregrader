"""
CLI entry point for the TCG Pre-Grader.

Design pattern: Thin CLI adapter over the same service layer used by the API.
The CLI does NOT go through HTTP — it instantiates services directly and calls
them in-process. This avoids the network hop and UploadFile abstraction while
keeping the business logic in one place (no duplication).

Data flow:
  Path validation → bytes read → magic bytes + resolution check →
  PreprocessingService.preprocess() → GraderService.predict() → JSON output

Why Typer over argparse / click?
- Typer derives the CLI interface from Python type hints, so the signature
  IS the documentation. No separate argument registration boilerplate.
- Enum support is built-in: CardType values become valid --card-type choices
  automatically, with a helpful error message for unknown values.

Technical Debt: The registry loads the model artifact synchronously at CLI
startup. For a batch of thousands of images this is fine, but if the CLI
is ever called in a tight loop (e.g., a shell script), the per-invocation
model load cost adds up. Fix: expose a persistent gRPC/REST server and have
the CLI call that instead.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer

from pregrader.config import load_settings
from pregrader.enums import CardType
from pregrader.exceptions import (
    ImageIngestionError,
    InferenceError,
    ModelNotFoundError,
)
from pregrader.logging_config import get_logger
from pregrader.registry import ModelRegistry
from pregrader.schemas import GradeResult
from pregrader.services.grader import GraderService
from pregrader.services.ingestion import ImageIngestionService
from pregrader.services.preprocessing import PreprocessingService

logger = get_logger(service="cli")

app = typer.Typer(
    name="pregrader",
    help="TCG card pre-grader — grade Pokemon (and future) card images via CNN.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ARTIFACT_PATH_MAP = {
    CardType.pokemon: "pokemon_model_artifact_path",
    CardType.one_piece: "one_piece_model_artifact_path",
    CardType.sports: "sports_model_artifact_path",
}


def _build_registry(settings, card_type: CardType) -> ModelRegistry:
    """Load the model for the requested card_type into a fresh registry.

    Raises:
        typer.Exit: If the artifact path is not configured or does not exist.
            Prints a descriptive message to stderr before exiting.
    """
    registry = ModelRegistry()
    attr = _ARTIFACT_PATH_MAP[card_type]
    artifact_path = getattr(settings, attr, None)

    if artifact_path is None:
        typer.echo(
            f"Error: No model artifact path configured for card type '{card_type.value}'. "
            f"Set {attr.upper()} in your .env file.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        registry.load(card_type, artifact_path)
    except ModelNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    return registry


def _validate_paths(images: list[Path]) -> None:
    """Abort with exit code 1 if any image path does not exist."""
    missing = [p for p in images if not p.exists()]
    if missing:
        for p in missing:
            typer.echo(f"Error: File not found: {p}", err=True)
        raise typer.Exit(code=1)


async def _run_pipeline(
    images: list[Path],
    card_type: CardType,
    settings,
    registry: ModelRegistry,
) -> list[GradeResult]:
    """Execute ingestion → preprocessing → inference for a list of image paths.

    Per-image InferenceError is caught and logged to stderr; the remaining
    images continue processing (Requirement 6.5).

    Returns:
        List of GradeResult objects — may be shorter than input on partial failure.
    """
    ingestion = ImageIngestionService(settings)
    preprocessing = PreprocessingService()
    grader = GraderService(registry, settings)

    preprocessed_cards = []

    for image_path in images:
        raw_bytes = image_path.read_bytes()
        image_id = image_path.name

        # Validate format and resolution using the ingestion service's
        # internal validators directly — we bypass validate_and_load()
        # because that method expects FastAPI UploadFile objects.
        try:
            ingestion._validate_magic_bytes(raw_bytes, image_id)
            ingestion._validate_resolution(raw_bytes, image_id)
        except ImageIngestionError as exc:
            typer.echo(f"Error [{image_id}]: {exc}", err=True)
            raise typer.Exit(code=1)

        preprocessed_cards.append(preprocessing.preprocess(raw_bytes, image_id))

    results: list[GradeResult] = []
    try:
        results = await grader.predict(preprocessed_cards, card_type)
    except ModelNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    return results


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def predict(
    images: list[Path] = typer.Argument(
        ...,
        help="One or more image file paths to grade (JPEG or PNG).",
        exists=False,  # We do our own check to give a better error message.
    ),
    card_type: CardType = typer.Option(
        CardType.pokemon,
        "--card-type",
        "-t",
        help="Card game type to use for grading.",
        case_sensitive=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write JSON results to this file instead of stdout.",
    ),
) -> None:
    """Grade one or more TCG card images and output PSA-scale predictions.

    Examples:
        pregrader predict card1.jpg card2.png
        pregrader predict card1.jpg --card-type pokemon --output results.json
    """
    # Validate all paths exist before loading the model — fail fast and cheap.
    _validate_paths(images)

    settings = load_settings()
    registry = _build_registry(settings, card_type)

    results = asyncio.run(
        _run_pipeline(images, card_type, settings, registry)
    )

    # Serialize results — GradeResult is a frozen Pydantic model so
    # model_dump() gives us a clean dict with enum values as strings.
    output_data = [r.model_dump(mode="json") for r in results]
    json_str = json.dumps(output_data, indent=2)

    if output is not None:
        output.write_text(json_str, encoding="utf-8")
        typer.echo(f"Results written to {output}")
    else:
        typer.echo(json_str)
