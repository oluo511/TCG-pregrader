"""
ManifestBuilder — append-only CSV writer for the training data manifest.

Why append-only and never overwrite?
  The manifest is the ground-truth label file consumed by ManifestLoader during
  training. Overwriting it mid-run would corrupt any concurrent training job
  reading the file. Appending is safe because CSV readers tolerate a growing
  file — they only see rows that were fully written before they opened the file.

Why validate with ManifestRow before writing?
  CertRecord grades are validated at PSA API parse time, but image_path is
  constructed at runtime. A second validation pass here catches any grade that
  slipped through (e.g., from an unverified listing) and ensures every row in
  the manifest is structurally sound before it reaches the training loader.
  A bad row silently skipped is far less damaging than a corrupt manifest that
  crashes the entire training run.
"""

import csv
from pathlib import Path

import structlog
from pydantic import BaseModel, Field, ValidationError

from data_pipeline.config import PipelineSettings
from data_pipeline.models import CertRecord

logger = structlog.get_logger(__name__)

# Canonical column order — must match ManifestLoader expectations exactly.
_CSV_HEADER = ["image_path", "overall_grade", "centering", "corners", "edges", "surface"]


class ManifestRow(BaseModel):
    """
    Pydantic v2 schema for a single manifest CSV row.

    Validates grade ranges before any bytes hit disk. Keeping this as a
    separate model (rather than reusing CertRecord) decouples the manifest
    format from the PSA API contract — they can evolve independently.
    """

    image_path: str
    overall_grade: int = Field(ge=1, le=10)
    centering: float = Field(ge=1.0, le=10.0)
    corners: float = Field(ge=1.0, le=10.0)
    edges: float = Field(ge=1.0, le=10.0)
    surface: float = Field(ge=1.0, le=10.0)


class ManifestBuilder:
    """
    Appends validated rows to the training manifest CSV.

    Lifecycle:
        builder = ManifestBuilder(settings, project_root=Path("/workspace"))
        builder.append_row(cert_record, image_path)  # called per downloaded image
    """

    def __init__(
        self,
        settings: PipelineSettings,
        project_root: Path | None = None,
    ) -> None:
        # project_root anchors relative path computation — defaults to the
        # working directory so the manifest is portable across machines.
        self._manifest_path: Path = settings.manifest_path
        self._project_root: Path = project_root if project_root is not None else Path.cwd()

    def append_row(self, cert_record: CertRecord, image_path: Path) -> None:
        """
        Validate and append a single row to the manifest CSV.

        Steps:
          1. Convert absolute image_path to a project-relative string so the
             manifest is portable — absolute paths break on any other machine.
          2. Validate all fields via ManifestRow; skip and log ERROR on failure
             rather than raising — one bad cert must not abort the entire run.
          3. Create the CSV with a header row on first write; subsequent writes
             append data rows only (header must appear exactly once).
        """
        # --- Step 1: Make image path relative to project root ---
        try:
            relative_path = str(image_path.relative_to(self._project_root))
        except ValueError:
            # image_path is not under project_root — use absolute path as fallback.
            # This can happen when output_dir is configured as an absolute path
            # outside the workspace (e.g. /tmp/slabs). The manifest will still
            # work on this machine; portability is reduced.
            relative_path = str(image_path)

        # --- Step 2: Validate the row before touching disk ---
        try:
            row = ManifestRow(
                image_path=relative_path,
                overall_grade=cert_record.overall_grade,
                centering=cert_record.centering,
                corners=cert_record.corners,
                edges=cert_record.edges,
                surface=cert_record.surface,
            )
        except ValidationError as exc:
            # Extract the first failing field name for a targeted log message.
            # Logging cert_number lets operators cross-reference the PSA API
            # response without having to grep through raw scraper output.
            first_error = exc.errors()[0] if exc.errors() else {}
            field_name = str(first_error.get("loc", ("unknown",))[0])
            logger.error(
                "manifest_row_validation_error",
                cert_number=cert_record.cert_number,
                field=field_name,
                error=str(exc),
            )
            return  # skip this row — do NOT re-raise

        # --- Step 3: Write header on first create, then append data row ---
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)

        needs_header = not self._manifest_path.exists()

        with open(self._manifest_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if needs_header:
                writer.writerow(_CSV_HEADER)
            writer.writerow([
                row.image_path,
                row.overall_grade,
                row.centering,
                row.corners,
                row.edges,
                row.surface,
            ])

        logger.debug(
            "manifest_row_appended",
            cert_number=cert_record.cert_number,
            image_path=relative_path,
        )
