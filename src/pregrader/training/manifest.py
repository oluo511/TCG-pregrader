"""
ManifestLoader — parses and validates the training CSV manifest.

Design pattern: Fail-fast on data integrity violations, skip-and-warn on
missing files. The distinction matters:
  - A grade value outside 1–10 is a data corruption issue that would silently
    poison the training distribution → halt immediately with ValidationError.
  - A missing image file is an ops issue (file not copied, path wrong) that
    affects only that sample → skip with WARNING so the rest of the dataset
    is still usable.

Why Pydantic for row validation instead of manual checks?
ManifestRow already encodes the full constraint set (ge/le on all grade
fields). Passing each CSV row through ManifestRow construction gives us
free constraint enforcement, clear error messages, and a typed object —
no hand-rolled range checks needed.
"""

import csv
from pathlib import Path

from pydantic import ValidationError

from pregrader.logging_config import get_logger
from pregrader.schemas import ManifestRow

logger = get_logger(service="manifest_loader")


class ManifestLoader:
    """Loads and validates a training manifest CSV into ManifestRow objects.

    CSV format (header required):
        image_path,overall_grade,centering,corners,edges,surface

    Behaviour:
      - Rows with grade values outside 1–10: raise ValidationError immediately.
        The entire load is aborted — a corrupt grade contaminates the dataset.
      - Rows whose image_path does not exist on disk: skip with WARNING.
        The remaining rows are still returned.
    """

    def load(self, csv_path: Path) -> list[ManifestRow]:
        """Parse csv_path and return validated ManifestRow objects.

        Args:
            csv_path: Path to the manifest CSV file.

        Returns:
            List of ManifestRow objects for rows whose image files exist.

        Raises:
            ValidationError: If any row contains a grade value outside 1–10.
                Propagates immediately — no rows are returned.
            FileNotFoundError: If csv_path itself does not exist.
        """
        rows: list[ManifestRow] = []

        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)

            for idx, raw in enumerate(reader):
                # Pydantic validates all field constraints here.
                # ValidationError on out-of-range grades propagates immediately
                # — do NOT catch it; the caller must see the corrupt row.
                row = ManifestRow(
                    image_path=Path(raw["image_path"]),
                    overall_grade=int(raw["overall_grade"]),
                    centering=float(raw["centering"]),
                    corners=float(raw["corners"]),
                    edges=float(raw["edges"]),
                    surface=float(raw["surface"]),
                )

                # Skip rows whose image file is missing — log with enough
                # context for an operator to locate and fix the gap.
                if not row.image_path.exists():
                    logger.warning(
                        "manifest_image_missing",
                        row_index=idx,
                        image_path=str(row.image_path),
                    )
                    continue

                rows.append(row)

        logger.info(
            "manifest_loaded",
            csv_path=str(csv_path),
            total_rows=len(rows),
        )

        return rows
