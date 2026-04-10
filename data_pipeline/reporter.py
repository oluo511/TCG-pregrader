"""
GradeReporter — reads the manifest CSV and produces a per-grade distribution report.

Why read from CSV rather than accepting in-memory counts?
  Requirement 7.4 is explicit: the report must reflect the *true persisted dataset*.
  If a previous pipeline run already wrote rows to the manifest, an in-memory counter
  would under-count. Reading from disk guarantees the report is always consistent with
  what the training loader will actually see.

Why csv.DictReader instead of pandas?
  This module has no pandas dependency and doesn't need one — we're doing a single
  linear scan to count rows per grade. Pulling in pandas for a group-by on a flat CSV
  would be a heavy dependency for a trivial operation.
"""

import csv
from pathlib import Path

import structlog
from pydantic import BaseModel

from data_pipeline.config import PipelineSettings

logger = structlog.get_logger(__name__)

# Grades the pipeline targets — PSA grades are integers 1 through 10 inclusive.
_ALL_GRADES: list[int] = list(range(1, 11))

# Thresholds from Req 7.2 and 7.3
_WARNING_THRESHOLD = 100
_TARGET_THRESHOLD = 500


class GradeReport(BaseModel):
    """
    Output of GradeReporter.report().

    counts_per_grade: grade (1–10) → number of rows in the manifest for that grade.
    rejection_counts: filter name → number of images rejected by that filter.
    grades_below_warning: grades whose count is below the WARNING_THRESHOLD (< 100).
    grades_at_target: grades that have reached or exceeded the TARGET_THRESHOLD (>= 500).
    total_images: total rows in the manifest (sum of all grade counts).
    """

    counts_per_grade: dict[int, int]
    rejection_counts: dict[str, int]
    grades_below_warning: list[int]
    grades_at_target: list[int]
    total_images: int


class GradeReporter:
    """
    Reads the manifest CSV from disk and reports per-grade image counts.

    Lifecycle:
        reporter = GradeReporter(settings)
        report = reporter.report(manifest_path, rejection_counts)
    """

    def __init__(self, settings: PipelineSettings) -> None:
        # Store settings for future extensibility (e.g., configurable thresholds).
        self._settings = settings

    def report(
        self,
        manifest_path: Path,
        rejection_counts: dict[str, int],
    ) -> GradeReport:
        """
        Read the manifest CSV, count rows per overall_grade, print a table, and
        emit structured log events for grades that are below warning or at target.

        Args:
            manifest_path: Path to the manifest CSV on disk.
            rejection_counts: Mapping of filter name → rejection count, forwarded
                              from ImagePreprocessor for inclusion in the report.

        Returns:
            GradeReport with counts, thresholds, and rejection breakdown.
        """
        # --- Guard: manifest does not exist yet (first run with no data) ---
        if not manifest_path.exists():
            logger.warning(
                "manifest_not_found",
                manifest_path=str(manifest_path),
            )
            return GradeReport(
                counts_per_grade={},
                rejection_counts=rejection_counts,
                grades_below_warning=[],
                grades_at_target=[],
                total_images=0,
            )

        # --- Step 1: Count rows per grade by scanning the CSV once ---
        # Initialise all grades to 0 so grades with no images still appear in the table.
        counts: dict[int, int] = {g: 0 for g in _ALL_GRADES}

        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    grade = int(row["overall_grade"])
                except (KeyError, ValueError):
                    # Malformed row — skip silently; ManifestBuilder already validated
                    # at write time, so this should never happen in practice.
                    continue
                if grade in counts:
                    counts[grade] += 1

        # --- Step 2: Print a formatted table to stdout (Req 7.1) ---
        self._print_table(counts, rejection_counts)

        # --- Step 3: Emit structured log events per grade (Req 7.2, 7.3) ---
        grades_below: list[int] = []
        grades_at: list[int] = []

        for grade in _ALL_GRADES:
            count = counts[grade]
            if count < _WARNING_THRESHOLD:
                # Req 7.2 — operator needs to know which grades are under-represented
                # so they can prioritise scraping effort for those grades.
                logger.warning(
                    "grade_below_warning_threshold",
                    grade=grade,
                    count=count,
                    threshold=_WARNING_THRESHOLD,
                )
                grades_below.append(grade)
            if count >= _TARGET_THRESHOLD:
                # Req 7.3 — positive signal: this grade has enough data for training.
                logger.info(
                    "grade_at_target",
                    grade=grade,
                    count=count,
                    target=_TARGET_THRESHOLD,
                )
                grades_at.append(grade)

        total = sum(counts.values())

        return GradeReport(
            counts_per_grade=counts,
            rejection_counts=rejection_counts,
            grades_below_warning=grades_below,
            grades_at_target=grades_at,
            total_images=total,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_table(
        counts: dict[int, int],
        rejection_counts: dict[str, int],
    ) -> None:
        """
        Print a human-readable grade distribution table to stdout.

        Why stdout and not the logger?
          The CLI contract (Req 10.4) requires the grade table to appear in the
          terminal output. Structured log events go to stderr or a log sink in
          production; stdout is reserved for operator-facing summaries.
        """
        print("\n-- Grade Distribution ------------------------------------------")
        print(f"{'Grade':>6}  {'Count':>6}  {'Status'}")
        print("-" * 44)
        for grade in _ALL_GRADES:
            count = counts.get(grade, 0)
            if count >= _TARGET_THRESHOLD:
                status = "[OK] target met"
            elif count < _WARNING_THRESHOLD:
                status = "[!] below warning"
            else:
                status = ""
            print(f"{grade:>6}  {count:>6}  {status}")
        print("-" * 44)
        print(f"{'TOTAL':>6}  {sum(counts.values()):>6}")

        if rejection_counts:
            print("\n-- Rejection Counts -------------------------------------")
            for filter_name, count in sorted(rejection_counts.items()):
                print(f"  {filter_name:<30} {count:>6}")
        print()
