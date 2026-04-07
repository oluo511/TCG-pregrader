"""
Feature: training-data-pipeline
Tests: 8.2 — GradeReporter unit tests (Requirements 7.1–7.4)

Why capsys instead of caplog for structlog assertions?
  structlog writes to stdout by default in test environments (no stdlib logging
  integration configured). caplog only captures Python's logging module output.
  capsys captures stdout/stderr directly, which is where structlog events land.
"""

import csv
from pathlib import Path

import pytest
from pydantic import SecretStr

from data_pipeline.config import PipelineSettings
from data_pipeline.reporter import GradeReport, GradeReporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path) -> PipelineSettings:
    return PipelineSettings(
        psa_api_token=SecretStr("test-token"),
        manifest_path=tmp_path / "manifest.csv",
    )


def _write_manifest(path: Path, grades: list[int]) -> None:
    """Write a minimal manifest CSV with the given grade values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "overall_grade", "centering", "corners", "edges", "surface"])
        for i, grade in enumerate(grades):
            writer.writerow([f"images/{i}.jpg", grade, 9.0, 9.0, 9.0, 9.0])


# ---------------------------------------------------------------------------
# 8.2a — WARNING logged for grade with count < 100 (Req 7.2)
# ---------------------------------------------------------------------------


def test_warning_logged_for_grade_below_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """
    When a grade has fewer than 100 images, GradeReporter must emit a WARNING
    event to stdout (via structlog) containing the grade and count (Req 7.2).
    """
    settings = _make_settings(tmp_path)
    manifest_path = tmp_path / "manifest.csv"

    # Grade 3 has only 5 images — well below the 100-image warning threshold.
    grades = [3] * 5
    _write_manifest(manifest_path, grades)

    reporter = GradeReporter(settings)
    report = reporter.report(manifest_path, rejection_counts={})

    captured = capsys.readouterr()
    # structlog emits to stdout; assert the warning event key is present.
    assert "grade_below_warning_threshold" in captured.out
    assert "3" in captured.out  # grade value must appear

    # The report model must also reflect the warning grade.
    assert 3 in report.grades_below_warning


# ---------------------------------------------------------------------------
# 8.2b — INFO logged for grade with count >= 500 (Req 7.3)
# ---------------------------------------------------------------------------


def test_info_logged_for_grade_at_target(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """
    When a grade reaches or exceeds 500 images, GradeReporter must emit an INFO
    event to stdout (via structlog) indicating the target has been met (Req 7.3).
    """
    settings = _make_settings(tmp_path)
    manifest_path = tmp_path / "manifest.csv"

    # Grade 9 has exactly 500 images — at the target threshold.
    grades = [9] * 500
    _write_manifest(manifest_path, grades)

    reporter = GradeReporter(settings)
    report = reporter.report(manifest_path, rejection_counts={})

    captured = capsys.readouterr()
    assert "grade_at_target" in captured.out
    assert "9" in captured.out

    assert 9 in report.grades_at_target


# ---------------------------------------------------------------------------
# 8.2c — Report reads from CSV, not in-memory state (Req 7.4)
# ---------------------------------------------------------------------------


def test_report_reads_from_csv_not_in_memory(tmp_path: Path) -> None:
    """
    GradeReporter must derive counts by reading the manifest CSV from disk.
    We write the CSV directly (bypassing any pipeline state), then instantiate
    a fresh GradeReporter and assert the counts match what was written (Req 7.4).
    """
    manifest_path = tmp_path / "manifest.csv"

    # Write the manifest directly — no pipeline instance involved.
    grades = [7] * 120 + [8] * 300 + [9] * 50
    _write_manifest(manifest_path, grades)

    # Fresh reporter with no prior in-memory state.
    settings = _make_settings(tmp_path)
    reporter = GradeReporter(settings)
    report = reporter.report(manifest_path, rejection_counts={})

    assert report.counts_per_grade[7] == 120
    assert report.counts_per_grade[8] == 300
    assert report.counts_per_grade[9] == 50
    assert report.total_images == 470


# ---------------------------------------------------------------------------
# 8.2d — counts_per_grade is accurate for a known grade distribution (Req 7.1)
# ---------------------------------------------------------------------------


def test_counts_per_grade_accurate_for_known_distribution(tmp_path: Path) -> None:
    """
    GradeReport.counts_per_grade must exactly match the row counts in the CSV
    for every grade 1–10, including grades with zero rows (Req 7.1).
    """
    manifest_path = tmp_path / "manifest.csv"

    # Deliberately uneven distribution across grades.
    grade_distribution = {
        1: 10,
        2: 25,
        3: 0,   # no images for grade 3
        4: 75,
        5: 150,
        6: 200,
        7: 400,
        8: 500,
        9: 600,
        10: 50,
    }
    grades: list[int] = []
    for grade, count in grade_distribution.items():
        grades.extend([grade] * count)

    _write_manifest(manifest_path, grades)

    settings = _make_settings(tmp_path)
    reporter = GradeReporter(settings)
    report = reporter.report(manifest_path, rejection_counts={})

    for grade, expected_count in grade_distribution.items():
        assert report.counts_per_grade[grade] == expected_count, (
            f"Grade {grade}: expected {expected_count}, got {report.counts_per_grade[grade]}"
        )

    expected_total = sum(grade_distribution.values())
    assert report.total_images == expected_total


# ---------------------------------------------------------------------------
# 8.2e — Missing manifest returns empty GradeReport with WARNING (Req 7.4)
# ---------------------------------------------------------------------------


def test_missing_manifest_returns_empty_report(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """
    When the manifest file does not exist, GradeReporter must return an empty
    GradeReport (all zeros/empty collections) and log a WARNING.
    """
    settings = _make_settings(tmp_path)
    manifest_path = tmp_path / "nonexistent_manifest.csv"

    reporter = GradeReporter(settings)
    report = reporter.report(manifest_path, rejection_counts={"sharpness": 3})

    captured = capsys.readouterr()
    assert "manifest_not_found" in captured.out

    assert report.total_images == 0
    assert report.counts_per_grade == {}
    assert report.grades_below_warning == []
    assert report.grades_at_target == []
    # rejection_counts should still be forwarded even when manifest is missing
    assert report.rejection_counts == {"sharpness": 3}
