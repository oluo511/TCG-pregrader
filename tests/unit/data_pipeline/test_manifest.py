"""
Feature: training-data-pipeline
Tests: 7.3 — ManifestBuilder unit tests (Requirements 6.1–6.5)

Why capsys instead of caplog for structlog assertions?
  structlog writes to stdout by default in test environments (no stdlib logging
  integration configured). caplog only captures Python's logging module output.
  capsys captures stdout/stderr directly, which is where structlog events land.
"""

import csv
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from data_pipeline.config import PipelineSettings
from data_pipeline.manifest import ManifestBuilder, ManifestRow
from data_pipeline.models import CertRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(tmp_path: Path) -> PipelineSettings:
    return PipelineSettings(
        psa_api_token=SecretStr("test-token"),
        manifest_path=tmp_path / "manifest.csv",
    )


def _make_cert(cert_number: str = "12345678", grade: int = 9) -> CertRecord:
    return CertRecord(
        cert_number=cert_number,
        overall_grade=grade,
        centering=9.0,
        corners=9.0,
        edges=9.0,
        surface=9.0,
        verified=True,
    )


# ---------------------------------------------------------------------------
# 7.3a — New manifest is created with correct CSV header on first append_row
# ---------------------------------------------------------------------------

def test_creates_manifest_with_header_on_first_write(tmp_path: Path) -> None:
    """
    On the very first append_row call the manifest file must not exist yet.
    ManifestBuilder must create it and write the canonical header row before
    the data row (Req 6.1, 6.3).
    """
    settings = _make_settings(tmp_path)
    builder = ManifestBuilder(settings, project_root=tmp_path)

    image_path = tmp_path / "images" / "12345678.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    assert not settings.manifest_path.exists()

    builder.append_row(_make_cert(), image_path)

    assert settings.manifest_path.exists()

    rows = settings.manifest_path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "image_path,overall_grade,centering,corners,edges,surface"
    assert len(rows) == 2  # header + one data row


# ---------------------------------------------------------------------------
# 7.3b — Absolute image path is written as relative path from project_root
# ---------------------------------------------------------------------------

def test_image_path_written_as_relative(tmp_path: Path) -> None:
    """
    The manifest must store paths relative to project_root so the file is
    portable across machines and CI environments (Req 6.2).
    """
    settings = _make_settings(tmp_path)
    builder = ManifestBuilder(settings, project_root=tmp_path)

    image_path = tmp_path / "data" / "raw_slabs" / "12345678.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    builder.append_row(_make_cert(), image_path)

    with open(settings.manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    # Must be relative, not absolute
    assert not Path(row["image_path"]).is_absolute()
    assert row["image_path"] == str(Path("data") / "raw_slabs" / "12345678.jpg")


# ---------------------------------------------------------------------------
# 7.3c — ValidationError is caught, ERROR is logged, row is skipped
# ---------------------------------------------------------------------------

def test_validation_error_is_caught_logged_and_row_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """
    When ManifestRow raises ValidationError (e.g., out-of-range grade), the
    builder must:
      - Log an ERROR event to stdout (structlog)
      - NOT raise the exception to the caller
      - NOT write any row to the manifest (Req 6.4, 6.5)
    """
    settings = _make_settings(tmp_path)
    builder = ManifestBuilder(settings, project_root=tmp_path)

    image_path = tmp_path / "images" / "99999999.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    cert = _make_cert(cert_number="99999999", grade=9)

    # Patch ManifestRow to raise ValidationError so we can inject an
    # out-of-range scenario without relaxing CertRecord's own constraints.
    fake_errors = [{"loc": ("overall_grade",), "msg": "value must be <= 10", "type": "value_error"}]
    fake_exc = ValidationError.from_exception_data(
        title="ManifestRow",
        input_type="python",
        line_errors=[
            {
                "type": "greater_than_equal",
                "loc": ("overall_grade",),
                "msg": "Input should be greater than or equal to 1",
                "input": 0,
                "ctx": {"ge": 1},
                "url": "https://errors.pydantic.dev/2/v/greater_than_equal",
            }
        ],
    )

    with patch("data_pipeline.manifest.ManifestRow", side_effect=fake_exc):
        # Must NOT raise
        builder.append_row(cert, image_path)

    captured = capsys.readouterr()
    assert "manifest_row_validation_error" in captured.out
    assert "99999999" in captured.out

    # No file should have been created (row was skipped)
    assert not settings.manifest_path.exists()


# ---------------------------------------------------------------------------
# 7.3d — Appending multiple rows: file grows correctly, header appears once
# ---------------------------------------------------------------------------

def test_multiple_rows_header_appears_once(tmp_path: Path) -> None:
    """
    After N append_row calls the manifest must contain exactly N data rows
    and exactly one header row at the top (Req 6.4).
    """
    settings = _make_settings(tmp_path)
    builder = ManifestBuilder(settings, project_root=tmp_path)

    certs = [
        ("11111111", 7),
        ("22222222", 8),
        ("33333333", 9),
    ]

    for cert_number, grade in certs:
        image_path = tmp_path / "images" / f"{cert_number}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.touch()
        builder.append_row(_make_cert(cert_number=cert_number, grade=grade), image_path)

    lines = settings.manifest_path.read_text(encoding="utf-8").splitlines()

    # Exactly one header
    header_lines = [l for l in lines if l.startswith("image_path")]
    assert len(header_lines) == 1
    assert lines[0] == "image_path,overall_grade,centering,corners,edges,surface"

    # Three data rows
    assert len(lines) == 4  # 1 header + 3 data

    # Verify cert numbers appear in order
    with open(settings.manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    for (cert_number, grade), row in zip(certs, rows):
        assert cert_number in row["image_path"]
        assert int(row["overall_grade"]) == grade
