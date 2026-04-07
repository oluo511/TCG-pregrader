"""
Feature: training-data-pipeline
Tests: 14.1 — Orchestrator unit tests (Requirements 1.3, 1.4, 5.3)

Strategy: construct a real Orchestrator with a valid PipelineSettings, then
replace all component attributes with MagicMock(spec=...) instances. This
avoids real HTTP calls, disk I/O, and PSA API interactions while still
exercising the orchestrator's coordination logic.

Why patch attributes after construction?
  PSAClient.__init__ validates the token and creates an httpx.AsyncClient.
  Constructing with a dummy token (SecretStr("test-token")) satisfies the
  guard, then we immediately swap the attribute for a mock — so no real
  client is ever used.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from data_pipeline.config import PipelineSettings
from data_pipeline.deduplicator import Deduplicator
from data_pipeline.downloader import ImageDownloader
from data_pipeline.exceptions import DownloadError, QuotaExhaustedError
from data_pipeline.manifest import ManifestBuilder
from data_pipeline.models import CertRecord, ScrapedRecord
from data_pipeline.orchestrator import Orchestrator
from data_pipeline.preprocessor import ImagePreprocessor
from data_pipeline.reporter import GradeReport, GradeReporter
from data_pipeline.scrapers.cardladder import CardLadderScraper
from data_pipeline.scrapers.ebay import EbayScraper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path) -> PipelineSettings:
    return PipelineSettings(
        psa_api_token=SecretStr("test-token"),
        output_dir=tmp_path / "images",
        manifest_path=tmp_path / "manifest.csv",
    )


def _make_cert(cert_number: str = "12345678") -> CertRecord:
    return CertRecord(
        cert_number=cert_number,
        overall_grade=9,
        centering=9.0,
        corners=9.0,
        edges=9.0,
        surface=9.0,
    )


def _make_record(cert_number: str = "12345678", source: str = "ebay") -> ScrapedRecord:
    return ScrapedRecord(
        cert_record=_make_cert(cert_number),
        image_url=f"http://example.com/{cert_number}.jpg",
        source=source,  # type: ignore[arg-type]
    )


def _make_orchestrator(tmp_path: Path) -> Orchestrator:
    """
    Build an Orchestrator with all components replaced by mocks.

    Returns the orchestrator with mocked internals ready for assertion.
    """
    settings = _make_settings(tmp_path)
    orch = Orchestrator(settings)

    # Replace every component with a spec-constrained mock so attribute
    # access on the mock mirrors the real class interface.
    orch._deduplicator = MagicMock(spec=Deduplicator)
    orch._downloader = MagicMock(spec=ImageDownloader)
    orch._preprocessor = MagicMock(spec=ImagePreprocessor)
    orch._manifest = MagicMock(spec=ManifestBuilder)
    orch._reporter = MagicMock(spec=GradeReporter)
    orch._ebay = MagicMock(spec=EbayScraper)
    orch._cardladder = MagicMock(spec=CardLadderScraper)

    # Default preprocessor behaviour: accept all images (no rejection).
    import numpy as np
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
    mock_quality = MagicMock(rejected=False, rejection_reason=None)
    orch._preprocessor.filter_quality.return_value = (dummy_image, mock_quality)
    orch._preprocessor.rejection_counts = {}

    # Default reporter: return a minimal GradeReport.
    orch._reporter.report.return_value = GradeReport(
        counts_per_grade={},
        rejection_counts={},
        grades_below_warning=[],
        grades_at_target=[],
        total_images=0,
    )

    return orch


# ---------------------------------------------------------------------------
# 14.1a — QuotaExhaustedError breaks the loop; deduplicator.persist() still runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quota_exhausted_breaks_loop_but_persist_still_called(
    tmp_path: Path,
) -> None:
    """
    When the downloader raises QuotaExhaustedError on the first record,
    the orchestrator must stop processing further records AND still call
    deduplicator.persist() via the try/finally guard (Req 1.3, 5.3).
    """
    orch = _make_orchestrator(tmp_path)

    records = [_make_record(str(i) * 8) for i in range(1, 4)]  # 3 records
    orch._ebay.scrape = AsyncMock(return_value=records)
    orch._cardladder.scrape = AsyncMock(return_value=[])

    # First download raises QuotaExhaustedError — loop must break.
    orch._downloader.download = AsyncMock(side_effect=QuotaExhaustedError("quota hit"))

    await orch.run(grades=[9], max_per_grade=10)

    # deduplicator.persist() must have been called exactly once (try/finally).
    orch._deduplicator.persist.assert_called_once()

    # manifest.append_row must NOT have been called — no record completed.
    orch._manifest.append_row.assert_not_called()


# ---------------------------------------------------------------------------
# 14.1b — DownloadError on first record; remaining records are still processed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_error_is_caught_and_remaining_records_processed(
    tmp_path: Path,
) -> None:
    """
    When downloader.download raises DownloadError for the first record,
    the orchestrator must catch it, continue, and process the remaining
    two records — resulting in two manifest.append_row calls (Req 4.1).
    """
    orch = _make_orchestrator(tmp_path)

    records = [
        _make_record("11111111"),
        _make_record("22222222"),
        _make_record("33333333"),
    ]
    orch._ebay.scrape = AsyncMock(return_value=records)
    orch._cardladder.scrape = AsyncMock(return_value=[])

    # First call raises DownloadError; subsequent calls return a valid path.
    fake_path = tmp_path / "images" / "dummy.jpg"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)  # minimal JPEG magic

    orch._downloader.download = AsyncMock(
        side_effect=[DownloadError("timeout"), fake_path, fake_path]
    )

    await orch.run(grades=[9], max_per_grade=10)

    # Two records succeeded → two manifest rows written.
    assert orch._manifest.append_row.call_count == 2

    # deduplicator.persist() must still have been called.
    orch._deduplicator.persist.assert_called_once()


# ---------------------------------------------------------------------------
# 14.1c — Unexpected exception mid-run; deduplicator.persist() still runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deduplicator_persisted_on_unexpected_exception(
    tmp_path: Path,
) -> None:
    """
    If an unexpected exception (not a pipeline exception) is raised during
    record processing, the try/finally must still call deduplicator.persist()
    before the exception propagates (Req 5.3).
    """
    orch = _make_orchestrator(tmp_path)

    records = [_make_record("12345678")]
    orch._ebay.scrape = AsyncMock(return_value=records)
    orch._cardladder.scrape = AsyncMock(return_value=[])

    # Simulate an unexpected runtime error (not a PipelineError subclass).
    orch._downloader.download = AsyncMock(side_effect=RuntimeError("unexpected!"))

    # The unexpected exception should propagate out of run().
    with pytest.raises(RuntimeError, match="unexpected!"):
        await orch.run(grades=[9], max_per_grade=10)

    # Despite the exception, persist() must have been called.
    orch._deduplicator.persist.assert_called_once()
