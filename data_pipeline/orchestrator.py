"""
Orchestrator — top-level pipeline coordinator.

Design pattern: Facade — a single `run()` entry point hides the complexity of
wiring PSAClient, scrapers, downloader, preprocessor, manifest, and reporter.
Callers (CLI, tests) only need to know about `run(grades, max_per_grade)`.

Why asyncio.TaskGroup for scrapers?
  TaskGroup (Python 3.11+) provides structured concurrency: if either scraper
  raises an unhandled exception, the other task is cancelled and the exception
  propagates cleanly. This is safer than asyncio.gather(return_exceptions=True)
  which silently swallows failures and returns them as values.

Why try/finally around the record loop?
  deduplicator.persist() MUST run even if the loop is interrupted by an
  unexpected exception or QuotaExhaustedError. Without try/finally, a crash
  mid-run would leave the seen-certs state file stale, causing re-downloads
  on the next run and wasting PSA quota.

Per-record exception isolation:
  One bad cert (DownloadError, CertLookupError, InvalidImageError) must not
  abort the entire run. We catch these individually and continue — the pipeline
  is designed to be restartable and idempotent.
"""

import asyncio

import structlog

from data_pipeline.config import PipelineSettings
from data_pipeline.deduplicator import Deduplicator
from data_pipeline.downloader import ImageDownloader
from data_pipeline.exceptions import (
    CertLookupError,
    DownloadError,
    InvalidImageError,
    QuotaExhaustedError,
)
from data_pipeline.manifest import ManifestBuilder
from data_pipeline.models import ScrapedRecord
from data_pipeline.preprocessor import ImagePreprocessor
from data_pipeline.psa_client import PSAClient
from data_pipeline.reporter import GradeReport, GradeReporter
from data_pipeline.scrapers.cardladder import CardLadderScraper
from data_pipeline.scrapers.ebay import EbayScraper

logger = structlog.get_logger(__name__)


class Orchestrator:
    """
    Wires all pipeline components and runs the end-to-end collection flow.

    Lifecycle:
        orchestrator = Orchestrator(settings)
        report = await orchestrator.run(grades=[9, 10], max_per_grade=500)
    """

    def __init__(self, settings: PipelineSettings) -> None:
        self._settings = settings

        # Core utilities — shared across scrapers and the record-processing loop.
        self._psa_client = PSAClient(settings)
        self._deduplicator = Deduplicator(settings)
        self._downloader = ImageDownloader(settings)
        self._preprocessor = ImagePreprocessor(settings)
        self._manifest = ManifestBuilder(settings)
        self._reporter = GradeReporter(settings)

        # Scrapers receive shared components so they operate on the same
        # deduplication state and PSA quota window across both sources.
        self._ebay = EbayScraper(
            settings, self._psa_client, self._deduplicator, self._downloader
        )
        self._cardladder = CardLadderScraper(
            settings, self._psa_client, self._deduplicator, self._downloader
        )

    async def run(self, grades: list[int], max_per_grade: int) -> GradeReport:
        """
        Execute the full pipeline: scrape → download → filter → manifest → report.

        Args:
            grades: PSA grade integers to collect (e.g. [1, 2, ..., 10]).
            max_per_grade: Upper bound on records collected per grade per source.

        Returns:
            GradeReport with per-grade counts, rejection breakdown, and
            threshold flags — ready for CLI display or programmatic inspection.
        """
        # Step 1 — Hydrate deduplication state from disk.
        # Must happen before scrapers run so they skip already-seen certs.
        self._deduplicator.load()

        # Step 2 — Run both scrapers concurrently.
        # TaskGroup cancels the sibling task if either raises — prevents a
        # hung scraper from blocking the pipeline indefinitely.
        async with asyncio.TaskGroup() as tg:
            ebay_task = tg.create_task(self._ebay.scrape(grades))
            cardladder_task = tg.create_task(self._cardladder.scrape(grades))

        ebay_records: list[ScrapedRecord] = ebay_task.result()
        cardladder_records: list[ScrapedRecord] = cardladder_task.result()
        all_records: list[ScrapedRecord] = ebay_records + cardladder_records

        logger.info(
            "orchestrator_scrape_complete",
            ebay_count=len(ebay_records),
            cardladder_count=len(cardladder_records),
            total=len(all_records),
        )

        # Step 3 — Process each record: download → quality filter → manifest.
        # try/finally guarantees deduplicator.persist() runs even on exception.
        quota_exhausted = False
        try:
            for record in all_records:
                if quota_exhausted:
                    # PSA quota was hit earlier in this loop — stop making
                    # further calls. Images already downloaded are still
                    # processed; we just can't look up new certs.
                    break

                try:
                    await self._process_record(record)
                except QuotaExhaustedError:
                    # Quota hit mid-run: log, set flag, break out of the loop.
                    # Already-downloaded images are preserved in the manifest.
                    logger.error(
                        "orchestrator_quota_exhausted",
                        cert_number=record.cert_record.cert_number,
                    )
                    quota_exhausted = True
                    break
                except CertLookupError as exc:
                    logger.warning(
                        "orchestrator_cert_lookup_error",
                        cert_number=exc.cert_number,
                        status_code=exc.status_code,
                    )
                    continue
                except (DownloadError, InvalidImageError) as exc:
                    logger.warning(
                        "orchestrator_record_error",
                        cert_number=record.cert_record.cert_number,
                        error=str(exc),
                    )
                    continue

        finally:
            # Persist deduplication state regardless of how the loop exits.
            # This is the single most important invariant in the orchestrator —
            # without it, a crash mid-run causes re-downloads on the next run.
            self._deduplicator.persist()

        # Step 4 — Build and return the grade distribution report.
        # rejection_counts comes from the preprocessor's per-filter tallies.
        return self._reporter.report(
            self._settings.manifest_path,
            rejection_counts=self._preprocessor.rejection_counts,
        )

    async def _process_record(self, record: ScrapedRecord) -> None:
        """
        Download, quality-filter, and manifest-append a single scraped record.

        The cert_record is already populated by the scraper's PSA lookup —
        no additional PSA call is needed here. This method is extracted to keep
        the main loop readable and to make per-record error handling explicit.

        Raises:
            DownloadError: if the image cannot be fetched after retries.
            InvalidImageError: if the image fails magic-byte or decode checks.
            QuotaExhaustedError: propagated from downloader if PSA quota is hit
                                 (unlikely here, but kept for completeness).
        """
        cert = record.cert_record

        # Download the slab photo — raises DownloadError or InvalidImageError
        # on failure; the caller catches and continues to the next record.
        image_path = await self._downloader.download(
            record.image_url,
            cert.cert_number,
            self._settings.output_dir,
        )

        # Quality filter — returns (None, report) if the image is rejected.
        image_bytes = image_path.read_bytes()
        filtered_image, quality_report = self._preprocessor.filter_quality(
            image_bytes, cert.cert_number
        )

        if filtered_image is None:
            # Image failed quality checks — log and skip manifest write.
            logger.debug(
                "orchestrator_image_rejected",
                cert_number=cert.cert_number,
                rejection_reason=quality_report.rejection_reason,
            )
            return

        # All checks passed — append to manifest.
        self._manifest.append_row(cert, image_path)
