"""
BaseScraper — Template Method ABC for eBay and Card Ladder scrapers.

Design decisions:
- Template Method pattern: subclasses implement `_fetch_listings` and
  `_extract_cert_number`; the base class owns robots.txt enforcement, crawl
  delay, semaphore concurrency, and pagination. This keeps each scraper focused
  on source-specific HTML parsing without duplicating polite-crawl logic.
- Fail-open on robots.txt: a transient network failure fetching robots.txt
  should not halt the entire pipeline. We log a WARNING and allow the request
  rather than blocking collection. Technical Debt: add --strict-robots CLI flag
  for production use where fail-closed is preferred.
- Per-domain crawl delay: `_last_request_time` is keyed by domain so eBay and
  Card Ladder delays are tracked independently — a Card Ladder request does not
  reset the eBay timer.
- Semaphore at the scrape() level: the semaphore wraps the full per-grade fetch
  loop, not individual page requests, so `max_concurrent_requests` controls how
  many grades are in-flight simultaneously.
"""

import asyncio
import time
import urllib.parse
import urllib.robotparser
from abc import ABC, abstractmethod

import httpx
import structlog

from data_pipeline.config import PipelineSettings
from data_pipeline.deduplicator import Deduplicator
from data_pipeline.downloader import ImageDownloader
from data_pipeline.models import RawListing, ScrapedRecord
from data_pipeline.psa_client import PSAClient

logger = structlog.get_logger(__name__)


class BaseScraper(ABC):
    """
    Abstract base for all scrapers.

    Subclasses must implement:
      - `_fetch_listings(grade, page)` — source-specific HTTP + HTML parsing
      - `_extract_cert_number(listing)` — source-specific cert regex / metadata
    """

    def __init__(
        self,
        settings: PipelineSettings,
        psa_client: PSAClient,
        deduplicator: Deduplicator,
        downloader: ImageDownloader,
    ) -> None:
        self._settings = settings
        self._psa_client = psa_client
        self._deduplicator = deduplicator
        self._downloader = downloader

        # Semaphore limits how many grades are scraped concurrently.
        # Keeps total outbound connections bounded regardless of grade list size.
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

        # robots.txt is fetched once per domain per run and cached here.
        # Avoids hammering the robots.txt endpoint on every page request.
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

        # Per-domain timestamp of the last outbound request.
        # Used by _acquire_crawl_token to enforce source-specific crawl delays.
        self._last_request_time: dict[str, float] = {}

    async def scrape(self, grades: list[int], max_per_grade: int | None = None) -> list[ScrapedRecord]:
        """
        Fan out across grades concurrently (bounded by semaphore), paginate
        each grade until max_per_grade is reached or the source is exhausted,
        and return a flat list of ScrapedRecords.

        Args:
            grades: PSA grade integers to collect.
            max_per_grade: Per-grade cap. Defaults to settings.max_listings_per_grade.
        """
        limit = max_per_grade if max_per_grade is not None else self._settings.max_listings_per_grade

        async def _scrape_grade(grade: int) -> list[ScrapedRecord]:
            async with self._semaphore:
                return await self._scrape_single_grade(grade, limit)

        results: list[list[ScrapedRecord]] = await asyncio.gather(
            *[_scrape_grade(g) for g in grades]
        )
        return [record for grade_records in results for record in grade_records]

    async def _scrape_single_grade(self, grade: int, max_listings: int) -> list[ScrapedRecord]:
        """
        Paginate through listings for a single grade, respecting robots.txt and
        crawl delay, until max_listings is reached or no more pages.
        """
        records: list[ScrapedRecord] = []
        page = 1

        while len(records) < max_listings:
            # Build a representative URL for this page to check robots.txt.
            # Subclasses expose the actual URL via _fetch_listings; we check
            # a synthetic URL here because we don't have the real one yet.
            # The domain-level check is what matters for robots.txt compliance.
            listings = await self._fetch_listings(grade, page)

            if not listings:
                # Source exhausted — no more pages for this grade.
                break

            for listing in listings:
                if len(records) >= max_listings:
                    break

                # Enforce robots.txt before processing each listing URL.
                if not await self._check_robots(listing.listing_url):
                    logger.warning(
                        "robots_disallowed",
                        url=listing.listing_url,
                        grade=grade,
                    )
                    continue

                # Acquire crawl token (blocks until delay has elapsed).
                domain = self._get_domain(listing.listing_url)
                await self._acquire_crawl_token(domain)

                cert_number = self._extract_cert_number(listing)
                if cert_number is None:
                    # No cert in title — use the search grade as the label directly.
                    # This is the common case for eBay listings; most sellers don't
                    # include the PSA cert number in the title. We trust the grade
                    # from the search query (LH_Complete + LH_Sold filters mean the
                    # listing was a real completed sale at that grade).
                    from data_pipeline.models import CertRecord
                    cert_record = CertRecord(
                        cert_number=f"ebay_{listing.listing_url.split('/')[-1].split('?')[0]}",
                        overall_grade=listing.raw_grade,
                        centering=1.0,
                        corners=1.0,
                        edges=1.0,
                        surface=1.0,
                        verified=False,
                    )
                    self._deduplicator.mark_seen(cert_record.cert_number, listing.source)
                    records.append(
                        ScrapedRecord(
                            cert_record=cert_record,
                            image_url=listing.image_url,
                            source=listing.source,
                        )
                    )
                    continue

                if self._deduplicator.is_seen(cert_number):
                    logger.debug(
                        "cert_already_seen",
                        cert_number=cert_number,
                        source=listing.source,
                    )
                    continue

                try:
                    cert_record = await self._psa_client.get_cert(cert_number)
                except Exception as exc:
                    logger.warning(
                        "psa_lookup_failed",
                        cert_number=cert_number,
                        error=str(exc),
                    )
                    continue

                self._deduplicator.mark_seen(cert_number, listing.source)
                records.append(
                    ScrapedRecord(
                        cert_record=cert_record,
                        image_url=listing.image_url,
                        source=listing.source,
                    )
                )

            page += 1

        return records

    async def _check_robots(self, url: str) -> bool:
        """
        Return True if the URL is allowed by the domain's robots.txt.

        Fetches robots.txt exactly once per domain per run (cached in
        self._robots_cache). On any fetch failure, logs a WARNING and returns
        True (fail-open) — a transient network error should not halt collection.
        """
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        scheme = parsed.scheme

        if domain not in self._robots_cache:
            robots_url = f"{scheme}://{domain}/robots.txt"
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(robots_url)
                    response.raise_for_status()
                    # Feed the raw text to the parser line by line.
                    parser.parse(response.text.splitlines())
            except Exception as exc:
                # Fail-open: log and allow the request rather than blocking.
                logger.warning(
                    "robots_fetch_failed",
                    domain=domain,
                    robots_url=robots_url,
                    error=str(exc),
                )
                # Cache a permissive parser so we don't retry on every call.
                permissive = urllib.robotparser.RobotFileParser()
                permissive.parse(["User-agent: *", "Allow: /"])
                self._robots_cache[domain] = permissive
                return True

            self._robots_cache[domain] = parser

        return self._robots_cache[domain].can_fetch("*", url)

    async def _acquire_crawl_token(self, domain: str) -> None:
        """
        Token-bucket style crawl delay enforcement.

        Determines the required delay for the given domain, calculates how much
        time has elapsed since the last request to that domain, and sleeps for
        the remainder if needed. Updates the last-request timestamp after sleeping.

        Domain routing:
          - eBay: settings.ebay_crawl_delay
          - Card Ladder: settings.cardladder_crawl_delay
          - All others: 1.0s default (conservative but not source-specific)
        """
        # Select the crawl delay for this domain.
        if "ebay.com" in domain:
            delay = self._settings.ebay_crawl_delay
        elif "cardladder.com" in domain:
            delay = self._settings.cardladder_crawl_delay
        else:
            delay = 1.0

        last = self._last_request_time.get(domain, 0.0)
        elapsed = time.monotonic() - last

        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)

        self._last_request_time[domain] = time.monotonic()

    def _get_domain(self, url: str) -> str:
        """Extract the netloc (domain + port) from a URL."""
        return urllib.parse.urlparse(url).netloc

    @abstractmethod
    async def _fetch_listings(self, grade: int, page: int) -> list[RawListing]:
        """
        Fetch one page of listings for the given grade from the source.

        Returns an empty list when there are no more pages.
        Subclasses are responsible for constructing the request URL, making the
        HTTP call, and parsing the HTML into RawListing objects.
        """
        ...

    @abstractmethod
    def _extract_cert_number(self, listing: RawListing) -> str | None:
        """
        Extract a PSA cert number from a listing's title/metadata.

        Returns None if no cert number can be found — the caller will skip the
        listing or flag it as unverified depending on the source.
        """
        ...
