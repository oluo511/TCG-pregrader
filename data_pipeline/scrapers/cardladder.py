"""
CardLadderScraper — fetches PSA-graded card sale records from Card Ladder.

Design decisions:
- Template Method: this class only implements the two abstract methods from
  BaseScraper (_fetch_listings, _extract_cert_number). All robots.txt checking,
  crawl delay, dedup, PSA lookup, and semaphore logic live in the base class.
  This keeps CardLadderScraper focused purely on Card Ladder-specific HTML parsing.
- Crawl token before fetch: we acquire the crawl token for www.cardladder.com
  inside _fetch_listings (before the HTTP call) to enforce the 3-second inter-page
  delay required by Requirement 3.4. The base class also acquires a token per
  listing URL after the page is fetched; acquiring one here covers the page-level gap.
- html.parser over lxml: avoids a C-extension dependency. lxml is faster but adds
  a native build requirement that complicates Docker/CI images. For the volume of
  pages we scrape, html.parser is fast enough.
- Fail-open on parse errors: a single malformed Card Ladder page should not abort
  the entire grade's collection run. We log a WARNING and return [] so the
  pagination loop terminates gracefully for that grade.
- cert_number_raw field: Card Ladder sometimes embeds the cert directly in a
  dedicated metadata element (.sale-record__cert). We capture this raw text and
  also run CERT_PATTERN against the title, so both paths are covered.
"""

import re

import httpx
import structlog
from bs4 import BeautifulSoup

from data_pipeline.models import RawListing
from data_pipeline.scrapers.base import BaseScraper

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Cert extraction pattern (class-level constant)
#
# Why this pattern?
# Card Ladder embeds cert numbers in sale record metadata and titles in formats
# like "Cert: 12345678" or "PSA cert 12345678". The pattern anchors on "PSA"
# or "cert" as a signal word, skips any non-digit characters (spaces, #, :),
# then captures 7–10 consecutive digits. 7 digits is the minimum PSA cert
# length; 10 is the current maximum.
# ---------------------------------------------------------------------------
CERT_PATTERN = re.compile(r"(?:PSA|cert)[^\d]*(\d{7,10})", re.IGNORECASE)

# Card Ladder sales history URL filtered by PSA grade and paginated.
# grade= filters to the specified PSA grade; page= controls pagination.
_SEARCH_URL = "https://www.cardladder.com/sales?grade={grade}&page={page}"

# Realistic User-Agent to avoid trivial bot-detection blocks.
# We identify ourselves as TCG-pipeline so Card Ladder can contact us if needed.
_USER_AGENT = "Mozilla/5.0 (compatible; TCG-pipeline/1.0)"

# Card Ladder's domain — used for crawl token acquisition (3s delay enforced
# by settings.cardladder_crawl_delay in BaseScraper._acquire_crawl_token).
_DOMAIN = "www.cardladder.com"


class CardLadderScraper(BaseScraper):
    """
    Scrapes Card Ladder sales history for PSA-graded cards.

    Card Ladder aggregates sales from eBay, Goldin, Heritage, and other
    platforms, making it a valuable supplement to direct eBay scraping —
    especially for rare grades (1–4) where eBay volume is thin.

    Inherits all orchestration logic (robots.txt, crawl delay, dedup, PSA
    lookup, semaphore) from BaseScraper. Only Card Ladder-specific concerns
    live here.
    """

    async def _fetch_listings(self, grade: int, page: int) -> list[RawListing]:
        """
        Fetch one page of Card Ladder sale records for the given PSA grade.

        Acquires a crawl token for www.cardladder.com before making the HTTP
        request so the 3-second inter-page delay (Req 3.4) is respected in
        addition to the per-listing delay enforced by the base class.

        Returns [] on any HTTP or parse error (fail-open, logs WARNING).
        Returns [] when no sale-record elements are found (end of pagination).
        """
        url = _SEARCH_URL.format(grade=grade, page=page)

        # Acquire crawl token before the HTTP call to enforce the 3-second
        # inter-page delay required by Requirement 3.4. The base class also
        # acquires a token per listing URL, but that happens *after* the page
        # is already fetched — this covers the page-level gap.
        await self._acquire_crawl_token(_DOMAIN)

        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
                timeout=30.0,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                html = response.text
        except httpx.HTTPError as exc:
            logger.warning(
                "cardladder_fetch_failed",
                url=url,
                grade=grade,
                page=page,
                error=str(exc),
            )
            return []

        try:
            return self._parse_listings(html, grade, url)
        except Exception as exc:
            # Catch-all for unexpected BeautifulSoup parse errors.
            # A malformed page should not crash the entire pipeline run.
            logger.warning(
                "cardladder_parse_failed",
                url=url,
                grade=grade,
                page=page,
                error=str(exc),
            )
            return []

    def _parse_listings(
        self, html: str, grade: int, page_url: str
    ) -> list[RawListing]:
        """
        Parse Card Ladder sales page HTML into RawListing objects.

        Card Ladder uses <div class="sale-record"> for each sale entry.
        We extract title, image URL, listing URL, and optional cert metadata.

        Skipped records (logged at DEBUG):
          - No image found in .sale-record__image — record has no slab photo.
        """
        soup = BeautifulSoup(html, "html.parser")
        records = soup.find_all("div", class_="sale-record")

        listings: list[RawListing] = []

        for record in records:
            # --- Title ---
            title_el = record.select_one(".sale-record__title")
            if title_el is None:
                continue
            title = title_el.get_text(strip=True)

            # --- Image URL ---
            # Card Ladder may lazy-load images: the real URL may be in
            # data-src before JS hydration, or in src after. We check both
            # attributes so the scraper works on raw HTML without JS execution.
            image_wrapper = record.select_one(".sale-record__image")
            if image_wrapper is None:
                logger.debug(
                    "cardladder_record_no_image_wrapper",
                    page_url=page_url,
                    title=title,
                )
                continue
            img_el = image_wrapper.find("img")
            if img_el is None:
                logger.debug(
                    "cardladder_record_no_img_tag",
                    page_url=page_url,
                    title=title,
                )
                continue

            image_url: str | None = img_el.get("src") or img_el.get("data-src")
            if not image_url:
                logger.debug(
                    "cardladder_record_no_image_url",
                    page_url=page_url,
                    title=title,
                )
                continue

            # --- Listing URL ---
            link_el = record.select_one(".sale-record__link")
            listing_url: str = link_el["href"] if link_el else page_url

            listings.append(
                RawListing(
                    source="cardladder",
                    listing_url=listing_url,
                    image_url=image_url,
                    title=title,
                    raw_grade=grade,
                )
            )

        return listings

    def _extract_cert_number(self, listing: RawListing) -> str | None:
        """
        Extract a PSA cert number from the listing title using CERT_PATTERN.

        Card Ladder sometimes embeds the cert directly in the title as
        "Cert: 12345678" — the CERT_PATTERN handles this format alongside
        the standard "PSA ... cert NNN" format used by eBay sellers.

        Returns the first 7–10 digit sequence preceded by "PSA" or "cert".
        Returns None if no match is found — the base class will skip the
        listing (no cert → no PSA API call → record not added to dataset).
        """
        match = CERT_PATTERN.search(listing.title)
        if match:
            return match.group(1)
        return None
