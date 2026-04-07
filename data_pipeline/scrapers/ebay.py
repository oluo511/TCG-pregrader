"""
EbayScraper — fetches completed PSA-graded Pokémon card listings from eBay.

Design decisions:
- Template Method: this class only implements the two abstract methods from
  BaseScraper (_fetch_listings, _extract_cert_number). All robots.txt checking,
  crawl delay, dedup, PSA lookup, and semaphore logic live in the base class.
  This keeps EbayScraper focused purely on eBay-specific HTML parsing.
- Crawl token before fetch: we acquire the crawl token inside _fetch_listings
  (before the HTTP call) rather than relying solely on the base class token
  acquisition in _scrape_single_grade. The base class acquires a token per
  listing URL *after* the page is fetched; acquiring one here ensures the
  inter-page delay is also respected, preventing burst requests across pages.
- html.parser over lxml: avoids a C-extension dependency. lxml is faster but
  adds a native build requirement that complicates Docker/CI images. For the
  volume of pages we scrape, html.parser is fast enough.
- Fail-open on parse errors: a single malformed eBay page should not abort the
  entire grade's collection run. We log a WARNING and return [] so the
  pagination loop terminates gracefully for that grade.
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
# eBay listing titles are free-form text. Sellers write things like:
#   "PSA 9 Charizard cert 12345678"
#   "cert 9876543 PSA graded Pikachu"
#   "PSA cert#00012345678 Blastoise"
# The pattern anchors on "PSA" or "cert" as a signal word, skips any
# non-digit characters (spaces, #, :, etc.), then captures 7–10 consecutive
# digits. 7 digits is the minimum PSA cert length; 10 is the current maximum.
# ---------------------------------------------------------------------------
CERT_PATTERN = re.compile(r"(?:PSA|cert)[^\d]*(\d{7,10})", re.IGNORECASE)

# eBay completed/sold listings search URL template.
# LH_Complete=1 → completed listings; LH_Sold=1 → sold only (filters out
# unsold completed listings which have no transaction price signal).
_SEARCH_URL = (
    "https://www.ebay.com/sch/i.html"
    "?_nkw=PSA+{grade}+pokemon"
    "&LH_Complete=1"
    "&LH_Sold=1"
    "&_pgn={page}"
)

# Realistic browser User-Agent — eBay's bot detection checks this header.
# The "compatible; TCG-pipeline/1.0" string was too obviously a bot.
# Using a real Chrome UA significantly reduces 503 rejection rates.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Additional headers that real browsers send — absence of these is a
# common bot-detection signal.
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}


class EbayScraper(BaseScraper):
    """
    Scrapes eBay completed/sold listings for PSA-graded Pokémon cards.

    Inherits all orchestration logic (robots.txt, crawl delay, dedup, PSA
    lookup, semaphore) from BaseScraper. Only eBay-specific concerns live here.
    """

    async def _fetch_listings(self, grade: int, page: int) -> list[RawListing]:
        """
        Fetch one page of eBay completed listings for the given PSA grade.

        Acquires a crawl token for www.ebay.com before making the HTTP request
        so inter-page delays are respected in addition to inter-listing delays.

        Returns [] on any HTTP or parse error (fail-open, logs WARNING).
        Returns [] when no s-item listings are found (signals end of pagination).
        """
        url = _SEARCH_URL.format(grade=grade, page=page)

        # Acquire crawl token before the HTTP call to enforce inter-page delay.
        # The base class also acquires a token per listing URL, but that happens
        # *after* the page is already fetched — this covers the page-level gap.
        await self._acquire_crawl_token("www.ebay.com")

        try:
            async with httpx.AsyncClient(
                headers=_HEADERS,
                follow_redirects=True,
                timeout=30.0,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                html = response.text
        except httpx.HTTPError as exc:
            logger.warning(
                "ebay_fetch_failed",
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
                "ebay_parse_failed",
                url=url,
                grade=grade,
                page=page,
                error=str(exc),
            )
            return []

    def _parse_listings(self, html: str, grade: int, page_url: str) -> list[RawListing]:
        """
        Parse eBay search results HTML into RawListing objects.

        eBay's search results page uses <li class="s-item"> for each listing.
        We extract title, image URL, and listing URL from each item.

        Skipped listings (logged at DEBUG):
          - Title contains "Shop on eBay" — eBay injects a promotional item
            at the top of every results page that is not a real listing.
          - No image found in .s-item__image-wrapper — listing has no slab photo.
        """
        soup = BeautifulSoup(html, "html.parser")
        items = soup.find_all("li", class_="s-item")

        listings: list[RawListing] = []

        for item in items:
            # --- Title ---
            title_el = item.select_one(".s-item__title")
            if title_el is None:
                continue
            title = title_el.get_text(strip=True)

            # eBay injects a "Shop on eBay" promotional card — skip it.
            if "Shop on eBay" in title:
                continue

            # --- Image URL ---
            # eBay lazy-loads images: the real URL may be in data-src before
            # the page is scrolled, or in src after JS hydration. We check
            # both attributes so the scraper works on raw HTML without JS.
            image_wrapper = item.select_one(".s-item__image-wrapper")
            if image_wrapper is None:
                continue
            img_el = image_wrapper.find("img")
            if img_el is None:
                continue

            image_url: str | None = img_el.get("src") or img_el.get("data-src")
            if not image_url:
                logger.debug(
                    "ebay_listing_no_image",
                    page_url=page_url,
                    title=title,
                )
                continue

            # --- Listing URL ---
            link_el = item.select_one(".s-item__link")
            listing_url: str = link_el["href"] if link_el else page_url

            listings.append(
                RawListing(
                    source="ebay",
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

        Returns the first 7–10 digit sequence preceded by "PSA" or "cert".
        Returns None if no match is found — the base class will skip the listing.
        """
        match = CERT_PATTERN.search(listing.title)
        if match:
            return match.group(1)
        return None
