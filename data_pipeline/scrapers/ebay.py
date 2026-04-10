"""
EbayScraper — fetches completed PSA-graded Pokémon card listings from eBay.

Design decisions:
- Playwright async API: eBay's search results are fully JS-rendered. We use
  playwright.async_api which is designed for asyncio — no thread-affinity
  issues, no greenlet conflicts, and it composes naturally with the async
  scraper base class.
- Browser reuse: one browser instance is created at first use and shared
  across all page fetches in a pipeline run. Each fetch gets a fresh context
  (isolated cookies/storage) to avoid session-based bot detection heuristics.
- Resource blocking: images, fonts, media, and stylesheets are aborted during
  page load. We only need the DOM — blocking binary resources cuts load time
  from ~8s to ~2-3s per page.
- API-first: when eBay Browse API credentials are configured, Playwright is
  skipped entirely in favour of the structured JSON API.
"""

import re
from typing import TYPE_CHECKING

import structlog
from bs4 import BeautifulSoup

from data_pipeline.models import RawListing
from data_pipeline.scrapers.base import BaseScraper
from data_pipeline.scrapers.ebay_api import EbayAPIClient

if TYPE_CHECKING:
    from playwright.async_api import Browser, Playwright

logger = structlog.get_logger(__name__)

CERT_PATTERN = re.compile(r"(?:PSA|cert)[^\d]*(\d{7,10})", re.IGNORECASE)

_SEARCH_URL = (
    "https://www.ebay.com/sch/i.html"
    "?_nkw=PSA+{grade}+pokemon+slab"
    "&LH_Complete=1"
    "&LH_Sold=1"
    "&_pgn={page}"
)

_ITEM_SELECTOR = "li.s-card"
_PAGE_TIMEOUT_MS = 20_000
_BLOCKED_RESOURCE_TYPES = {"image", "media", "font", "stylesheet"}


class EbayScraper(BaseScraper):
    """
    Scrapes eBay completed/sold listings for PSA-graded Pokémon cards
    using Playwright's async API for JS-rendered page content.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._api_client: EbayAPIClient | None = None
        if self._settings.ebay_client_id:
            self._api_client = EbayAPIClient(self._settings)
        self._playwright: "Playwright | None" = None
        self._browser: "Browser | None" = None

    async def _ensure_browser(self) -> "Browser":
        """Launch Playwright + Chromium on first call; return cached instance."""
        if self._browser is None:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            )
            logger.info("playwright_browser_launched")
        return self._browser

    async def close(self) -> None:
        """Shut down the browser and Playwright runtime. Call after scraping."""
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def _fetch_listings(self, grade: int, page: int) -> list[RawListing]:
        """Route to Browse API if configured, otherwise use Playwright."""
        if self._api_client is not None:
            return await self._api_client.search_listings(grade=grade, page=page)
        return await self._fetch_with_playwright(grade, page)

    async def _fetch_with_playwright(self, grade: int, page: int) -> list[RawListing]:
        """
        Open a fresh browser context, navigate to the eBay search page,
        wait for listings to hydrate, and return parsed RawListing objects.
        """
        browser = await self._ensure_browser()
        url = _SEARCH_URL.format(grade=grade, page=page)

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        try:
            pw_page = await context.new_page()

            async def _block_resources(route):
                if route.request.resource_type in _BLOCKED_RESOURCE_TYPES:
                    await route.abort()
                else:
                    await route.continue_()

            await pw_page.route("**/*", _block_resources)
            await pw_page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            try:
                await pw_page.wait_for_selector(_ITEM_SELECTOR, timeout=_PAGE_TIMEOUT_MS)
            except Exception:
                # Timed out — could be bot detection or genuinely empty results.
                # Capture a snippet of the page title to help diagnose.
                title = await pw_page.title()
                logger.warning(
                    "ebay_playwright_no_items",
                    grade=grade,
                    page=page,
                    page_title=title,
                )
                return []

            html = await pw_page.content()
        finally:
            await context.close()

        listings = self._parse_listings(html, grade, url)
        logger.info(
            "ebay_playwright_page_fetched",
            grade=grade,
            page=page,
            listings_found=len(listings),
        )
        return listings

    def _parse_listings(self, html: str, grade: int, page_url: str) -> list[RawListing]:
        """Parse eBay search results HTML into RawListing objects.

        eBay's current DOM (2024+) uses li.s-card inside ul.srp-results.
        Title is in the first element with a class containing 'title'.
        Image is the first img tag in the card.
        Listing URL is the first anchor pointing to ebay.com/itm.
        """
        soup = BeautifulSoup(html, "html.parser")

        ul = soup.find("ul", class_="srp-results")
        if ul is None:
            return []

        cards = [
            li for li in ul.find_all("li", recursive=False)
            if "s-card" in (li.get("class") or [])
        ]

        listings: list[RawListing] = []
        for card in cards:
            # Title — first element whose class contains "title"
            title_el = card.select_one("[class*=title]")
            if title_el is None:
                continue
            title = title_el.get_text(strip=True)
            if not title or "Shop on eBay" in title:
                continue

            # Image
            img_el = card.find("img")
            if img_el is None:
                continue
            image_url: str | None = img_el.get("src") or img_el.get("data-src")
            if not image_url or not image_url.startswith("http"):
                continue

            # Listing URL — first anchor pointing to an eBay item page
            link_el = card.select_one('a[href*="ebay.com/itm"]') or card.select_one("a.s-card__link")
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
        match = CERT_PATTERN.search(listing.title)
        return match.group(1) if match else None
