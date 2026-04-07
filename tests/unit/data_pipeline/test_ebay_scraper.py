"""
Feature: training-data-pipeline
Tests: 10.3 — EbayScraper unit tests (Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7)

Why these tests?
- Routing logic: EbayScraper now delegates to EbayAPIClient when credentials
  are present. Tests verify the correct path is taken in both cases so a
  misconfiguration doesn't silently produce zero listings.
- Cert extraction is the critical signal that links a listing to a PSA record.
  Both false-positives (wrong cert) and false-negatives (None when cert exists)
  corrupt the training dataset.
- _parse_listings is tested directly (not via _fetch_listings) because the HTML
  parsing logic is still the fallback path and must remain correct even though
  it's no longer the primary production path.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from data_pipeline.config import PipelineSettings
from data_pipeline.deduplicator import Deduplicator
from data_pipeline.downloader import ImageDownloader
from data_pipeline.models import RawListing
from data_pipeline.psa_client import PSAClient
from data_pipeline.scrapers.ebay import EbayScraper
from data_pipeline.scrapers.ebay_api import EbayAPIClient


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_scraper(ebay_client_id: str = "") -> EbayScraper:
    """
    Instantiate EbayScraper with fully mocked dependencies.

    ebay_client_id="" → no API client (degraded mode, returns []).
    ebay_client_id="app-id" → EbayAPIClient is instantiated.
    """
    settings = PipelineSettings(
        psa_api_token=SecretStr("test-token"),
        ebay_client_id=ebay_client_id,
        ebay_client_secret=SecretStr("test-secret" if ebay_client_id else ""),
    )
    return EbayScraper(
        settings=settings,
        psa_client=MagicMock(spec=PSAClient),
        deduplicator=MagicMock(spec=Deduplicator),
        downloader=MagicMock(spec=ImageDownloader),
    )


def _make_listing(title: str = "PSA 9 Charizard cert 12345678 Pokemon") -> RawListing:
    return RawListing(
        source="ebay",
        listing_url="https://www.ebay.com/itm/123",
        image_url="https://i.ebayimg.com/images/g/abc/s-l500.jpg",
        title=title,
        raw_grade=9,
    )


# ---------------------------------------------------------------------------
# 10.3a — _fetch_listings delegates to EbayAPIClient when credentials present
#
# Why test delegation explicitly?
# The routing branch (api_client is not None) is the production path. If the
# delegation is broken, the scraper silently falls through to the warning path
# and returns [] — no error, just missing data.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_listings_delegates_to_api_client_when_configured() -> None:
    """
    When ebay_client_id is set, _fetch_listings must call
    EbayAPIClient.search_listings and return its result (Req 2.1).
    """
    scraper = _make_scraper(ebay_client_id="my-app-id")
    assert scraper._api_client is not None

    expected = [_make_listing()]
    with patch.object(
        scraper._api_client,
        "search_listings",
        new_callable=AsyncMock,
        return_value=expected,
    ) as mock_search:
        result = await scraper._fetch_listings(grade=9, page=2)

    mock_search.assert_called_once_with(grade=9, page=2)
    assert result == expected


@pytest.mark.asyncio
async def test_fetch_listings_passes_grade_and_page_to_api_client() -> None:
    """
    _fetch_listings must forward grade and page unchanged to search_listings.
    An off-by-one on page would silently skip or double-fetch pages.
    """
    scraper = _make_scraper(ebay_client_id="my-app-id")

    with patch.object(
        scraper._api_client,  # type: ignore[union-attr]
        "search_listings",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_search:
        await scraper._fetch_listings(grade=7, page=3)

    mock_search.assert_called_once_with(grade=7, page=3)


# ---------------------------------------------------------------------------
# 10.3b — _fetch_listings returns [] with WARNING when no credentials
#
# Why test the degraded path?
# Without credentials, the scraper must not silently return [] as if there are
# no listings — it must log a WARNING so operators know the pipeline is
# misconfigured. The [] return is intentional (fail-open), but the log is the
# signal that something needs fixing.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_listings_returns_empty_when_no_api_credentials() -> None:
    """
    When ebay_client_id is empty, _fetch_listings must return [] (Req 2.5).
    No EbayAPIClient should be instantiated.
    """
    scraper = _make_scraper(ebay_client_id="")
    assert scraper._api_client is None

    result = await scraper._fetch_listings(grade=9, page=1)
    assert result == []


@pytest.mark.asyncio
async def test_no_api_client_instantiated_without_credentials() -> None:
    """
    EbayScraper.__init__ must leave _api_client as None when ebay_client_id
    is empty — instantiating it with blank credentials would cause a 401 on
    every token fetch rather than a clear startup-time misconfiguration signal.
    """
    scraper = _make_scraper(ebay_client_id="")
    assert scraper._api_client is None


def test_api_client_instantiated_when_credentials_present() -> None:
    """
    EbayScraper.__init__ must create an EbayAPIClient instance when
    ebay_client_id is non-empty.
    """
    scraper = _make_scraper(ebay_client_id="real-app-id")
    assert isinstance(scraper._api_client, EbayAPIClient)


# ---------------------------------------------------------------------------
# 10.3c — Cert number NOT extracted when title has no PSA pattern → None
#
# Why test the None case explicitly?
# The base class skips listings where _extract_cert_number returns None.
# A false-negative here silently drops valid listings from the dataset.
# ---------------------------------------------------------------------------

def test_extract_cert_number_returns_none_when_no_pattern() -> None:
    """
    Listings with no PSA cert pattern in the title must return None (Req 2.2).
    """
    scraper = _make_scraper()
    listing = _make_listing(title="Pokemon Card Lot Vintage")
    assert scraper._extract_cert_number(listing) is None


def test_extract_cert_number_returns_none_for_short_digit_sequence() -> None:
    """
    Digit sequences shorter than 7 digits must not match (PSA certs are 7–10 digits).
    """
    scraper = _make_scraper()
    listing = _make_listing(title="PSA 9 Charizard grade 123456")  # only 6 digits
    assert scraper._extract_cert_number(listing) is None


# ---------------------------------------------------------------------------
# 10.3d — Cert number IS extracted from valid PSA patterns
#
# Why multiple title formats?
# Sellers use inconsistent formatting. The regex must handle "PSA ... cert NNN",
# "cert NNN PSA graded", and similar variations without false-negatives.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title, expected_cert", [
    ("PSA 9 Charizard cert 12345678 Pokemon", "12345678"),
    ("cert 9876543 PSA graded Pikachu", "9876543"),
    ("PSA cert 00012345678 Blastoise", "0001234567"),  # 11 digits — regex captures max 10
    ("Pokemon PSA cert#98765432 Venusaur", "98765432"),
    ("Sold PSA 10 cert: 1234567 Mewtwo", "1234567"),  # 7-digit minimum
])
def test_extract_cert_number_valid_patterns(title: str, expected_cert: str) -> None:
    """
    _extract_cert_number must return the cert number string for all valid
    PSA cert title formats (Req 2.2).
    """
    scraper = _make_scraper()
    listing = _make_listing(title=title)
    result = scraper._extract_cert_number(listing)
    assert result == expected_cert


# ---------------------------------------------------------------------------
# 10.3e — _parse_listings: "Shop on eBay" promotional item is skipped
#
# Why test _parse_listings directly?
# The HTML parsing logic is still present (it's the fallback path) and must
# remain correct. Testing it directly avoids the routing layer and keeps
# these tests independent of credential configuration.
# ---------------------------------------------------------------------------

def test_parse_listings_skips_shop_on_ebay_item() -> None:
    """
    eBay injects a "Shop on eBay" promotional card into every results page.
    It must be filtered out so it doesn't appear as a real listing (Req 2.3).
    """
    scraper = _make_scraper()

    html = """
    <html><body><ul>
      <li class="s-item">
        <span class="s-item__title">Shop on eBay</span>
        <div class="s-item__image-wrapper"><img src="https://img.ebay.com/promo.jpg"/></div>
        <a class="s-item__link" href="https://www.ebay.com/promo">link</a>
      </li>
      <li class="s-item">
        <span class="s-item__title">PSA 9 Charizard cert 12345678</span>
        <div class="s-item__image-wrapper"><img src="https://i.ebayimg.com/images/g/abc/s-l500.jpg"/></div>
        <a class="s-item__link" href="https://www.ebay.com/itm/123">link</a>
      </li>
    </ul></body></html>
    """

    listings = scraper._parse_listings(html, grade=9, page_url="https://www.ebay.com/sch/i.html")

    assert len(listings) == 1
    assert listings[0].title == "PSA 9 Charizard cert 12345678"


# ---------------------------------------------------------------------------
# 10.3f — _parse_listings: listing without image is skipped
# ---------------------------------------------------------------------------

def test_parse_listings_skips_listing_without_image() -> None:
    """
    Listings with no image URL must be skipped — we cannot train on a record
    with no slab photo (Req 2.4).
    """
    scraper = _make_scraper()

    html = """
    <html><body><ul>
      <li class="s-item">
        <span class="s-item__title">PSA 9 Charizard cert 12345678</span>
        <div class="s-item__image-wrapper"><img /></div>
        <a class="s-item__link" href="https://www.ebay.com/itm/123">link</a>
      </li>
    </ul></body></html>
    """

    listings = scraper._parse_listings(html, grade=9, page_url="https://www.ebay.com/sch/i.html")
    assert listings == []


# ---------------------------------------------------------------------------
# 10.3g — _parse_listings: data-src fallback for lazy-loaded images
# ---------------------------------------------------------------------------

def test_parse_listings_uses_data_src_fallback() -> None:
    """
    eBay lazy-loads images: the real URL may be in data-src before JS hydration.
    _parse_listings must fall back to data-src when src is absent (Req 2.4).
    """
    scraper = _make_scraper()

    html = """
    <html><body><ul>
      <li class="s-item">
        <span class="s-item__title">PSA 9 Charizard cert 12345678</span>
        <div class="s-item__image-wrapper">
          <img data-src="https://i.ebayimg.com/images/g/abc/s-l500.jpg"/>
        </div>
        <a class="s-item__link" href="https://www.ebay.com/itm/123">link</a>
      </li>
    </ul></body></html>
    """

    listings = scraper._parse_listings(html, grade=9, page_url="https://www.ebay.com/sch/i.html")

    assert len(listings) == 1
    assert listings[0].image_url == "https://i.ebayimg.com/images/g/abc/s-l500.jpg"


# ---------------------------------------------------------------------------
# 10.3h — _parse_listings: empty page returns [] (pagination stop signal)
# ---------------------------------------------------------------------------

def test_parse_listings_returns_empty_on_no_items() -> None:
    """
    When a page contains no s-item elements, _parse_listings returns [].
    The base class pagination loop uses this as the stop signal (Req 2.6).
    """
    scraper = _make_scraper()
    html = "<html><body><ul class='srp-results'></ul></body></html>"
    result = scraper._parse_listings(html, grade=9, page_url="https://www.ebay.com/sch/i.html")
    assert result == []
