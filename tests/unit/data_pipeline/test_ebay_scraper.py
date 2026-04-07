"""
Feature: training-data-pipeline
Tests: 10.3 — EbayScraper unit tests (Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7)

Why these tests?
- URL construction is the contract between the scraper and eBay's search API.
  A wrong parameter (e.g., missing LH_Sold) silently returns wrong data.
- Cert extraction is the critical signal that links a listing to a PSA record.
  Both false-positives (wrong cert) and false-negatives (None when cert exists)
  corrupt the training dataset.
- HTTP error handling must be fail-open: a single bad page should not abort
  the entire grade's collection run.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from data_pipeline.config import PipelineSettings
from data_pipeline.deduplicator import Deduplicator
from data_pipeline.downloader import ImageDownloader
from data_pipeline.models import RawListing
from data_pipeline.psa_client import PSAClient
from data_pipeline.scrapers.ebay import EbayScraper


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_scraper() -> EbayScraper:
    """Instantiate EbayScraper with fully mocked dependencies."""
    settings = PipelineSettings(psa_api_token=SecretStr("test-token"))
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
# 10.3a — Correct search URL is constructed for each grade 1–10
#
# Why test URL construction?
# The URL is the only interface between the scraper and eBay. Wrong parameters
# (e.g., missing LH_Complete or wrong grade) silently return incorrect data
# that would corrupt the training set without any error signal.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("grade", range(1, 11))
async def test_fetch_listings_url_construction(grade: int) -> None:
    """
    _fetch_listings must build a URL containing the correct grade, LH_Complete,
    and LH_Sold parameters for every PSA grade 1–10 (Req 2.1).
    """
    scraper = _make_scraper()

    captured_urls: list[str] = []

    # Minimal HTML with no s-item listings — we only care about the URL called.
    empty_html = "<html><body><ul class='srp-results'></ul></body></html>"

    mock_response = MagicMock()
    mock_response.text = empty_html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def _capture_get(url: str, **kwargs) -> MagicMock:
        captured_urls.append(url)
        return mock_response

    mock_client.get = _capture_get

    with patch("data_pipeline.scrapers.ebay.httpx.AsyncClient", return_value=mock_client):
        # Patch crawl token so the test doesn't sleep.
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            await scraper._fetch_listings(grade, page=1)

    assert len(captured_urls) == 1
    url = captured_urls[0]
    assert f"PSA+{grade}+pokemon" in url
    assert "LH_Complete=1" in url
    assert "LH_Sold=1" in url


# ---------------------------------------------------------------------------
# 10.3b — Cert number NOT extracted when title has no PSA pattern → None
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
# 10.3c — Cert number IS extracted from valid PSA patterns
#
# Why multiple title formats?
# Sellers use inconsistent formatting. The regex must handle "PSA ... cert NNN",
# "cert NNN PSA graded", and similar variations without false-negatives.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title, expected_cert", [
    ("PSA 9 Charizard cert 12345678 Pokemon", "12345678"),
    ("cert 9876543 PSA graded Pikachu", "9876543"),
    ("PSA cert 00012345678 Blastoise", "0001234567"),  # 11 digits — regex captures max 10, so first 10 matched
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
# 10.3d — _fetch_listings returns [] on HTTP error (no exception propagates)
#
# Why test fail-open behaviour?
# A single eBay page returning a 5xx or a network timeout must not abort the
# entire grade's collection run. The base class pagination loop treats [] as
# "no more pages" and moves on (Req 2.5).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_listings_returns_empty_on_transport_error() -> None:
    """
    httpx.TransportError must be caught, a WARNING logged, and [] returned.
    No exception should propagate to the caller (Req 2.5).
    """
    scraper = _make_scraper()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=httpx.TransportError("connection reset"))

    with patch("data_pipeline.scrapers.ebay.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            result = await scraper._fetch_listings(grade=9, page=1)

    assert result == []


@pytest.mark.asyncio
async def test_fetch_listings_returns_empty_on_http_status_error() -> None:
    """
    A non-2xx HTTP response (e.g., 403 Forbidden) must be caught and [] returned.
    """
    scraper = _make_scraper()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "403 Forbidden",
            request=MagicMock(),
            response=MagicMock(),
        )
    )

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("data_pipeline.scrapers.ebay.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            result = await scraper._fetch_listings(grade=9, page=1)

    assert result == []


# ---------------------------------------------------------------------------
# 10.3e — HTML parsing: "Shop on eBay" promotional item is skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_listings_skips_shop_on_ebay_item() -> None:
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

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("data_pipeline.scrapers.ebay.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            listings = await scraper._fetch_listings(grade=9, page=1)

    assert len(listings) == 1
    assert listings[0].title == "PSA 9 Charizard cert 12345678"


# ---------------------------------------------------------------------------
# 10.3f — HTML parsing: listing without image is skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_listings_skips_listing_without_image() -> None:
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

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("data_pipeline.scrapers.ebay.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            listings = await scraper._fetch_listings(grade=9, page=1)

    assert listings == []


# ---------------------------------------------------------------------------
# 10.3g — HTML parsing: data-src fallback for lazy-loaded images
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_listings_uses_data_src_fallback() -> None:
    """
    eBay lazy-loads images: the real URL may be in data-src before JS hydration.
    _fetch_listings must fall back to data-src when src is absent (Req 2.4).
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

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("data_pipeline.scrapers.ebay.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            listings = await scraper._fetch_listings(grade=9, page=1)

    assert len(listings) == 1
    assert listings[0].image_url == "https://i.ebayimg.com/images/g/abc/s-l500.jpg"


# ---------------------------------------------------------------------------
# 10.3h — Empty page returns [] (pagination stop signal)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_listings_returns_empty_on_no_items() -> None:
    """
    When a page contains no s-item elements, _fetch_listings returns [].
    The base class pagination loop uses this as the stop signal (Req 2.6).
    """
    scraper = _make_scraper()

    html = "<html><body><ul class='srp-results'></ul></body></html>"

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("data_pipeline.scrapers.ebay.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            result = await scraper._fetch_listings(grade=9, page=99)

    assert result == []
