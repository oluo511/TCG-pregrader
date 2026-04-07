"""
Feature: training-data-pipeline
Tests: 11.3 — CardLadderScraper unit tests (Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7)

Why these tests?
- Cert extraction is the critical signal that links a sale record to a PSA cert.
  Both false-positives (wrong cert) and false-negatives (None when cert exists)
  corrupt the training dataset.
- HTTP error handling must be fail-open: a single bad page should not abort
  the entire grade's collection run.
- Crawl delay domain verification ensures the 3-second Card Ladder delay
  (Req 3.4) is enforced — using the wrong domain key would silently apply
  the wrong delay.
- HTML parsing tests guard against Card Ladder markup changes that would
  silently drop all records or include malformed ones.
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
from data_pipeline.scrapers.cardladder import CardLadderScraper


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_scraper() -> CardLadderScraper:
    """Instantiate CardLadderScraper with fully mocked dependencies."""
    settings = PipelineSettings(psa_api_token=SecretStr("test-token"))
    return CardLadderScraper(
        settings=settings,
        psa_client=MagicMock(spec=PSAClient),
        deduplicator=MagicMock(spec=Deduplicator),
        downloader=MagicMock(spec=ImageDownloader),
    )


def _make_listing(title: str = "PSA 9 Charizard Cert: 12345678") -> RawListing:
    return RawListing(
        source="cardladder",
        listing_url="https://www.cardladder.com/sales/12345678",
        image_url="https://cdn.cardladder.com/img/12345678.jpg",
        title=title,
        raw_grade=9,
    )


# ---------------------------------------------------------------------------
# 11.3a — _extract_cert_number returns None when cert absent from title
#
# Why test the None case explicitly?
# The base class skips listings where _extract_cert_number returns None.
# A false-negative here silently drops valid listings from the dataset.
# ---------------------------------------------------------------------------

def test_extract_cert_number_returns_none_when_no_pattern() -> None:
    """
    Listings with no PSA cert pattern in the title must return None (Req 3.2).
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
# 11.3b — _extract_cert_number returns cert when "Cert: 12345678" present
#
# Why test the Card Ladder-specific "Cert: NNN" format?
# Card Ladder embeds certs in titles as "Cert: 12345678" — a format not
# common on eBay. The CERT_PATTERN must handle this to avoid false-negatives
# on Card Ladder's primary cert embedding style.
# ---------------------------------------------------------------------------

def test_extract_cert_number_returns_cert_from_cert_colon_format() -> None:
    """
    "Cert: 12345678" format (Card Ladder's primary style) must be extracted (Req 3.2).
    """
    scraper = _make_scraper()
    listing = _make_listing(title="PSA 9 Charizard Cert: 12345678")
    assert scraper._extract_cert_number(listing) == "12345678"


def test_extract_cert_number_returns_cert_from_psa_prefix() -> None:
    """
    Standard "PSA cert NNN" format must also be extracted (Req 3.2).
    """
    scraper = _make_scraper()
    listing = _make_listing(title="PSA cert 98765432 Pikachu Holo")
    assert scraper._extract_cert_number(listing) == "98765432"


def test_extract_cert_number_case_insensitive() -> None:
    """
    Pattern must be case-insensitive — "CERT: 1234567" and "cert: 1234567" both match.
    """
    scraper = _make_scraper()
    listing = _make_listing(title="CERT: 1234567 PSA 10 Mewtwo")
    assert scraper._extract_cert_number(listing) == "1234567"


# ---------------------------------------------------------------------------
# 11.3c — _fetch_listings returns [] on HTTP error (no exception propagates)
#
# Why test fail-open behaviour?
# A single Card Ladder page returning a 5xx or a network timeout must not
# abort the entire grade's collection run. The base class pagination loop
# treats [] as "no more pages" and moves on (Req 3.6).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_listings_returns_empty_on_transport_error() -> None:
    """
    httpx.TransportError must be caught, a WARNING logged, and [] returned.
    No exception should propagate to the caller (Req 3.6).
    """
    scraper = _make_scraper()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=httpx.TransportError("connection reset"))

    with patch("data_pipeline.scrapers.cardladder.httpx.AsyncClient", return_value=mock_client):
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

    with patch("data_pipeline.scrapers.cardladder.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            result = await scraper._fetch_listings(grade=9, page=1)

    assert result == []


# ---------------------------------------------------------------------------
# 11.3d — Crawl delay domain is www.cardladder.com
#
# Why verify the domain string explicitly?
# BaseScraper._acquire_crawl_token routes to settings.cardladder_crawl_delay
# only when "cardladder.com" is in the domain string. Passing the wrong domain
# (e.g., "cardladder.com" without "www.") would still work, but passing an
# unrelated domain would silently apply the default 1.0s delay instead of the
# required 3.0s (Req 3.4). This test pins the exact domain used.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_listings_acquires_crawl_token_for_cardladder_domain() -> None:
    """
    _acquire_crawl_token must be called with "www.cardladder.com" to ensure
    the 3-second crawl delay is applied (Req 3.4).
    """
    scraper = _make_scraper()

    empty_html = "<html><body></body></html>"
    mock_response = MagicMock()
    mock_response.text = empty_html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("data_pipeline.scrapers.cardladder.httpx.AsyncClient", return_value=mock_client):
        with patch.object(
            scraper, "_acquire_crawl_token", new_callable=AsyncMock
        ) as mock_token:
            await scraper._fetch_listings(grade=9, page=1)

    mock_token.assert_called_once_with("www.cardladder.com")


# ---------------------------------------------------------------------------
# 11.3e — HTML parsing: record without image is skipped
#
# Why test image-less records?
# We cannot train on a record with no slab photo. Silently including a record
# with a None image_url would cause a DownloadError downstream and waste a
# PSA API quota call (Req 3.1).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_listings_skips_record_without_image() -> None:
    """
    Sale records with no image element must be skipped (Req 3.1).
    """
    scraper = _make_scraper()

    # Record has a title and link but no .sale-record__image element.
    html = """
    <html><body>
      <div class="sale-record">
        <span class="sale-record__title">PSA 9 Charizard Cert: 12345678</span>
        <a class="sale-record__link" href="https://www.cardladder.com/sales/12345678">link</a>
      </div>
    </body></html>
    """

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("data_pipeline.scrapers.cardladder.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            listings = await scraper._fetch_listings(grade=9, page=1)

    assert listings == []


@pytest.mark.asyncio
async def test_fetch_listings_skips_record_with_empty_img_src() -> None:
    """
    Records where the img element has no src or data-src must be skipped.
    """
    scraper = _make_scraper()

    html = """
    <html><body>
      <div class="sale-record">
        <span class="sale-record__title">PSA 9 Charizard Cert: 12345678</span>
        <div class="sale-record__image"><img /></div>
        <a class="sale-record__link" href="https://www.cardladder.com/sales/12345678">link</a>
      </div>
    </body></html>
    """

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("data_pipeline.scrapers.cardladder.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            listings = await scraper._fetch_listings(grade=9, page=1)

    assert listings == []


# ---------------------------------------------------------------------------
# 11.3f — HTML parsing: valid record is parsed correctly
#
# Why test the happy path with the canonical fixture?
# This is the primary contract test — it verifies that the scraper correctly
# maps Card Ladder's HTML structure to a RawListing with the right field values.
# A regression here means zero records are collected from Card Ladder.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_listings_parses_valid_record_correctly() -> None:
    """
    A well-formed sale record must be parsed into a RawListing with correct
    title, image_url, listing_url, and source="cardladder" (Req 3.1).
    """
    scraper = _make_scraper()

    # Canonical Card Ladder HTML fixture from the task spec.
    html = """
    <html><body>
      <div class="sale-record">
        <span class="sale-record__title">PSA 9 Charizard Cert: 12345678</span>
        <div class="sale-record__image"><img src="https://cdn.cardladder.com/img/12345678.jpg"/></div>
        <a class="sale-record__link" href="https://www.cardladder.com/sales/12345678">link</a>
      </div>
    </body></html>
    """

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("data_pipeline.scrapers.cardladder.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            listings = await scraper._fetch_listings(grade=9, page=1)

    assert len(listings) == 1
    listing = listings[0]
    assert listing.title == "PSA 9 Charizard Cert: 12345678"
    assert listing.image_url == "https://cdn.cardladder.com/img/12345678.jpg"
    assert listing.listing_url == "https://www.cardladder.com/sales/12345678"
    assert listing.source == "cardladder"
    assert listing.raw_grade == 9


@pytest.mark.asyncio
async def test_fetch_listings_uses_data_src_fallback() -> None:
    """
    Card Ladder lazy-loads images: the real URL may be in data-src before JS
    hydration. _fetch_listings must fall back to data-src when src is absent.
    """
    scraper = _make_scraper()

    html = """
    <html><body>
      <div class="sale-record">
        <span class="sale-record__title">PSA 9 Charizard Cert: 12345678</span>
        <div class="sale-record__image">
          <img data-src="https://cdn.cardladder.com/img/12345678.jpg"/>
        </div>
        <a class="sale-record__link" href="https://www.cardladder.com/sales/12345678">link</a>
      </div>
    </body></html>
    """

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("data_pipeline.scrapers.cardladder.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            listings = await scraper._fetch_listings(grade=9, page=1)

    assert len(listings) == 1
    assert listings[0].image_url == "https://cdn.cardladder.com/img/12345678.jpg"


@pytest.mark.asyncio
async def test_fetch_listings_falls_back_to_page_url_when_no_link() -> None:
    """
    When .sale-record__link is absent, listing_url should fall back to the page URL.
    """
    scraper = _make_scraper()

    html = """
    <html><body>
      <div class="sale-record">
        <span class="sale-record__title">PSA 9 Charizard Cert: 12345678</span>
        <div class="sale-record__image"><img src="https://cdn.cardladder.com/img/12345678.jpg"/></div>
      </div>
    </body></html>
    """

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("data_pipeline.scrapers.cardladder.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            listings = await scraper._fetch_listings(grade=9, page=1)

    assert len(listings) == 1
    # listing_url falls back to the page URL when no anchor is present
    assert "cardladder.com" in listings[0].listing_url


@pytest.mark.asyncio
async def test_fetch_listings_returns_empty_on_no_records() -> None:
    """
    When a page contains no sale-record elements, _fetch_listings returns [].
    The base class pagination loop uses this as the stop signal (Req 3.7).
    """
    scraper = _make_scraper()

    html = "<html><body></body></html>"

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("data_pipeline.scrapers.cardladder.httpx.AsyncClient", return_value=mock_client):
        with patch.object(scraper, "_acquire_crawl_token", new_callable=AsyncMock):
            result = await scraper._fetch_listings(grade=9, page=99)

    assert result == []
