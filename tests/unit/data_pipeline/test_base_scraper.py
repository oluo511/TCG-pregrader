# Feature: training-data-pipeline
# Tests: 9.1 — BaseScraper unit tests (Requirements 8.1, 8.2, 8.3, 8.4)
#
# Why a ConcreteScraper fixture?
# BaseScraper is abstract — we need a minimal concrete subclass to instantiate
# it. ConcreteScraper returns empty listings and None cert numbers, which is
# sufficient for testing the base-class behaviour in isolation.
#
# Why patch httpx.AsyncClient rather than the method?
# _check_robots creates its own AsyncClient internally. Patching at the class
# level intercepts the constructor so we control the response without touching
# the method's internal structure.

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from data_pipeline.config import PipelineSettings
from data_pipeline.deduplicator import Deduplicator
from data_pipeline.downloader import ImageDownloader
from data_pipeline.models import RawListing, ScrapedRecord
from data_pipeline.psa_client import PSAClient
from data_pipeline.scrapers.base import BaseScraper


# ---------------------------------------------------------------------------
# Concrete subclass — minimal implementation to allow instantiation
# ---------------------------------------------------------------------------

class ConcreteScraper(BaseScraper):
    async def _fetch_listings(self, grade: int, page: int) -> list[RawListing]:
        return []

    def _extract_cert_number(self, listing: RawListing) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

ROBOTS_TXT = "\n".join([
    "User-agent: *",
    "Disallow: /search",
    "Allow: /",
])


def _make_scraper() -> ConcreteScraper:
    """Build a ConcreteScraper with minimal mocked dependencies."""
    settings = PipelineSettings(psa_api_token=SecretStr("test-token"))
    return ConcreteScraper(
        settings=settings,
        psa_client=MagicMock(spec=PSAClient),
        deduplicator=MagicMock(spec=Deduplicator),
        downloader=MagicMock(spec=ImageDownloader),
    )


def _make_robots_response(body: str, status: int = 200) -> MagicMock:
    """Return a mock httpx.Response for a robots.txt fetch."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = body
    resp.raise_for_status = MagicMock()  # no-op for 200
    return resp


# ---------------------------------------------------------------------------
# 9.1a — _check_robots returns False for disallowed path, True for allowed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_robots_disallowed_path_returns_false() -> None:
    """
    /search is explicitly disallowed in the robots.txt fixture.
    _check_robots must return False so the scraper skips that URL (Req 8.4).
    """
    scraper = _make_scraper()

    mock_response = _make_robots_response(ROBOTS_TXT)
    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("data_pipeline.scrapers.base.httpx.AsyncClient", return_value=mock_client_instance):
        result = await scraper._check_robots("https://www.example.com/search/results")

    assert result is False


@pytest.mark.asyncio
async def test_check_robots_allowed_path_returns_true() -> None:
    """
    /products is not disallowed — _check_robots must return True (Req 8.4).
    """
    scraper = _make_scraper()

    mock_response = _make_robots_response(ROBOTS_TXT)
    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("data_pipeline.scrapers.base.httpx.AsyncClient", return_value=mock_client_instance):
        result = await scraper._check_robots("https://www.example.com/products")

    assert result is True


# ---------------------------------------------------------------------------
# 9.1b — _check_robots fetches robots.txt exactly once per domain (cached)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_robots_fetches_once_per_domain() -> None:
    """
    robots.txt must be fetched exactly once per domain per run regardless of
    how many URLs are checked for that domain (Req 8.4). Subsequent calls must
    use the cached RobotFileParser without making another HTTP request.
    """
    scraper = _make_scraper()

    mock_response = _make_robots_response(ROBOTS_TXT)
    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("data_pipeline.scrapers.base.httpx.AsyncClient", return_value=mock_client_instance):
        await scraper._check_robots("https://www.example.com/products")
        await scraper._check_robots("https://www.example.com/other-page")

    # Two calls to _check_robots for the same domain → only one HTTP GET
    assert mock_client_instance.get.call_count == 1


# ---------------------------------------------------------------------------
# 9.1c — _acquire_crawl_token sleeps for the remaining delay
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acquire_crawl_token_sleeps_remaining_delay() -> None:
    """
    If only 0.5s has elapsed since the last request to a domain that requires
    a 1.0s delay, _acquire_crawl_token must sleep for ~0.5s (Req 8.2).

    We patch time.monotonic to return controlled values:
      - First call (inside _acquire_crawl_token to read last time): returns 100.0
        (simulating the stored last_request_time)
      - Second call (elapsed calculation): returns 100.5 (0.5s elapsed)
      - Third call (update after sleep): returns 101.0

    The domain is not eBay or Card Ladder so the default 1.0s delay applies.
    """
    scraper = _make_scraper()
    domain = "www.somesite.com"

    # Pre-seed the last request time so elapsed can be calculated.
    scraper._last_request_time[domain] = 100.0

    # monotonic sequence: first read (elapsed calc) → 100.5, post-sleep update → 101.0
    monotonic_values = iter([100.5, 101.0])

    with patch("data_pipeline.scrapers.base.time.monotonic", side_effect=monotonic_values):
        with patch("data_pipeline.scrapers.base.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await scraper._acquire_crawl_token(domain)

    # elapsed = 100.5 - 100.0 = 0.5; delay = 1.0; sleep = 1.0 - 0.5 = 0.5
    mock_sleep.assert_called_once()
    sleep_arg = mock_sleep.call_args[0][0]
    assert abs(sleep_arg - 0.5) < 1e-9


@pytest.mark.asyncio
async def test_acquire_crawl_token_no_sleep_when_delay_elapsed() -> None:
    """
    If enough time has already elapsed, _acquire_crawl_token must NOT sleep.
    This confirms the token bucket doesn't add unnecessary latency (Req 8.2).
    """
    scraper = _make_scraper()
    domain = "www.somesite.com"
    scraper._last_request_time[domain] = 100.0

    # 2.0s elapsed — more than the 1.0s default delay
    monotonic_values = iter([102.0, 102.0])

    with patch("data_pipeline.scrapers.base.time.monotonic", side_effect=monotonic_values):
        with patch("data_pipeline.scrapers.base.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await scraper._acquire_crawl_token(domain)

    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# 9.1d — _check_robots fail-open on fetch failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_robots_fail_open_on_fetch_error() -> None:
    """
    If the robots.txt fetch raises an exception, _check_robots must return True
    (fail-open) and log a WARNING rather than propagating the error (Req 8.4).
    """
    scraper = _make_scraper()

    mock_client_instance = AsyncMock()
    mock_client_instance.get = AsyncMock(side_effect=Exception("connection refused"))
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("data_pipeline.scrapers.base.httpx.AsyncClient", return_value=mock_client_instance):
        result = await scraper._check_robots("https://www.example.com/products")

    assert result is True


# ---------------------------------------------------------------------------
# 9.1e — eBay and Card Ladder domains use their configured crawl delays
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acquire_crawl_token_uses_ebay_delay() -> None:
    """eBay domain must use settings.ebay_crawl_delay, not the 1.0s default."""
    scraper = _make_scraper()
    domain = "www.ebay.com"
    expected_delay = scraper._settings.ebay_crawl_delay  # 1.0 by default in settings

    scraper._last_request_time[domain] = 100.0
    # elapsed = 0.1s — less than ebay_crawl_delay
    monotonic_values = iter([100.1, 101.0])

    with patch("data_pipeline.scrapers.base.time.monotonic", side_effect=monotonic_values):
        with patch("data_pipeline.scrapers.base.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await scraper._acquire_crawl_token(domain)

    expected_sleep = expected_delay - 0.1
    sleep_arg = mock_sleep.call_args[0][0]
    assert abs(sleep_arg - expected_sleep) < 1e-9


@pytest.mark.asyncio
async def test_acquire_crawl_token_uses_cardladder_delay() -> None:
    """Card Ladder domain must use settings.cardladder_crawl_delay (3.0s default)."""
    scraper = _make_scraper()
    domain = "www.cardladder.com"
    expected_delay = scraper._settings.cardladder_crawl_delay  # 3.0 by default

    scraper._last_request_time[domain] = 100.0
    # elapsed = 0.5s — less than cardladder_crawl_delay
    monotonic_values = iter([100.5, 103.0])

    with patch("data_pipeline.scrapers.base.time.monotonic", side_effect=monotonic_values):
        with patch("data_pipeline.scrapers.base.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await scraper._acquire_crawl_token(domain)

    expected_sleep = expected_delay - 0.5
    sleep_arg = mock_sleep.call_args[0][0]
    assert abs(sleep_arg - expected_sleep) < 1e-9
