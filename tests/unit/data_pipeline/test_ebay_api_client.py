"""
Unit tests for EbayAPIClient (data_pipeline/scrapers/ebay_api.py).

Why these tests?
- Token fetch correctness: the Basic auth header is the only credential sent to
  eBay's OAuth endpoint. A wrong encoding silently produces 401s on every call.
- Token caching: without caching, every search_listings call would hit the token
  endpoint — burning rate limits and adding ~200ms latency per page.
- search_listings contract: the Browse API response shape is the integration
  boundary. Tests pin the mapping from API JSON → RawListing so a schema change
  in the API is caught immediately.
- Edge cases (no image, absent itemSummaries): these are the two most common
  "silent data loss" paths — items that look valid but produce no RawListing.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from data_pipeline.config import PipelineSettings
from data_pipeline.scrapers.ebay_api import EbayAPIClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(
    client_id: str = "test-app-id",
    client_secret: str = "test-secret",
) -> PipelineSettings:
    """Build minimal PipelineSettings with eBay credentials pre-filled."""
    return PipelineSettings(
        psa_api_token=SecretStr("psa-token"),
        ebay_client_id=client_id,
        ebay_client_secret=SecretStr(client_secret),
    )


def _make_token_response(
    access_token: str = "test-access-token",
    expires_in: int = 7200,
) -> MagicMock:
    """Mock httpx.Response for the OAuth token endpoint."""
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "access_token": access_token,
        "expires_in": expires_in,
        "token_type": "Application Access Token",
    }
    return resp


def _make_search_response(items: list[dict] | None = None) -> MagicMock:
    """Mock httpx.Response for the Browse API search endpoint."""
    resp = MagicMock(spec=httpx.Response)
    resp.raise_for_status = MagicMock()
    body: dict = {}
    if items is not None:
        body["itemSummaries"] = items
    resp.json.return_value = body
    return resp


def _sample_item(
    title: str = "PSA 9 Charizard Pokemon",
    image_url: str = "https://i.ebayimg.com/images/g/abc/s-l500.jpg",
    item_url: str = "https://www.ebay.com/itm/123456789",
) -> dict:
    """Minimal Browse API item summary dict."""
    return {
        "itemId": "v1|123456789|0",
        "title": title,
        "image": {"imageUrl": image_url},
        "itemWebUrl": item_url,
    }


# ---------------------------------------------------------------------------
# _get_app_token — correct POST with Basic auth header
#
# Why test the auth header encoding?
# The Basic auth value is Base64("client_id:client_secret"). A wrong separator
# (e.g., space instead of colon) or wrong encoding silently produces 401s.
# We assert the exact header value so any encoding regression is caught here.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_app_token_posts_correct_basic_auth_header() -> None:
    """
    _get_app_token must POST to the token URL with a correctly encoded
    Basic auth header: Base64("client_id:client_secret").
    """
    import base64

    settings = _make_settings(client_id="my-app-id", client_secret="my-secret")
    client = EbayAPIClient(settings)

    captured_headers: list[dict] = []
    token_resp = _make_token_response()

    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    async def _capture_post(url: str, headers: dict, content: str) -> MagicMock:
        captured_headers.append(headers)
        return token_resp

    mock_http.post = _capture_post

    with patch("data_pipeline.scrapers.ebay_api.httpx.AsyncClient", return_value=mock_http):
        token = await client._get_app_token()

    assert token == "test-access-token"
    assert len(captured_headers) == 1

    expected_encoded = base64.b64encode(b"my-app-id:my-secret").decode()
    assert captured_headers[0]["Authorization"] == f"Basic {expected_encoded}"


@pytest.mark.asyncio
async def test_get_app_token_caches_token_and_expiry() -> None:
    """
    After _get_app_token succeeds, self._token and self._token_expires_at
    must be set so subsequent calls within the window skip the POST.
    """
    settings = _make_settings()
    client = EbayAPIClient(settings)

    token_resp = _make_token_response(access_token="cached-token", expires_in=3600)
    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.post = AsyncMock(return_value=token_resp)

    with patch("data_pipeline.scrapers.ebay_api.httpx.AsyncClient", return_value=mock_http):
        await client._get_app_token()

    assert client._token == "cached-token"
    # expires_at should be roughly now + 3600 - 60 = now + 3540
    assert client._token_expires_at > time.time() + 3500


# ---------------------------------------------------------------------------
# _ensure_token — caching behaviour
#
# Why test that the token is NOT re-fetched?
# The eBay token endpoint has its own rate limit. Fetching a new token on every
# search_listings call would exhaust it quickly in a high-volume pipeline run.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_token_returns_cached_token_without_refetch() -> None:
    """
    _ensure_token must return the cached token without calling _get_app_token
    when the token is still valid (time.time() < _token_expires_at).
    """
    settings = _make_settings()
    client = EbayAPIClient(settings)

    # Pre-populate cache with a token that expires far in the future
    client._token = "already-valid-token"
    client._token_expires_at = time.time() + 3600

    with patch.object(client, "_get_app_token", new_callable=AsyncMock) as mock_fetch:
        result = await client._ensure_token()

    assert result == "already-valid-token"
    mock_fetch.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_token_refetches_when_expired() -> None:
    """
    _ensure_token must call _get_app_token when the cached token has expired.
    """
    settings = _make_settings()
    client = EbayAPIClient(settings)

    # Simulate an expired token
    client._token = "expired-token"
    client._token_expires_at = time.time() - 1  # already in the past

    with patch.object(
        client, "_get_app_token", new_callable=AsyncMock, return_value="fresh-token"
    ) as mock_fetch:
        result = await client._ensure_token()

    assert result == "fresh-token"
    mock_fetch.assert_called_once()


# ---------------------------------------------------------------------------
# search_listings — correct RawListing objects from mock response
#
# Why assert each field individually?
# RawListing is the contract between the scraper and the rest of the pipeline.
# A wrong field mapping (e.g., swapping image_url and listing_url) would
# silently produce bad training data without any type error.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_listings_returns_correct_raw_listings() -> None:
    """
    search_listings must map Browse API item summaries to RawListing objects
    with correct source, listing_url, image_url, title, and raw_grade.
    """
    settings = _make_settings()
    client = EbayAPIClient(settings)

    items = [
        _sample_item(
            title="PSA 9 Charizard Pokemon",
            image_url="https://i.ebayimg.com/images/g/abc/s-l500.jpg",
            item_url="https://www.ebay.com/itm/111",
        ),
        _sample_item(
            title="PSA 9 Pikachu Pokemon",
            image_url="https://i.ebayimg.com/images/g/xyz/s-l500.jpg",
            item_url="https://www.ebay.com/itm/222",
        ),
    ]
    search_resp = _make_search_response(items=items)

    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.get = AsyncMock(return_value=search_resp)

    with patch.object(client, "_ensure_token", new_callable=AsyncMock, return_value="tok"):
        with patch("data_pipeline.scrapers.ebay_api.httpx.AsyncClient", return_value=mock_http):
            listings = await client.search_listings(grade=9, page=1)

    assert len(listings) == 2

    first = listings[0]
    assert first.source == "ebay"
    assert first.title == "PSA 9 Charizard Pokemon"
    assert first.image_url == "https://i.ebayimg.com/images/g/abc/s-l500.jpg"
    assert first.listing_url == "https://www.ebay.com/itm/111"
    assert first.raw_grade == 9


@pytest.mark.asyncio
async def test_search_listings_uses_bearer_token_in_header() -> None:
    """
    search_listings must pass the token as 'Authorization: Bearer <token>'.
    A missing or malformed auth header produces 401s silently.
    """
    settings = _make_settings()
    client = EbayAPIClient(settings)

    captured_headers: list[dict] = []
    search_resp = _make_search_response(items=[_sample_item()])

    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    async def _capture_get(url: str, params: dict, headers: dict) -> MagicMock:
        captured_headers.append(headers)
        return search_resp

    mock_http.get = _capture_get

    with patch.object(client, "_ensure_token", new_callable=AsyncMock, return_value="my-bearer"):
        with patch("data_pipeline.scrapers.ebay_api.httpx.AsyncClient", return_value=mock_http):
            await client.search_listings(grade=9, page=1)

    assert captured_headers[0]["Authorization"] == "Bearer my-bearer"


# ---------------------------------------------------------------------------
# search_listings — absent itemSummaries → []
#
# Why test this specifically?
# The Browse API omits itemSummaries (rather than returning an empty list) when
# there are no more results. Treating this as an error would raise an exception;
# treating it as [] correctly signals "end of pagination" to the caller.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_listings_returns_empty_when_item_summaries_absent() -> None:
    """
    When the Browse API response has no 'itemSummaries' key, search_listings
    must return [] — this is the pagination stop signal, not an error.
    """
    settings = _make_settings()
    client = EbayAPIClient(settings)

    # _make_search_response(items=None) produces {} — no itemSummaries key
    search_resp = _make_search_response(items=None)

    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.get = AsyncMock(return_value=search_resp)

    with patch.object(client, "_ensure_token", new_callable=AsyncMock, return_value="tok"):
        with patch("data_pipeline.scrapers.ebay_api.httpx.AsyncClient", return_value=mock_http):
            result = await client.search_listings(grade=9, page=99)

    assert result == []


@pytest.mark.asyncio
async def test_search_listings_returns_empty_when_item_summaries_is_empty_list() -> None:
    """
    An empty itemSummaries list must also return [] (last page of results).
    """
    settings = _make_settings()
    client = EbayAPIClient(settings)

    search_resp = _make_search_response(items=[])

    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.get = AsyncMock(return_value=search_resp)

    with patch.object(client, "_ensure_token", new_callable=AsyncMock, return_value="tok"):
        with patch("data_pipeline.scrapers.ebay_api.httpx.AsyncClient", return_value=mock_http):
            result = await client.search_listings(grade=9, page=2)

    assert result == []


# ---------------------------------------------------------------------------
# search_listings — items with no image URL are skipped
#
# Why test this?
# An item with no image would produce a RawListing with image_url="" which
# would pass Pydantic validation but fail at download time. Filtering here
# prevents silent bad records from entering the manifest.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_listings_skips_items_with_no_image_url() -> None:
    """
    Items where image.imageUrl is absent or empty must be skipped.
    Items with a valid image URL must still be returned.
    """
    settings = _make_settings()
    client = EbayAPIClient(settings)

    items = [
        # No image key at all
        {
            "itemId": "v1|1|0",
            "title": "PSA 9 No Image Card",
            "itemWebUrl": "https://www.ebay.com/itm/1",
        },
        # image key present but imageUrl is empty string
        {
            "itemId": "v1|2|0",
            "title": "PSA 9 Empty Image Card",
            "image": {"imageUrl": ""},
            "itemWebUrl": "https://www.ebay.com/itm/2",
        },
        # Valid item — should be included
        _sample_item(title="PSA 9 Valid Card"),
    ]
    search_resp = _make_search_response(items=items)

    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.get = AsyncMock(return_value=search_resp)

    with patch.object(client, "_ensure_token", new_callable=AsyncMock, return_value="tok"):
        with patch("data_pipeline.scrapers.ebay_api.httpx.AsyncClient", return_value=mock_http):
            listings = await client.search_listings(grade=9, page=1)

    assert len(listings) == 1
    assert listings[0].title == "PSA 9 Valid Card"


# ---------------------------------------------------------------------------
# search_listings — HTTP error → [] (fail-open)
#
# Why test fail-open?
# A single API error (e.g., 503 during a eBay maintenance window) must not
# abort the entire pipeline run. Returning [] signals "no more pages" to the
# caller, which gracefully moves on to the next grade.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_listings_returns_empty_on_http_error() -> None:
    """
    An httpx.HTTPError during the Browse API call must be caught, a WARNING
    logged, and [] returned. No exception should propagate to the caller.
    """
    settings = _make_settings()
    client = EbayAPIClient(settings)

    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.get = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=MagicMock(),
            response=MagicMock(),
        )
    )

    with patch.object(client, "_ensure_token", new_callable=AsyncMock, return_value="tok"):
        with patch("data_pipeline.scrapers.ebay_api.httpx.AsyncClient", return_value=mock_http):
            result = await client.search_listings(grade=9, page=1)

    assert result == []


# ---------------------------------------------------------------------------
# search_listings — correct pagination offset
#
# Why test offset calculation?
# page=1 → offset=0, page=2 → offset=50. An off-by-one here silently skips
# the first page or double-fetches the last, corrupting dataset coverage.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_listings_correct_offset_for_page() -> None:
    """
    search_listings must compute offset = (page - 1) * limit.
    page=1 → offset=0; page=3 with limit=50 → offset=100.
    """
    settings = _make_settings()
    client = EbayAPIClient(settings)

    captured_params: list[dict] = []
    search_resp = _make_search_response(items=[])

    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)

    async def _capture_get(url: str, params: dict, headers: dict) -> MagicMock:
        captured_params.append(params)
        return search_resp

    mock_http.get = _capture_get

    with patch.object(client, "_ensure_token", new_callable=AsyncMock, return_value="tok"):
        with patch("data_pipeline.scrapers.ebay_api.httpx.AsyncClient", return_value=mock_http):
            await client.search_listings(grade=9, page=3, limit=50)

    assert captured_params[0]["offset"] == 100
    assert captured_params[0]["limit"] == 50
