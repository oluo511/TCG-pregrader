"""
EbayAPIClient — eBay Browse API client using OAuth 2.0 client credentials flow.

Why replace HTML scraping with the official API?
eBay's bot detection (Akamai/PerimeterX) increasingly blocks headless HTTP
clients, producing 403/CAPTCHA responses that silently return zero listings.
The Browse API is the sanctioned, stable interface — it returns structured JSON,
has predictable pagination, and won't get blocked by bot detection.

Architecture: OAuth client credentials → cached app token → Browse API search.
- Token is cached in-memory with a 60-second expiry buffer to avoid clock-skew
  edge cases where a token expires mid-request.
- No persistent token storage: app tokens are cheap to re-fetch (one POST) and
  storing them on disk introduces secret-management complexity for minimal gain.
- httpx.AsyncClient is created per-call (not shared) to avoid connection-pool
  state leaking between token fetches and search calls.
"""

import base64
import time
from typing import Any

import httpx
import structlog

from data_pipeline.config import PipelineSettings
from data_pipeline.models import RawListing

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# eBay OAuth 2.0 token endpoint (client credentials grant)
# Scope grants read-only access to the Browse API — the minimum required.
# ---------------------------------------------------------------------------
_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
_BROWSE_SCOPE = "https://api.ebay.com/oauth/api_scope"

# ---------------------------------------------------------------------------
# Browse API search endpoint
# v1 is the current stable version; v2 is in beta and not yet GA.
# ---------------------------------------------------------------------------
_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"


class EbayAPIClient:
    """
    Thin async client for the eBay Browse API.

    Handles OAuth token lifecycle (fetch + cache) and translates Browse API
    item summaries into RawListing objects consumed by EbayScraper.

    Thread-safety note: token caching uses simple attribute assignment.
    This is safe for asyncio (single-threaded event loop) but NOT for
    multi-threaded use. If this ever runs in a thread pool, add a lock.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        self._settings = settings
        # Cached token state — None means "not yet fetched"
        self._token: str | None = None
        # Unix timestamp after which the cached token must be refreshed
        self._token_expires_at: float = 0.0

    async def _get_app_token(self) -> str:
        """
        Fetch a fresh OAuth 2.0 app token via the client credentials grant.

        Why Basic auth over form-encoded credentials?
        eBay's token endpoint follows RFC 6749 §2.3.1: client credentials are
        sent as a Base64-encoded "client_id:client_secret" Authorization header.
        Sending them in the request body is also valid per the spec but eBay's
        docs explicitly show the header approach — we follow the official example.

        The 60-second buffer on expiry prevents a race where a token is valid
        when we check but expires before the downstream Browse API call completes.
        """
        client_id = self._settings.ebay_client_id
        # SecretStr.get_secret_value() is the only way to read the raw secret —
        # this is intentional: it makes secret access explicit and grep-able.
        client_secret = self._settings.ebay_client_secret.get_secret_value()

        # Build Basic auth header: Base64("client_id:client_secret")
        credentials = f"{client_id}:{client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        body = (
            "grant_type=client_credentials"
            f"&scope={_BROWSE_SCOPE.replace(':', '%3A').replace('/', '%2F')}"
        )

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(_TOKEN_URL, headers=headers, content=body)
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        token: str = data["access_token"]
        expires_in: int = data.get("expires_in", 7200)

        # Cache with 60s buffer so we never use a token that's about to expire
        self._token = token
        self._token_expires_at = time.time() + expires_in - 60

        logger.info(
            "ebay_token_fetched",
            expires_in=expires_in,
            expires_at=self._token_expires_at,
        )
        return token

    async def _ensure_token(self) -> str:
        """
        Return a valid cached token, refreshing it if expired or absent.

        Why not use a lock here?
        In an asyncio context there's no concurrent access to this method —
        the event loop is single-threaded. If we ever move to a thread pool
        executor, this needs an asyncio.Lock to prevent duplicate token fetches.
        """
        if self._token is not None and time.time() < self._token_expires_at:
            return self._token
        return await self._get_app_token()

    async def search_listings(
        self,
        grade: int,
        page: int,
        limit: int = 50,
    ) -> list[RawListing]:
        """
        Search eBay for PSA-graded Pokémon card listings via the Browse API.

        Why FIXED_PRICE|AUCTION filter?
        We want both buy-it-now and auction listings to maximise dataset size.
        The pipe-separated syntax is eBay's Browse API filter format for OR
        conditions within a single filter field.

        Why offset-based pagination over cursor?
        The Browse API supports both. Offset is simpler to reason about and
        maps directly to the page/limit model used by BaseScraper. Cursor-based
        pagination would require threading state through the scraper loop.

        Returns [] on any HTTP error or when itemSummaries is absent (signals
        end of pagination to the caller).
        """
        token = await self._ensure_token()

        params: dict[str, str | int] = {
            "q": f"PSA {grade} pokemon",
            "filter": "buyingOptions:{FIXED_PRICE|AUCTION}",
            "limit": limit,
            "offset": (page - 1) * limit,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(_SEARCH_URL, params=params, headers=headers)
                response.raise_for_status()
                data: dict[str, Any] = response.json()
        except httpx.HTTPError as exc:
            logger.warning(
                "ebay_api_search_failed",
                grade=grade,
                page=page,
                error=str(exc),
            )
            return []

        # itemSummaries is absent when there are no more results — this is the
        # Browse API's pagination stop signal (not an error condition).
        items: list[dict[str, Any]] = data.get("itemSummaries", [])
        if not items:
            return []

        listings: list[RawListing] = []
        for item in items:
            image_url: str = item.get("image", {}).get("imageUrl", "")

            # Skip items with no slab photo — we cannot train on imageless records.
            if not image_url:
                logger.debug(
                    "ebay_api_listing_no_image",
                    item_id=item.get("itemId"),
                    title=item.get("title", ""),
                )
                continue

            listings.append(
                RawListing(
                    source="ebay",
                    listing_url=item["itemWebUrl"],
                    image_url=image_url,
                    title=item["title"],
                    raw_grade=grade,
                )
            )

        return listings
