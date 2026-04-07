"""
PSAClient — async HTTP client for the PSA Public API.

Responsibilities:
  1. Quota enforcement: atomic check+increment via asyncio.Lock prevents TOCTOU
     races when multiple scraper tasks call get_cert() concurrently (Req 1.3, 1.4).
  2. Retry with exponential backoff: 5xx / connection errors → 3 retries at 2s/4s/8s;
     429 → Retry-After header sleep; 4xx (not 429) → immediate CertLookupError (Req 1.5–1.7).
  3. Quota state persistence: JSON file survives process restarts so the 24-hour
     window is enforced across multiple pipeline runs (Req 1.3).

Why asyncio.Lock for quota, not a semaphore?
  A semaphore limits concurrency; a Lock serializes the read-modify-write of the
  quota counter. Without the Lock, two concurrent get_cert() calls could both read
  calls_today=99 (under quota), both increment to 100, and both proceed — making
  101 total calls. The Lock makes the check+increment atomic.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import structlog
from pydantic import BaseModel

from data_pipeline.config import PipelineSettings
from data_pipeline.exceptions import (
    CertLookupError,
    ConfigurationError,
    QuotaExhaustedError,
)
from data_pipeline.models import CertRecord

logger = structlog.get_logger(__name__)


class QuotaState(BaseModel):
    """
    Persisted daily quota counter.

    reset_at is timezone-aware (UTC) so comparisons with datetime.now(UTC) are
    unambiguous across DST transitions and server timezone changes.
    """

    calls_today: int = 0
    reset_at: datetime


def _now_utc() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class PSAClient:
    """
    Async client for the PSA Public API cert lookup endpoint.

    Usage:
        client = PSAClient(settings)
        cert = await client.get_cert("12345678")
    """

    def __init__(self, settings: PipelineSettings) -> None:
        # Fail fast at construction time — no point wiring up the rest of the
        # pipeline if the API token is missing (Req 1.2, 9.2).
        raw_token = settings.psa_api_token.get_secret_value()
        if not raw_token.strip():
            raise ConfigurationError(
                "PSA_API_TOKEN is required but was empty. "
                "Set it in your .env file or environment."
            )

        self._settings = settings
        self._quota_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {raw_token}"},
            timeout=30.0,
        )

    async def get_cert(self, cert_number: str) -> CertRecord:
        """
        Retrieve a verified CertRecord from the PSA API.

        Flow: acquire lock → load quota → maybe reset window → check quota →
              increment + persist → release lock → HTTP GET with retry → parse.

        Raises:
            QuotaExhaustedError: daily quota reached, no HTTP call made.
            CertLookupError: non-retryable 4xx response.
        """
        async with self._quota_lock:
            state = await self._load_quota_state()

            # Reset the window if 24 hours have elapsed since last reset.
            if _now_utc() >= state.reset_at:
                logger.info(
                    "psa_quota_window_reset",
                    previous_calls=state.calls_today,
                    new_reset_at=(_now_utc() + timedelta(hours=24)).isoformat(),
                )
                state = QuotaState(
                    calls_today=0,
                    reset_at=_now_utc() + timedelta(hours=24),
                )

            if state.calls_today >= self._settings.psa_daily_quota:
                logger.warning(
                    "psa_quota_exhausted",
                    calls_today=state.calls_today,
                    quota=self._settings.psa_daily_quota,
                    reset_at=state.reset_at.isoformat(),
                )
                raise QuotaExhaustedError(
                    f"PSA daily quota of {self._settings.psa_daily_quota} calls exhausted. "
                    f"Resets at {state.reset_at.isoformat()}."
                )

            state = QuotaState(
                calls_today=state.calls_today + 1,
                reset_at=state.reset_at,
            )
            await self._persist_quota_state(state)
            logger.debug(
                "psa_quota_incremented",
                calls_today=state.calls_today,
                quota=self._settings.psa_daily_quota,
            )
        # Lock released — now make the actual HTTP call outside the critical section.
        url = f"{self._settings.psa_base_url}{cert_number}"

        async def _do_request() -> httpx.Response:
            return await self._client.get(url)

        response = await self._retry_with_backoff(_do_request, cert_number=cert_number)
        return self._parse_response(response, cert_number)

    async def _load_quota_state(self) -> QuotaState:
        """
        Read quota state from disk. Returns a fresh state (reset_at = now+24h)
        if the state file does not exist yet.
        """
        path: Path = self._settings.psa_quota_state_path
        if not path.exists():
            return QuotaState(reset_at=_now_utc() + timedelta(hours=24))

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return QuotaState.model_validate(data)
        except Exception as exc:
            logger.warning(
                "psa_quota_state_read_error",
                path=str(path),
                error=str(exc),
            )
            # Fail-safe: treat as fresh window rather than blocking all calls.
            return QuotaState(reset_at=_now_utc() + timedelta(hours=24))

    async def _persist_quota_state(self, state: QuotaState) -> None:
        """
        Write quota state to disk as JSON.

        model_dump(mode="json") serialises datetime as ISO-8601 string,
        which round-trips cleanly through model_validate.
        """
        path: Path = self._settings.psa_quota_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(state.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

    async def _retry_with_backoff(
        self,
        fn: Callable[[], Awaitable[httpx.Response]],
        max_retries: int = 3,
        base_delay: float = 2.0,
        cert_number: str = "<unknown>",
    ) -> httpx.Response:
        """
        Execute fn() with exponential backoff retry.

        Retry policy (Req 1.5–1.7):
          - 5xx or httpx.TransportError → retry; delays: base_delay * 2**attempt
            (attempt 0 → 2s, attempt 1 → 4s, attempt 2 → 8s)
          - 429 → read Retry-After header (default 60s), sleep, retry
          - 4xx (not 429) → raise CertLookupError immediately (no retry)
          - Retries exhausted → raise CertLookupError

        The callable pattern (fn: Callable) makes this reusable by ImageDownloader
        without coupling it to cert numbers or URLs.
        """
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = await fn()
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "psa_transport_error_retry",
                        cert_number=cert_number,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
                    continue
                break

            status = response.status_code

            if status == 429:
                retry_after = float(response.headers.get("Retry-After", 60))
                logger.warning(
                    "psa_rate_limited",
                    cert_number=cert_number,
                    attempt=attempt + 1,
                    retry_after=retry_after,
                )
                if attempt < max_retries:
                    await asyncio.sleep(retry_after)
                    continue
                # Exhausted retries on 429 — fall through to raise below.
                raise CertLookupError(cert_number, status)

            if 500 <= status < 600:
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "psa_5xx_retry",
                        cert_number=cert_number,
                        status_code=status,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise CertLookupError(cert_number, status)

            if 400 <= status < 500:
                # Non-retryable client error — raise immediately (Req 1.6).
                logger.error(
                    "psa_4xx_error",
                    cert_number=cert_number,
                    status_code=status,
                )
                raise CertLookupError(cert_number, status)

            # 2xx / 3xx — success.
            return response

        # All retries exhausted via TransportError path.
        raise CertLookupError(cert_number, 0) from last_exc

    def _parse_response(self, response: httpx.Response, cert_number: str) -> CertRecord:
        """
        Parse the PSA API JSON response into a CertRecord.

        Expected shape (Req 1.1):
        {
          "PSAcert": {
            "CertNumber": "12345678",
            "OverallGrade": "10",
            "GradeDescription": "GEM MT",
            "Centering": "9",
            "Corners": "9",
            "Edges": "9",
            "Surface": "9"
          }
        }

        Grades arrive as strings from the API — cast to int/float explicitly.
        """
        try:
            payload = response.json()
            cert_data = payload["PSAcert"]
            return CertRecord(
                cert_number=cert_data["CertNumber"],
                overall_grade=int(cert_data["OverallGrade"]),
                centering=float(cert_data["Centering"]),
                corners=float(cert_data["Corners"]),
                edges=float(cert_data["Edges"]),
                surface=float(cert_data["Surface"]),
                verified=True,
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.error(
                "psa_response_parse_error",
                cert_number=cert_number,
                error=str(exc),
                response_text=response.text[:500],
            )
            raise CertLookupError(cert_number, response.status_code) from exc
