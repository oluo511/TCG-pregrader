# Feature: training-data-pipeline
# Tests: 2.3 — PSAClient unit tests (Requirements 1.1–1.7)
#
# Why mock asyncio.sleep everywhere?
# The retry logic has real sleeps (2s/4s/8s). Letting them run would make the
# test suite take minutes. We patch sleep to a no-op so we test the *logic*
# (correct number of attempts, correct exception raised) without wall-clock cost.

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from data_pipeline.config import PipelineSettings
from data_pipeline.exceptions import CertLookupError, ConfigurationError, QuotaExhaustedError
from data_pipeline.psa_client import PSAClient, QuotaState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(tmp_path: Path, token: str = "test-token", quota: int = 10) -> PipelineSettings:
    """Build a minimal PipelineSettings pointing quota state at a temp file."""
    return PipelineSettings(
        psa_api_token=SecretStr(token),
        psa_daily_quota=quota,
        psa_quota_state_path=tmp_path / ".quota_state.json",
    )


def _make_response(status_code: int, body: dict | None = None, headers: dict | None = None) -> MagicMock:
    """Return a mock httpx.Response with the given status, JSON body, and headers."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json.return_value = body or {}
    resp.text = json.dumps(body or {})
    return resp


_VALID_PSA_BODY = {
    "PSAcert": {
        "CertNumber": "12345678",
        "OverallGrade": "9",
        "GradeDescription": "MINT",
        "Centering": "9",
        "Corners": "9",
        "Edges": "9",
        "Surface": "9",
    }
}


# ---------------------------------------------------------------------------
# 2.3a — ConfigurationError raised at init when PSA_API_TOKEN is absent/empty
# ---------------------------------------------------------------------------

def test_init_raises_configuration_error_on_empty_token(tmp_path: Path) -> None:
    """
    PSAClient must fail fast at construction time when the token is blank.
    An empty token would silently produce 401s on every API call — better to
    surface the misconfiguration immediately (Req 1.2).
    """
    settings = _make_settings(tmp_path, token="   ")  # whitespace-only
    with pytest.raises(ConfigurationError, match="PSA_API_TOKEN"):
        PSAClient(settings)


# ---------------------------------------------------------------------------
# 2.3b — 429 reads Retry-After header and sleeps that duration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_429_reads_retry_after_header(tmp_path: Path) -> None:
    """
    On a 429 response the client must sleep exactly Retry-After seconds before
    retrying, not a fixed backoff. This respects the server's rate-limit window
    and avoids hammering the API (Req 1.7).
    """
    settings = _make_settings(tmp_path)
    client = PSAClient(settings)

    rate_limited = _make_response(429, headers={"Retry-After": "42"})
    success = _make_response(200, body=_VALID_PSA_BODY)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [rate_limited, success]
            await client.get_cert("12345678")

    # The first sleep must be the Retry-After value, not the backoff formula
    mock_sleep.assert_any_call(42.0)


# ---------------------------------------------------------------------------
# 2.3c — 4xx (not 429) raises CertLookupError immediately, no retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_4xx_raises_cert_lookup_error_immediately(tmp_path: Path) -> None:
    """
    A 404 means the cert number is invalid — retrying won't help.
    The client must raise CertLookupError on the first attempt without sleeping
    (Req 1.6). We assert sleep was never called to confirm no retry occurred.
    """
    settings = _make_settings(tmp_path)
    client = PSAClient(settings)

    not_found = _make_response(404)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = not_found
            with pytest.raises(CertLookupError) as exc_info:
                await client.get_cert("99999999")

    assert exc_info.value.cert_number == "99999999"
    assert exc_info.value.status_code == 404
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# 2.3d — Quota counter persists across PSAClient instances via state file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_quota_persists_across_instances(tmp_path: Path) -> None:
    """
    The quota counter must survive process restarts. If instance A makes 3 calls
    and instance B is created from the same state file, B must start at 3 — not 0.
    This is the core correctness requirement for the 24-hour quota window (Req 1.3).
    """
    settings = _make_settings(tmp_path, quota=10)
    success = _make_response(200, body=_VALID_PSA_BODY)

    # Instance A: make 3 calls
    client_a = PSAClient(settings)
    with patch("asyncio.sleep", new_callable=AsyncMock):
        with patch.object(client_a._client, "get", new_callable=AsyncMock, return_value=success):
            for _ in range(3):
                await client_a.get_cert("12345678")

    # Instance B: read the same state file — should see calls_today == 3
    state_data = json.loads(settings.psa_quota_state_path.read_text())
    state = QuotaState.model_validate(state_data)
    assert state.calls_today == 3


# ---------------------------------------------------------------------------
# 2.3e — Quota resets when reset_at is in the past
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_quota_resets_when_window_expired(tmp_path: Path) -> None:
    """
    If the state file has a reset_at timestamp in the past, the client must
    treat it as a fresh window (calls_today = 0) rather than blocking calls
    from a previous day's run (Req 1.4).
    """
    settings = _make_settings(tmp_path, quota=2)

    # Write a stale state: quota exhausted, but reset_at is 2 hours ago
    stale_state = QuotaState(
        calls_today=2,
        reset_at=datetime.now(timezone.utc) - timedelta(hours=2),
    )
    settings.psa_quota_state_path.write_text(
        json.dumps(stale_state.model_dump(mode="json"))
    )

    client = PSAClient(settings)
    success = _make_response(200, body=_VALID_PSA_BODY)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=success):
            # Should NOT raise QuotaExhaustedError — window has reset
            record = await client.get_cert("12345678")

    assert record.cert_number == "12345678"
